
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func #flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import sys
from pathlib import Path
import numpy as np

# yixing: not using this
def mod_Longchat_Attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cur_prompt_id: int = None, 
    save_attn_dir: str = None, 
    layer_idx: int = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
            Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel
    
    attention_mask: [bsz, q_len]
    """
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bsz, q_len, nh, hd]
    # [bsz, nh, q_len, hd]

    kv_seq_len = key_states.shape[-2]
    assert past_key_value is None, "past_key_value is not supported"

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]
    assert not output_attentions, "output_attentions is not supported"

    # yixing:
    # assert not use_cache, "use_cache is not supported"

    # Flash attention codes from
    # https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py

    # transform the data into the format required by flash attention
    qkv = torch.stack([query_states, key_states, value_states], dim=2) # [bsz, nh, 3, q_len, hd]
    qkv = qkv.transpose(1, 3) # [bsz, q_len, 3, nh, hd]
    # We have disabled _prepare_decoder_attention_mask in LlamaModel
    # the attention_mask should be the same as the key_padding_mask
    key_padding_mask = attention_mask


    if key_padding_mask is None:
        qkv = rearrange(qkv, 'b s ... -> (b s) ...')
        max_s = q_len
        cu_q_lens = torch.arange(0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32,
                                device=qkv.device)
        # flash_attn_qkvpacked_func
        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(output, '(b s) ... -> b s ...', b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, 0.0,
            softmax_scale=None, causal=True
        )
        output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                    indices, bsz, q_len),
                        'b s (h d) -> b s h d', h=nheads)
    return self.o_proj(rearrange(output,
                                    'b s h d -> b s (h d)')), None, None

# class LlamaAttention(nn.Module):
def mod_LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cur_prompt_id: int = None, 
    save_attn_dir: str = None, 
    layer_idx: int = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    # start: yixing
    assert cur_prompt_id is not None
    assert save_attn_dir is not None
    if layer_idx is None:
        print(f'block_idx is None: runnning without saving attn_score.')
    # end: yixing

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


    # torch.Size([1, 40, 3591, 128])
    # torch.Size([1, 40, 3591, 128])

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # start: yixing
    query_states_t = query_states.transpose(1, 2)
    key_states_t = key_states.transpose(1, 2)
    query_states_t = rearrange(query_states_t, '... h d -> ... (h d)')
    key_states_t = rearrange(key_states_t, '... h d -> ... (h d)')
    # attn_weights.shape: (bsz, num_heads, seq_lenth, seq_lenth). seq is the input token_ids.
    # value_states.shape: (bsz, num_heads, seq_lenth, head_dim). seq is the input token_ids.
    '''
    # sum1 = torch.sum(attn_weights, dim = -1)
    # sum2 = torch.sum(attn_weights, dim = [1, 3])
    # sum3 = torch.squeeze(sum2, 0)
    # print(sum1.shape, sum2.shape, sum3.shape, sum4.shape, (sum1.equal(sum2)), (sum3.equal(sum4)))
    # torch.Size([1, 40, 3591]) torch.Size([1, 3591]) torch.Size([3591]) torch.Size([3591]) False True
    '''
    (_, seq_lenth, _) = query_states_t.shape
    save_attn_score = True
    if save_attn_score and (seq_lenth > 1) and (layer_idx is not None):        
        # attn_weights_docu = torch.mean(attn_weights, dim=[0, 1, 2])#3])
        attn_weights_docu = torch.mul(query_states_t, key_states_t)
        del query_states_t
        del key_states_t
        attn_weights_docu = torch.mean(attn_weights_docu, dim = -1)
        # print(torch.min(attn_weights_mean), torch.max(attn_weights_mean))
        # print(attn_weights_mean.shape)
        # sys.exit(0)

        save_attn_dir = f'{save_attn_dir}/prompt-{cur_prompt_id}'
        Path(save_attn_dir).mkdir(exist_ok = True, parents = True)

        attn_weights_docu = attn_weights_docu.detach().cpu().numpy()
        np.save(f'{save_attn_dir}/layer-{layer_idx}.npy', attn_weights_docu)
        # sys.exit(0)
    # end: yixing

    # torch.Size([1, 40, 3591, 128])
    # torch.Size([1, 40, 3591, 128])

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

        
    # yixing-test
    # torch.Size([1, 40, 3591, 128])
    # torch.Size([1, 40, 3591, 128])

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # yixing-test
    # torch.Size([1, 40, 3591, 128])
    # torch.Size([1, 40, 3591, 128])



    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    # print(attn_weights)
    # print(torch.min(attn_weights), torch.max(attn_weights))
    # sys.exit(0)
    # yixing: above: qk, mask. following: softmax

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask


    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)



    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    attn_output = torch.matmul(attn_weights, value_states)


    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


### yixing:
# longchat: transformers.models.llama.modeling_llama.LlamaAttention.forward = forward