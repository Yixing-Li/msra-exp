'''
modified from attention
'''
import math
import warnings
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
from torch import nn
import os
import sys

import numpy as np
from pathlib import Path

# class GroupedQueryAttention(nn.Module):
def mod_attn_forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, is_causal: bool=True, needs_weights: bool=False, cur_prompt_id: int = None, save_attn_dir: str = None, block_idx: int = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    # start: yixing
    assert cur_prompt_id is not None
    assert save_attn_dir is not None
    if block_idx is None:
        print(f'block_idx is None: runnning without saving attn_score.')
    # end: yixing

    qkv = self.Wqkv(x)
    if self.clip_qkv:
        qkv = qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)
    (query, key, value) = qkv.split([self.d_model, self.kv_n_heads * self.head_dim, self.kv_n_heads * self.head_dim], dim=2)
    key_padding_mask = attention_mask
    if self.qk_ln:
        dtype = query.dtype
        query = self.q_ln(query).to(dtype)
        key = self.k_ln(key).to(dtype)

    # yixing-test
    (_, seqlen_q, _) = query.shape
    save_attn_score = True
    if save_attn_score and (seqlen_q > 1) and (block_idx is not None):
        # print(f'{query.shape}, {key.shape}, {value.shape}')
        # torch.Size([1, 4464, 7168]), torch.Size([1, 4464, 7168]), torch.Size([1, 4464, 7168])
        attn_score = torch.mul(query, key)

        # yixing:
        # tried to use the value_sign to make sure attn_score are +. but got worse. (in main_dir/utils_test/output/attn_score-nq_open-docu_30-gold_4-m_3_i/2023_12_20-12_00)
        # so not use value_sign.
        # value_sign = torch.sign(value)

        # attn_score_signed = torch.mul(attn_score, value_sign)

        attn_score_docu = torch.sum(attn_score, dim = -1)

        # print(f'{attn_score.shape}, {attn_score_docu.shape}, ')
        # torch.Size([1, 4464, 7168]), torch.Size([1, 4464]),
        save_attn_dir = f'{save_attn_dir}/prompt-{cur_prompt_id}'
        Path(save_attn_dir).mkdir(exist_ok = True, parents = True)

        attn_score_docu = attn_score_docu.detach().cpu().numpy()
        np.save(f'{save_attn_dir}/block-{block_idx}.npy', attn_score_docu)

    # yixing: we can get qkv here. (or deeper, check the journal.)
    # n_heads = self.n_heads
    # kv_n_heads = self.kv_n_heads if self.kv_n_heads is not None else self.n_heads
    # q = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    # k = rearrange(key, 'b s (h d) -> b s h d', h=kv_n_heads)
    # v = rearrange(value, 'b s (h d) -> b s h d', h=kv_n_heads)
    # (batch, seqlen_q, nheads, d) = q.shape
    # (_, seqlen_k, _, _) = k.shape

    # yixing: self.attn_impl == 'triton', and self.attn_fn = triton_flash_attn_fn

    (context, attn_weights, past_key_value) = self.attn_fn(query, key, value, self.n_heads, self.kv_n_heads, past_key_value=past_key_value, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
    return (self.out_proj(context), attn_weights, past_key_value)

# used
def mod_triton_flash_attn_fn(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, n_heads: int, kv_n_heads: Optional[int]=None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, softmax_scale: Optional[float]=None, attn_bias: Optional[torch.Tensor]=None, key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False, dropout_p: float=0.0, training: bool=False, needs_weights: bool=False, multiquery: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    try:
        from .flash_attn_triton import flash_attn_func
    except:
        _installed = False
        if version.parse(torch.__version__) < version.parse('2.0.0'):
            _installed = True
            try:
                from flash_attn.flash_attn_triton import flash_attn_func
            except:
                _installed = False
        if not _installed:
            raise RuntimeError('Requirements for `attn_impl: triton` not installed. Either (1) have a CUDA-compatible GPU ' + 'and `pip install .[gpu]` if installing from llm-foundry source or ' + '`pip install triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python` ' + 'if installing from pypi, or (2) use torch attn model.attn_config.attn_impl=torch (torch attn_impl will be slow). ' + 'Note: (1) requires you have CMake and PyTorch already installed.')
    check_valid_inputs(query, key, value)
    if multiquery:
        warnings.warn(DeprecationWarning('The direct use of the multiquery arg is deprecated. Setting kv_n_heads=1 automatically. Please set kv_n_heads=1 explicitly to remove this warning.'))
        kv_n_heads = 1
    elif kv_n_heads is None:
        warnings.warn(DeprecationWarning('Not specifying a value for the kv_n_heads arg is deprecated. Setting kv_n_heads=n_heads automatically. Please set kv_n_heads=n_heads explicitly to remove this warning.'))
        kv_n_heads = n_heads
    if past_key_value is not None:
        if len(past_key_value) != 0:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        past_key_value = (key, value)
    if attn_bias is not None:
        _s_q = max(0, attn_bias.size(2) - query.size(1))
        _s_k = max(0, attn_bias.size(3) - key.size(1))
        attn_bias = attn_bias[:, :, _s_q:, _s_k:]
    if dropout_p:
        raise NotImplementedError(f'Dropout not implemented for attn_impl: triton.')
    dropout_p = dropout_p if training else 0.0
    if needs_weights:
        raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
    if key_padding_mask is not None:
        warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=kv_n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=kv_n_heads)

    # yixing-test
    print(f'in mod_triton_flash_attn_fn:{query.shape}, {key.shape}, {value.shape}')
    sys.exit(0)

    if kv_n_heads == 1:
        key = key.repeat(1, 1, n_heads, 1)
        value = value.repeat(1, 1, n_heads, 1)
    elif kv_n_heads < n_heads:
        key = repeat_kv_for_gqa(key, n_heads // kv_n_heads)
        value = repeat_kv_for_gqa(value, n_heads // kv_n_heads)
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None, past_key_value)


# used
# flash_attn_func: from .flash_attn_triton import flash_attn_func
# this is a special func ! TODO
