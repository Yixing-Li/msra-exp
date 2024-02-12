'''
Modified from mpt model.
'''

import math
import warnings
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# from .blocks import MPTBlock

# class MPTForCausalLM(MPTPreTrainedModel):
def mod_MPTForCausalLM_forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.FloatTensor]=None) -> CausalLMOutputWithPast:
    # yixing:
    assert self.cur_prompt_id is not None
    self.transformer.cur_prompt_id = self.cur_prompt_id
    self.transformer.save_attn_dir = self.save_attn_dir

    return_dict = return_dict if return_dict is not None else self.config.return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if inputs_embeds is not None:
        raise NotImplementedError('inputs_embeds has to be None (for hf/peft support).')

        
    # yixing: self.transformer is MPTModel.
    # ( self.transformer: MPTModel = MPTModel(config) )
    outputs = self.transformer(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id, return_dict=return_dict, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache)

    logits = self.transformer.wte(outputs.last_hidden_state.to(self.transformer.wte.weight.device), True)

    if self.logit_scale is not None:
        if self.logit_scale == 0:
            warnings.warn(f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
        logits *= self.logit_scale
    loss = None
    if labels is not None:
        _labels = torch.roll(labels, shifts=-1)
        _labels[:, -1] = -100
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), _labels.to(logits.device).view(-1))
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# class MPTModel(MPTPreTrainedModel):
def mod_MPTModel_forward(self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.FloatTensor]]]=None, attention_mask: Optional[torch.ByteTensor]=None, prefix_mask: Optional[torch.ByteTensor]=None, sequence_id: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, use_cache: Optional[bool]=None, inputs_embeds: Optional[torch.Tensor]=None) -> BaseModelOutputWithPast:
    # yixing
    assert self.cur_prompt_id is not None
    assert self.save_attn_dir is not None

    return_dict = return_dict if return_dict is not None else self.config.return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    if attention_mask is not None:
        attention_mask = attention_mask.bool()
    if prefix_mask is not None:
        prefix_mask = prefix_mask.bool()
    if not return_dict:
        raise NotImplementedError('return_dict False is not implemented yet for MPT')
    if output_attentions:
        if self.attn_impl != 'torch':
            raise NotImplementedError('output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.')
    if self.training and attention_mask is not None and (attention_mask[:, 0].sum() != attention_mask.shape[0]):
        raise NotImplementedError('MPT does not support training with left padding.')
    if self.prefix_lm and prefix_mask is None:
        raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')
    if inputs_embeds is not None:
        raise NotImplementedError('inputs_embeds is not implemented for MPT.')
    if self.training:
        if self.attn_uses_sequence_id and sequence_id is None:
            raise ValueError('sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
        elif self.attn_uses_sequence_id is False and sequence_id is not None:
            warnings.warn('MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' + 'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.')
    S = input_ids.size(1)
    assert S <= self.config.max_seq_len, f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

    tok_emb = self.wte(input_ids)
    if self.learned_pos_emb:
        past_position = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(f'past_key_values must provide a past_key_value for each attention ' + f'layer in the network (len(past_key_values)={len(past_key_values)!r}; self.config.n_layers={self.config.n_layers!r}).')
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)
        if S + past_position > self.config.max_seq_len:
            raise ValueError(f'Cannot forward input with past sequence length {past_position} and current sequence length ' + f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.')
        pos = torch.arange(past_position, S + past_position, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        if attention_mask is not None:
            pos = torch.clamp(pos - torch.cumsum((~attention_mask).to(torch.int32), dim=1)[:, past_position:], min=0)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
    else:
        x = tok_emb
    if self.embedding_fraction == 1:
        x = self.emb_drop(x)
    else:
        x_shrunk = x * self.embedding_fraction + x.detach() * (1 - self.embedding_fraction)
        assert isinstance(self.emb_drop, nn.Module)
        x = self.emb_drop(x_shrunk)
    (attn_bias, attention_mask) = self._attn_bias(device=x.device, dtype=torch.float32, attention_mask=attention_mask, prefix_mask=prefix_mask, sequence_id=sequence_id)
    presents = () if use_cache else None
    if use_cache and past_key_values is None:
        past_key_values = [() for _ in range(self.config.n_layers)]
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # yixing: self.blocks includes the attn block.
    # ( self.blocks = nn.ModuleList([MPTBlock(device=config.init_device, **config.to_dict()) for _ in range(config.n_layers)]) )
    for (b_idx, block) in enumerate(self.blocks):
        if output_hidden_states:
            assert all_hidden_states is not None
            all_hidden_states = all_hidden_states + (x,)
        past_key_value = past_key_values[b_idx] if past_key_values is not None else None

        # yixing:
        (x, attn_weights, present) = block(x, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=self.is_causal, output_attentions=bool(output_attentions), cur_prompt_id = self.cur_prompt_id, save_attn_dir = self.save_attn_dir, block_idx = b_idx)

        if presents is not None:
            presents += (present,)
        if output_attentions:
            assert all_self_attns is not None
            all_self_attns = all_self_attns + (attn_weights,)
    x = self.norm_f(x)
    if output_hidden_states:
        assert all_hidden_states is not None
        all_hidden_states = all_hidden_states + (x,)
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_self_attns)


