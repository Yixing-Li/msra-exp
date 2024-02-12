'''
modified from block
'''
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

# from .attention import ATTN_CLASS_REGISTRY

# class MPTBlock(nn.Module):
def mod_MPTBlock_forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.ByteTensor]=None, is_causal: bool=True, output_attentions: bool=False, cur_prompt_id: int = None, save_attn_dir: str = None, block_idx: int = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    # yixing
    assert cur_prompt_id is not None
    assert save_attn_dir is not None

    a = self.norm_1(x)

    # yixing: self.attn is defined as:
    # (attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']])
    # (self.attn = attn_class(d_model=d_model, n_heads=n_heads, fc_type=fc_type, device=device, **attn_config_subset_for_attn_class, bias=not no_bias))

    # and the forward of these class are the same.
    (b, attn_weights, past_key_value) = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal, needs_weights=output_attentions, cur_prompt_id = cur_prompt_id, save_attn_dir = save_attn_dir, block_idx = block_idx)

    x = x + self.resid_attn_dropout(b)
    m = x
    if self.norm_2 is not None:
        m = self.norm_2(x)
    n = self.ffn(m)
    x = x + self.resid_ffn_dropout(n)
    return (x, attn_weights, past_key_value)

