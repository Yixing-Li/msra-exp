import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import sys
import math

import numpy as np
from scipy.linalg import dft
import matplotlib.pyplot as plt
from pathlib import Path

# yixing
from rms_norm import RMSNorm
from headwise_GroupNorm import HeadWiseGroupNorm

RMS_norm = RMSNorm(0,  eps = 1e-7, elementwise_affine = False) # eps = 0, 
got_dft = False
DFT = None

got_HW_GN = False
HW_GN = None # HeadWiseGroupNorm()


def calc_spectrum(A):
    return DFT @ A @ DFT.T

def plot_spectrum(a, exp_folder, exp_name, layer = 0, head = 0, ith_images = None, ith_image_ith = None):
    if not got_dft:
        global DFT
        len_tokens = a.shape[-1]
        DFT = dft(len_tokens, scale='sqrtn')

    a = a.cpu().numpy()
    s = calc_spectrum(a)
    s = np.linalg.norm(s, ord=2, axis=1)
    s = np.concatenate([s[-math.floor(len_tokens/2):], s[0:1], s[1:math.floor(len_tokens/2)]], axis=0)
    # (s.shape) # (198,)
    # max_x, max_value = s.argmax(), s.max()
    dc_value = s[ int(a.shape[-1] / 2) ]
    # print(max_x, a.shape[-1])
    all_sqr = np.sum(s**2)
    hc_dc = (all_sqr - (dc_value ** 2+1e-7)) / (dc_value ** 2 + 1e-7)
    # print(hc_dc)

    if (ith_images is not None) and ith_images <= 2:
        if (ith_images is not None) and (ith_image_ith is not None):
            ith_image = f'-ith_{ith_images}_{ith_image_ith}'
        save_folder = f'{exp_folder}/spectrum{ith_image}'
        Path(save_folder).mkdir(parents = True, exist_ok = True)
        plt.plot(s)
        plt.savefig(f'{save_folder}/{exp_name}-layer_{layer}-head_{head}.png') 
        plt.clf() 
    return hc_dc

def compute_similarity(x, y):
    s = torch.matmul(torch.transpose(x, -2, -1), y) # [bs, h, l, l]
    l1, l2 = torch.linalg.norm(x, ord=2, dim=-2), torch.linalg.norm(y, ord=2, dim=-2)
    l = l1[..., :, None] * l2[..., None, :] # [bs, h, l, l]
    # N = s.shape[-1]
    s = torch.abs(s / l)

    return s.mean().item()

# def matrix_similarity(matrix):
    # Compute Frobenius norm for each matrix in the batch
    # norms = torch.norm(matrix.view(matrix.size(0), -1), dim=1) 
    
    # # Compute pairwise differences
    # diff_matrix = matrix.unsqueeze(1) - matrix.unsqueeze(0)
    
    # # Compute Frobenius norm of pairwise differences
    # diff_norms = torch.norm(diff_matrix.view(diff_matrix.size(0), -1), dim=1)
    
    # # Calculate similarity as inverse of the difference norm
    # similarity_matrix = 1 / (1 + diff_norms.view(-1, 1))

    # sys.exit(0)
    
    return similarity_matrix

class mod_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(mod_Attention).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        print(f'in mod attention !!!')
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q.shape: B, self.num_heads, N, C
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# def exp_clip(x, *y):
def swish(x, b = 1):
    return x * F.sigmoid(b * x)

def mod_Attention_forward(self, x, observe = False, ith_images = None, get_res_args = None):
    # great! got into here.
    # yixing-test

    B, N, C = x.shape # C = dim. 
    # eval: x.shape: [384, 198, 384]. self.qkv(x): [384, 198, 1152]
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
    # q.shape:    B, self.num_heads, N, C // self.num_heads
    # attn.shape: B, self.num_heads, N, N
    num_heads = q.shape[1]
    num_groups = 4
    # num_features = 64
    attn = (q @ k.transpose(-2, -1)) * self.scale

    if observe:
        pass
        # print(f'observe mode.')
        # print(f'before double_softmax:{attn}')

    # softmax
    if self.model_args['double_softmax']:
        math_1e = math.exp(-1)
        math_e = math.exp(1)
        # get org_attn -> calculate sum_attn -> do attn softmax -> do attn double-norm

        if not self.model_args['inside_exp']:
            sum_attn = torch.sum(attn, dim = -1, keepdim=True) 
        else:
            exp_attn = attn.exp()
            if self.model_args['inside_clip']:
                # print(self.inside_clip)
                exp_attn = torch.clamp(exp_attn, math_1e, math_e)

            sum_attn = torch.sum(exp_attn, dim = -1, keepdim=True) 

        if not self.model_args['outside_exp']: 
            total_sum = torch.sum(sum_attn, dim=-2, keepdim=True)
            normed_sum_attn = sum_attn / total_sum
        else:
            normed_sum_attn = sum_attn.softmax(dim = -2)

        # same device:
        # sum_attn.device, normed_sum_attn.device, attn.device

        normed_sum_attn = normed_sum_attn * attn.shape[-2]
        attn = attn.softmax(dim=-1)
        attn = attn * normed_sum_attn 
        
        if observe:
            pass
            # print(f'scale factor:{normed_sum_attn}\n')
            # print(f'after double_softmax:{attn}\n')
            # sys.exit(0)

        # diag_sumnormed_attn: # torch.Size([96, 12, 197, 197]) <class 'torch.Tensor'>
    elif self.model_args['swish']:
        # self.scale = qk_scale or head_dim ** -0.5
        # attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.model_args['swish_first']:
            attn = swish(attn)
            A_norm = torch.abs(torch.sum(attn, dim = -1, keepdim = True))
            A_norm = torch.maximum(A_norm, torch.ones_like(A_norm))
            attn = attn / A_norm
        else:
            A_norm = torch.abs(torch.sum(attn, dim = -1, keepdim = True))
            A_norm = torch.maximum(A_norm, torch.ones_like(A_norm))
            attn = attn / A_norm
            attn = swish(attn)

    else:
        # train: (attn.shape) # torch.Size([96, 12, 197, 197]) <class 'torch.Tensor'>
        # test:  (attn.shape) # torch.Size([384, 12, 198, 198])
        attn = attn.softmax(dim=-1)

    if self.model_args['get_spectrum'] and get_res_args['get_spectrum']:
        # ith_images
        ith_image_ith = 0
        hc_dcs = []
        for ith_head_idx, ith_head_attn in enumerate(attn[ith_image_ith]):
            hc_dc = plot_spectrum(ith_head_attn, self.model_args['exp_folder'], self.model_args['exp_name'], self.ith_blk, ith_head_idx, ith_images, ith_image_ith)
            hc_dcs.append(hc_dc)
        self.spectrum[ith_images] = np.mean(hc_dcs)

        
    # end: yixing
    
    attn = self.attn_drop(attn)
    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    value = attn @ v
    if self.model_args['add_rmsnorm']:
        # TODO: add rms_norm or group_norm
        value = RMS_norm(value) # if eps=0, would cause nan.
        if self.model_args['HW_GN']:
            raise RuntimeError('set both: RMS and HeadWise_GN') 
    if self.model_args['HW_GN']:
        if not got_HW_GN:
            global HW_GN
            # num_heads = 8
            # num_groups = 4
            # num_features = 64
            HW_GN = HeadWiseGroupNorm(num_heads = num_heads, num_groups = num_groups, num_features = value.shape[-1])
        # q, k, v, value.shape:    B, self.num_heads, N, C // self.num_heads
        # attn.shape:              B, self.num_heads, N, N
        value = HW_GN(value)
    x = value.transpose(1, 2).reshape(B, N, C)

    x = self.proj(x)
    x = self.proj_drop(x)

    if self.model_args['get_batch_simi'] and get_res_args['get_batch_simi']:
        batch_simi = compute_similarity(x, x) # one float for each block.
        # batch_simi = matrix_similarity(x)
        self.batch_simi[ith_images] = batch_simi
    return x
    
    