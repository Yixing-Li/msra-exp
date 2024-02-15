import torch
import torch.nn as nn
from functools import partial


def mod_DistViT_forward_features(self, x, observe = False, ith_images = None, get_res_args = None):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add the dist_token
    ith_image_dict = {'ith_images': ith_images, 'get_res_args': get_res_args} if ith_images is not None else {}

    if observe:
        print(f'0 torch.isnan(x).any():{torch.isnan(x).any()}') 
        print(x.dtype) # torch.float32
    B = x.shape[0]
    x = self.patch_embed(x)
    if observe:
        # got x have nan already here. what is patch_embed? here????
        print(f'1 torch.isnan(x).any():{torch.isnan(x).any()}') 

    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    dist_token = self.dist_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, dist_token, x), dim=1)
    if observe:
        print(f'2 torch.isnan(x).any():{torch.isnan(x).any()}') 

    x = x + self.pos_embed
    if observe:
        print(f'3 torch.isnan(x).any():{torch.isnan(x).any()}') 
    x = self.pos_drop(x)
    if observe:
        print(f'4 torch.isnan(x).any():{torch.isnan(x).any()}') 

    for ith_blk, blk in enumerate(self.blocks):
        x = blk(x, observe, **ith_image_dict)
        if observe:
            # print(f'5-{ith_blk}th_blk torch.isnan(x).any():{torch.isnan(x).any()}') 
            pass

    x = self.norm(x)
    if observe:
        print(f'6 torch.isnan(x).any():{torch.isnan(x).any()}') 
    return x[:, 0], x[:, 1]

def mod_DistViT_forward(self, x, observe = False, ith_images = None, get_res_args = None):
    if observe:
        print(f'0 before: torch.isnan(x).any():{torch.isnan(x).any()}') 
    ith_image_dict = {'ith_images': ith_images, 'get_res_args': get_res_args} if ith_images is not None else {}
    x, x_dist = self.forward_features(x, observe, **ith_image_dict)
    if observe:
        print(f'final: torch.isnan(x).any():{torch.isnan(x).any()}') 
    if observe:
        print(f'final: torch.isnan(x_dist).any():{torch.isnan(x_dist).any()}') 
    x = self.head(x)
    if observe:
        print(f'final: torch.isnan(x).any():{torch.isnan(x).any()}') 
    x_dist = self.head_dist(x_dist)
    if observe:
        print(f'final: torch.isnan(x_dist).any():{torch.isnan(x_dist).any()}')
    if self.training:
        return x, x_dist
    else:
        # during inference, return the average of both classifier predictions
        return (x + x_dist) / 2


# def mod_DistViT_forward_features(self, x, observe = False, ith_images = None):
#     # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     # with slight modifications to add the dist_token
#     ith_image_dict = {'ith_images': ith_images} if ith_images is not None else {}

#     B = x.shape[0]
#     x = self.patch_embed(x)

#     cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#     dist_token = self.dist_token.expand(B, -1, -1)
#     x = torch.cat((cls_tokens, dist_token, x), dim=1)

#     x = x + self.pos_embed
#     x = self.pos_drop(x)

#     for blk in self.blocks:
#         x = blk(x, observe, **ith_image_dict)

#     x = self.norm(x)
#     return x[:, 0], x[:, 1]

# def mod_DistViT_forward(self, x, observe = False, ith_images = None):
#     ith_image_dict = {'ith_images': ith_images} if ith_images is not None else {}
#     x, x_dist = self.forward_features(x, observe, **ith_image_dict)
#     x = self.head(x)
#     x_dist = self.head_dist(x_dist)
#     if self.training:
#         return x, x_dist
#     else:
#         # during inference, return the average of both classifier predictions
#         return (x + x_dist) / 2