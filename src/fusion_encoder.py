# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py


from typing import List
import torch
import torch.nn as nn

from .vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from .attention import AttentionLayer


class FusionEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        out_layers: List[int],
        fusion_layers: List[int],
        alt_start: int = -1,
        qknorm_start: int = -1,
        rope_start: int = -1,
        cat_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        assert name in {"vits", "vitb", "vitl", "vitg"}
        self.name = name
        self.out_layers = out_layers
        self.fusion_layers = fusion_layers
        self.alt_start = alt_start
        self.qknorm_start = qknorm_start
        self.rope_start = rope_start
        self.cat_token = cat_token
        encoder_map = {
            "vits": vit_small,
            "vitb": vit_base,
            "vitl": vit_large,
            "vitg": vit_giant2,
        }
        encoder_fn = encoder_map[self.name]
        ffn_layer = "swiglufused" if self.name == "vitg" else "mlp"
        self.pretrained = encoder_fn(
            img_size=518,
            patch_size=14,
            ffn_layer=ffn_layer,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
        )

        # for depth_fusion blk
        #self.fuse_gate = nn.Parameter(torch.tensor(0.0))
        self.depth_fusion_blk = AttentionLayer(
            num_blocks=1,
            dim=self.pretrained.embed_dim,
            num_heads=self.pretrained.embed_dim//64,
            expansion=4.0,
            dropout=0.0,
            layer_scale=-1,
            context_dim=self.pretrained.embed_dim,
            use_bias=False,
        )

    def cross_attn_batch(self, x, d):
        B, S = x.shape[0], x.shape[1]
        x_flat  = x.reshape(B * S, x.shape[2],  x.shape[3])
        d_flat = d.reshape(B * S, d.shape[2], d.shape[3])
        x_fused = self.depth_fusion_blk(x_flat, d_flat)
        return x_fused.reshape(B, S, x_fused.shape[1], x_fused.shape[2]) # (B,S,C,N)

    def forward(self, x, depth_tokens, **kwargs):
        out_layers = list(self.out_layers)
        out_set = set(out_layers)

        state = self.pretrained.init_layer_state(x, **kwargs)

        collected_outx = {}
        collected_cam = {}

        for i in range(len(self.pretrained.blocks)):
            # depth_fusion if depth_tokens is given
            if depth_tokens is not None and i in self.fusion_layers:
                #fusion_idx = self.fusion_layers.index(i)
                state["x"] = self.cross_attn_batch(state["x"], depth_tokens)
                #_x = self.cross_attn_batch(state["x"], depth_tokens)
                #delta = self.cross_attn_batch(state["x"], depth_tokens) - state["x"]
                #state["x"] = state["x"] + torch.sigmoid(self.fuse_gate) * delta
            
            state, out_x = self.pretrained.process_one_layer(state, **kwargs)
            if i in out_set:
                collected_outx[i] = out_x
                collected_cam[i] = out_x[:, :, 0]

        outs = [collected_outx[i] for i in out_layers]
        cams = [collected_cam[i] for i in out_layers]

        # norm（元実装と同じ）
        embed_dim = self.pretrained.embed_dim
        norm = self.pretrained.norm

        if outs[0].shape[-1] == embed_dim:
            outs = [norm(o) for o in outs]
        elif outs[0].shape[-1] == (embed_dim * 2):
            outs = [
                torch.cat([o[..., :embed_dim], norm(o[..., embed_dim:])], dim=-1)
                for o in outs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outs[0].shape}")

        # cls + register 除去（元実装と同じ）
        start = 1 + self.pretrained.num_register_tokens
        outs = [o[..., start:, :] for o in outs]

        return tuple(zip(outs, cams)), []