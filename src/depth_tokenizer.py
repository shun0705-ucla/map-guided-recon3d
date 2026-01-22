import torch.nn as nn
from .depth_modules import Encoder, Decoder, model_config
import torch
from typing import Literal, List

class SPNet(nn.Module):
    def __init__(
        self,
        model_type: Literal["Tiny", "Small", "Base", "Large"] = "Tiny",
    ):
        super().__init__()
        dims, depths, dp_rate = model_config(model_type)
        self.encoder = Encoder(in_chans=5,
            dims=dims, depths=depths, dp_rate=dp_rate)
        self.decoder = Decoder(
            out_chans=1, dims=dims)
        self.dims = [dim //2 for dim in dims]
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    
    def get_intermediate_layers(self, x):
        features = self.encoder(x)
        features = self.decoder.get_intermediate_layers(features)
        return features # [1/2, 1/4, 1/8, 1/16]

    def get_encoded_feature(self, x):
        feature = self.encoder(x)
        return feature[-1] # [1/32]
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

class DepthTokenizer(nn.Module):
    def __init__(
        self,
        model_type: Literal["Tiny", "Small", "Base", "Large"] = "Tiny",
        output_dim: int = 768,
    ):
        super().__init__()
        self.spnet =SPNet(model_type=model_type)
    
    def forward(self, rgb: torch.Tensor, d:torch.Tensor) -> torch.Tensor:
        # reshape inputs for SPNet
        B, S, _, H, W = rgb.shape
        rgb_flat = rgb.reshape(B * S, rgb.shape[2], H, W)
        d_flat = d.reshape(B * S, H, W).unsqueeze(1)
        valid = (d_flat > 0).type_as(d_flat)

        # Get feature from SPNet
        x = torch.cat([rgb_flat, d_flat, valid], dim=1)  # (B*S, 5, H, W)
        feat = self.spnet.get_encoded_feature(x)         # (B*S, Cctx, H/32, W/32)

        # (B*S, Cctx, Hf, Wf) -> (B,S,N,Cctx)
        _, Cctx, Hf, Wf = feat.shape
        feat = feat.reshape(B, S, Cctx, Hf, Wf)
        depth_tokens = feat.flatten(3).transpose(2, 3)   # (B,S,N,Cctx)

        return depth_tokens