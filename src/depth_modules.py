import torch
import functools
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal
from torchvision.ops import StochasticDepth

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    
    We use channel_first mode for SP-Norm.
    """

    def __init__(self, normalized_shape, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1))

    def forward(self, x):
        s, u = torch.std_mean(x, dim=1, keepdim=True)
        x = (x - u) / (s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x

class NormLayer(nn.Module):
    def __init__(self, normalized_shape, norm_type):
        super(NormLayer, self).__init__()
        self.norm_type = norm_type

        if self.norm_type == 'LN':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'BN':
            self.norm = nn.BatchNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'IN':
            self.norm = nn.InstanceNorm2d(normalized_shape, affine=True)
        elif self.norm_type == 'RZ':
            self.norm = nn.Identity()
        elif self.norm_type in ['CNX', 'CN+X', 'GRN']:
            self.norm = LayerNorm(normalized_shape, affine=False)
            # Use 1*1 conv to implement SLP in the channel dimension. 
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        elif self.norm_type == 'NX':
            self.norm = LayerNorm(normalized_shape, affine=True)
        elif self.norm_type == 'CX':
            self.conv = nn.Conv2d(normalized_shape, normalized_shape, kernel_size=1)
        else:
            raise ValueError('norm_type error')

    def forward(self, x):
        if self.norm_type in ['LN', 'BN', 'IN', 'RZ']:
            x = self.norm(x)
        elif self.norm_type in ['CNX', 'GRN']:
            x = self.conv(self.norm(x)) * x
        elif self.norm_type == 'CN+X':
            x = self.conv(self.norm(x)) + x
        elif self.norm_type == 'NX':
            x = self.norm(x) * x
        elif self.norm_type == 'CX':
            x = self.conv(x) * x
        else:
            raise ValueError('norm_type error')
        return x

class CNBlock(nn.Module):
    def __init__(self, dim: int, norm_type: str, dp_rate: float):
        super(CNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            NormLayer(dim, norm_type),
            nn.Conv2d(dim, 4 * dim, kernel_size=1),
            nn.ReLU(inplace=True),
            GRN(4 * dim) if norm_type == 'GRN' else nn.Identity(),
            nn.Conv2d(4 * dim, dim, kernel_size=1),
        )
        self.drop_path = StochasticDepth(dp_rate, mode='batch')
        self.norm_type = norm_type
        if self.norm_type == 'RZ':
            self.alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        res = self.block(x)
        if self.norm_type == 'RZ':
            res = self.alpha * res
        x = x + self.drop_path(res)
        return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        x = (1 + self.gamma * Nx) * x + self.beta
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        in_chans=5,
        dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        dp_rate=0.0,
        norm_type="CNX",
    ):
        super(Encoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Conv2d(in_chans, all_dims[0], kernel_size=3, padding=1)

        self.downsample_layers.append(stem)
        for i in range(5):
            downsample_layer = nn.Sequential(
                NormLayer(all_dims[i], norm_type),
                nn.Conv2d(all_dims[i], all_dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    CNBlock(dims[i], norm_type, dp_rates[cur + j])
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        outputs = []
        for i in range(6):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs

class Decoder(nn.Module):
    def __init__(self, out_chans=1, dims=[96, 192, 384, 768], norm_type="CNX"):
        super(Decoder, self).__init__()
        all_dims = [dims[0] // 4, dims[0] // 2] + dims
        self.upsample_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()
        for i in range(5):
            upsample_layer = nn.ConvTranspose2d(
                all_dims[i + 1], all_dims[i], kernel_size=2, stride=2
            )
            fusion_layer = nn.Conv2d(2 * all_dims[i], all_dims[i], kernel_size=1)
            self.upsample_layers.append(upsample_layer)
            self.fusion_layers.append(fusion_layer)

        self.stages = nn.ModuleList()
        self.stages.append(nn.Identity())
        self.stages.append(nn.Identity())
        for i in range(3):
            stage = CNBlock(dims[i], norm_type, 0.0)
            self.stages.append(stage)
        
        self.head = nn.Conv2d(all_dims[0], 1, kernel_size=3, padding=1)
    
    def get_intermediate_layers(self, ins):
        x = ins[-1]
        output = []
        for i in range(4, 0, -1):
            x = self.upsample_layers[i](x)
            if x.shape[-2:] != ins[i].shape[-2:]:
                x = F.interpolate(x, size=ins[i].shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([ins[i], x], dim=1)
            x = self.fusion_layers[i](x)
            x = self.stages[i](x)
            output.append(x)
        output.reverse()
        return output

    def forward(self, ins):
        x = ins[-1]
        for i in range(4, -1, -1):
            x = self.upsample_layers[i](x)
            x = torch.cat([ins[i], x], dim=1)
            x = self.fusion_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x

def model_config(model_type):
    if model_type == "Tiny":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 9, 3]  # block number
        dp_rate = 0.0
    if model_type == "Small":
        dims = [96, 192, 384, 768]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Base":
        dims = [128, 256, 512, 1024]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.1
    elif model_type == "Large":
        dims = [192, 384, 768, 1536]  # dimensions
        depths = [3, 3, 27, 3]  # block number
        dp_rate = 0.2
    return dims, depths, dp_rate

def log(x):
    valid_mask = x > 0
    result = torch.zeros_like(x)
    result[valid_mask] = torch.log(x[valid_mask])
    return result

def median(x: torch.Tensor, default_value: float=1.0) -> torch.Tensor:
    """
    Computes the median of values greater than 0 for each batch in the input tensor.
    
    For each batch, only values greater than 0 are considered. If no such values exist,
    a default value is used instead.

    Args:
        x (torch.Tensor): Input tensor of shape [B, C, H, W]
        default_value (float): Value to use when no valid values (> 0) are found in a batch

    Returns:
        torch.Tensor: A tensor of shape [B, 1, 1, 1], containing the computed median (or default)
                      for each batch
    """
    
    B = x.shape[0]
    ndim = x.ndim
    medians = []

    for b in range(B):
        valid_values = x[b][x[b] > 0.0]
        if valid_values.numel() > 0:
            medians.append(torch.median(valid_values))
        else:
            medians.append(torch.tensor(default_value, device=x.device))

    result = torch.stack(medians)
    return result.view(B, *(1,) * (ndim - 1))