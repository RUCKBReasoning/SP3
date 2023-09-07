import torch
import torch.nn as nn
from torch import Tensor

class PackedLayerNorm(nn.LayerNorm):
    
    def __init__(self, 
        normalized_shape: int,
        packed_size: int,
        eps: float = 0.00001, 
        elementwise_affine: bool = True, 
        device=None, 
        dtype=None,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        self.packed_size = packed_size
        self.pad = nn.ConstantPad1d((0, normalized_shape - packed_size), 0)
        self.unpad = nn.ConstantPad1d((0, packed_size - normalized_shape), 0)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = super().forward(x)
        x = self.unpad(x)
        return x
