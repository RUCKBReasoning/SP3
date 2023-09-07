import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import Union, Tuple, Optional
from .packed_layer_norm import PackedLayerNorm


def sample(inps: Union[Tuple[torch.Tensor, ...], torch.Tensor], size: int):
    if isinstance(inps, torch.Tensor):
        assert len(inps.shape) == 3 and inps.shape[0] == 1
        inps = inps.squeeze(0).cpu()  # (seq_length, hidden_size)
        size = min(inps.shape[0], size)
        indices = np.random.choice(inps.shape[0], size, replace=False)
        indices = torch.from_numpy(indices)
        return inps[indices]
    else:
        return tuple(sample(x, size) for x in inps)


class Mask(nn.Module):
    min_s = -0.1
    max_s = 1.1
    eps = 1e-6
    magical_number = 0.8
    
    def __init__(self, features: int, repeat: int = 1) -> None:
        super().__init__()
        self.activate = nn.Parameter(torch.tensor(False), requires_grad=False)
        self.features = features * repeat
        self.repeat = repeat
        self.beta = 0.5
        self.log_alpha = nn.Parameter(torch.zeros((features,)))
        self.sampler = torch.distributions.Uniform(self.eps, 1. - self.eps)
        self.set_params(10.0)
    
    def set_state(self, activate: bool):
        self.activate.copy_(activate)
    
    @torch.no_grad()
    def set_params(self, mean: float, indices: Optional[torch.LongTensor] = None):  # [-10, 10]
        if indices is None:
            self.log_alpha.normal_(mean=mean, std=1e-2)
        else:
            self.log_alpha[indices].normal_(mean=mean, std=1e-2)
    
    def L(self):
        log_alpha = self.log_alpha.repeat(self.repeat)
        x = (0 - self.min_s) / (self.max_s - self.min_s)
        logits = math.log(x) - math.log(1 - x)
        return torch.sigmoid(log_alpha - logits * self.beta).clamp(min=self.eps, max=1-self.eps)

    def sample_z(self):  # z -> mask
        log_alpha = self.log_alpha.repeat(self.repeat)
        u = self.sampler.sample((self.features,)).type_as(log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        s_bar = s * (self.max_s - self.min_s) + self.min_s
        z = F.hardtanh(s_bar, min_val=0, max_val=1)
        return z
    
    def deterministic_z(self):
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        num_zeros = round(sub_features - Lc.item()) * self.repeat
        log_alpha = self.log_alpha.repeat(self.repeat)
        z = torch.sigmoid(log_alpha / self.beta * self.magical_number)
        if num_zeros > 0:
            _, indices = torch.topk(z, k=num_zeros, largest=False)
            z[indices] = 0
        return z
    
    def forward(self):
        if self.activate.item():
            if self.training:
                return self.sample_z()
            else:
                return self.deterministic_z()
        else:
            return 1.

    @torch.no_grad()
    def parse(self):
        sub_features = self.features // self.repeat
        Lc = self.L().sum() / self.repeat
        num_zeros = round(sub_features - Lc.item())
        num_non_zeros = sub_features - num_zeros
        z = torch.sigmoid(self.log_alpha / self.beta * self.magical_number)

        _, indices = torch.topk(z, k=num_non_zeros)
        if self.repeat > 1:  # shape: (num_heads, head_dim)
            z = z.repeat(self.repeat)
            indices = torch.concat(tuple(indices + i * sub_features for i in range(self.repeat)))
        return z[indices], indices


class Compactor(nn.Linear):
    
    def __init__(self, 
        features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        mask_pos="no",  # [no, input, output]
        mask: Mask = None
    ) -> None:
        super().__init__(features, features, bias, device, dtype)
        self.mask_position = mask_pos
        self.in_mask = mask  if mask_pos == "input"  else lambda: 1.0
        self.out_mask = mask if mask_pos == "output" else lambda: 1.0
        self.kw_args = {
            "in_features": features,
            "out_features": features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.in_mask()
        x = super().forward(x)
        x = x * self.out_mask()
        return x

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.weight.copy_(torch.eye(self.in_features).type_as(self.weight))
        if self.bias is not None:
            self.bias.zero_()
    
    @torch.no_grad()
    def extract(self, 
        mask_pos: str, 
        indices: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> nn.Linear:
        if values is None:
            values = torch.ones_like(indices, dtype=self.weight.dtype)
        values = torch.diag(values)
        if mask_pos == "input":
            self.kw_args["in_features"] = indices.shape[0]
            new_linear = nn.Linear(**self.kw_args)        
            new_linear.weight.copy_(self.weight[:, indices] @ values)
            if new_linear.bias is not None:
                new_linear.bias.copy_(self.bias)
        elif mask_pos == "output":
            self.kw_args["out_features"] = indices.shape[0]
            new_linear = nn.Linear(**self.kw_args)        
            new_linear.weight.copy_(values @ self.weight[indices, :])
            if new_linear.bias is not None:
                new_linear.bias.copy_(values @ self.bias[indices])
        else:
            raise ValueError
        return new_linear


class LayerNormWithCompactor(nn.LayerNorm):
    
    def __init__(self, 
        normalized_shape: int,
        eps: float = 0.00001, 
        elementwise_affine: bool = True, 
        device=None, 
        dtype=None,
        compactor_bias=True,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
        assert isinstance(normalized_shape, int)
        # comp -> Compactor
        self.mask = Mask(normalized_shape)
        self.in_comp = Compactor(
            normalized_shape, bias=compactor_bias, mask_pos="output", mask=self.mask)
        self.out_comp = Compactor(
            normalized_shape, bias=compactor_bias, mask_pos="input", mask=self.mask)
        self.kw_args = {
            "normalized_shape": normalized_shape,
            "packed_size": normalized_shape,
            "eps": eps, 
            "elementwise_affine": elementwise_affine, 
            "device": device, 
            "dtype": dtype,
        }
    
    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.in_comp(x)
        x = super().forward(x)
        x = self.out_comp(x)
        return x

    def forward_with_outputs(self, x: Tensor) -> Tensor:
        x0 = self.in_comp(x)
        x1 = super().forward(x0)
        x2 = self.out_comp(x1)
        return x2, (x0, x1, x2)

    
    def get_hook_fn(self, act_dict, name, size):
        return lambda module, inp, outp: act_dict[name].append(sample((inp[0], outp), size))
    
    def extract(self, indices: torch.Tensor) -> PackedLayerNorm:
        self.kw_args["packed_size"] = indices.shape[0]
        pad_size = self.kw_args["normalized_shape"] - self.kw_args["packed_size"]
        new_layernorm = PackedLayerNorm(**self.kw_args)
        new_layernorm.weight.copy_(F.pad(self.weight[indices], (0, pad_size), value=0))
        new_layernorm.bias.copy_(F.pad(self.bias[indices], (0, pad_size), value=0))
        return new_layernorm


class LinearWithCompactorBefore(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        compactor: Compactor = None,
        mask: Optional[Mask] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = mask
        self.compactor = compactor
        self.kw_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }

    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)  

    def forward(self, x: Tensor) -> Tensor:
        x = self.compactor(x)
        x = super().forward(x)
        return x
    
    def get_hook_fn(self, act_dict, name, size):
        return lambda module, inp, outp: act_dict[name].append(sample(inp[0], size))

    def extract(self) -> nn.Linear:
        new_linear = nn.Linear(**self.kw_args)
        new_linear.load_state_dict(self.state_dict(), strict=False)
        return new_linear


class LinearWithCompactorAfter(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
        compactor: Compactor = None,
        mask: Optional[Mask] = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = mask
        self.compactor = compactor
        self.kw_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }

    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)  

    def forward(self, x: Tensor) -> Tensor:
        x = super().forward(x)
        x = self.compactor(x)
        return x
    
    def get_hook_fn(self, act_dict, name, size):
        return lambda module, inp, outp: act_dict[name].append(sample(outp, size))

    def extract(self) -> nn.Linear:
        new_linear = nn.Linear(**self.kw_args)
        new_linear.load_state_dict(self.state_dict(), strict=False)
        return new_linear


class LinearWithMaskBefore(nn.Linear):
    
    def __init__(self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.mask = Mask(in_features)
        self.kw_args = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "device": device, 
            "dtype": dtype,
        }

    def super_forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.mask()
        x = super().forward(x)
        return x

    def get_hook_fn(self, act_dict, name):
        def fn(module, inp, outp):
            assert len(inp[0].shape) == 3 and inp[0].shape[0] == 1
            inp: Tensor = inp[0].squeeze(0).cpu()
            act_values = inp.abs().max(dim=0).values
            filter_weights = module.weight.norm(dim=0)
            values = act_values * filter_weights
            if name not in act_dict:
                act_dict[name] = values
            else:
                act_dict[name] = torch.max(act_dict[name], values)            
        return fn

    def extract(self,
        indices: torch.Tensor,
        values: Optional[torch.Tensor] = None,
    ) -> nn.Linear:
        if values is None:
            values = torch.ones_like(indices, dtype=self.weight.dtype)
        values = torch.diag(values)

        self.kw_args["in_features"] = indices.shape[0]
        new_linear = nn.Linear(**self.kw_args)        
        new_linear.weight.copy_(self.weight[:, indices] @ values)
        if new_linear.bias is not None:
            new_linear.bias.copy_(self.bias)
        return new_linear
