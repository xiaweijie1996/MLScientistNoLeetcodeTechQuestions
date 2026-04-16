
from typing import Optional
import copy
import torch
import torch.nn as nn
import numpy as np

from scipy.stats import norm
# Define a simple model
class SimpleMap(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 hid_dim: int,
                 layer_n:int,
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_dim, hid_dim),
                nn.SELU(),
                nn.Linear(hid_dim, in_dim),
                nn.SELU(),
               
            ) for _ in range(layer_n)
            ]
        )
    def forward(self, 
                x: torch.Tensor
                )-> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
    
class Adaptor(nn.Module):
    def __init__(self, 
                r: int,
                in_dim: int
                ):
        super().__init__()
        self.adaptor = nn.Sequential(
            nn.Linear(in_dim, r),
            nn.SELU(),
            nn.Linear(r, in_dim))
        
    def forward(self, 
                x: torch.Tensor
                )-> torch.Tensor:
        return self.adaptor(x)
        
    
class Quantization4bit(nn.Module):
    def __init__(self, 
                 arbitrary_range: list=[-1,1]
                ):
        super().__init__()
        self.arbitrary_range = arbitrary_range
        self.codebook = self._double_quantization_gaussian_()
        
    def fit(self, x: torch.Tensor):
        self.x_max = torch.max(x)
        self.x_min = torch.min(x)
        
    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        x_clipped = torch.clamp(x, self.x_min, self.x_max)
        x_normalized = (x_clipped - self.x_min) / (self.x_max - self.x_min) * (self.arbitrary_range[1] - self.arbitrary_range[0]) + self.arbitrary_range[0]
        return x_normalized
        
    def _k_bit_quantization_gaussian_(self, 
                            q_range: list=[-1,1],
                            n_bit: int=4,
                            dtype: torch.dtype = torch.float16
                            )-> torch.Tensor:
        
        quantlies = np.arange(1, (n_bit))/(n_bit)
        values = []
        for p in quantlies:
            _value = norm.ppf(p, loc=0, scale=1)
            values.append(torch.tensor(float(_value), dtype=dtype))
            
        values = torch.stack(values)
        values_nomalized = q_range[0] + (values - values[0]) / (values[-1] - values[0]) * (q_range[1] - q_range[0])
        return values_nomalized
    
    def _double_quantization_gaussian_(self, 
                            dtype: torch.dtype = torch.float16
                            )-> torch.Tensor:
        quantiles_negative = self._k_bit_quantization_gaussian_([self.arbitrary_range[0], 0], 2**3, dtype)
        quantiles_positive = self._k_bit_quantization_gaussian_([0, self.arbitrary_range[1]], 2**3+1, dtype)[1:]
        quantiles = torch.cat([quantiles_negative, quantiles_positive])
        return quantiles
        
    def map_to_quantized_values(self, x: torch.Tensor):
        x_trans = self.fit_transform(x)
        x_expanded = x_trans.reshape(-1, 1)                   # (N, 1)
        codebook = self.codebook.to(x.device)          # (Q,)
        distances = torch.abs(x_expanded - codebook)   # broadcast -> (N, Q)
        indices = torch.argmin(distances, dim=-1)      # (N,)
        quantized_values = codebook[indices].view_as(x)
        return quantized_values
        
    def dequantize_values(self, input_quantized: torch.Tensor) -> torch.Tensor:
        output = (
            (input_quantized - self.arbitrary_range[0])
            / (self.arbitrary_range[1] - self.arbitrary_range[0])
            * (self.x_max - self.x_min)
            + self.x_min
        )
        return output
        
    
class QuantizedModel(nn.Module):
    def __init__(self, 
                 model: Optional[nn.Module] = None,
                ):
        super().__init__()
        self.model = model
        self.quantizer_list = []
        if self.model is not None:
            self._fit_to_model()
        print("quantizer_list: ", self.quantizer_list)
        
    def _fit_to_model(self):
        if self.model is None:
            raise ValueError("QuantizedModel requires a model before fitting quantizers.")

        for param in self.model.parameters():
            quantizer = Quantization4bit()
            quantizer.fit(param.data)
            self.quantizer_list.append(quantizer)
            
    def quantized_model_parameters(self):
        if self.model is None:
            raise ValueError("QuantizedModel has no model to quantize.")

        quantized_params = []
        for param, quantizer in zip(self.model.parameters(), self.quantizer_list):
            quantized_param = quantizer.map_to_quantized_values(param.data)
            quantized_params.append(quantized_param)
        return quantized_params
    
    def dequantized_model_parameters(self, quantized_params):
        dequantized_params = []
        for quantized_param, quantizer in zip(quantized_params, self.quantizer_list):
            dequantized_param = quantizer.dequantize_values(quantized_param)
            dequantized_params.append(dequantized_param)
        return dequantized_params
    
    def dequantize_model(self, quantized_params):
        # Copy a model
        model = copy.deepcopy(self.model)
        
        dequantized_params = self.dequantized_model_parameters(quantized_params)
        
        with torch.no_grad():
            for param, dequantized_param in zip(model.parameters(), dequantized_params):
                # Asisgen dequantized parameters to the model
                param.copy_(dequantized_param)
        
        return model

if __name__ == "__main__":
    b ,d = 1, 2
    x = torch.randn((b, d), dtype=torch.float16)
    map = SimpleMap(d, d*3, 2)
    print(next(map.parameters()).dtype)
    
    map = map.to(torch.float16)
    print(next(map.parameters()).dtype)
    
    adaptor = Adaptor(2, d)
    print(next(adaptor.parameters()).dtype)
    
    # Change to float16
    adaptor = adaptor.to(torch.float16)
    print(next(adaptor.parameters()).dtype)
    
    # Parameters
    num_para = sum(p.numel() for p in map.parameters())
    print("map para:", num_para)
    
    num_adaptor = sum(p.numel() for p in adaptor.parameters())
    print("adaptor para:", num_adaptor)
    
    y = map(x) + adaptor(x)
    print(y.shape)
    
    # -----------------------
    quantizer = Quantization4bit()
    print("codebook: ", quantizer.codebook)
    quantized_y = quantizer.map_to_quantized_values(y)
    print("y: ", y, "\nquantized_y: ", quantized_y)
    dequantized_y = quantizer.dequantize_values(quantized_y)
    print("\ndequantized_y: ", dequantized_y)
    
    # ------------------------
    quantized_model = QuantizedModel(map)
    quantized_params = quantized_model.quantized_model_parameters()
    print("\nquantized_params: ", quantized_params)
    dequantized_params = quantized_model.dequantized_model_parameters(quantized_params)
    print("\ndequantized_params: ", dequantized_params)
    dequantized_model = quantized_model.dequantize_model(quantized_params)
    y_dequantized = dequantized_model(x) + adaptor(x)
    print("\ndequantized_model output: ", y_dequantized)
    print("\noriginal model output: ", y)