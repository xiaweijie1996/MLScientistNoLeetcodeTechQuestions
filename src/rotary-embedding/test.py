import math

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmb(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 emb_dim: int
                 ):
        super().__init__()  
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        
        self.register_buffer("cos_cased", None, persistent=False)
        self.register_buffer("sin_cased", None, persistent=False)
        
    def _get_cos_sin(self)-> Tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.emb_dim // 2
        div = 10000**(-torch.arange(0, half_dim)/half_dim) # (half_dim,)
        seq = torch.arange(self.seq_len)
        value = torch.outer(seq, div)
        value = torch.repeat_interleave(value, repeats=2, dim=-1)
        
        cos = torch.cos(value) # (seq_len, emb_dim)
        sin = torch.sin(value) # (seq_len, emb_dim)
        return cos, sin

    def forward(self, 
                input: torch.tensor # (B, seq_len, emb_dim)
                )-> torch.tensor:
        # Chekc if busffer
        if self.cos_cased is None or self.sin_cased is None:
            self.cos_cased, self.sin_cased = self._get_cos_sin() #  (seq_len, emb_dim)
            print("Buffer generated")
            
        input_cos = input * self.cos_cased.unsqueeze(0) # (B, seq_len, emb_dim)
        
        input_odd = input[:, :, 1::2].unsqueeze(-1) # (B, seq_len, emb_dim/2, 1)
        input_even = input[:, :, 0::2].unsqueeze(-1) # (B, seq_len, emb_dim/2, 1)
        input_cat = torch.cat(( -input_odd, input_even), dim=-1) # (B, seq_len, emb_dim/2, 2)
        input_cat = input_cat.reshape(-1, self.seq_len, self.emb_dim) # (B, seq_len, emb_dim)
        
        input_sin = input_cat * self.sin_cased.unsqueeze(0) # (B, seq_len, emb_dim)
        
        return input_cos + input_sin
        
                
        



emb = RotaryEmb(5, 4)

value = emb._get_cos_sin()

print(value[0].shape, value)

x = torch.randn((3, 5, 4))
encoded = emb(x)
print(encoded.shape)