"""
Create a Multi-head attention from srach

1-Mannual normalization
2-Mannual Mult-head attention
3-Mannual positonal encoding

"""
from typing import Optional

import torch
import torch.nn as nn

# Define normlization
class LayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape: int, 
                 eps: float=1e-05,
                 ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.lamda = nn.Parameter(torch.ones(self.normalized_shape)) # (1, 1, normalized_shape) -> (B, S_L, normalized_shape)
        self.beta = nn.Parameter(torch.zeros(self.normalized_shape)) # (1, 1, normalized_shape) -> (B, S_L, normalized_shape)
        
    def forward(self, 
                input: torch.Tensor, # (B, S_L, normalized_shape)
                )-> torch.Tensor:
        _mean = input.mean(dim=-1, keepdim=True)
        _var = input.var(dim=-1, keepdim=True,  unbiased=False) + self.eps
        # print(input.shape, _mean.shape, _var.shape, self.lamda.shape)
        return ((input-_mean)/torch.sqrt(_var))*self.lamda + self.beta

class RMSNorm(nn.Module):
    def __init__(self,
            normalized_shape: int, 
            eps: float=1e-05,
            ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.lamda = nn.Parameter(torch.ones(self.normalized_shape)) # (1, 1, normalized_shape) -> (B, S_L, normalized_shape)
        
    def forward(self, 
                input: torch.Tensor, # (B, S_L, normalized_shape)
                )-> torch.Tensor:
        
        _var_sum = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        
        return (input/torch.sqrt(_var_sum))*self.lamda

class SoftMax(nn.Module):
    def __init__(self, 
                 dim: int=-1,
                 ):
        super().__init__()
        self.dim = dim

    def forward(self, 
                input: torch.Tensor, # (B, S_L, normalized_shape)
                ) -> torch.Tensor:
        # naive way is exp(x_i)/\sum exp(x_i)
        # max x_i -> exp(x_i - max{x})/ sum exp(x_i -max{x})
        _max = input.max(dim=self.dim, keepdim=True).values # (B, S_L, 1)
        input = input - _max # (B, S_L, normalized_shape)
        _exp = torch.exp(input)
        return _exp/_exp.sum(dim=self.dim, keepdim=True)

def casual_mask(seq_len: int)->torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)) # (emb_dim, emb_dim)
        
def positional_encoding(seq_len: int, emb_dim:int)->torch.Tensor:
    # data (-1, s_l. emb_dim)
    # (pos,2i ) sin(pos/1000**2i/d) , pos is row, 2i is colum
    # (pos,2i+1 ) sin(pos/1000**2i/d) , pos is row, 2i+1 is colum
    
    position = torch.arange(seq_len).unsqueeze(1) # (seq_len, 1)
    
    # (emb_dim/2,)
    div_term = torch.arange(0, emb_dim, 2, dtype=torch.float32)
    div_term = 1.0 / (10000 ** (div_term / emb_dim))
    
    # div_term = torch.exp(
    # torch.arange(0, emb_dim, 2, dtype=torch.float32) * 
    # (-math.log(10000.0) / emb_dim)
    # )
    
    pe = torch.zeros(seq_len, emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 emb_dim: int,
                 head_num: int,
                 dropout: float=0.1
                 ):
        super().__init__()

        self.emb_dim = emb_dim
        self.head_num = head_num
        
        self.dropoutlayer = nn.Dropout(dropout)
        self.map = nn.Linear(emb_dim, emb_dim*3)
        
        if emb_dim%head_num!=0:
            raise ValueError("Embedding dimsion can not be divided  by head number")
        else:
            self.head_dim = emb_dim//head_num
            
        self.softmaxlayer = SoftMax(dim=-1)
        self.fnn = nn.Linear(emb_dim, emb_dim)
      
    def forward(self, 
                input: torch.Tensor, # (B, S_K, emb_dim)
                mask: Optional[torch.Tensor]=None,
                )->torch.Tensor:
        b, s_l, _ = input.shape
        
        qkv = self.map(input) # (B, S_K, emb_dim*3)
        q = qkv[:, :, :self.emb_dim] # (B, S_K, emb_dim)
        k = qkv[:, :, self.emb_dim:2*self.emb_dim]  # (B, S_K, emb_dim)
        v = qkv[:, :, self.emb_dim*2:]  # (B, S_K, emb_dim)
        
        q = q.view(b, s_l, self.head_num, self.head_dim).transpose(1, 2) # (B, head_num, S_K, head_dim)
        k = k.view(b, s_l, self.head_num, self.head_dim).transpose(1, 2) # (B, head_num, S_K, head_dim)
        v = v.view(b, s_l, self.head_num, self.head_dim).transpose(1, 2) # (B, head_num, S_K, head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1))/self.head_dim**0.5
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
            
        scores = self.softmaxlayer(scores)
        scores = self.dropoutlayer(scores)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(b, s_l, self.emb_dim)
        
        output = self.fnn(output)
        return output

# Test
x = torch.randn((10, 5, 4))
# layernorm = LayerNorm(4)
# rmsnorm = RMSNorm(4)
# print(layernorm(x).shape)
# print(rmsnorm(x).shape)
# softmaxlayer = SoftMax()
# print(softmaxlayer(x).shape)
# print(softmaxlayer(x)[0].sum(-1))
attention = MultiHeadAttention(4, 2)
output = attention(x, casual_mask(5))
print(output.shape)