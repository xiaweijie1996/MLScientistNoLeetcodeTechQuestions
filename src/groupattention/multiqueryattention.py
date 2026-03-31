from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
# from torch import Tensor   # then bare `Tensor` works

class MultiQueryattention(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 head_dim: int):
        super().__init__()

        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        if self.hidden_dim % self.head_dim != 0:
            raise ValueError("hidden_dim must be divisible by head_dim")

        self.head_num = self.hidden_dim // self.head_dim

        self.map_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.map_k = nn.Linear(self.hidden_dim, self.head_dim)
        self.map_v = nn.Linear(self.hidden_dim, self.head_dim)
        
    def forward(self, 
                q: torch.Tensor,  # (b, sl, hidden_dim)
                k: torch.Tensor,  # (b, sk, hidden_dim)
                v: torch.Tensor,  # (b, sk, hidden_dim)
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor :
        # Get the shape
        b, sl, _ = q.shape
        
        q = self.map_q(q) # (b, sl, hidden_dim)
        k = self.map_k(k) # (b, sk, head_dim)
        v = self.map_v(v) # (b, sk, head_dim)
        
        # Reshape the q, k ,v
        q = q.view(b, sl, self.head_num, self.head_dim).transpose(1, 2)  # (b, head_num, sl, head_dim)
        k = k.unsqueeze(1) # (b, 1, sk, hidden_dim)
        v = v.unsqueeze(1)  # (b, 1, sk, hidden_dim)
        
        score = torch.matmul(q, k.transpose(-2, -1))/(self.head_dim**0.5) # (b, self.head_num, sl, sk)
        
        if mask is not None:
            score = score.masked_fill(mask==0, float("-inf"))
        
        attention_weight = torch.softmax(score, dim=-1) # (b, self.head_num, sl, sk)
        
        output = torch.matmul(attention_weight, v) # (b, self.head_num, sl, head_dim)
        output = output.transpose(1, 2).contiguous().view(b, sl, self.hidden_dim) # (b, sl, hidden_dim)
        
        return output
        
            

if __name__ == "__main__":
    qattention = MultiQueryattention(4, 2)
    print(qattention.head_num)
    
    q = torch.randn(2, 5, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    
    out = qattention(q, k, v)
    print(out.shape)
    
    mask = torch.ones((1, 1, 5, 3))  # must broadcast to (b, head_num, sl, sk)
    out = qattention(q, k, v, mask)
    print(out.shape)
        