from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
# from torch import Tensor   # then bare `Tensor` works

class GroupQueryattention(nn.Module):
    def __init__(self, 
                 hidden_dim: int,
                 head_dim: int,
                 group_num: int,
                 
                 ):
        super().__init__()

        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.group_num = group_num

        if self.hidden_dim % self.head_dim != 0:
            raise ValueError("hidden_dim must be divisible by head_dim")
        self.head_num = self.hidden_dim // self.head_dim

        if self.head_num % self.group_num != 0:
            raise ValueError("head_num must be divisible by group_num")
        self.head_per_group = self.head_num // self.group_num

        self.map_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.map_k = nn.Linear(self.hidden_dim, self.head_dim*self.group_num)
        self.map_v = nn.Linear(self.hidden_dim, self.head_dim*self.group_num)
        
    def forward(self, 
                q: torch.Tensor,  # (b, sl, hidden_dim)
                k: torch.Tensor,  # (b, sk, hidden_dim)
                v: torch.Tensor,  # (b, sk, hidden_dim)
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor :
        # Get the shape
        b, sl, _ = q.shape
        _, sk, _ = k.shape
        
        q = self.map_q(q) # (b, sl, hidden_dim) or (b, sl, head_dim*head_num)
        k = self.map_k(k) # (b, sk, head_dim*group_num)
        v = self.map_v(v) # (b, sk, head_dim*group_num)
        
        # Reshape the q, k ,v
        q = q.view(b, sl, self.head_num, self.head_dim).transpose(1, 2)  # (b, head_num, sl, head_dim)
        k = k.view(b, sk, self.group_num, self.head_dim).transpose(1, 2) # (b, group_num, sk, head_dim)
        v = v.view(b, sk, self.group_num, self.head_dim).transpose(1, 2) # (b, group_num, sk, head_dim)
        
        # Expand k v to match the shape of q
        k = k.repeat_interleave(self.head_per_group, dim=1)
        v = v.repeat_interleave(self.head_per_group, dim=1)
        
        score = torch.matmul(q, k.transpose(-2, -1))/(self.head_dim**0.5) # (b, self.head_num, sl, sk)
        
        if mask is not None:
            score = score.masked_fill(~mask, float("-inf"))
        
        attention_weight = torch.softmax(score, dim=-1) # (b, self.head_num, sl, sk)
        
        output = torch.matmul(attention_weight, v) # (b, self.head_num, sl, head_dim)
        output = output.transpose(1, 2).contiguous().view(b, sl, self.hidden_dim) # (b, sl, hidden_dim)
        
        return output
        
            

if __name__ == "__main__":
    qattention = GroupQueryattention(12, 2, 3)
    print(qattention.head_num)
    
    q = torch.randn(2, 5, 12)
    k = torch.randn(2, 3, 12)
    v = torch.randn(2, 3, 12)
    
    out = qattention(q, k, v)
    print(out.shape)
    
    mask = torch.ones((1, 1, 5, 3), dtype=torch.bool)  # must broadcast to (b, head_num, sl, sk)
    out = qattention(q, k, v, mask)
    print(out.shape)
        