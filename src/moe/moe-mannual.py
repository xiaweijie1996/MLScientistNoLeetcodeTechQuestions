
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Experts
# Define router
# Define attention
# Define loss 

class Expert(nn.Module):
    def __init__(self, 
                 emb_dim: int,
                 hid_dim: int,
                 dropout: float=0.1
                 ):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, emb_dim),
        )
    
    def forward(self,
                input: torch.Tensor # (B, s_l, emb_dim)
                )-> torch.Tensor:
        return self.model(input) # (B, s_l, emb_dim)


class Router(nn.Module):
    def __init__(self, 
                 expert_num: int,
                 top_k: int,
                 emb_dim: int,
                 hid_dim: int
                 ):
        super().__init__()
        
        self.expert_num = expert_num
        self.top_k = top_k
        
        self.router = nn.Sequential(
            nn.Linear(emb_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, expert_num),
        )
    
    def forward(self, 
                input: torch.Tensor # (B, s_l, emb_dim)
                )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(input)  # (B, s_l, expert_num)
        k_logits, k_index = torch.topk(logits, self.top_k, dim=-1) # (B, s_l, k)
        k_prob = F.softmax(k_logits, dim=-1)
        return logits, k_logits, k_prob,  k_index
    
    
class MoeLayer(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 hid_dim: int,
                 expert_num: int,
                 top_k: int,
                 dropout: float=0.1
                 ):
        super().__init__()
        self.expert_num = expert_num
        self.top_k = top_k
        self.emb_dim = emb_dim
        
        self.router = Router(expert_num, top_k, emb_dim, hid_dim)
        
        self.experts = nn.ModuleList(
            [Expert(emb_dim, hid_dim, dropout) for _ in range(expert_num)]
        )
    
    def forward(self, 
                input: torch.Tensor # (B, s_l, emb_dim)
                ):
        B, s_l, _ = input.shape
        
        logits, k_logits, k_prob,  k_index = self.router(input)
        
        input_flat = input.reshape(B * s_l, self.emb_dim) # (B * s_l, self.emb_dim)
        k_index_flat = k_index.reshape(B * s_l, -1) # (B * s_l, k)
        k_prob_flat = k_prob.reshape(B * s_l, -1) # (B * s_l, k)
        
        out_tokens = torch.zeros_like(input_flat)
        for _index, _expert in enumerate(self.experts):
            _pos, _kth = torch.where(k_index_flat==_index)

            if _pos.numel() == 0:
                continue
            
            _exp_input_flat = input_flat[_pos] # (len(_pos), self.emb_dim)
            _expert_out = _expert(_exp_input_flat) # (len(_pos), self.emb_dim)
            _weighted_expert_out = _expert_out* k_prob_flat[_pos, _kth].unsqueeze(1) # (len(_pos), self.emb_dim)
            
            out_tokens[_pos] += _weighted_expert_out

        out_tokens = out_tokens.reshape(B, s_l, self.emb_dim)
        aus = {}
        return out_tokens, aus


if __name__ == "__main__":
    x = torch.randn((2,4,4))
    moelayer = MoeLayer(emb_dim=4,
                        hid_dim=5,
                        expert_num=5,
                        top_k=3)

    out, _ = moelayer(x)
    print(out.shape)
