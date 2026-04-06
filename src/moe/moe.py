import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """
    单个专家网络
    输入: [N, d_model]
    输出: [N, d_model]
    """
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKRouter(nn.Module):
    """
    路由器：
    对每个 token 选择 top-k 个专家
    """
    def __init__(self, d_model: int, num_experts: int, k: int = 2):
        super().__init__()
        assert k <= num_experts, "k must be <= num_experts"
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]

        返回:
            topk_idx: [B, T, k]
            topk_weight: [B, T, k]
            router_logits: [B, T, E]
        """
        router_logits = self.gate(x)                      # [B, T, E]
        topk_logits, topk_idx = torch.topk(router_logits, k=self.k, dim=-1)
        topk_weight = F.softmax(topk_logits, dim=-1)     # [B, T, k]
        return topk_idx, topk_weight, router_logits


class MoELayer(nn.Module):
    """
    稀疏 MoE 层
    每个 token 只路由到 top-k 个专家
    """
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int,
        k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k

        self.router = TopKRouter(d_model, num_experts, k)
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, d_hidden, dropout=dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]

        返回:
            out: [B, T, D]
            aux: 路由辅助信息
        """
        B, T, D = x.shape
        assert D == self.d_model

        topk_idx, topk_weight, router_logits = self.router(x)

        x_flat = x.reshape(B * T, D)                           # [N, D]
        topk_idx_flat = topk_idx.reshape(B * T, self.k)       # [N, k]
        topk_weight_flat = topk_weight.reshape(B * T, self.k) # [N, k]

        out_flat = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            token_pos, kth = torch.where(topk_idx_flat == expert_id)

            if token_pos.numel() == 0:
                continue

            expert_input = x_flat[token_pos]                   # [M, D]
            expert_output = expert(expert_input)               # [M, D]
            weights = topk_weight_flat[token_pos, kth].unsqueeze(-1)  # [M, 1]

            out_flat[token_pos] += expert_output * weights

        out = out_flat.reshape(B, T, D)

        aux = {
            "router_logits": router_logits,
            "topk_idx": topk_idx,
            "topk_weight": topk_weight
        }
        return out, aux


class MultiHeadSelfAttention(nn.Module):
    """
    简化版多头自注意力
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, T, d]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)  # [B, H, T, T]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = attn_probs @ v  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)


class TransformerBlockWithMoE(nn.Module):
    """
    用 MoE 替代普通 FFN 的 Transformer Block
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_hidden: int,
        num_experts: int,
        k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            d_hidden=d_hidden,
            num_experts=num_experts,
            k=k,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.dropout(self.attn(self.ln1(x)))
        moe_out, aux = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x, aux


def load_balance_loss(router_logits: torch.Tensor, topk_idx: torch.Tensor, num_experts: int):
    """
    简单负载均衡损失
    用来防止少数专家过载、其他专家闲置
    """
    probs = F.softmax(router_logits, dim=-1)  # [B, T, E]

    # importance: soft 路由概率总和
    importance = probs.sum(dim=(0, 1))  # [E]
    importance = importance / importance.sum()

    # load: 实际被 top-k 选中的频率
    one_hot = F.one_hot(topk_idx, num_classes=num_experts).float()  # [B, T, k, E]
    load = one_hot.sum(dim=(0, 1, 2))  # [E]
    load = load / load.sum()

    uniform = torch.full_like(load, 1.0 / num_experts)

    loss = F.mse_loss(importance, uniform) + F.mse_loss(load, uniform)
    return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    # 输入参数
    B, T, D = 4, 16, 64
    x = torch.randn(B, T, D)

    # 构建模型
    model = TransformerBlockWithMoE(
        d_model=64,
        n_heads=4,
        d_hidden=128,
        num_experts=6,
        k=2,
        dropout=0.1
    )

    # 前向传播
    out, aux = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Top-k indices shape:", aux["topk_idx"].shape)
    print("Top-k weights shape:", aux["topk_weight"].shape)

    # 负载均衡损失
    aux_loss = load_balance_loss(
        aux["router_logits"],
        aux["topk_idx"],
        num_experts=6
    )
    print("Load balance loss:", aux_loss.item())

    # 假设任务损失
    target = torch.randn_like(out)
    task_loss = F.mse_loss(out, target)

    total_loss = task_loss + 0.01 * aux_loss
    print("Task loss:", task_loss.item())
    print("Total loss:", total_loss.item())

    # 反向传播
    total_loss.backward()
    print("Backward success.")