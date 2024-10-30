"""
Scaled-Dot Product Attention Layer and Multi-Head Attention.
"""

import torch
from torch import nn, Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.attention = ScaledDotProductAttention()
        self.num_head = num_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        # 1. Apply linear projection.
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        # 2. Split tensor by number of heads.
        q, k, v = self.split(q), self.split(k), self.split(v)
        # 3. Do scaled dot-product attention.
        out = self.attention(q, k, v, mask)
        # 4. Concat and project
        out = self.concat(out)
        out = self.w_o(out)
        return out

    def split(self, x: Tensor):
        """Split tensor by number of heads.

        Args:
            x (Tensor): input tensor of shape (batch_size, length, d_model)

        Returns:
            Tensor: output tensor of shape (batch_size, length, num_head, d_head)
        """
        batch_size, length, d_model = x.size()
        d_head = d_model // self.num_head
        return x.view(batch_size, length, self.num_head, d_head).transpose(1, 2)

    def concat(self, x: Tensor):
        """Inverse function of self.split.

        Args:
            x (Tensor): input tensor of shape (batch_size, length, num_head, d_head)

        Returns:
            Tensor: output tensor of shape (batch_size, length, d_model)
        """
        batch_size, num_head, length, d_head = x.size()
        d_model = num_head * d_head
        x = x.transpose(1, 2)
        return x.view(batch_size, length, d_model)


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        dk = k.size()[3]
        kt = k.transpose(2, 3)
        # 1. Scale and dot product.
        score = torch.matmul(q, kt) / torch.sqrt(dk)
        # 2. Apply mask(opt.)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        # 3. Apply softmax to get attention score.
        score = self.softmax(score)
        # 4. Compute weighted sum.
        v = torch.matmul(score, v)
        return v
