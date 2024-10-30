from mlp import MLP
from attention import MultiHeadAttention

from torch import nn, Tensor


class EncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int, num_head: int, drop_prob: float = 0.1
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.mlp = MLP(d_model=d_model, d_hidden=d_hidden, drop_prob=drop_prob)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: Tensor, mask: Tensor | None = None):
        shortcut = x
        # 1. Self-Attention.
        out = self.attention(q=x, k=x, v=x, mask=mask)

        # 2. Add and norm.
        out = self.dropout1(out)
        out = self.ln1(out + shortcut)

        shortcut = out
        # 3. MLP
        out = self.mlp(out)
        # 4. Add and norm.
        out = self.ln2(out + shortcut)
        return out


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, d_hidden: int, num_head: int, drop_prob: float = 0.1
    ) -> None:
        super(DecoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_head=num_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.mlp = MLP(d_model=d_model, d_hidden=d_hidden, drop_prob=drop_prob)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self,
        dec: Tensor,
        enc: Tensor,
        target_mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ):
        # 1. Compute self-attention.
        shortcut = dec
        out = self.self_attention(q=dec, k=dec, v=dec, mask=target_mask)
        # 2. Add and Norm
        out = self.dropout1(out)
        out = self.ln1(out + shortcut)

        if enc is not None:
            # 3. Compute encoder - decoder attention.
            shortcut = out
            out = self.enc_dec_attention(q=dec, k=enc, v=enc, mask=src_mask)

            # 4. Add and Norm
            out = self.dropout2(out)
            out = self.ln2(out + shortcut)
        # 5. MLP
        shortcut = out
        out = self.mlp(out)
        # 6. Add and Norm
        out = self.dropout3(out)
        out = self.ln3(out + shortcut)

        return out
