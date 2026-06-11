# model.py - SpiderWeb Self-Attention Model Definition
#
# Core formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + lambda * M_web) * V
#
# M_web = M_center + M_hierarchy + M_position
#
# Four attention variants for ablation study:
#   A: Pure Transformer (no bias)
#   B: Position bias only
#   C: Simple structure (hierarchy + position)
#   D: Full SpiderWeb (center + hierarchy + position)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Multi-Head Attention with optional M_web structural bias
# ---------------------------------------------------------------------------

class SpiderWebMultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional M_web structural bias."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, M_web=None, lambda_=0.5, return_attention=False):
        """
        Args:
            x: (B, N, D) input
            mask: (B, 1, 1, N) attention mask (True = attend)
            M_web: (B, N, N) structural bias matrix, or None for pure attention
            lambda_: bias strength coefficient
            return_attention: if True, also returns (attn_weights, pre_softmax_scores)
        """
        B, N, D = x.shape

        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        pre_bias_scores = scores.clone()

        if M_web is not None and lambda_ > 0:
            scores = scores + lambda_ * M_web.unsqueeze(1)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        if return_attention:
            return self.W_o(out), attn_weights.detach()
        return self.W_o(out)


# ---------------------------------------------------------------------------
# Transformer Encoder Layer with SpiderWeb attention
# ---------------------------------------------------------------------------

class SpiderWebEncoderLayer(nn.Module):
    """Transformer encoder layer supporting SpiderWeb structural bias."""

    def __init__(self, d_model, n_heads, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = SpiderWebMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None, M_web=None, lambda_=0.5, return_attention=False):
        result = self.self_attn(x, mask, M_web, lambda_, return_attention=return_attention)
        if return_attention:
            attn_out, attn_weights = result
        else:
            attn_out = result
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        if return_attention:
            return x, attn_weights
        return x


# ---------------------------------------------------------------------------
# Full SpiderWeb Classifier
# ---------------------------------------------------------------------------

class SpiderWebClassifier(nn.Module):
    """
    Transformer classifier supporting multiple attention bias modes.

    bias_mode:
      "none"   -> Model A: Pure Transformer
      "pos"    -> Model B: Position bias only
      "simple" -> Model C: Hierarchy + Position bias
      "full"   -> Model D: SpiderWeb (Center + Hierarchy + Position)
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, n_classes=8, max_len=80, dropout=0.1,
                 bias_mode="full"):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.bias_mode = bias_mode

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            SpiderWebEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids, mask, M_web=None, lambda_=0.5, return_attention=False):
        B, N = token_ids.shape
        positions = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(token_ids) + self.pos_embedding(positions)

        all_attentions = []
        for layer in self.layers:
            result = layer(x, mask, M_web, lambda_, return_attention=return_attention)
            if return_attention:
                x, attn_w = result
                all_attentions.append(attn_w)
            else:
                x = result

        valid_mask = mask.squeeze(1).squeeze(1).float()
        x = (x * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)

        logits = self.classifier(x)

        if return_attention:
            return logits, all_attentions
        return logits


def create_model(bias_mode, vocab_size=500, d_model=128, n_heads=4,
                 n_layers=2, d_ff=512, n_classes=8, max_len=80, dropout=0.1):
    return SpiderWebClassifier(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=d_ff, n_classes=n_classes,
        max_len=max_len, dropout=dropout, bias_mode=bias_mode,
    )


if __name__ == "__main__":
    from data import create_dataloaders, build_m_web

    train_loader, test_loader, ds = create_dataloaders(num_samples=200, batch_size=16)

    for mode in ["none", "full"]:
        model = create_model(bias_mode=mode)
        batch = next(iter(train_loader))
        token_ids = batch["token_ids"]
        mask = (token_ids != 0).unsqueeze(1).unsqueeze(2)

        M = None
        if mode == "full":
            M = build_m_web(batch["levels"], batch["segments"], batch["seq_len"])

        logits, attns = model(token_ids, mask, M, lambda_=0.5, return_attention=True)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Mode={mode:8s} | logits={tuple(logits.shape)} | "
              f"attn_layers={len(attns)} | attn_shape={attns[-1].shape} | params={n_params:,}")

    print("model.py OK")
