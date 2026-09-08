import math
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F


@dataclass
class VaaniConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12        # < n_head enables GQA
    d_model: int = 768
    ffn_mult: float = 8 / 3    # SwiGLU hidden ≈ 8/3·d, rounded to 64
    block_size: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        dt = x.dtype; x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.w


def precompute_rope(hd, end, theta):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    t = torch.arange(end).float()
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)          # [end, hd]
    return emb.cos(), emb.sin()


def apply_rope(q, k, cos, sin):
    cos = cos[None, None]; sin = sin[None, None]     # [1,1,T,hd]
    def rot(x):
        x1, x2 = x.chunk(2, dim=-1); return torch.cat((-x2, x1), dim=-1)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh, self.nkv = c.n_head, c.n_kv_head
        self.hd = c.d_model // c.n_head
        self.wq = nn.Linear(c.d_model, self.nh * self.hd, bias=False)
        self.wk = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wv = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.wo = nn.Linear(self.nh * self.hd, c.d_model, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k = k.repeat_interleave(r, dim=1); v = v.repeat_interleave(r, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = int(c.ffn_mult * c.d_model); h = 64 * ((h + 63) // 64)
        self.w1 = nn.Linear(c.d_model, h, bias=False)   # gate
        self.w3 = nn.Linear(c.d_model, h, bias=False)   # up
        self.w2 = nn.Linear(h, c.d_model, bias=False)   # down
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d_model, c.norm_eps), Attention(c)
        self.n2, self.mlp  = RMSNorm(c.d_model, c.norm_eps), MLP(c)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class Vaani(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.tie_embeddings:
            self.lm_head.weight = self.tok.weight
        cos, sin = precompute_rope(c.d_model // c.n_head, c.block_size, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():          # scaled residual init
            if n.endswith("wo.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))
    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx)
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        for b in self.blocks:
            x = b(x, cos, sin)
        x = self.norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
