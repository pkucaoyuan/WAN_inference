"""
WAN2.2 Model Implementation
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from .normalization import WanRMSNorm
from .rope import apply_rope


class WanFFN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.activation = nn.GELU(approximate='tanh')
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., eps=1e-6, use_rope=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        
        # Flash attention implementation
        self.attn = flash_attention

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if self.use_rope:
            q, k = apply_rope(q, k, freqs)

        q = q.permute(0, 2, 1, 3).contiguous()  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        attn_output = self.attn(q, k, v, seq_lens, grid_sizes)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(b, s, -1)

        return self.o(attn_output)


class WanCrossAttention(WanSelfAttention):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., eps=1e-6):
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, eps, use_rope=False)

    def forward(self, x, context, context_lens):
        B, L, C = x.shape
        H = self.num_heads
        
        q = self.norm_q(self.q(x)).view(B, L, H, -1)
        k = self.norm_k(self.k(context)).view(B, context.size(1), H, -1)
        v = self.v(context).view(B, context.size(1), H, -1)
        
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, H, L, D]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        
        attn_output = self.attn(q, k, v, context_lens, None)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, L, C)
        
        return self.o(attn_output)


class WanAttentionBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_dim,
                 qkv_bias=False,
                 qk_norm=False,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.norm1 = WanRMSNorm(dim, eps=eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qkv_bias, qk_norm, eps=eps)
        self.norm2 = WanRMSNorm(dim, eps=eps)
        self.ffn = WanFFN(dim, ffn_dim)
        self.norm3 = WanRMSNorm(dim, eps=eps) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, qkv_bias, qk_norm, eps=eps)

    def forward(self, x, context, context_lens, e, seq_lens, grid_sizes, freqs):
        # Self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # Cross-attention
        cross_out = self.cross_attn(self.norm3(x), context, context_lens)
        x = x + cross_out

        # FFN
        ffn_input = self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
        ffn_out = self.ffn(ffn_input)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + ffn_out * e[5].squeeze(2)

        return x


class WanModel(ModelMixin, ConfigMixin):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_dim,
                 num_layers,
                 qkv_bias=False,
                 qk_norm=False,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, num_heads, ffn_dim, qkv_bias, qk_norm, 
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # Final norm
        self.final_norm = WanRMSNorm(dim, eps=eps)

    def forward(self, x, context, context_lens, e, seq_lens, grid_sizes, freqs):
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, context, context_lens, e, seq_lens, grid_sizes, freqs)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x