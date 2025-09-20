# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, active_mask=None, cached_qkv=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            active_mask(Tensor): Shape [L], True for active tokens
            cached_qkv(dict): Cached Q,K,V for frozen tokens
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # CAT算法：冻结token复用QKV，激活token计算新QKV
        if active_mask is not None and cached_qkv is not None:
            # 使用缓存的QKV + 新计算的激活token QKV
            active_indices = torch.where(active_mask)[0]
            frozen_indices = torch.where(~active_mask)[0]
            
            # 只对激活token计算新的Q,K,V
            x_active = x[:, active_indices, :]
            q_active = self.norm_q(self.q(x_active)).view(b, len(active_indices), n, d)
            k_active = self.norm_k(self.k(x_active)).view(b, len(active_indices), n, d)
            v_active = self.v(x_active).view(b, len(active_indices), n, d)
            
            # 构建完整的Q,K,V矩阵
            q = torch.zeros(b, s, n, d, device=x.device, dtype=q_active.dtype)
            k = torch.zeros(b, s, n, d, device=x.device, dtype=k_active.dtype)
            v = torch.zeros(b, s, n, d, device=x.device, dtype=v_active.dtype)
            
            # 填入激活token的新QKV
            q[:, active_indices, :, :] = q_active
            k[:, active_indices, :, :] = k_active
            v[:, active_indices, :, :] = v_active
            
            # 填入冻结token的缓存QKV
            if len(frozen_indices) > 0:
                q[:, frozen_indices, :, :] = cached_qkv['q_frozen']
                k[:, frozen_indices, :, :] = cached_qkv['k_frozen']
                v[:, frozen_indices, :, :] = cached_qkv['v_frozen']
            
            print(f"   ⚡ QKV投影节省: 只计算{len(active_indices)}/{s}个token的QKV")
        else:
            # 标准计算：所有token计算QKV
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 enable_token_pruning=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.enable_token_pruning = enable_token_pruning

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # QKV缓存用于冻结token优化
        self._frozen_qkv_cache = None

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        active_mask=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # Token裁剪优化：只计算激活token
        if active_mask is not None and self.enable_token_pruning:
            # 获取激活token的索引
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) < x.size(1):  # 确实有token被裁剪
                # 精确计时：分析每个环节的性能
                import time
                layer_start = time.time()
                # 只对激活token进行计算
                x_active = x[:, active_indices, :]
                # e是tuple，需要分别处理每个元素
                e_active = tuple(e_elem[:, active_indices, :] if e_elem.size(1) == x.size(1) else e_elem 
                               for e_elem in e)
                
                # Self-attention：优化版CAT算法，复用冻结token的QKV
                # 只计算激活token的Q，复用冻结token的K,V
                
                # 高效获取冻结token的索引（避免Python循环）
                all_indices = torch.arange(x.size(1), device=x.device, dtype=torch.long)
                frozen_mask = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                frozen_mask[active_indices] = False
                frozen_indices = torch.where(frozen_mask)[0]
                
                # 计算激活token的归一化输入
                x_norm = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
                
                # 检查QKV缓存：缓存的冻结token是否仍在当前冻结集合中
                cache_valid = False
                if (hasattr(self, '_frozen_qkv_cache') and 
                    self._frozen_qkv_cache and 
                    len(frozen_indices) > 0 and
                    self._frozen_qkv_cache['valid']):
                    
                    cached_indices = self._frozen_qkv_cache['frozen_indices']
                    # 检查缓存的token是否都仍然被冻结（允许新增冻结token）
                    cached_still_frozen = torch.isin(cached_indices, frozen_indices).all()
                    cache_valid = cached_still_frozen.item()
                    
                    if cache_valid:
                        print(f"   🔄 QKV缓存有效: {len(cached_indices)}个token复用")
                    else:
                        if hasattr(self, '_frozen_qkv_cache') and self._frozen_qkv_cache:
                            cached_indices = self._frozen_qkv_cache['frozen_indices']
                            print(f"   ❌ QKV缓存失效: 缓存{len(cached_indices)}个 vs 当前{len(frozen_indices)}个")
                        else:
                            print(f"   ❌ QKV缓存失效: 无缓存数据")
                
                if cache_valid:
                    # 使用QKV缓存的混合attention计算
                    y_mixed = self._compute_mixed_attention(x_norm, active_indices, frozen_indices, 
                                                          seq_lens, grid_sizes, freqs)
                    # 简化QKV缓存输出
                    pass  # QKV缓存命中，无需输出
                else:
                    # CAT算法：传递active_mask和cached_qkv实现真正的QKV复用
                    cached_qkv_data = self._frozen_qkv_cache if hasattr(self, '_frozen_qkv_cache') and self._frozen_qkv_cache else None
                    
                    # 创建完整的active_mask
                    full_active_mask = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                    if len(frozen_indices) > 0:
                        full_active_mask[frozen_indices] = False
                    
                    attn_start = time.time()
                    y_mixed = self.self_attn(x_norm, seq_lens, grid_sizes, freqs, 
                                           active_mask=full_active_mask, cached_qkv=cached_qkv_data)
                    attn_time = time.time() - attn_start
                
                # 缓存当前的Q,K,V用于下一步（基于预测的冻结token）
                cache_start = time.time()
                self._cache_frozen_qkv(x_norm, frozen_indices, seq_lens, grid_sizes, freqs)
                cache_time = time.time() - cache_start
                
                # Algorithm 1: Line 3-5: 只有选中token使用attention结果更新
                y = torch.zeros_like(x)
                y[:, active_indices, :] = y_mixed[:, active_indices, :].to(x.dtype)
                
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    # 只更新激活token，冻结token保持原值
                    x_new = x + y * e[2].squeeze(2)
                    x[:, active_indices, :] = x_new[:, active_indices, :].to(x.dtype)
                    # 冻结token保持x[:, frozen_indices, :]不变
                
                # Cross-attention & FFN：按CAT算法实现
                def cross_attn_ffn_pruned(x, context, context_lens, e, active_indices):
                    # Algorithm 1: Cross-attention所有token参与，但只更新激活token
                    cross_out_full = self.cross_attn(self.norm3(x), context, context_lens)
                    
                    # ✅ 修复：只有激活token接收cross-attention结果
                    cross_out = torch.zeros_like(x)
                    cross_out[:, active_indices, :] = cross_out_full[:, active_indices, :]
                    x = x + cross_out  # 只有激活token接收cross-attention结果
                    
                    # Algorithm 1: Line 4: 只有选中token (Ts,t) 通过MLP(FFN)更新
                    x_active = x[:, active_indices, :]
                    e_ffn_active = tuple(e_elem[:, active_indices, :] if e_elem.size(1) == x.size(1) else e_elem 
                                       for e_elem in e)
                    ffn_input = self.norm2(x_active).float() * (1 + e_ffn_active[4].squeeze(2)) + e_ffn_active[3].squeeze(2)
                    ffn_out = self.ffn(ffn_input)  # 🔥 只计算激活token的FFN
                    print(f"   ⚡ FFN节省: 只计算{len(active_indices)}/{x.size(1)}token ({100*len(active_indices)/x.size(1):.1f}%)")
                    
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        x_active = x_active + ffn_out * e_ffn_active[5].squeeze(2)
                    
                    # Algorithm 1: Line 4: 更新选中token的hidden state
                    x[:, active_indices, :] = x_active.to(x.dtype)
                    # Algorithm 1: Line 7: 未选中token保持上一步状态（已经在x中）
                    return x
                
                ffn_start = time.time()
                x = cross_attn_ffn_pruned(x, context, context_lens, e, active_indices)
                ffn_time = time.time() - ffn_start
                
                layer_total_time = time.time() - layer_start
                print(f"   📊 层计时: Attention={attn_time:.3f}s, Cache={cache_time:.3f}s, FFN={ffn_time:.3f}s, Total={layer_total_time:.3f}s")
            else:
                # 没有token被裁剪，正常计算
                y = self.self_attn(
                    self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                    seq_lens, grid_sizes, freqs)
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + y * e[2].squeeze(2)

                def cross_attn_ffn(x, context, context_lens, e):
                    x = x + self.cross_attn(self.norm3(x), context, context_lens)
                    y = self.ffn(
                        self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        x = x + y * e[5].squeeze(2)
                    return x

                x = cross_attn_ffn(x, context, context_lens, e)
        else:
            # 标准计算路径（无裁剪）
            y = self.self_attn(
                self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
                seq_lens, grid_sizes, freqs)
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[2].squeeze(2)

            def cross_attn_ffn(x, context, context_lens, e):
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                y = self.ffn(
                    self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    x = x + y * e[5].squeeze(2)
                return x

            x = cross_attn_ffn(x, context, context_lens, e)
        return x
    
    def _cache_frozen_qkv(self, x_norm, frozen_indices, seq_lens, grid_sizes, freqs):
        """缓存冻结token的Q,K,V用于下一步复用"""
        if len(frozen_indices) > 0:
            # 计算并缓存冻结token的Q,K,V
            x_frozen = x_norm[:, frozen_indices, :]
            
            # 计算冻结token的Q,K,V（用于下一步复用）
            b, s_frozen = x_frozen.size(0), len(frozen_indices)
            n, d = self.self_attn.num_heads, self.self_attn.head_dim
            q_frozen = self.self_attn.norm_q(self.self_attn.q(x_frozen)).view(b, s_frozen, n, d)
            k_frozen = self.self_attn.norm_k(self.self_attn.k(x_frozen)).view(b, s_frozen, n, d)
            v_frozen = self.self_attn.v(x_frozen).view(b, s_frozen, n, d)
            
            self._frozen_qkv_cache = {
                'frozen_indices': frozen_indices.clone(),
                'q_frozen': q_frozen.clone(),
                'k_frozen': k_frozen.clone(), 
                'v_frozen': v_frozen.clone(),
                'valid': True
            }
        else:
            self._frozen_qkv_cache = None
    
    def _compute_mixed_attention(self, x_norm, active_indices, frozen_indices, seq_lens, grid_sizes, freqs):
        """计算混合attention：新的激活token QKV + 缓存的冻结token QKV"""
        if self._frozen_qkv_cache and self._frozen_qkv_cache['valid']:
            # 计算激活token的Q,K,V
            x_active = x_norm[:, active_indices, :]
            b, s_active = x_active.size(0), len(active_indices)
            n, d = self.self_attn.num_heads, self.self_attn.head_dim
            
            q_active = self.self_attn.norm_q(self.self_attn.q(x_active)).view(b, s_active, n, d)
            k_active = self.self_attn.norm_k(self.self_attn.k(x_active)).view(b, s_active, n, d)
            v_active = self.self_attn.v(x_active).view(b, s_active, n, d)
            
            # 获取缓存的冻结token Q,K,V
            q_frozen = self._frozen_qkv_cache['q_frozen']
            k_frozen = self._frozen_qkv_cache['k_frozen'] 
            v_frozen = self._frozen_qkv_cache['v_frozen']
            
            # 重新组合完整的Q,K,V矩阵
            full_seq_len = x_norm.size(1)
            q_full = torch.zeros(b, full_seq_len, n, d, device=x_norm.device, dtype=q_active.dtype)
            k_full = torch.zeros(b, full_seq_len, n, d, device=x_norm.device, dtype=k_active.dtype)
            v_full = torch.zeros(b, full_seq_len, n, d, device=x_norm.device, dtype=v_active.dtype)
            
            # 填入激活token的新Q,K,V
            q_full[:, active_indices, :, :] = q_active
            k_full[:, active_indices, :, :] = k_active
            v_full[:, active_indices, :, :] = v_active
            
            # 填入冻结token的缓存Q,K,V
            q_full[:, frozen_indices, :, :] = q_frozen
            k_full[:, frozen_indices, :, :] = k_frozen
            v_full[:, frozen_indices, :, :] = v_frozen
            
            # 使用混合的Q,K,V计算attention（简化版本）
            # 注意：这里需要手动实现attention，因为flash_attention不支持混合输入
            scale = (d ** -0.5)
            attention_scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_out = torch.matmul(attention_weights, v_full)  # [B, L, H, d]
            
            return attention_out.flatten(-2).to(x_norm.dtype)  # [B, L, H*d] 保持数据类型一致
        
        # 回退到完整计算
        return self.self_attn(x_norm, seq_lens, grid_sizes, freqs)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps, enable_token_pruning=True) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        active_mask=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            active_mask=active_mask)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
