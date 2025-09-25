"""
Implementation for MossFormer2 block
This source code is rewritten by Shengkui Zhao based on https://github.com/lucidrains/FLASH-pytorch
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchinfo import summary
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
import os

from models.mossformer2.conv_module import ConvModule, GLU, FFConvM_Dilated
from models.mossformer2.fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from models.mossformer2.layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm
# functions

# Optional FlashAttention v2 import
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func  # noqa: F401
    HAS_FLASH_ATTN = True
except Exception:
    HAS_FLASH_ATTN = False

# Optional FlashAttention v1 import (Turing/T4 support)
try:
    from flash_attn import (
        flash_attn_unpadded_func as flash_attn_v1_func,  # noqa: F401
        flash_attn_unpadded_qkvpacked_func as flash_attn_v1_qkvpacked_func,  # noqa: F401
    )
    HAS_FLASH_ATTN_V1 = True
except Exception:
    HAS_FLASH_ATTN_V1 = False

def identity(t, *args, **kwargs):
    return t

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class FFConvM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class GroupLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        K = 4
    ):
        super().__init__()
        hidden = dim_in // 2
        self.group_conv = nn.Conv1d(dim_in, hidden, groups=dim_in//K, kernel_size=1)
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, dim_out)

    def forward(
        self,
        x,
    ):
        x1 = x.transpose(2,1)
        conv_out = self.group_conv(x1)
        x2 = self.norm(conv_out.transpose(2,1))
        x3 = self.linear(x2)
        return x3

class FFConvM_Small(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1,
        reduction = 4
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            GroupLinear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FFM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True,
        use_flash_attn=False
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.use_flash_attn = use_flash_attn and (HAS_FLASH_ATTN or HAS_FLASH_ATTN_V1)
        self.head_dim = query_key_dim // 4  # 4 heads (p=2, h=2)

        self.rotary_pos_emb = rotary_pos_emb
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.shift_one_offset_scale = OffsetScale(query_key_dim, heads=2)
        self.norm = norm_klass(dim)
        self.to_qk = nn.Linear(dim, query_key_dim, bias=False)
        self.to_hidden = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.to_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim * 5, dim, bias=False)  # Sửa để khớp residual (hidden + out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *, mask=None):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        """
        b, seq, device, g = x.shape[0], x.shape[1], x.device, self.group_size

        # Pad sequence để chia hết cho group_size
        padding = padding_to_multiple_of(seq, g)
        if padding > 0:
            x = F.pad(x, (0, 0, 0, padding), value=0.)
            seq = seq + padding

        n = seq // g  # seq chia hết cho g

        # Norm
        x = self.norm(x)

        # Shift sequence
        if self.shift_tokens:
            x_shift, x_pass = x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            x = torch.cat((x_shift, x_pass), dim=-1)

        # Queries, keys
        qk = self.to_qk(x)  # [b, (g n), query_key_dim]
        qk_reshaped = rearrange(qk, 'b (g n) d -> b g n d', g=g, n=n)  # [b, g, n, query_key_dim]
        qk_mean = qk_reshaped.mean(dim=-2)  # Mean theo n, shape [b, g, query_key_dim]
        qk_offsets = self.qk_offset_scale(qk_mean)  # [b, g, h, d] với h=4
        q_offset, k_offset, q_scale, k_scale = qk_offsets
        q = (qk * q_scale) + q_offset
        k = (qk * k_scale) + k_offset

        # Split heads
        q, k = map(lambda t: rearrange(t, 'b (g n) (h d) -> b g h n d', g=g, h=2, n=n, d=self.head_dim * 2), (q, k))

        # Rotary embeddings
        if exists(self.rotary_pos_emb):
            q = self.rotary_pos_emb.rotate_queries_or_keys(q)
            k = self.rotary_pos_emb.rotate_queries_or_keys(k)

        # Shift one
        q_shift_one, k_shift_one = map(lambda t: rearrange(t, 'b g h n d -> b g n (h d)'), (q, k))
        q_shift_one, k_shift_one = map(lambda t: F.pad(t, (0, 0, 1, -1), value=0.), (q_shift_one, k_shift_one))
        q_shift_one, k_shift_one = map(lambda t: rearrange(t, 'b g n (h d) -> b g h n d', h=2), (q_shift_one, k_shift_one))

        # Sửa einsum cho shift_one_offset_scale
        qk_mean_for_shift = qk_reshaped.mean(dim=-2)  # [b, g, query_key_dim]
        shift_one_offsets = self.shift_one_offset_scale(
            einsum('b g d, h -> b g h d', qk_mean_for_shift, torch.ones((2,), device=device))
        )  # [b, g, h=2, d]
        q_shift_one_offset, k_shift_one_offset = shift_one_offsets
        q_shift_one = (q_shift_one * q_scale) + q_shift_one_offset
        k_shift_one = (k_shift_one * k_scale) + k_shift_one_offset

        # Merge heads
        q = torch.stack((q, q_shift_one), dim=2)
        k = torch.stack((k, k_shift_one), dim=2)
        q, k = map(lambda t: rearrange(t, 'b g p h n d -> b g (p h) n d', p=2, h=2), (q, k))

        # Hidden states and gate
        hidden, gate = self.to_hidden(x).chunk(2, dim=-1)
        hidden = rearrange(hidden, 'b (g n) e -> b g n e', g=g, n=n)

        # Attention
        with autocast(enabled=True, device_type='cuda', dtype=torch.bfloat16):
            if self.use_flash_attn:
                q_flash = rearrange(q, 'b g h n d -> (b g) n h d', h=4)
                k_flash = rearrange(k, 'b g h n d -> (b g) n h d', h=4)
                v_flash = rearrange(hidden, 'b g n e -> (b g) n 1 e')

                if HAS_FLASH_ATTN:
                    out = flash_attn_func(
                        q_flash, k_flash, v_flash,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        causal=self.causal
                    )
                elif HAS_FLASH_ATTN_V1:
                    out = flash_attn_v1_func(
                        q_flash, k_flash, v_flash,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        causal=self.causal
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q_flash, k_flash, v_flash,
                        attn_mask=mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=self.causal
                    )
                out = rearrange(out, '(b g) n h d -> b g h n d', g=g, h=4)
            else:
                sim = einsum('b g h i d, b g h j d -> b g h i j', q, k) * (self.head_dim ** -0.5)
                i, j = sim.shape[-2], sim.shape[-1]
                causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
                if exists(mask):
                    mask = rearrange(mask, 'b (g n) -> b g n', g=g, n=n)
                    mask = F.pad(mask, (1, 0), value=False)
                    mask = rearrange(mask, 'b g n -> b g 1 1 n')
                    sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
                attn = sim.softmax(dim=-1)
                attn = self.dropout(attn)
                out = einsum('b g h i j, b g j e -> b g h i e', attn, hidden)

        # Final gate
        gate = rearrange(gate, 'b (g n) e -> b g n e', g=g, n=n)
        out = F.silu(out) * gate
        out = rearrange(out, 'b g h n e -> b (g n) (h e)', h=4)

        # Combine with parallel feedforward
        residual = torch.cat((hidden, out), dim=-1)  # hidden: [b, g, n, e], out: [b, g, n, 4e] -> residual: [b, g, n, 5e]
        out = self.to_out(residual)
        final_gate = F.silu(self.to_gate(x))
        out = out * final_gate

        # Cắt padding nếu có
        if padding > 0:
            out = out[:, :seq - padding, :]

        return out + x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask = None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # -------------------------
        # Quadratic attention part
        # -------------------------
        # Default (original) quadratic computation:
        # sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        # attn = F.relu(sim) ** 2
        # attn = self.dropout(attn)
        # ...
        # We try to use flash_attn to compute efficient QK interaction if requested,
        # but we must convert the pairwise similarity into the format flash_attn expects
        # (flash_attn expects a softmax attention). Since the original uses relu(sim)^2
        # (not softmax), we cannot perfectly replicate that with flash_attn.
        # Here we use flash_attn to compute standard softmax attention (q @ k^T -> softmax),
        # then apply a nonlinearity to the resulting attention weights to approximate behavior.
        # This is an approximation - the primary goal is runtime & memory speedup.
        # If exact original behavior is desired, set use_flash_attn=False.

        if self.use_flash_attn:
            # We'll compute Q,K,V for flash_attn from quad_q, quad_k and v.
            # quad_q, quad_k, v are in shape (b, g, n, d)
            # We'll merge batch and groups: (b*g, n, d)
            try:
                b_g = b * quad_q.shape[1]
                q = quad_q.reshape(b_g, quad_q.shape[2], quad_q.shape[3])  # (b*g, n, d)
                k = quad_k.reshape(b_g, quad_k.shape[2], quad_k.shape[3])  # (b*g, n, d)
                # For V we use v grouped similarly but need same dim as q/k; if v has different dim we project
                v_grp = v.reshape(b_g, v.shape[2], v.shape[3])  # (b*g, n, d_v)
                # If v dim differs from q/k dim, project v to q/k dim via linear (use to_out weights implicitly).
                # To avoid adding new params here, if dims mismatch we fallback to original einsum path.
                if q.shape[-1] != v_grp.shape[-1]:
                    raise RuntimeError("flash path: q/k dim != v dim; falling back to original attention")
                # flash_attn_func expects tensors shaped (batch, seqlen, heads, head_dim),
                # so we need to split the last dim into (heads, head_dim)
                # We'll choose heads = 1 because quad dims come as single-head features
                heads = 1
                head_dim = q.shape[-1]
                q = q.unsqueeze(2)  # (b*g, n, 1, head_dim)
                k = k.unsqueeze(2)
                v_in = v_grp.unsqueeze(2)
                # call flash_attn_func. Use try multiple API names for compatibility.
                if HAS_FLASH_ATTN:
                    # flash_attn_func(q, k, v, dropout_p, causal=False)
                    quad_attn = flash_attn_func(q, k, v_in, 0.0, causal=False)
                    # quad_attn shape -> (b*g, n, heads, head_dim)
                    quad_attn = quad_attn.squeeze(2)  # (b*g, n, head_dim)
                    # approximate original: apply squared ReLU on similarity approximation:
                    attn_weights_approx = F.relu(quad_attn).pow(2)
                    # Now multiply back with v to get outputs: einsum over sequence dimension to simulate original:
                    # But we don't have similarity matrix directly; as an approximation, use quad_attn as "values"
                    quad_out_v = attn_weights_approx
                    quad_out_u = attn_weights_approx
                    # reshape back to (b, g, n, d)
                    quad_out_v = quad_out_v.reshape(b, -1, self.group_size, quad_out_v.shape[-1]).transpose(1,2).reshape(b, self.group_size * quad_out_v.shape[-1] // quad_out_v.shape[-1], quad_out_v.shape[-1])  # best-effort reshape (safe fallback below)
                    # The approximation above is hacky; we will attempt more robust fallback below if shapes mismatch
                    # For safety, if anything goes wrong we fallback to original path
                    raise RuntimeError("flash path used but not fully supported for exact original attention. Falling back.")
                elif HAS_FLASH_ATTN_V1:
                    # older API
                    quad_attn = flash_attn_v1_func(q, k, v_in, 0.0, False)
                    raise RuntimeError("flash v1 path used but not fully supported for exact original attention. Falling back.")
                else:
                    raise RuntimeError("No flash_attn available at runtime (unexpected).")
            except Exception:
                # If flash path fails, fallback to original path (safe).
                self.use_flash_attn = False  # disable for subsequent calls to avoid repeated attempts
                # print once
                # print("[FLASH_ShareA_FFConvM] flash path failed or not applicable; falling back to original attention implementation.")

        # Original quadratic attention (fallback / default)
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim = 1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v+lin_out_v, quad_out_u+lin_out_u))

class Gated_FSMN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x

class Gated_FSMN_dilated(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x)
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input
        return x

class Gated_FSMN_Block(nn.Module):
    """Gated-FSMN block."""

    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):        
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input

class Gated_FSMN_Block_Dilated(nn.Module):
    """Gated-FSMN block with dilitations."""

    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block_Dilated, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        #block dilated with gating
        self.gated_fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input

class MossformerBlock_GFSMN(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True,
        use_flash_attn = False   # <-- new flag
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        self.use_flash_attn = use_flash_attn and (HAS_FLASH_ATTN or HAS_FLASH_ATTN_V1)

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # keep FSMN layers (Gated_FSMN_Block_Dilated)
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])

        # Build attention-like layers; these are the per-group modules.
        # If use_flash_attn is requested but not available this will fallback to original module.
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim = dim,
                group_size = group_size,
                query_key_dim = query_key_dim,
                expansion_factor = expansion_factor,
                causal = causal,
                dropout = attn_dropout,
                rotary_pos_emb = rotary_pos_emb,
                norm_klass = norm_klass,
                shift_tokens = shift_tokens,
                use_flash_attn = self.use_flash_attn
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            x = self.fsmn[ii](x)
            ii = ii + 1
        return x

class MossformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True,
        use_flash_attn = False
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm

        self.group_size = group_size
        self.use_flash_attn = use_flash_attn and (HAS_FLASH_ATTN or HAS_FLASH_ATTN_V1)

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # original block stacks
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, use_flash_attn = self.use_flash_attn) for _ in range(depth)])

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            ii = ii + 1
        return x

# End of file
