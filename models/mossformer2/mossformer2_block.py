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

from models.mossformer2.conv_module import ConvModule, GLU, FFConvM_Dilated
from models.mossformer2.fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from models.mossformer2.layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm

# Import Flash Attention functions
try:
    from flash_attn import (
        flash_attn_func, 
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_func,
        flash_attn_with_kvcache
    )
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention available with full feature set")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_kvpacked_func = None
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
    print("Warning: Flash Attention not available. Using PyTorch native attention.")

# functions

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
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 1.,
        causal = False,
        dropout = 0.1,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = True,
        use_flash_attn = True,
        flash_attn_config = None,
        debug_fa: bool = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        self.debug_fa = debug_fa
        
        # Flash Attention configuration with enhanced defaults
        self.flash_attn_config = flash_attn_config or {}
        self.use_qkv_packed = self.flash_attn_config.get('use_qkv_packed', True)
        self.use_alibi = self.flash_attn_config.get('use_alibi', True)  # Enable ALiBi by default
        self.use_sliding_window = self.flash_attn_config.get('use_sliding_window', True)  # Enable sliding window by default
        self.window_size = self.flash_attn_config.get('window_size', (64, 64))  # Default window size
        self.use_softcap = self.flash_attn_config.get('use_softcap', True)  # Enable softcap by default
        self.softcap_value = self.flash_attn_config.get('softcap_value', 0.1)  # Default softcap value
        self.deterministic = self.flash_attn_config.get('deterministic', False)

        # positional embeddings
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        
        self.to_hidden = FFConvM(
            dim_in = dim,
            dim_out = hidden_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        self.to_qk = FFConvM(
            dim_in = dim,
            dim_out = query_key_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)

        self.to_out = FFConvM(
            dim_in = dim*2,
            dim_out = dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        
        self.gateActivate=nn.Sigmoid()
        
        # Flash Attention specific parameters
        if self.use_flash_attn:
            # Find the largest divisor of query_key_dim that's <= 16 (for efficiency)
            max_heads = min(16, query_key_dim)
            self.num_heads = 1
            for i in range(1, max_heads + 1):
                if query_key_dim % i == 0:
                    self.num_heads = i
            
            self.head_dim = query_key_dim // self.num_heads
            
            # No extra params: we'll slice/pad V/U at runtime to match Q/K dim for FlashAttention
            
            # ALiBi slopes for positional bias
            if self.use_alibi:
                self.alibi_slopes = self._get_alibi_slopes(self.num_heads)
            else:
                self.alibi_slopes = None

    def _get_alibi_slopes(self, num_heads):
        """Generate ALiBi slopes for positional bias"""
        def get_slopes(heads):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            
            if math.log2(heads).is_integer():
                return get_slopes_power_of_2(heads)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(heads))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:heads-closest_power_of_2]
        
        slopes = get_slopes(num_heads)
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(
        self,
        x,
        *,
        mask = None
    ):

        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        # prenorm
        #x = self.fsmn(x)
        normed_x = x #self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        residual = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        # Handle optimized attention output format
        if self.use_flash_attn and FLASH_ATTN_AVAILABLE:
            # Optimized attention already returns the final attention outputs
            # Ensure output dimension matches expected dim*2
            if att_v.shape[-1] + att_u.shape[-1] != x.shape[-1] * 2:
                # If dimensions don't match, use original gating mechanism
                out = (att_u*v ) * self.gateActivate(att_v*u)
            else:
                out = torch.cat([att_v, att_u], dim=-1)
        else:
            # Original MossFormer2 gating mechanism
            out = (att_u*v ) * self.gateActivate(att_v*u)
        
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask = None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        if self.use_flash_attn:
            return self.cal_flash_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)
        else:
            return self.cal_standard_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

    def cal_flash_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """Enhanced Flash Attention implementation for MossFormer2 with advanced features"""
        try:
            import time
            b, n, device = x.shape[0], x.shape[-2], x.device
            g = self.group_size
            # Keep originals for safe fallback (ungrouped)
            quad_q_orig, lin_q_orig, quad_k_orig, lin_k_orig = quad_q, lin_q, quad_k, lin_k
            v_orig, u_orig = v, u
            
            # Prepare copies for FlashAttention (slice/pad to match Q/K dim; no params added)
            if self.use_flash_attn:
                qkv_dim = quad_q.shape[-1]
                v_dim = v.shape[-1]
                if v_dim == qkv_dim:
                    v_att, u_att = v, u
                elif v_dim > qkv_dim:
                    v_att = v[..., :qkv_dim]
                    u_att = u[..., :qkv_dim]
                else:
                    pad = qkv_dim - v_dim
                    v_att = F.pad(v, (0, pad), value=0.0)
                    u_att = F.pad(u, (0, pad), value=0.0)
            else:
                v_att, u_att = v, u
            
            # Apply grouped attention like original MossFormer2
            # Pad sequence to multiple of group_size
            # timers/counters
            def _sync():
                if x.is_cuda:
                    torch.cuda.synchronize(device)
            t0_total = time.perf_counter(); _sync()
            fa_ok, fa_fb = 0, 0
            fa_kernel_ms = 0.0
            padding = padding_to_multiple_of(n, g)
            if padding > 0:
                quad_q, quad_k, lin_q, lin_k, v_att, u_att = map(
                    lambda t: F.pad(t, (0, 0, 0, padding), value=0.), 
                    (quad_q, quad_k, lin_q, lin_k, v_att, u_att)
                )
                mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
                mask = F.pad(mask, (0, padding), value=False)
            
            # Group along sequence dimension
            quad_q, quad_k, lin_q, lin_k, v_att, u_att = map(
                lambda t: rearrange(t, 'b (g n) d -> b g n d', n=g), 
                (quad_q, quad_k, lin_q, lin_k, v_att, u_att)
            )
            
            if exists(mask):
                mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)
            
            # Process each group with enhanced FlashAttention
            attn_out_v_list = []
            attn_out_u_list = []
            
            for group_idx in range(quad_q.shape[1]):  # Iterate over groups
                # Extract current group
                q_group = quad_q[:, group_idx]  # (b, n, d)
                k_group = quad_k[:, group_idx]  # (b, n, d)
                v_group = v_att[:, group_idx]       # (b, n, d)
                u_group = u_att[:, group_idx]       # (b, n, d)
                # Corresponding original dims for gating
                v_orig_group = v_orig[:, group_idx]
                u_orig_group = u_orig[:, group_idx]
                
                # Reshape for FlashAttention with dynamic head configuration
                num_heads = self.num_heads
                head_dim = self.head_dim
                
                # Ensure dimensions are compatible
                if q_group.shape[-1] % (num_heads * head_dim) == 0:
                    # Reshape to (b, n_per_group, num_heads, head_dim)
                    n_per_group = q_group.shape[1]
                    q_reshaped = q_group.view(b, n_per_group, num_heads, head_dim)
                    k_reshaped = k_group.view(b, n_per_group, num_heads, head_dim)
                    v_reshaped = v_group.view(b, n_per_group, num_heads, head_dim)
                    u_reshaped = u_group.view(b, n_per_group, num_heads, head_dim)
                else:
                    # Pad to compatible size
                    target_dim = num_heads * head_dim
                    if q_group.shape[-1] < target_dim:
                        pad_size = target_dim - q_group.shape[-1]
                        q_reshaped = F.pad(q_group, (0, pad_size), value=0.0)
                        k_reshaped = F.pad(k_group, (0, pad_size), value=0.0)
                        v_reshaped = F.pad(v_group, (0, pad_size), value=0.0)
                        u_reshaped = F.pad(u_group, (0, pad_size), value=0.0)
                    else:
                        # Truncate to target size
                        q_reshaped = q_group[:, :, :target_dim]
                        k_reshaped = k_group[:, :, :target_dim]
                        v_reshaped = v_group[:, :, :target_dim]
                        u_reshaped = u_group[:, :, :target_dim]
                    
                    n_per_group = q_reshaped.shape[1]
                    q_reshaped = q_reshaped.view(b, n_per_group, num_heads, head_dim)
                    k_reshaped = k_reshaped.view(b, n_per_group, num_heads, head_dim)
                    v_reshaped = v_reshaped.view(b, n_per_group, num_heads, head_dim)
                    u_reshaped = u_reshaped.view(b, n_per_group, num_heads, head_dim)
                
                # Apply enhanced FlashAttention to current group
                try:
                    # Ensure supported dtype
                    attn_dtype = torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16
                    qf = q_reshaped.to(attn_dtype).contiguous()
                    kf = k_reshaped.to(attn_dtype).contiguous()
                    vf = v_reshaped.to(attn_dtype).contiguous()
                    uf = u_reshaped.to(attn_dtype).contiguous()
                    
                    softmax_scale = 1.0 / math.sqrt(head_dim)
                    
                    # Prepare ALiBi slopes if enabled
                    alibi_slopes = None
                    if self.use_alibi and self.alibi_slopes is not None:
                        alibi_slopes = self.alibi_slopes.to(device)
                    
                    # Prepare window size
                    window_size = (-1, -1)
                    if self.use_sliding_window:
                        window_size = self.window_size
                    
                    # Prepare softcap (disable when dropout>0, unsupported)
                    softcap = 0.0
                    if self.use_softcap and not bool(self.training and (self.dropout.p > 0.0)):
                        softcap = self.softcap_value
                    
                    # Enhanced FlashAttention for v with all features
                    t1 = time.perf_counter(); _sync()
                    attn_v = flash_attn_func(
                        qf, kf, vf,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        softmax_scale=softmax_scale,
                        causal=self.causal,
                        window_size=window_size,
                        softcap=softcap,
                        alibi_slopes=alibi_slopes,
                        deterministic=self.deterministic
                    )
                    _sync(); t2 = time.perf_counter(); fa_kernel_ms += (t2 - t1) * 1000.0
                    
                    # Enhanced FlashAttention for u with all features
                    t3 = time.perf_counter(); _sync()
                    attn_u = flash_attn_func(
                        qf, kf, uf,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        softmax_scale=softmax_scale,
                        causal=self.causal,
                        window_size=window_size,
                        softcap=softcap,
                        alibi_slopes=alibi_slopes,
                        deterministic=self.deterministic
                    )
                    _sync(); t4 = time.perf_counter(); fa_kernel_ms += (t4 - t3) * 1000.0
                    
                    # Convert back to original dtype and reshape
                    attn_v = attn_v.to(v_group.dtype).view(b, n_per_group, -1)
                    attn_u = attn_u.to(u_group.dtype).view(b, n_per_group, -1)
                    
                    # Ensure attention outputs match original v/u dims for gating
                    if attn_v.shape[-1] != v_orig_group.shape[-1]:
                        if attn_v.shape[-1] < v_orig_group.shape[-1]:
                            pad_size = v_orig_group.shape[-1] - attn_v.shape[-1]
                            attn_v = F.pad(attn_v, (0, pad_size), value=0.0)
                        else:
                            attn_v = attn_v[:, :, :v_orig_group.shape[-1]]
                    
                    if attn_u.shape[-1] != u_orig_group.shape[-1]:
                        if attn_u.shape[-1] < u_orig_group.shape[-1]:
                            pad_size = u_orig_group.shape[-1] - attn_u.shape[-1]
                            attn_u = F.pad(attn_u, (0, pad_size), value=0.0)
                        else:
                            attn_u = attn_u[:, :, :u_orig_group.shape[-1]]
                    
                    attn_out_v_list.append(attn_v)
                    attn_out_u_list.append(attn_u)
                    fa_ok += 1
                    
                except Exception as e:
                    # Fallback to standard attention for this group
                    print(f"Enhanced FlashAttention failed for group {group_idx}: {e}")
                    # Use standard attention for this group
                    sim = einsum('... i d, ... j d -> ... i j', q_group, k_group) / g
                    attn = F.relu(sim) ** 2
                    attn = self.dropout(attn)
                    
                    if exists(mask) and group_idx < mask.shape[1]:
                        attn = attn.masked_fill(~mask[:, group_idx:group_idx+1], 0.)
                    
                    if self.causal:
                        causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
                        attn = attn.masked_fill(causal_mask, 0.)
                    
                    attn_v = einsum('... i j, ... j d -> ... i d', attn, v_group)
                    attn_u = einsum('... i j, ... j d -> ... i d', attn, u_group)
                    
                    attn_out_v_list.append(attn_v)
                    attn_out_u_list.append(attn_u)
            
            # Concatenate all groups back
            attn_out_v = torch.cat(attn_out_v_list, dim=1)  # (b, n, d)
            attn_out_u = torch.cat(attn_out_u_list, dim=1)  # (b, n, d)
            
            # Remove padding
            attn_out_v = attn_out_v[:, :n]
            attn_out_u = attn_out_u[:, :n]
            
            _sync(); t_end = time.perf_counter()
            if getattr(self, 'debug_fa', False):
                total_ms = (t_end - t0_total) * 1000.0
                print(f"[FA] groups={quad_q.shape[1]} ok={fa_ok} fb={fa_fb} fa_kernel_ms={fa_kernel_ms:.3f} total_ms={total_ms:.3f}")

            return attn_out_v, attn_out_u
            
        except Exception as e:
            print(f"Enhanced Flash Attention failed: {e}")
            # Fallback with ungrouped original tensors
            return self.cal_standard_attention(x, quad_q_orig, lin_q_orig, quad_k_orig, lin_k_orig, v_orig, u_orig, mask)

    def cal_standard_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask = None):
        """Original MossFormer2 attention implementation"""
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

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

        # calculate quadratic attention output
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

class DenoiseLayer(nn.Module):
    """Denoise layer for noise reduction in MossFormer2 architecture.
    
    This layer is designed to be inserted within the Repeat R times loop
    to improve training with noisy data by applying noise reduction techniques.
    
    Arguments
    ---------
    dim : int
        Input/output dimension
    inner_channels : int
        Inner channel dimension for processing
    dropout : float
        Dropout rate for regularization
    use_spectral_subtraction : bool
        Whether to use spectral subtraction for noise reduction
    use_wiener_filter : bool
        Whether to use Wiener filtering approach
    """
    
    def __init__(
        self,
        dim,
        inner_channels=256,
        dropout=0.1,
        use_spectral_subtraction=True,
        use_wiener_filter=True,
        norm_type='scalenorm'
    ):
        super(DenoiseLayer, self).__init__()
        
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm
        else:
            norm_klass = nn.LayerNorm
            
        self.dim = dim
        self.inner_channels = inner_channels
        self.use_spectral_subtraction = use_spectral_subtraction
        self.use_wiener_filter = use_wiener_filter
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(dim, inner_channels),
            norm_klass(inner_channels),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        
        # Noise estimation branch
        self.noise_estimator = nn.Sequential(
            nn.Linear(inner_channels, inner_channels // 2),
            norm_klass(inner_channels // 2),
            nn.PReLU(),
            nn.Linear(inner_channels // 2, inner_channels),
            nn.Sigmoid()
        )
        
        # Spectral subtraction module
        if self.use_spectral_subtraction:
            self.spectral_subtraction = nn.Sequential(
                nn.Linear(inner_channels, inner_channels),
                norm_klass(inner_channels),
                nn.PReLU(),
                nn.Linear(inner_channels, inner_channels),
                nn.Sigmoid()
            )
        
        # Wiener filter module
        if self.use_wiener_filter:
            self.wiener_filter = nn.Sequential(
                nn.Linear(inner_channels, inner_channels),
                norm_klass(inner_channels),
                nn.PReLU(),
                nn.Linear(inner_channels, inner_channels),
                nn.Tanh()
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(inner_channels, dim),
            nn.Dropout(dropout)
        )
        
        # Residual connection with learnable scaling
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Forward pass of the denoise layer.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape [B, L, D]
            where B = batch size, L = sequence length, D = dimension
            
        Returns
        -------
        torch.Tensor
            Denoised output tensor of shape [B, L, D]
        """
        residual = x
        
        # Input projection
        x_proj = self.input_proj(x)
        
        # Noise estimation
        noise_mask = self.noise_estimator(x_proj)
        
        # Apply noise reduction techniques
        denoised = x_proj
        
        # Spectral subtraction
        if self.use_spectral_subtraction:
            spectral_mask = self.spectral_subtraction(x_proj)
            denoised = denoised * (1 - noise_mask * spectral_mask)
        
        # Wiener filtering
        if self.use_wiener_filter:
            wiener_coeff = self.wiener_filter(x_proj)
            denoised = denoised * wiener_coeff
        
        # Output projection
        output = self.output_proj(denoised)
        
        # Residual connection with learnable scaling
        return residual + self.residual_scale * output

class AdaptiveDenoiseLayer(nn.Module):
    """Adaptive denoise layer that learns noise characteristics dynamically.
    
    This layer adapts its denoising strategy based on the input signal characteristics,
    making it more effective for different types of noise.
    
    Arguments
    ---------
    dim : int
        Input/output dimension
    inner_channels : int
        Inner channel dimension for processing
    num_heads : int
        Number of attention heads for adaptive processing
    dropout : float
        Dropout rate for regularization
    """
    
    def __init__(
        self,
        dim,
        inner_channels=256,
        num_heads=8,
        dropout=0.1,
        norm_type='scalenorm'
    ):
        super(AdaptiveDenoiseLayer, self).__init__()
        
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm
        else:
            norm_klass = nn.LayerNorm
            
        self.dim = dim
        self.inner_channels = inner_channels
        self.num_heads = num_heads
        
        # Input processing
        self.input_norm = norm_klass(dim)
        self.input_proj = nn.Linear(dim, inner_channels)
        
        # Multi-head attention for noise pattern recognition
        self.noise_attention = nn.MultiheadAttention(
            embed_dim=inner_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Adaptive noise reduction
        self.adaptive_reduction = nn.Sequential(
            nn.Linear(inner_channels, inner_channels),
            norm_klass(inner_channels),
            nn.PReLU(),
            nn.Linear(inner_channels, inner_channels),
            nn.Sigmoid()
        )
        
        # Output processing
        self.output_norm = norm_klass(inner_channels)
        self.output_proj = nn.Linear(inner_channels, dim)
        
        # Learnable parameters for adaptation
        self.adaptation_scale = nn.Parameter(torch.ones(1))
        self.adaptation_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of the adaptive denoise layer.
        
        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape [B, L, D]
            
        Returns
        -------
        torch.Tensor
            Denoised output tensor of shape [B, L, D]
        """
        residual = x
        
        # Input normalization and projection
        x_norm = self.input_norm(x)
        x_proj = self.input_proj(x_norm)
        
        # Multi-head attention for noise pattern recognition
        attn_out, _ = self.noise_attention(x_proj, x_proj, x_proj)
        
        # Adaptive noise reduction
        reduction_mask = self.adaptive_reduction(attn_out)
        denoised = x_proj * reduction_mask
        
        # Output processing
        output_norm = self.output_norm(denoised)
        output = self.output_proj(output_norm)
        
        # Adaptive residual connection
        return residual + self.adaptation_scale * output + self.adaptation_bias

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
        flash_attn_config = None,
        use_denoise = False,
        denoise_type = 'adaptive',  # 'basic' or 'adaptive'
        denoise_config = None
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        self.use_denoise = use_denoise
        self.denoise_type = denoise_type

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        
        # Enhanced Flash Attention configuration for better performance
        enhanced_flash_config = {
            'use_qkv_packed': True,
            'use_alibi': True,  # Enable ALiBi for better positional encoding
            'use_sliding_window': True,  # Enable sliding window for local attention
            'window_size': (64, 64),  # 64 tokens left and right
            'use_softcap': True,  # Enable softcapping for attention scores
            'softcap_value': 0.1,  # Softcap value
            'deterministic': False
        }
        
        # Merge with user config if provided
        if flash_attn_config:
            enhanced_flash_config.update(flash_attn_config)
        
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, use_flash_attn = True, flash_attn_config = enhanced_flash_config) for _ in range(depth)])
        
        # Add denoise layers if enabled
        if self.use_denoise:
            denoise_config = denoise_config or {}
            if self.denoise_type == 'basic':
                self.denoise_layers = nn.ModuleList([
                    DenoiseLayer(
                        dim=dim,
                        inner_channels=denoise_config.get('inner_channels', 256),
                        dropout=denoise_config.get('dropout', 0.1),
                        use_spectral_subtraction=denoise_config.get('use_spectral_subtraction', True),
                        use_wiener_filter=denoise_config.get('use_wiener_filter', True),
                        norm_type=norm_type
                    ) for _ in range(depth)
                ])
            elif self.denoise_type == 'adaptive':
                self.denoise_layers = nn.ModuleList([
                    AdaptiveDenoiseLayer(
                        dim=dim,
                        inner_channels=denoise_config.get('inner_channels', 256),
                        num_heads=denoise_config.get('num_heads', 8),
                        dropout=denoise_config.get('dropout', 0.1),
                        norm_type=norm_type
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
            
            # Apply denoise layer if enabled
            if self.use_denoise and hasattr(self, 'denoise_layers'):
                x = self.denoise_layers[ii](x)
            
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
        flash_attn_config = None,
        use_denoise = False,
        denoise_type = 'adaptive',  # 'basic' or 'adaptive'
        denoise_config = None
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size
        self.use_denoise = use_denoise
        self.denoise_type = denoise_type

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        
        # Enhanced Flash Attention configuration for better performance
        enhanced_flash_config = {
            'use_qkv_packed': True,
            'use_alibi': True,  # Enable ALiBi for better positional encoding
            'use_sliding_window': True,  # Enable sliding window for local attention
            'window_size': (64, 64),  # 64 tokens left and right
            'use_softcap': True,  # Enable softcapping for attention scores
            'softcap_value': 0.1,  # Softcap value
            'deterministic': False
        }
        
        # Merge with user config if provided
        if flash_attn_config:
            enhanced_flash_config.update(flash_attn_config)
        
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, use_flash_attn = True, flash_attn_config = enhanced_flash_config) for _ in range(depth)])
        
        # Add denoise layers if enabled
        if self.use_denoise:
            denoise_config = denoise_config or {}
            if self.denoise_type == 'basic':
                self.denoise_layers = nn.ModuleList([
                    DenoiseLayer(
                        dim=dim,
                        inner_channels=denoise_config.get('inner_channels', 256),
                        dropout=denoise_config.get('dropout', 0.1),
                        use_spectral_subtraction=denoise_config.get('use_spectral_subtraction', True),
                        use_wiener_filter=denoise_config.get('use_wiener_filter', True),
                        norm_type=norm_type
                    ) for _ in range(depth)
                ])
            elif self.denoise_type == 'adaptive':
                self.denoise_layers = nn.ModuleList([
                    AdaptiveDenoiseLayer(
                        dim=dim,
                        inner_channels=denoise_config.get('inner_channels', 256),
                        num_heads=denoise_config.get('num_heads', 8),
                        dropout=denoise_config.get('dropout', 0.1),
                        norm_type=norm_type
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
            
            # Apply denoise layer if enabled
            if self.use_denoise and hasattr(self, 'denoise_layers'):
                x = self.denoise_layers[ii](x)
            
            ii = ii + 1
        return x

# Example usage with enhanced Flash Attention configuration
def create_enhanced_mossformer_block(
    dim=512,
    depth=6,
    group_size=256,
    query_key_dim=128,
    use_alibi=True,
    use_sliding_window=False,
    use_softcap=False,
    use_denoise=False,
    denoise_type='adaptive',
    denoise_config=None,
    **kwargs
):
    """
    Create MossFormer2 block with enhanced Flash Attention features and optional denoise layers
    
    Args:
        dim: Model dimension
        depth: Number of layers
        group_size: Group size for attention
        query_key_dim: Query/Key dimension
        use_alibi: Enable ALiBi positional bias
        use_sliding_window: Enable sliding window attention
        use_softcap: Enable softcapping
        use_denoise: Enable denoise layers in Repeat R times loop
        denoise_type: Type of denoise layer ('basic' or 'adaptive')
        denoise_config: Configuration for denoise layers
        **kwargs: Additional arguments
    """
    
    # Enhanced Flash Attention configuration
    flash_attn_config = {
        'use_qkv_packed': True,
        'use_alibi': use_alibi,
        'use_sliding_window': use_sliding_window,
        'window_size': (64, 64) if use_sliding_window else (-1, -1),
        'use_softcap': use_softcap,
        'softcap_value': 0.1 if use_softcap else 0.0,
        'deterministic': False
    }
    
    return MossformerBlock(
        dim=dim,
        depth=depth,
        group_size=group_size,
        query_key_dim=query_key_dim,
        flash_attn_config=flash_attn_config,
        use_denoise=use_denoise,
        denoise_type=denoise_type,
        denoise_config=denoise_config,
        **kwargs
    )

def create_denoise_mossformer_block(
    dim=512,
    depth=6,
    group_size=256,
    query_key_dim=128,
    denoise_type='adaptive',
    denoise_config=None,
    **kwargs
):
    """
    Create MossFormer2 block with denoise layers for training with noisy data
    
    Args:
        dim: Model dimension
        depth: Number of layers
        group_size: Group size for attention
        query_key_dim: Query/Key dimension
        denoise_type: Type of denoise layer ('basic' or 'adaptive')
        denoise_config: Configuration for denoise layers
        **kwargs: Additional arguments
        
    Example:
        >>> # Basic denoise configuration
        >>> basic_config = {
        ...     'inner_channels': 256,
        ...     'dropout': 0.1,
        ...     'use_spectral_subtraction': True,
        ...     'use_wiener_filter': True
        ... }
        >>> model = create_denoise_mossformer_block(
        ...     dim=512, depth=6, denoise_type='basic', denoise_config=basic_config
        ... )
        
        >>> # Adaptive denoise configuration
        >>> adaptive_config = {
        ...     'inner_channels': 256,
        ...     'num_heads': 8,
        ...     'dropout': 0.1
        ... }
        >>> model = create_denoise_mossformer_block(
        ...     dim=512, depth=6, denoise_type='adaptive', denoise_config=adaptive_config
        ... )
    """
    
    # Default denoise configuration
    if denoise_config is None:
        if denoise_type == 'basic':
            denoise_config = {
                'inner_channels': 256,
                'dropout': 0.1,
                'use_spectral_subtraction': True,
                'use_wiener_filter': True
            }
        else:  # adaptive
            denoise_config = {
                'inner_channels': 256,
                'num_heads': 8,
                'dropout': 0.1
            }
    
    return MossformerBlock_GFSMN(
        dim=dim,
        depth=depth,
        group_size=group_size,
        query_key_dim=query_key_dim,
        use_denoise=True,
        denoise_type=denoise_type,
        denoise_config=denoise_config,
        **kwargs
    )
