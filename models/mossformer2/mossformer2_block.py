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
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("Flash Attention available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None
    print("Warning: Flash Attention not available. Using PyTorch native attention.")

# Use PyTorch's native scaled_dot_product_attention as fallback
try:
    from torch.nn.functional import scaled_dot_product_attention
    TORCH_NATIVE_ATTN_AVAILABLE = True
    print("PyTorch native scaled_dot_product_attention available")
except ImportError:
    TORCH_NATIVE_ATTN_AVAILABLE = False
    scaled_dot_product_attention = None
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
        use_flash_attn = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        self.use_flash_attn = use_flash_attn and (FLASH_ATTN_AVAILABLE or TORCH_NATIVE_ATTN_AVAILABLE)

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
        if self.use_flash_attn and (FLASH_ATTN_AVAILABLE or TORCH_NATIVE_ATTN_AVAILABLE):
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
        """Flash Attention implementation for MossFormer2"""
        try:
            b, n, device = x.shape[0], x.shape[-2], x.device
            original_dtype_v = v.dtype
            original_dtype_u = u.dtype
            
            # Debug tensor shapes
            # print(f"Debug: quad_q shape: {quad_q.shape}, v shape: {v.shape}")
            # print(f"Debug: num_heads: {self.num_heads}, head_dim: {self.head_dim}")
            
            # Use quad_q and quad_k for Flash Attention (simpler approach)
            # Reshape for multi-head attention
            head_dim = self.head_dim
            
            # Check if dimensions are compatible
            expected_size = b * n * self.num_heads * head_dim
            actual_size = quad_q.numel()
            
            print(f"Debug: expected_size = {expected_size}, actual_size = {actual_size}")
            print(f"Debug: quad_q.shape = {quad_q.shape}")
            print(f"Debug: b={b}, n={n}, num_heads={self.num_heads}, head_dim={head_dim}")
            
            if actual_size != expected_size:
                print(f"Dimension mismatch: expected {expected_size}, got {actual_size}")
                # Try to fix the dimension by adjusting head_dim
                actual_head_dim = actual_size // (b * n * self.num_heads)
                if actual_head_dim * b * n * self.num_heads == actual_size:
                    print(f"Adjusted head_dim from {head_dim} to {actual_head_dim}")
                    head_dim = actual_head_dim
                else:
                    raise ValueError(f"Tensor size mismatch: {actual_size} != {expected_size}")
            
            q_flash = quad_q.view(b, n, self.num_heads, head_dim)
            k_flash = quad_k.view(b, n, self.num_heads, head_dim)
            
            # Handle v and u with different dimensions
            # FlashAttention expects: (batch_size, seqlen_k, num_heads_k, head_size)
            # where num_heads_k should match the query num_heads for proper attention
            
            # Calculate head dimensions for v and u
            v_head_dim = v.shape[-1] // self.num_heads if v.shape[-1] % self.num_heads == 0 else v.shape[-1]
            u_head_dim = u.shape[-1] // self.num_heads if u.shape[-1] % self.num_heads == 0 else u.shape[-1]
            
            print(f"Debug: v_head_dim = {v_head_dim}, u_head_dim = {u_head_dim}")
            print(f"Debug: v.shape = {v.shape}, u.shape = {u.shape}")
            print(f"Debug: num_heads = {self.num_heads}, v.shape[-1] = {v.shape[-1]}")
            
            # Reshape v for FlashAttention: (batch_size, seqlen_k, num_heads_k, head_size)
            # Ensure v has the same number of heads as q/k
            if v.shape[-1] % self.num_heads == 0:
                v_flash = v.view(b, n, self.num_heads, v_head_dim)
            else:
                # If v dimension doesn't divide evenly, we need to adjust
                # Try to find a compatible number of heads for v
                v_total_dim = v.shape[-1]
                # Find the largest divisor of v_total_dim that's <= self.num_heads
                v_num_heads = 1
                for i in range(1, min(self.num_heads, v_total_dim) + 1):
                    if v_total_dim % i == 0:
                        v_num_heads = i
                
                v_head_dim = v_total_dim // v_num_heads
                v_flash = v.view(b, n, v_num_heads, v_head_dim)
                
                # If we have fewer heads than expected, we need to repeat or pad
                if v_num_heads < self.num_heads:
                    # Repeat the last head to match num_heads
                    repeat_factor = self.num_heads // v_num_heads
                    remainder = self.num_heads % v_num_heads
                    v_flash = v_flash.repeat(1, 1, repeat_factor, 1)
                    if remainder > 0:
                        # Add remaining heads by repeating the first few heads
                        v_flash = torch.cat([v_flash, v_flash[:, :, :remainder, :]], dim=2)
                
            # Reshape u for FlashAttention: (batch_size, seqlen_k, num_heads_k, head_size)
            if u.shape[-1] % self.num_heads == 0:
                u_flash = u.view(b, n, self.num_heads, u_head_dim)
            else:
                # If u dimension doesn't divide evenly, we need to adjust
                # Try to find a compatible number of heads for u
                u_total_dim = u.shape[-1]
                # Find the largest divisor of u_total_dim that's <= self.num_heads
                u_num_heads = 1
                for i in range(1, min(self.num_heads, u_total_dim) + 1):
                    if u_total_dim % i == 0:
                        u_num_heads = i
                
                u_head_dim = u_total_dim // u_num_heads
                u_flash = u.view(b, n, u_num_heads, u_head_dim)
                
                # If we have fewer heads than expected, we need to repeat or pad
                if u_num_heads < self.num_heads:
                    # Repeat the last head to match num_heads
                    repeat_factor = self.num_heads // u_num_heads
                    remainder = self.num_heads % u_num_heads
                    u_flash = u_flash.repeat(1, 1, repeat_factor, 1)
                    if remainder > 0:
                        # Add remaining heads by repeating the first few heads
                        u_flash = torch.cat([u_flash, u_flash[:, :, :remainder, :]], dim=2)
            
            # Apply Flash Attention with multiple fallback options
            softmax_scale = 1.0 / math.sqrt(head_dim)
            
            try:
                # Check GPU architecture and choose appropriate attention
                use_flash_attn = False
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    # Flash Attention 2.x only supports Ampere (8.0+) and newer
                    use_flash_attn = major >= 8
                    print(f"GPU SM {major}.{minor}: {'Ampere+ (Flash Attention supported)' if use_flash_attn else 'Turing or older (PyTorch native)'}")
                
                # Try Flash Attention for supported GPUs
                if use_flash_attn and FLASH_ATTN_AVAILABLE and flash_attn_func is not None:
                    print("Using Flash Attention")
                    # Ensure supported dtype for FlashAttention (fp16/bf16)
                    attn_dtype = torch.bfloat16 if getattr(torch.cuda, 'is_bf16_supported', lambda: False)() else torch.float16
                    qf = q_flash.to(attn_dtype).contiguous()
                    kf = k_flash.to(attn_dtype).contiguous()
                    vf = v_flash.to(attn_dtype).contiguous()
                    uf = u_flash.to(attn_dtype).contiguous()
                    # Flash attention for v
                    attn_out_v = flash_attn_func(
                        qf, kf, vf,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        softmax_scale=softmax_scale,
                        causal=self.causal
                    )
                    
                    # Flash attention for u
                    attn_out_u = flash_attn_func(
                        qf, kf, uf,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        softmax_scale=softmax_scale,
                        causal=self.causal
                    )
                    # Cast back to original dtypes
                    attn_out_v = attn_out_v.to(original_dtype_v)
                    attn_out_u = attn_out_u.to(original_dtype_u)
                    
                # Use PyTorch native attention for Turing and older GPUs
                elif TORCH_NATIVE_ATTN_AVAILABLE and scaled_dot_product_attention is not None:
                    print("Using PyTorch native attention (optimized for Turing/older GPUs)")
                    # PyTorch native attention for v
                    attn_out_v = scaled_dot_product_attention(
                        q_flash, k_flash, v_flash,
                        attn_mask=None,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=self.causal,
                        scale=softmax_scale
                    )
                    
                    # PyTorch native attention for u
                    attn_out_u = scaled_dot_product_attention(
                        q_flash, k_flash, u_flash,
                        attn_mask=None,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=self.causal,
                        scale=softmax_scale
                    )
                    
                else:
                    raise ValueError("No optimized attention available")
                    
            except Exception as e:
                print(f"Optimized attention failed: {e}")
                raise e
            
            # Reshape back to original dimensions
            # FlashAttention output shape: (batch_size, seqlen_q, num_heads, head_size)
            # We need to reshape to (batch_size, seqlen_q, total_features)
            
            print(f"Debug: attn_out_v.shape = {attn_out_v.shape}, attn_out_u.shape = {attn_out_u.shape}")
            
            # For v output - handle the case where we may have repeated heads
            if attn_out_v.shape[-2] == self.num_heads:
                # We have the expected number of heads, but may need to reduce if we repeated
                if v.shape[-1] % self.num_heads != 0:
                    # We repeated heads, so we need to take only the original heads
                    v_num_heads = v.shape[-1] // (v.shape[-1] // self.num_heads) if v.shape[-1] // self.num_heads > 0 else 1
                    attn_out_v = attn_out_v[:, :, :v_num_heads, :]
                
                # Reshape to original v dimensions
                attn_out_v = attn_out_v.view(b, n, -1)
            else:
                # Fallback: just flatten the last two dimensions
                attn_out_v = attn_out_v.view(b, n, -1)
                
            # For u output - handle the case where we may have repeated heads
            if attn_out_u.shape[-2] == self.num_heads:
                # We have the expected number of heads, but may need to reduce if we repeated
                if u.shape[-1] % self.num_heads != 0:
                    # We repeated heads, so we need to take only the original heads
                    u_num_heads = u.shape[-1] // (u.shape[-1] // self.num_heads) if u.shape[-1] // self.num_heads > 0 else 1
                    attn_out_u = attn_out_u[:, :, :u_num_heads, :]
                
                # Reshape to original u dimensions
                attn_out_u = attn_out_u.view(b, n, -1)
            else:
                # Fallback: just flatten the last two dimensions
                attn_out_u = attn_out_u.view(b, n, -1)
            
            return attn_out_v, attn_out_u
            
        except Exception as e:
            print(f"Flash Attention failed: {e}")
            # print("Falling back to standard attention...")
            return self.cal_standard_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

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
        shift_tokens = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, use_flash_attn = True) for _ in range(depth)])
  
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
        shift_tokens = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, use_flash_attn = True) for _ in range(depth)])

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
            ii = ii + 1
        return x