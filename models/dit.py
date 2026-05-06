"""
dit.py
======
Diffusion Transformer (DiT) architecture for conditional image generation.

The implementation follows the paper
    "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
    https://arxiv.org/abs/2212.09748

Module overview
---------------
FourierEncoder              – Maps scalar time t → Fourier embedding.
Patchifier                  – image (b,1,H,W) → token sequence (b,N,D).
MHA                         – Multi-headed self-attention.
DiffusionTransformerLayer   – Single DiT block with adaLN-Zero conditioning.
DiffusionTransformer        – Stack of DiT layers with positional embeddings.
Depatchifier                – Token sequence (b,N,D) → image (b,1,H,W).
MNISTDiffusionTransformer   – End-to-end CFG vector-field model for MNIST.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from models.vector_fields import ConditionalVectorField, MLP


# ---------------------------------------------------------------------------
# Fourier time encoder
# ---------------------------------------------------------------------------

class FourierEncoder(nn.Module):
    """
    Random Fourier feature time embedding.

    For a scalar time t ∈ [0, 1] produces

        t^emb = [cos(2π w_1 t), …, cos(2π w_{d/2} t),
                 sin(2π w_1 t), …, sin(2π w_{d/2} t)]^T

    where w_i ~ N(0, 1) are fixed at construction time.

    Parameters
    ----------
    dim : int
        Output dimension (must be even).
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        # Fixed (non-learnable) random frequencies stored as a buffer so that
        # the encoder moves correctly when .to(device) is called.
        self.register_buffer("weights", torch.randn(dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (b,)
        Returns:
            embedding: (b, dim)
        """
        # (b, 1) * (d/2,) → (b, d/2)
        freqs = 2.0 * math.pi * self.weights * t.unsqueeze(-1)
        return torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)


# ---------------------------------------------------------------------------
# Patchifier
# ---------------------------------------------------------------------------

class Patchifier(nn.Module):
    """
    Splits an image into non-overlapping patches and projects them to a
    *dim*-dimensional embedding space.

        (b, 1, H, W)  →  (b, N, D)

    where N = (H // patch_size) * (W // patch_size) and D = dim.

    Parameters
    ----------
    img_size   : int  – spatial resolution (assumes square images)
    patch_size : int  – side length of each patch
    dim        : int  – embedding dimension
    """

    def __init__(self, img_size: int, patch_size: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.conv = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, 1, H, W)
        Returns:
            tokens: (b, N, D)
        """
        x = self.conv(x)                        # (b, D, H/p, W/p)
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


# ---------------------------------------------------------------------------
# Multi-Headed Self-Attention
# ---------------------------------------------------------------------------

class MHA(nn.Module):
    """
    Multi-headed self-attention.

    Parameters
    ----------
    dim   : int – model / embedding dimension
    heads : int – number of attention heads (must divide dim)
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, n, d)
        Returns:
            out: (b, n, d)
        """
        b, n, d = x.shape
        h = self.heads

        # 1. Compute queries, keys, values in one matmul
        qkv = self.qkv_proj(x)                          # (b, n, 3d)
        q, k, v = qkv.chunk(3, dim=-1)                  # each (b, n, d)

        # 2. Reshape: fold heads into the batch dimension
        #    (b, n, d) → (b, h, n, head_dim)
        q = q.view(b, n, h, self.head_dim).transpose(1, 2)
        k = k.view(b, n, h, self.head_dim).transpose(1, 2)
        v = v.view(b, n, h, self.head_dim).transpose(1, 2)

        # 3. Scaled dot-product attention
        #    attn_weights: (b, h, n, n)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 4. Combine with values → (b, h, n, head_dim)
        out = torch.matmul(attn_weights, v)

        # 5. Unfold heads → (b, n, d)
        out = out.transpose(1, 2).contiguous().view(b, n, d)

        # 6. Output projection
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# DiT layer (adaLN-Zero)
# ---------------------------------------------------------------------------

class DiffusionTransformerLayer(nn.Module):
    """
    Single DiT block with adaptive layer-norm (adaLN-Zero) conditioning.

    The conditioning vector c (shape b, dim) is used to produce six modulation
    parameters:  γ₁, β₁, α₁  (for the attention branch)
                 γ₂, β₂, α₂  (for the feed-forward branch)

    The modulation is  x ← x * (1 + γ) + β  before each sub-layer, and
    α is used to gate the residual connection.

    The final linear layer of the MLP that produces (γ, β, α) is
    zero-initialised (adaLN-Zero trick) to stabilise early training.

    Parameters
    ----------
    dim   : int – model dimension
    heads : int – attention heads
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()

        # Layer norms (no affine parameters – modulation is done externally)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Multi-headed self-attention
        self.attn = MHA(dim, heads)

        # Point-wise feed-forward (hidden dim = 4 × dim, as in the DiT paper)
        self.ff = MLP([dim, 4 * dim, dim])

        # Conditioning MLP: c → (γ₁, β₁, α₁, γ₂, β₂, α₂)
        # Zero-initialise the last layer for stable training (adaLN-Zero).
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN_mlp[-1].weight)
        nn.init.zeros_(self.adaLN_mlp[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, n, d)  – token sequence
            c: (b, d)     – conditioning embedding
        Returns:
            x: (b, n, d)
        """
        # 1. Derive per-sample modulation parameters from conditioning
        mods = self.adaLN_mlp(c)                        # (b, 6d)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mods.chunk(6, dim=-1)
        # Expand to (b, 1, d) for broadcast over the sequence dimension
        gamma1 = gamma1.unsqueeze(1)
        beta1  = beta1.unsqueeze(1)
        alpha1 = alpha1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2  = beta2.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)

        # 2. Attention branch
        x_attn = self.norm1(x) * (1 + gamma1) + beta1   # modulated norm
        x = x + alpha1 * self.attn(x_attn)              # gated residual

        # 3. Feed-forward branch
        x_ff = self.norm2(x) * (1 + gamma2) + beta2     # modulated norm
        x = x + alpha2 * self.ff(x_ff)                  # gated residual

        return x


# ---------------------------------------------------------------------------
# Diffusion Transformer
# ---------------------------------------------------------------------------

class DiffusionTransformer(nn.Module):
    """
    Stack of ``depth`` DiT layers with learnable positional embeddings.

    Parameters
    ----------
    depth    : int – number of transformer layers
    n_tokens : int – sequence length (number of patches)
    dim      : int – model dimension
    **layer_kwargs : forwarded to each ``DiffusionTransformerLayer``
    """

    def __init__(self, depth: int, n_tokens: int, dim: int, **layer_kwargs):
        super().__init__()

        # Learnable positional embeddings: one vector per token position
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)

        # Stack of DiT blocks
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(dim=dim, **layer_kwargs) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, n, d)
            c: (b, d)
        Returns:
            x: (b, n, d)
        """
        # 1. Add positional encodings (broadcast over batch)
        x = x + self.pos_emb

        # 2. Pass through each DiT layer
        for layer in self.layers:
            x = layer(x, c)

        return x


# ---------------------------------------------------------------------------
# Depatchifier
# ---------------------------------------------------------------------------

class Depatchifier(nn.Module):
    """
    Converts a token sequence back to pixel space.

        (b, N, D)  →  (b, 1, H, W)

    Implementation
    --------------
    1. RMS-normalise along the feature dimension.
    2. Project each token to dimension ``final_dim * patch_size²`` with a linear
       layer (i.e., one pixel-group per token).
    3. Rearrange ``(b, N, final_dim·p²) → (b, final_dim, H, W)``.
    4. Apply a final 1×1 convolution to get a single output channel.

    Parameters
    ----------
    img_size  : int
    patch_size : int
    dim        : int – incoming token dimension
    final_dim  : int – intermediate channel count before the output conv
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        dim: int,
        final_dim: int = 16,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        h = w = img_size // patch_size

        self.norm = nn.RMSNorm(dim)
        # Project to final_dim * p * p values per token
        self.proj = nn.Linear(dim, final_dim * patch_size * patch_size)
        # (b, N, final_dim*p*p) → (b, final_dim, H, W)
        self.rearrange = Rearrange(
            "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h=h, w=w, p1=patch_size, p2=patch_size,
        )
        # Final 1×1 conv: final_dim → 1 channel
        self.out_conv = nn.Conv2d(final_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (b, N, D)
        Returns:
            out: (b, 1, H, W)
        """
        x = self.norm(x)                          # (b, N, D)
        x = self.proj(x)                          # (b, N, final_dim·p²)
        x = self.rearrange(x)                     # (b, final_dim, H, W)
        return self.out_conv(x)                   # (b, 1, H, W)


# ---------------------------------------------------------------------------
# Full MNIST DiT model
# ---------------------------------------------------------------------------

class MNISTDiffusionTransformer(ConditionalVectorField):
    """
    Classifier-free-guidance vector-field model  u_t^θ(x | y)  for MNIST.

    Architecture (following the DiT paper)
    ----------------------------------------
    1. Embed time  t        → FourierEncoder → Linear → t_emb  (b, dim)
    2. Embed class y        → nn.Embedding             → y_emb  (b, dim)
    3. Conditioning:          c = t_emb + y_emb               (b, dim)
    4. Patchify x:            Patchifier            → (b, N, dim)
    5. DiT blocks:            DiffusionTransformer  → (b, N, dim)
    6. Depatchify:            Depatchifier          → (b, 1, 32, 32)

    Parameters
    ----------
    patch_size : int – side length of each patch  (img_size must divide evenly)
    num_layers : int – number of DiT blocks
    dim        : int – model dimension
    heads      : int – attention heads
    img_size   : int – spatial resolution (default 32 for MNIST)
    num_classes: int – number of real classes (null = num_classes)
    """

    def __init__(
        self,
        patch_size: int = 4,
        num_layers: int = 8,
        dim: int = 256,
        heads: int = 8,
        img_size: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        n_tokens = (img_size // patch_size) ** 2

        # 0. Time and class embedders
        fourier_dim = dim  # Fourier output directly matches model dim
        self.time_embedder = nn.Sequential(
            FourierEncoder(fourier_dim),
            nn.Linear(fourier_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # num_classes + 1 to accommodate the null label (index = num_classes)
        self.class_embedder = nn.Embedding(num_classes + 1, dim)

        # 1. Patchifier
        self.patchifier = Patchifier(img_size=img_size, patch_size=patch_size, dim=dim)

        # 2. Diffusion transformer
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )

        # 3. Depatchifier
        self.depatchifier = Depatchifier(
            img_size=img_size, patch_size=patch_size, dim=dim
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (b, 1, 32, 32)  – noisy image at time t
            t: (b,)             – continuous time in [0, 1]
            y: (b,)             – integer class labels (null = 10)
        Returns:
            u_t^θ(x|y): (b, 1, 32, 32)
        """
        # Squeeze any extra singleton dims that may come from the trainer
        if t.dim() > 1:
            t = t.view(-1)

        # 1. Conditioning embedding c = time_emb + class_emb
        t_emb = self.time_embedder(t)           # (b, dim)
        y_emb = self.class_embedder(y)          # (b, dim)
        c = t_emb + y_emb                       # (b, dim)

        # 2. Patchify
        tokens = self.patchifier(x)             # (b, N, dim)

        # 3. DiT
        tokens = self.dit(tokens, c)            # (b, N, dim)

        # 4. Depatchify
        return self.depatchifier(tokens)        # (b, 1, 32, 32)

