"""Step 2 (v6b): Chiron3D with a hierarchical "tower" trunk.

Geometry (r = 800 bp, flank = 1024 bp, input 525,248 bp):
    one-hot (B, 4, L)
    -> ASAP ConvNeXtV2 stem (4 -> 256, k=15)            @ 1 bp
    -> pad-left + MaxPool(2)                              @ 2 bp   (same as the v4 trunk so far)
    -> 4 x [GroupNorm, GELU, Conv1d k=5, MaxPool(2)]      2 -> 4 -> 8 -> 16 -> 32 bp, 256 -> ... -> 512 ch
    -> n dilated residual blocks (ASAP pattern) @ 32 bp   receptive field ~2 * sum(dilations) * 32 bp
    -> drop the flank (1024 bp = 32 positions per side)
    -> attention pool 25 : 1  (32 bp -> 800 bp)           (or max / mean+max)
    -> 1x1 projector 512 -> 128
    -> same head as before: positional encoding, 4-layer transformer, outer concat, 2D decoder

Every long-range step the v4 model had to do with a 128-d transformer on composition fingerprints
now gets kilobase-scale, motif-aware context before the transformer, at about the same compute.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import src.models.model.blocks as blocks
from src.models.model.chiron_model import diagonalize_small, move_feature_forward
from src.models.model.asap_vendor.layers.convnext import ConvNeXtV2Block


class TowerStage(nn.Module):
    """GroupNorm -> GELU -> Conv1d(k=5, same) -> MaxPool(2).  Halves the length, grows the channels."""

    def __init__(self, c_in, c_out, kernel_size=5):
        super().__init__()
        self.norm = nn.GroupNorm(1, c_in)
        self.act = nn.GELU()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size, padding="same")
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        return self.pool(self.conv(self.act(self.norm(x))))


class DilatedResBlock(nn.Module):
    """ASAP / Basenji residual block: dilated conv (C -> C/2) then pointwise conv (C/2 -> C), zero-init norm gain."""

    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        mid = channels // 2
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(channels, mid, 3, dilation=dilation, padding="same", bias=False)
        self.norm2 = nn.GroupNorm(1, mid)
        self.conv2 = nn.Conv1d(mid, channels, 1, bias=False)
        self.norm3 = nn.GroupNorm(1, channels)
        nn.init.zeros_(self.norm3.weight)          # block starts as identity (as in ASAP's bn_gamma='zeros')
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + self.drop(self.norm3(h))


class AttentionPool1d(nn.Module):
    """Enformer-style attention pooling over non-overlapping groups of `pool_size` positions.
    Initialised (identity * 2) so it behaves like a soft max-pool from the start."""

    def __init__(self, channels, pool_size):
        super().__init__()
        self.pool_size = pool_size
        self.to_logits = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.eye_(self.to_logits.weight.squeeze())
        with torch.no_grad():
            self.to_logits.weight.mul_(2.0)

    def forward(self, x):                           # (B, C, L), L divisible by pool_size
        b, c, l = x.shape
        x = x.view(b, c, l // self.pool_size, self.pool_size)
        attn = self.to_logits(x).softmax(dim=-1)
        return (x * attn).sum(-1)                   # (B, C, L / pool_size)


class TowerTrunk(nn.Module):
    """(B, 4, L) -> (B, channels[-1], L // 32)."""
    BIN = 32

    def __init__(self, channels=(256, 320, 384, 448, 512), n_dil_blocks=8, dil_mult=2.0,
                 dropout=0.1, use_checkpoint=True):
        super().__init__()
        assert len(channels) == 5, "stem channels + 4 tower stages"
        self.stem = ConvNeXtV2Block(channels_in=4, channels_out=channels[0], kernel_size=15)
        self.stem_pool = nn.MaxPool1d(2)
        self.stages = nn.ModuleList([TowerStage(channels[i], channels[i + 1]) for i in range(4)])
        dils, d = [], 1.0
        for _ in range(n_dil_blocks):
            dils.append(int(round(d))); d *= dil_mult
        self.dilations = dils
        self.dil_blocks = nn.ModuleList([DilatedResBlock(channels[-1], dd, dropout) for dd in dils])
        self.out_channels = channels[-1]
        self.use_checkpoint = use_checkpoint
        self.receptive_field_bp = 15 + 2 * (2 + 4 + 8 + 16) * 2 + 2 * sum(dils) * self.BIN
        print(f"[TowerTrunk] channels={list(channels)} dilations={dils} "
              f"RF ~{self.receptive_field_bp/1000:.1f} kb  out=(B, {self.out_channels}, L/{self.BIN})")

    def _ck(self, fn, *a):
        if self.use_checkpoint and self.training:
            return cp.checkpoint(fn, *a, use_reentrant=False)
        return fn(*a)

    def forward(self, x):
        assert x.shape[-1] % self.BIN == 0, f"input length {x.shape[-1]} must be a multiple of {self.BIN}"
        x = self._ck(self.stem, x)
        x = self.stem_pool(F.pad(x, (1, 0)))         # @ 2 bp, identical to the v4 trunk
        for st in self.stages:
            x = self._ck(st, x)                      # 2 -> 4 -> 8 -> 16 -> 32 bp
        for blk in self.dil_blocks:
            x = self._ck(blk, x)
        return x


class Chiron3D_Tower(nn.Module):

    def __init__(self, mid_hidden=128, resolution=800, n_bins=654, flank=1024,
                 tower_channels=(256, 320, 384, 448, 512), tower_blocks=8, tower_dil_mult=2.0,
                 tower_pool="attn", dropout=0.1, loop_head=False):
        super().__init__()
        self.resolution, self.n_bins, self.flank = int(resolution), int(n_bins), int(flank)
        self.target_span = self.n_bins * self.resolution
        self.input_width = self.target_span + 2 * self.flank
        self.trunk = TowerTrunk(tower_channels, tower_blocks, tower_dil_mult, dropout)
        B = self.trunk.BIN
        assert self.resolution % B == 0, f"resolution {self.resolution} must be a multiple of {B}"
        assert self.flank % B == 0, f"flank {self.flank} must be a multiple of {B}"
        self.pool_factor = self.resolution // B           # 25
        self.trim = self.flank // B                       # 32
        self.trunk_from_scratch = True

        assert tower_pool in ("attn", "max", "meanmax"), tower_pool
        self.tower_pool = tower_pool
        C = self.trunk.out_channels
        if tower_pool == "attn":
            self.bin_pool = AttentionPool1d(C, self.pool_factor)
        proj_in = C * (2 if tower_pool == "meanmax" else 1)
        self.act = nn.GELU()
        self.projector = nn.Conv1d(proj_in, mid_hidden, kernel_size=1, bias=True)
        print(f"[Chiron3D_Tower] resolution={self.resolution} n_bins={self.n_bins} span={self.target_span} "
              f"flank={self.flank} input={self.input_width} pool_factor={self.pool_factor} "
              f"bin_pool={tower_pool} projector {proj_in}->{mid_hidden}")

        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.loop_head = bool(loop_head)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128, num_blocks=8, grad_ckpt=True,
                                      out_channels=2 if loop_head else 1)
        if self.loop_head:
            import math
            with torch.no_grad():
                self.decoder.conv_end.bias[1] = -math.log((1 - 0.01) / 0.01)

    def embed(self, x):
        """Bin embeddings the transformer receives: (B, mid_hidden, n_bins)."""
        if x.shape[-1] != self.input_width:
            off = (x.shape[-1] - self.input_width) // 2
            assert off >= 0, f"input {x.shape[-1]} bp is narrower than required {self.input_width} bp"
            x = x[..., off:off + self.input_width]
        x = self.trunk(x)                                          # (B, 512, input/32)
        x = x[..., self.trim:x.shape[-1] - self.trim]              # (B, 512, span/32)
        x = self.act(x)
        if self.tower_pool == "attn":
            x = self.bin_pool(x)
        else:
            b, c, l = x.shape
            x = x.view(b, c, self.n_bins, self.pool_factor)
            x = x.amax(-1) if self.tower_pool == "max" else torch.cat([x.mean(-1), x.amax(-1)], 1)
        return self.projector(x)                                   # (B, 128, N)

    def forward(self, x):
        x = self.embed(x)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x)
        return x if self.loop_head else x[:, 0]