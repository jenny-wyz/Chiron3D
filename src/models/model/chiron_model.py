import torch
import torch.nn as nn
import src.models.model.blocks as blocks
from borzoi_pytorch import Borzoi
from borzoi_pytorch.config_borzoi import BorzoiConfig
from pathlib import Path


def diagonalize_small(x):
    n = x.shape[-1]
    x_i = x.unsqueeze(2).repeat(1, 1, n, 1)
    x_j = x.unsqueeze(3).repeat(1, 1, 1, n)
    input_map = torch.cat([x_i, x_j], dim=1)
    return input_map

def move_feature_forward(x):
    # Input: (B, L, C) -> Output: (B, C, L)
    return x.transpose(1, 2).contiguous()

def get_borzoi_backbone(local: bool, model_type: str):
    assert model_type in ["borzoi", "flashzoi"], "Invalid model type. Choose 'borzoi' or 'flashzoi'."

    repo_root = Path(__file__).resolve().parents[3]
    model_dir = repo_root / "data" / model_type

    cfg = BorzoiConfig.from_pretrained(str(model_dir))
    cfg.return_center_bins_only = False
    borzoi = Borzoi.from_pretrained(str(model_dir), config=cfg)
    return borzoi


class Chiron3D(nn.Module):

    # Borzoi geometry -- fixed by the backbone, do not change.
    BORZOI_INPUT = 524288   # required input width; TargetLengthCrop raises below 523264
    EMB_BIN      = 32       # bp per embedding bin (conv-tower stride 2 * 2^4)
    EMB_BINS     = 16352    # 16384 - 32, from TargetLengthCrop(16384 - 32)
    EMB_HEAD_OFF = 512      # 16 bins trimmed off the front by that crop

    def __init__(self, mid_hidden=128, local=True, model_type="borzoi",
                 resolution=400, n_bins=256):
        super().__init__()

        self.resolution = int(resolution)
        self.n_bins = int(n_bins)
        self.target_span = self.n_bins * self.resolution

        assert self.target_span <= self.EMB_BINS * self.EMB_BIN, (
            f"target span {self.target_span} bp exceeds embedding span "
            f"{self.EMB_BINS * self.EMB_BIN} bp")
        assert self.target_span % 64 == 0, (
            f"n_bins*resolution = {self.target_span} must be a multiple of 64 so the "
            f"centre crop lands on an embedding-bin edge")

        self.flank = (self.BORZOI_INPUT - self.target_span) // 2
        assert (self.flank - self.EMB_HEAD_OFF) % self.EMB_BIN == 0
        self.emb_lo = (self.flank - self.EMB_HEAD_OFF) // self.EMB_BIN
        self.emb_hi = self.emb_lo + self.target_span // self.EMB_BIN

        # exact reshape-mean when resolution is a whole number of 32 bp bins
        self.exact_pool = (self.resolution % self.EMB_BIN == 0)
        self.pool_factor = self.resolution // self.EMB_BIN if self.exact_pool else None
        print(f"[Chiron3D] resolution={self.resolution} n_bins={self.n_bins} "
              f"span={self.target_span} flank={self.flank} "
              f"emb[{self.emb_lo}:{self.emb_hi}] exact_pool={self.exact_pool}")

        self.borzoi = get_borzoi_backbone(local, model_type)

        for param in self.borzoi.parameters():
            param.requires_grad = False
        self.borzoi.eval()

        self.activation = nn.ReLU()
        self.projector = nn.Conv1d(1536, mid_hidden, kernel_size=1, stride=1, padding=0, bias=True)

        self.length_reducer = nn.AdaptiveAvgPool1d(self.n_bins)

        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128,
                                      num_blocks=8, grad_ckpt=True)    # new

    def forward(self, x):
        x = self.borzoi.get_embs_after_crop(x)          # (B, 1536, 16352) @ 32 bp
        x = x[..., self.emb_lo:self.emb_hi]             # centre crop -> (B, 1536, span/32)
        x = self.projector(x)
        if self.exact_pool:
            b, c, l = x.shape
            x = x.view(b, c, self.n_bins, self.pool_factor).mean(-1)
        else:
            x = self.length_reducer(x)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        return x


class ResidualDownBlock(nn.Module):
    def __init__(self, ch, kernel_size, stride):
        super().__init__()
        # main conv path
        self.conv = nn.Conv1d(ch, ch, kernel_size, stride=stride, padding=0)
        self.bn = nn.GroupNorm(num_groups=1, num_channels=ch)
        # project the skip to match time‐length & channels
        self.skip = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=1, stride=stride, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=ch)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)       # [batch,128,L_in] → [batch,128,L_out]
        out = self.conv(x)            # → [batch,128,L_out]
        out = self.bn(out)            # normalize
        identity = identity[..., :out.size(-1)]
        out = out + identity          # merge
        out = self.act(out)           # nonlinearity
        return out



from src.models.model.dcnn_trunk import DCNNTrunk


class Chiron3D_DCNN(nn.Module):

    TRUNK_BIN = 2      # bp per trunk position (stem conv + MaxPool1d(2))

    def __init__(self, mid_hidden=128, resolution=400, n_bins=256,
                 flank=1024, asap_ckpt=None, dropout=0.1):
        super().__init__()

        self.resolution = int(resolution)
        self.n_bins = int(n_bins)
        self.target_span = self.n_bins * self.resolution
        self.flank = int(flank)
        self.input_width = self.target_span + 2 * self.flank

        assert self.resolution % self.TRUNK_BIN == 0, (
            f"resolution {self.resolution} must be even; trunk sits on a {self.TRUNK_BIN} bp grid")
        assert self.flank % self.TRUNK_BIN == 0, f"flank {self.flank} must be even"
        self.pool_factor = self.resolution // self.TRUNK_BIN
        self.trim = self.flank // self.TRUNK_BIN
        print(f"[Chiron3D_DCNN] resolution={self.resolution} n_bins={self.n_bins} "
              f"span={self.target_span} flank={self.flank} input={self.input_width} "
              f"pool_factor={self.pool_factor}")

        self.trunk = DCNNTrunk(asap_ckpt=asap_ckpt, dropout=dropout)

        self.activation = nn.ReLU()
        self.projector = nn.Conv1d(self.trunk.out_channels, mid_hidden, kernel_size=1, bias=True)

        self.attn = blocks.AttnModuleSmall(hidden=mid_hidden, record_attn=False)
        self.decoder = blocks.Decoder(mid_hidden * 2, hidden=128, num_blocks=8, grad_ckpt=True)

    def forward(self, x):
        if x.shape[-1] != self.input_width:          # tolerate a wider window (e.g. a Borzoi-width bed)
            off = (x.shape[-1] - self.input_width) // 2
            assert off >= 0, f"input {x.shape[-1]} bp is narrower than required {self.input_width} bp"
            x = x[..., off:off + self.input_width]
        x = self.trunk(x)                            # (B, 256, input_width/2) @ 2 bp
        x = x[..., self.trim:x.shape[-1] - self.trim]   # drop flank -> (B, 256, span/2)
        x = self.projector(x)
        b, c, l = x.shape
        x = x.view(b, c, self.n_bins, self.pool_factor).mean(-1)
        x = move_feature_forward(x)
        x = self.attn(x)
        x = move_feature_forward(x)
        x = diagonalize_small(x)
        x = self.decoder(x).squeeze(1)
        return x