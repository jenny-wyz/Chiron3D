import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from src.models.model.asap_vendor.convnext_dcnn import ConvNeXtDCNN


class DCNNTrunk(nn.Module):
    """ASAP ConvNeXtDCNN without its ATAC head. (B, 4, L) -> (B, 256, L//2), 2 bp/position."""

    def __init__(self, asap_ckpt=None, dropout=0.1, use_checkpoint=True):
        super().__init__()
        m = ConvNeXtDCNN(dropout=dropout)
        if asap_ckpt is not None:
            sd = torch.load(asap_ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(sd, "module.")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            print(f"[DCNNTrunk] missing={len(missing)} unexpected={len(unexpected)}")
            w = m.init_conv.dwconv.weight.data              # ASAP one-hot order A,G,C,T
            m.init_conv.dwconv.weight.data = w[:, [0, 2, 1, 3]].clone()   # -> A,C,G,T
        self.stem, self.pool, self.core = m.init_conv, m.init_pool, m.core
        self.use_checkpoint = use_checkpoint
        self.out_channels = 256

    def _block(self, x, i):
        c = self.core
        return c.dropout(c.conv_blocks[i](c.dconv_blocks[i](x)))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(self.stem, x, use_reentrant=False)
        else:
            x = self.stem(x)
        x = self.pool(F.pad(x, (1, 0)))
        for i in range(self.core.nr_res_blocks):
            if self.use_checkpoint and self.training:
                x = x + cp.checkpoint(self._block, x, i, use_reentrant=False)
            else:
                x = x + self._block(x, i)
        return x