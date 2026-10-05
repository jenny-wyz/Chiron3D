import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from src.models.dataset.genomic_dataset import GenomicDataset
from src.models.training.utils import get_learnable_params, get_model
from einops import rearrange
import os
import numpy as np
from src.models.evaluation.metrics import mse, insulation_corr
import math

LOOP_POS_MASS = 20.67

def sigmoid_focal_loss(logits, targets, alpha=0.999, gamma=2.0, normalizer=None):
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    if normalizer is None:
        normalizer = LOOP_POS_MASS * logits.shape[0]      # constant, never batch-dependent
    return loss.sum() / normalizer
    

class TrainModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.model = get_model(args)
        self.args = args
        self.save_hyperparameters()
        self._last_preview_epoch = -1

    def forward(self, x):
        return self.model(x)

    def proc_batch(self, batch):
        inputs = batch["sequence"]

        if "features" in batch:
            inputs = torch.cat((inputs, batch["features"]), dim=1)  # TODO: Check this works with COrigami model too.

        return inputs, batch["matrix"]

    def on_validation_epoch_start(self):
        self._val_pearsons = []
        self._val_spearmans = []

    def _accumulate_corr(self, outputs, mats, store: str):
        with torch.no_grad():
            if not getattr(self.args, 'oe_target', False):
                outputs = torch.clamp(outputs, min=0)
            for out, true in zip(outputs, mats):
                out_c = out.detach().to(torch.float32).cpu()
                true_c = true.detach().to(torch.float32).cpu()

                r_pearson, r_spearman = insulation_corr(out_c, true_c, res=self.args.resolution)

                if store == "val":
                    if not np.isnan(r_pearson):
                        self._val_pearsons.append(float(r_pearson))
                    if not np.isnan(r_spearman):
                        self._val_spearmans.append(float(r_spearman))

    def _compute_loss(self, outputs, mat, batch):
        """Returns (total, loss_map, loss_loop). Reduces to v2/v3a when no loop head."""
        mask = batch["mask"].to(outputs.device)
        if getattr(self.args, "loop_file", None):
            pred_map, pred_loop = outputs[:, 0], outputs[:, 1]
        else:
            pred_map, pred_loop = outputs, None

        loss_map = (((pred_map - mat) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        if pred_loop is None:
            zero = torch.zeros((), device=outputs.device)
            return loss_map, loss_map, zero

        loss_loop = sigmoid_focal_loss(pred_loop, batch["loop_hm"].to(outputs.device),
                                       alpha=float(getattr(self.args, "loop_alpha", 0.999)),
                                       gamma=2.0)
        w = float(getattr(self.args, "loop_weight", 50.0))

        with torch.no_grad():
            hm = batch["loop_hm"].to(outputs.device)
            pos = hm > 0.5
            p = torch.sigmoid(pred_loop)
            B = pred_loop.shape[0]
            self.log("bs", float(B), prog_bar=True)
            self.log("npos_per_win", pos.sum().float() / B, prog_bar=True)
            self.log("loop_p_neg", p[~pos].mean(), prog_bar=True)
            if pos.any():
                self.log("loop_p_pos", p[pos].mean(), prog_bar=True)
            
        return loss_map + w * loss_loop, loss_map, loss_loop

    def _map_channel(self, outputs):
        return outputs[:, 0] if getattr(self.args, "loop_file", None) else outputs

    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        inputs.requires_grad_()
        outputs = self(inputs)

        loss, loss_map, loss_loop = self._compute_loss(outputs, mat, batch)

        metrics = {'train_step_loss': loss,
                   'train_step_map': loss_map,
                   'train_step_loop': loss_loop}
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)

        if getattr(self.trainer, "sanity_checking", False):
            return ret_metrics
        inputs, mat = self.proc_batch(batch)
        with torch.no_grad():
            outputs = self(inputs)
        self._accumulate_corr(self._map_channel(outputs), mat, store="val")
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        loss, _, _ = self._compute_loss(outputs, mat, batch)
        return loss

    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)

        metrics = {
            'train_loss': ret_metrics['loss'] * self.args.accum,
        }

        self.log_dict(metrics, prog_bar=True)

    def validation_epoch_end(self, step_outputs):
        ret_metrics = self._shared_epoch_end(step_outputs)

        val_p = float(np.mean(self._val_pearsons)) if len(self._val_pearsons) else float("nan")
        val_s = float(np.mean(self._val_spearmans)) if len(self._val_spearmans) else float("nan")

        metrics = {
            'val_loss': ret_metrics['loss'],
            'pearson_corr_val': torch.tensor(val_p, device=self.device),
            'spearman_corr_val': torch.tensor(val_s, device=self.device),
        }
        self.log_dict(metrics, prog_bar=True)

    def _shared_epoch_end(self, step_outputs):
        loss = torch.tensor(step_outputs).mean()
        return {'loss': loss}

    def configure_optimizers(self):
        params = get_learnable_params(self.model, weight_decay=0)
        optimizer = torch.optim.AdamW(params)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=1e-7,
        )

        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_loss',
            'strict': True,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    def get_dataset(self, args, mode):
        if args.num_genom_feat == 0:
            args.genom_feat_path = None  # Dont set it if num genomic features is 0

        use_pretrained_backbone = args.trunk in ('borzoi', 'dcnn')   # both want 4-channel one-hot
        flank = None if args.trunk == 'borzoi' else args.flank

        dataset = GenomicDataset(
            regions_file_path=args.regions_file,
            cool_file_path=args.cool_file,
            fasta_dir=args.fasta_dir,
            genomic_feature_path=args.genom_feat_path,
            mode=mode,
            # val_chroms=["chr5", "chr12", "chr13", "chr21"],
            # test_chroms=["chr2", "chr6", "chr19"],
            val_chroms=["chr2L"],
            test_chroms=["chrX"],
            use_pretrained_backbone=use_pretrained_backbone,
            resolution=args.resolution,
            n_bins=args.n_bins,
            flank=flank,
            oe_target=getattr(args, 'oe_target', False),
            loop_file=getattr(args, 'loop_file', None),
            balance=getattr(args, 'balance', False),
            matrix_scale=getattr(args, 'matrix_scale', 1.0),
            expected_file=getattr(args, 'expected_file', None),
            bad_bins_file=getattr(args, 'bad_bins_file', None),
        )

        return dataset

    def get_dataloader(self, args, mode):
        dataset = self.get_dataset(args, mode)

        shuffle = False  # validation and test settings
        if mode == 'train':
            shuffle = True

        batch_size = args.dataloader_batch_size
        num_workers = args.dataloader_num_workers
        if not args.dataloader_ddp_disabled:
            gpus = args.trainer_num_gpu
            batch_size = int(args.dataloader_batch_size / gpus)
            num_workers = int(args.dataloader_num_workers / gpus)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=None,
            persistent_workers=False
        )
        return dataloader
