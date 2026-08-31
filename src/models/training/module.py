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

    def training_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        inputs.requires_grad_()
        outputs = self(inputs)

        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)

        metrics = {'train_step_loss': loss}
        self.log_dict(metrics, batch_size=inputs.shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)

        if getattr(self.trainer, "sanity_checking", False):
            return ret_metrics
        inputs, mat = self.proc_batch(batch)
        with torch.no_grad():
            outputs = self(inputs)
        self._accumulate_corr(outputs, mat, store="val")
        return ret_metrics

    def test_step(self, batch, batch_idx):
        ret_metrics = self._shared_eval_step(batch, batch_idx)
        return ret_metrics

    def _shared_eval_step(self, batch, batch_idx):
        inputs, mat = self.proc_batch(batch)
        outputs = self(inputs)
        criterion = torch.nn.MSELoss()
        loss = criterion(outputs, mat)
        return loss

    def training_epoch_end(self, step_outputs):
        step_outputs = [out['loss'] for out in step_outputs]
        ret_metrics = self._shared_epoch_end(step_outputs)

        metrics = {
            'train_loss': ret_metrics['loss'] * 8,
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

        use_pretrained_backbone = False
        if args.borzoi:
            use_pretrained_backbone = True

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
