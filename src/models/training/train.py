import sys
import torch
import argparse
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
from src.models.training.module import TrainModule
torch.serialization.add_safe_globals([argparse.Namespace])


def init_parser():
    parser = argparse.ArgumentParser(description='C.Origami Training Module.')

    # Data and Run Directories
    parser.add_argument('--seed', dest='run_seed', default=2077, type=int, help='Random seed for training')
    parser.add_argument('--save_path', dest='run_save_path', default='checkpoints', help='Path to the model checkpoint')
    parser.add_argument('--ckpt-path', dest='ckpt_path', default=None, help='Path to resume checkpoint')

    # Data directories
    parser.add_argument('--regions-file', dest='regions_file', help='Regions for training, validation and test data',
                        required=True)
    parser.add_argument('--fasta-dir', dest='fasta_dir', required=True, help='Directory with chromosome fasta files')
    parser.add_argument('--cool-file', dest='cool_file', required=True, help='Interaction matrix path')
    parser.add_argument('--genom-feat-path', dest='genom_feat_path', default=None,
                        help='Path to the genomic feature file (default: None)')

    # Model parameters
    parser.add_argument('--num-genom-feat', dest='num_genom_feat', type=int, default=0,
                        help='Number of genomic features to consider (default: 0)')

    # Training Parameters
    parser.add_argument('--patience', dest='trainer_patience', default=30, type=int,
                        help='Epoches before early stopping')
    parser.add_argument('--max-epochs', dest='trainer_max_epochs', default=60, type=int, help='Max epochs')
    parser.add_argument('--save-top-n', dest='trainer_save_top_n', default=5, type=int, help='Top n models to save')
    parser.add_argument('--num-gpu', dest='trainer_num_gpu', default=1, type=int, help='Number of GPUs to use')
    parser.add_argument('--local', action='store_true', help='Local test run (with CPU)')

    # Dataloader Parameters
    parser.add_argument('--batch-size', dest='dataloader_batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--ddp-disabled', dest='dataloader_ddp_disabled', action='store_false',
                        help='Using ddp, adjust batch size')
    parser.add_argument('--num-workers', dest='dataloader_num_workers', default=16, type=int, help='Dataloader workers')

    # If using backbone for embeddings
    parser.add_argument('--borzoi', action='store_true', help='Use borzoi backbone for embeddings')

    # Output-matrix geometry
    parser.add_argument('--resolution', dest='resolution', type=int, default=800,
                        help='Contact-map bin size in bp; must equal the --cool-file binsize')
    parser.add_argument('--n-bins', dest='n_bins', type=int, default=256,
                        help='Side length N of the predicted map; predicted span C = N * resolution')


    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    return args


def init_training(args):
    # Early_stopping
    early_stop_callback = callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.00,
                                                  patience=args.trainer_patience,
                                                  verbose=False,
                                                  mode="min")
    # Checkpoints
    checkpoint_callback = callbacks.ModelCheckpoint(dirpath=f'{args.run_save_path}/models',
                                                    save_top_k=20,
                                                    monitor='val_loss')

    # LR monitor
    lr_monitor = callbacks.LearningRateMonitor(logging_interval='epoch')

    # Logger
    csv_logger = pl.loggers.CSVLogger(save_dir=f'{args.run_save_path}/csv')

    # Assign seed
    pl.seed_everything(args.run_seed, workers=True)
    pl_module = TrainModule(args)

    pl_trainer = get_trainer(args, csv_logger, early_stop_callback, checkpoint_callback, lr_monitor)

    trainloader = pl_module.get_dataloader(args, 'train')
    valloader = pl_module.get_dataloader(args, 'val')

    if args.ckpt_path:
        print(f"Resuming from checkpoint: {args.ckpt_path}")
        pl_trainer.fit(pl_module, trainloader, valloader, ckpt_path=args.ckpt_path)
    else:
        pl_trainer.fit(pl_module, trainloader, valloader)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Training finished. Best checkpoint (lowest val_loss) is:\n {best_ckpt_path}")
    else:
        print("[DEBUG]: No best checkpoint found.")


def get_trainer(args, all_loggers, early_stop_callback, checkpoint_callback, lr_monitor):
    # Local test run
    if args.local:
        return pl.Trainer(accelerator="cpu",
                          devices=1,
                          gradient_clip_val=1,
                          logger=all_loggers,
                          precision="bf16",
                          callbacks=[early_stop_callback,
                                     checkpoint_callback,
                                     lr_monitor],
                          max_epochs=args.trainer_max_epochs)
    else:
        return pl.Trainer(strategy="ddp",
                          accelerator="gpu", devices=args.trainer_num_gpu,
                          accumulate_grad_batches=8,
                          gradient_clip_val=1,
                          logger=all_loggers,
                          precision="bf16",
                          callbacks=[early_stop_callback,
                                     checkpoint_callback,
                                     lr_monitor],
                          max_epochs=args.trainer_max_epochs)


def main():
    args = init_parser()
    init_training(args)


if __name__ == '__main__':
    main()
