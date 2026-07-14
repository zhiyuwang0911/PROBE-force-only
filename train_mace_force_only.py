"""
Train force-only PROBE on MACE-OFF23.

Classifies per-atom and structure-level force reliability only (no energy head).
Structure-level force predictions mean-aggregate per-atom force logits.

MACE forces are computed via live forward pass (not read from extxyz).

Edit the CONFIG block below, then run:
    python train_mace_force_only.py
    python train_mace_force_only.py --enable-cueq
    python train_mace_force_only.py --lambda-force-atom 1.0 --lambda-force-mol 0.3
    python train_mace_force_only.py --resume   # continue from output_dir/last_checkpoint.pt
"""

import argparse
from pathlib import Path

import torch

from probe.model import ForceOnlyPROBEModel
from probe.backends.mace import (
    load_mace, get_z_table,
    train_val_split_loader,
    process_batch_mace_multitask,
    scan_force_error_boundaries,
)
from probe.train import run_force_only_training

CONFIG = {
    'mace_model_path':   '/path/to/MACE-OFF23_large.model',
    'train_xyz':         '/path/to/train.xyz',
    'output_dir':        './probe_mace_force_only_outputs',

    'device':            'cuda' if torch.cuda.is_available() else 'cpu',
    'enable_cueq':       False,
    'batch_size':        128,
    'valid_fraction':    0.1,
    'error_boundary_percentile': 50,

    'lambda_force_atom': 1.0,
    'lambda_force_mol':  1.0,

    'lr':                5e-5,
    'weight_decay':      1e-4,
    'epochs':            1000,
    'early_stopping_patience': 10,
    'scheduler_patience':      5,
    'scheduler_factor':        0.9,
    'min_lr':            5e-6,
    'gradient_clip_norm':1.0,
    'checkpoint_every':  1,

    'atom_encoder_hidden':       [256, 128],
    'atom_encoder_output_dim':   256,
    'mol_attention_heads':       32,
    'atom_force_head_hidden':    [128, 32],
    'dropout':                   0.1,

    'high_conf_cutoffs': {0: 0.8, 1: 0.8},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train force-only PROBE on MACE-OFF23')
    parser.add_argument('--enable-cueq', action='store_true',
                        help='Enable NVIDIA cuEquivariance CUDA acceleration')
    parser.add_argument('--lambda-force-atom', type=float, default=None,
                        help='Loss weight for per-atom force (default: CONFIG)')
    parser.add_argument('--lambda-force-mol', type=float, default=None,
                        help='Loss weight for structure force (default: CONFIG)')
    parser.add_argument(
        '--resume', nargs='?', const='AUTO', default=None,
        help='Resume training. With no path, uses '
             'CONFIG[output_dir]/last_checkpoint.pt',
    )
    parser.add_argument(
        '--checkpoint-every', type=int, default=None,
        help='Save last_checkpoint.pt every N epochs '
             '(default: CONFIG checkpoint_every)',
    )
    return parser.parse_args()


def _resolve(cli_value, config_key: str) -> float:
    return cli_value if cli_value is not None else CONFIG[config_key]


def main():
    args = parse_args()
    device = CONFIG['device']
    enable_cueq = CONFIG['enable_cueq'] or args.enable_cueq
    lambda_force_atom = _resolve(args.lambda_force_atom, 'lambda_force_atom')
    lambda_force_mol = _resolve(args.lambda_force_mol, 'lambda_force_mol')
    checkpoint_every = (args.checkpoint_every
                        if args.checkpoint_every is not None
                        else CONFIG.get('checkpoint_every', 1))

    resume_path = None
    if args.resume is not None:
        resume_path = (Path(CONFIG['output_dir']) / 'last_checkpoint.pt'
                       if args.resume == 'AUTO' else Path(args.resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {resume_path}")

    print(
        f"Loss weights: lambda_force_atom={lambda_force_atom}, "
        f"lambda_force_mol={lambda_force_mol}"
    )

    extractor = load_mace(CONFIG['mace_model_path'], device, enable_cueq=enable_cueq)
    z_table = get_z_table(extractor)
    r_max = float(extractor.mace_model.r_max)

    print("Loading data...")
    train_loader, val_loader = train_val_split_loader(
        CONFIG['train_xyz'], z_table, r_max,
        CONFIG['batch_size'], CONFIG['valid_fraction'],
    )

    if resume_path is not None:
        print(f"Loading error bins from resume checkpoint {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        error_bins_f_atom = torch.tensor(
            resume_ckpt['error_bins_force_atom'], device=device, dtype=torch.float32)
        error_bins_f_mol = torch.tensor(
            resume_ckpt['error_bins_force_mol'], device=device, dtype=torch.float32)
        print(f"  force_atom bins={error_bins_f_atom.tolist()}")
        print(f"  force_mol bins={error_bins_f_mol.tolist()}")
    else:
        print("Computing force error boundaries on training set...")
        boundary_f_atom, boundary_f_mol = scan_force_error_boundaries(
            train_loader, device, extractor, CONFIG['error_boundary_percentile'])
        error_bins_f_atom = torch.tensor([0.0, boundary_f_atom], device=device)
        error_bins_f_mol = torch.tensor([0.0, boundary_f_mol], device=device)

    model = ForceOnlyPROBEModel(
        backbone_dim=extractor.feat_dim,
        atom_encoder_hidden=CONFIG['atom_encoder_hidden'],
        atom_encoder_output_dim=CONFIG['atom_encoder_output_dim'],
        mol_attention_heads=CONFIG['mol_attention_heads'],
        atom_force_head_hidden=CONFIG['atom_force_head_hidden'],
        dropout=CONFIG['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Force-only PROBE parameters: {total_params:,}")

    process_fn = lambda batch, dev: process_batch_mace_multitask(batch, dev, extractor)
    history = run_force_only_training(
        model=model,
        process_batch_fn=process_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        error_bins_f_atom=error_bins_f_atom,
        error_bins_f_mol=error_bins_f_mol,
        device=device,
        output_dir=CONFIG['output_dir'],
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
        epochs=CONFIG['epochs'],
        early_stopping_patience=CONFIG['early_stopping_patience'],
        scheduler_patience=CONFIG['scheduler_patience'],
        scheduler_factor=CONFIG['scheduler_factor'],
        min_lr=CONFIG['min_lr'],
        gradient_clip_norm=CONFIG['gradient_clip_norm'],
        lambda_force_atom=lambda_force_atom,
        lambda_force_mol=lambda_force_mol,
        high_conf_cutoffs=CONFIG['high_conf_cutoffs'],
        resume_path=str(resume_path) if resume_path else None,
        checkpoint_every=checkpoint_every,
    )

    print(f"\nTraining complete. Best epoch: {history['best_epoch']}")
    print(f"Checkpoint saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
