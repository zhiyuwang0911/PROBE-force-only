"""MACE backend for force-only PROBE."""

from .mace import (
    load_mace,
    get_z_table,
    train_val_split_loader,
    process_batch_mace_multitask,
    scan_force_error_boundaries,
)

# Alias for force-only training scripts
process_batch_mace_force_only = process_batch_mace_multitask
