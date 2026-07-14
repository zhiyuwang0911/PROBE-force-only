# PROBE-force-only

Force reliability classifier built on frozen MACE embeddings.

Extends the [PROBE](https://github.com/isayevlab/PROBE) framework to classify **force prediction reliability** only — no energy reliability head. Two outputs:

- **Per-atom force reliability** — is the force on atom *i* reliable?
- **Structure-level force reliability** — mean-aggregated from per-atom logits (no extra head)

Based on [PROBE-forces](https://github.com/zhiyuwang0911/PROBE-forces) multitask code, with the energy classifier removed.

---

## Architecture

```
Frozen MACE backbone
        │
        ▼  per-atom embeddings [B, N, d]
  Atom Encoder MLP  (d → 256 → 128 → 256)
        │
        ▼
  Multi-Head Self-Attention  (32 heads)
        │
        ├─ atom force head MLP [256 → 128 → 32 → 2]  →  logits per atom [B, N, 2]
        └─ mean aggregate P(unreliable) over atoms   →  logits per structure [B, 2]
```

Trainable parameters: ~426K (shared encoder + attention + atom force head; no energy pool/proj/classifier).

---

## Installation

```bash
conda env create -f environment_mace.yml
conda activate probe_mace
```

**CUDA acceleration (optional):**

```bash
conda env create -f environment_mace_cueq.yml
conda activate probe_mace_cueq
```

---

## Training

1. Edit `CONFIG` in `train_mace_force_only.py` (paths, batch size, etc.).

2. Run:

```bash
python train_mace_force_only.py
```

**Optional flags:**

```bash
python train_mace_force_only.py --enable-cueq

python train_mace_force_only.py \
  --lambda-force-atom 1.0 \
  --lambda-force-mol 0.3
```

| Flag | `CONFIG` key | Default | Task |
|------|----------------|---------|------|
| `--lambda-force-atom` | `lambda_force_atom` | `1.0` | Per-atom force reliability |
| `--lambda-force-mol` | `lambda_force_mol` | `1.0` | Structure force reliability |

`L_Fs` is derived from the same per-atom head as `L_Fa`; lowering `lambda_force_mol` (e.g. `0.3`) is often reasonable.

Labels use the 50th percentile of force component MAE on the training set (class 0 = reliable, class 1 = unreliable). Checkpoint: `output_dir/best_force_only_model_<timestamp>.pt`.

**Resume after walltime / crash:** each epoch also writes `output_dir/last_checkpoint.pt`. Continue without re-scanning force boundaries:

```bash
python train_mace_force_only.py --resume
python train_mace_force_only.py --resume /path/to/last_checkpoint.pt --checkpoint-every 1
```

---

## Repository layout

```
PROBE-force-only/
├── probe/
│   ├── model.py          # ForceOnlyPROBEModel
│   ├── train.py          # force-only training / evaluation
│   ├── labels.py         # force error metrics + boundaries
│   ├── metrics.py
│   ├── io_extxyz.py
│   └── backends/mace.py
├── train_mace_force_only.py
├── infer_mace_force_only.py
├── environment_mace.yml
└── environment_mace_cueq.yml
```

---

## Inference

## Inference (CLI)

```bash
python infer_mace_force_only.py \
  --mace-model /path/to/MACE-OFF23_large.model \
  --checkpoint /path/to/best_force_only_model_YYYYMMDD_HHMMSS.pt \
  --test-xyz /path/to/test.xyz \
  --output-dir ./probe_force_only_inference
```

Writes `predictions_structure.csv`, `predictions_atom.csv`, `predictions.npz`, and `metrics.json` when reference forces are available.


```python
import torch
from probe.model import ForceOnlyPROBEModel

model = ForceOnlyPROBEModel(backbone_dim=224)  # match your MACE feat_dim
ckpt = torch.load('best_force_only_model.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# atom_feats [B, N, d], atom_mask [B, N] bool
with torch.no_grad():
    logits_atom, logits_mol = model(atom_feats, atom_mask)
    p_unrel_atom = torch.softmax(logits_atom, dim=-1)[..., 1]   # [B, N]
    p_unrel_mol  = torch.softmax(logits_mol, dim=-1)[..., 1]    # [B]
```

---

## License

MIT — see [LICENSE](LICENSE).

Extends original [PROBE](https://github.com/isayevlab/PROBE) (Isayev Lab). Force-only modifications Copyright (c) 2026 Zhiyu Wang.
