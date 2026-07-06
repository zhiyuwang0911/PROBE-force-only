"""
PROBE force-only model architecture.

Classifies per-atom and structure-level force reliability from frozen
MLIP per-atom embeddings. No energy reliability head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None,
                return_attention: bool = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        if return_attention:
            return out, attn_weights
        return out


def build_mlp(input_dim: int, hidden_dims: list, output_dim: int,
              dropout: float = 0.1, use_layernorm: bool = True,
              last_activation: bool = False, last_layernorm: bool = False) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    if last_layernorm:
        layers.append(nn.LayerNorm(output_dim))
    if last_activation:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


def aggregate_atom_force_logits(logits_atom: torch.Tensor,
                                atom_mask: torch.Tensor) -> torch.Tensor:
    """Mean-aggregate per-atom force logits to structure-level logits."""
    probs = F.softmax(logits_atom, dim=-1)
    p_unrel = probs[..., 1]
    mask_f = atom_mask.float()
    p_mol = (p_unrel * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
    p_mol = p_mol.clamp(1e-6, 1 - 1e-6)
    return torch.stack([torch.log(1 - p_mol), torch.log(p_mol)], dim=-1)


class ForceOnlyPROBEModel(nn.Module):
    """
    Force-only PROBE: per-atom + structure force reliability.

    Architecture:
        atom encoder MLP  →  [B, N, D]
        multi-head self-attention  →  [B, N, D]
        ├─ atom force head: per-atom MLP → [B, N, 2]
        └─ structure force: mean aggregate atom logits → [B, 2]  (no extra head)
    """

    def __init__(
        self,
        backbone_dim: int,
        n_classes: int = 2,
        atom_encoder_hidden: list = [256, 128],
        atom_encoder_output_dim: int = 256,
        mol_attention_heads: int = 32,
        atom_force_head_hidden: list = [128, 32],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.atom_encoder_output_dim = atom_encoder_output_dim

        self.atom_encoder = build_mlp(
            backbone_dim, atom_encoder_hidden, atom_encoder_output_dim,
            dropout, use_layernorm=True, last_layernorm=True,
        )
        self.mol_attention = MultiHeadSelfAttention(
            atom_encoder_output_dim, mol_attention_heads, dropout
        )
        self.mol_attention_norm = nn.LayerNorm(atom_encoder_output_dim)
        self.atom_force_head = build_mlp(
            atom_encoder_output_dim, atom_force_head_hidden, n_classes,
            dropout, use_layernorm=True,
        )
        self._last_attention_weights = None

    def encode_atoms(self, atom_feats: torch.Tensor,
                     atom_mask: torch.Tensor) -> torch.Tensor:
        z = self.atom_encoder(atom_feats)
        attended, attn_w = self.mol_attention(z, mask=atom_mask, return_attention=True)
        attended = self.mol_attention_norm(attended + z)
        self._last_attention_weights = attn_w.detach()
        return attended

    def forward(self, atom_feats: torch.Tensor, atom_mask: torch.Tensor,
                return_attention: bool = False):
        """
        Returns:
            logits_force_atom: [B, N, 2]
            logits_force_mol:  [B, 2]
        """
        attended = self.encode_atoms(atom_feats, atom_mask)
        logits_force_atom = self.atom_force_head(attended)
        logits_force_mol = aggregate_atom_force_logits(logits_force_atom, atom_mask)

        if return_attention:
            return logits_force_atom, logits_force_mol, self._last_attention_weights
        return logits_force_atom, logits_force_mol

    def get_atom_importance(self, atom_feats: torch.Tensor,
                            atom_mask: torch.Tensor) -> torch.Tensor:
        _, _, attn_w = self.forward(atom_feats, atom_mask, return_attention=True)
        importance = attn_w.mean(dim=1).sum(dim=1)
        importance = importance * atom_mask.float()
        importance = importance / importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return importance

    def get_attention_weights(self) -> torch.Tensor:
        return self._last_attention_weights
