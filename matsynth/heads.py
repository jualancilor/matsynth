"""Tahap B: shared MLP trunk + head_synth di atas embeddings tersimpan.

head_synth: trunk [d -> h] lalu linear [h -> 1], BCEWithLogitsLoss (sigmoid
implisit). Input hanya tensor embedding dari disk — encoder tidak disentuh.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, Z: np.ndarray) -> "Standardizer":
        assert Z.ndim == 2
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std < 1e-8] = 1.0  # kolom konstan
        return cls(mean=mean, std=std)

    def transform(self, Z: np.ndarray) -> np.ndarray:
        assert Z.ndim == 2 and Z.shape[1] == self.mean.shape[0], \
            f"Z shape {Z.shape} vs mean {self.mean.shape}"
        out = (Z - self.mean) / self.std
        assert np.isfinite(out).all()
        return out


class SynthHead(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.trunk = nn.Sequential(  # shared trunk [d -> h]
            nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(p_drop),
        )
        self.head_synth = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.dim() == 2 and z.size(1) == self.d_in, f"z shape {tuple(z.shape)}"
        h = self.trunk(z)
        assert h.shape == (z.size(0), self.hidden)
        logits = self.head_synth(h).squeeze(-1)
        assert logits.shape == (z.size(0),)
        return logits


def train_synth_head(Z_train: np.ndarray, y_train: np.ndarray,
                     hidden: int = 64, epochs: int = 800, lr: float = 3e-3,
                     weight_decay: float = 1e-3, seed: int = 0) -> SynthHead:
    assert Z_train.ndim == 2 and y_train.ndim == 1
    assert Z_train.shape[0] == y_train.shape[0]
    assert set(np.unique(y_train)) <= {0.0, 1.0}
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    assert n_pos > 0 and n_neg > 0, "train butuh kedua kelas"

    torch.manual_seed(seed)
    model = SynthHead(d_in=Z_train.shape[1], hidden=hidden)
    zt = torch.from_numpy(Z_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / n_pos))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):  # full-batch: dataset kecil
        opt.zero_grad()
        loss = loss_fn(model(zt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_proba(model: SynthHead, Z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(Z.astype(np.float32)))).numpy()
    assert p.shape == (Z.shape[0],)
    return p


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    assert y_true.shape == y_prob.shape
    y_pred = (y_prob >= thr).astype(int)
    yt = y_true.astype(int)
    tp = int(((y_pred == 1) & (yt == 1)).sum())
    tn = int(((y_pred == 0) & (yt == 0)).sum())
    fp = int(((y_pred == 1) & (yt == 0)).sum())
    fn = int(((y_pred == 0) & (yt == 1)).sum())
    n = len(yt)
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}
