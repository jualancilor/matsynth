"""Langkah 4 (Tahap B): head_synth di embeddings MACE argyrodite.

Protokol split: leave-one-family-out (LOFO) — tiap family (anonymized formula)
bergantian jadi test set penuh, train = family lain. assert_no_leakage dicek di
tiap fold. Multi-seed untuk stabilitas. Metrik dilaporkan PER FAMILY (tiap fold
= satu family), tanpa agregat tunggal yang menyembunyikan kegagalan per family.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matsynth.families import assert_no_leakage
from matsynth.heads import (Standardizer, binary_metrics, predict_proba,
                            train_synth_head)
from matsynth.utils import save_json

EMB_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "embeddings_argyrodite.pt"
SEEDS = [0, 1, 2]
HIDDEN = 64
EPOCHS = 800


def load_embeddings(path: Path = EMB_PATH):
    data = torch.load(path, weights_only=False)
    mids = sorted(data.keys())
    Z = np.stack([data[m]["z"].numpy() for m in mids])
    y = np.array([float(data[m]["label"]) for m in mids])
    fams = np.array([data[m]["family"] for m in mids])
    formulas = np.array([data[m]["formula"] for m in mids])
    assert Z.ndim == 2 and Z.shape[0] == len(mids) == len(y) == len(fams)
    assert np.isfinite(Z).all()
    return mids, Z, y, fams, formulas


def lofo_eval(Z, y, fams, formulas, seeds=SEEDS) -> dict:
    per_family: dict[str, dict] = {}
    for fam in sorted(set(fams)):
        te = fams == fam
        tr = ~te
        assert_no_leakage(list(formulas[tr]), list(formulas[te]))
        assert set(np.unique(y[tr])) == {0.0, 1.0}, \
            f"train fold {fam} tidak punya dua kelas"

        seed_metrics = []
        for seed in seeds:
            std = Standardizer.fit(Z[tr])
            model = train_synth_head(std.transform(Z[tr]), y[tr],
                                     hidden=HIDDEN, epochs=EPOCHS, seed=seed)
            p = predict_proba(model, std.transform(Z[te]))
            seed_metrics.append(binary_metrics(y[te], p))

        agg = {k: float(np.mean([m[k] for m in seed_metrics]))
               for k in ("accuracy", "f1", "precision", "recall")}
        agg.update({
            "n_test": int(te.sum()),
            "n_test_pos": int(y[te].sum()),
            "n_test_neg": int((1 - y[te]).sum()),
            "n_train": int(tr.sum()),
            "accuracy_std_over_seeds": float(np.std([m["accuracy"] for m in seed_metrics])),
            "f1_std_over_seeds": float(np.std([m["f1"] for m in seed_metrics])),
            "per_seed": seed_metrics,
            "test_formulas": sorted(formulas[te]),
        })
        per_family[fam] = agg
        print(f"  family={fam:<8} n_test={agg['n_test']:>2} "
              f"acc={agg['accuracy']:.3f}±{agg['accuracy_std_over_seeds']:.3f} "
              f"f1={agg['f1']:.3f}±{agg['f1_std_over_seeds']:.3f}")
    return per_family


def main() -> None:
    mids, Z, y, fams, formulas = load_embeddings()
    print(f"n={len(mids)} d={Z.shape[1]} pos={int(y.sum())} neg={int(len(y)-y.sum())}")
    per_family = lofo_eval(Z, y, fams, formulas)
    save_json("step4_synth_mace.json", {
        "encoder": "mace_invariant_last_layer_meanpool",
        "embedding_dim": int(Z.shape[1]),
        "protocol": "leave-one-family-out, family=anonymized_formula, "
                    "standardizer fit di train, multi-seed",
        "seeds": SEEDS, "hidden": HIDDEN, "epochs": EPOCHS,
        "n_total": len(mids), "n_pos": int(y.sum()),
        "per_family": per_family,
    })


if __name__ == "__main__":
    main()
