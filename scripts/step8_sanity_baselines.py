"""Langkah 8: FLOOR CHECK — apakah hasil langkah 4-6 berarti apa-apa?

Dua diagnostik yang menentukan interpretasi seluruh proyek:

(A) Baseline heuristik unsur untuk retrieval precursor. Untuk tiap unsur di
    target, pakai precursor yang paling sering dipakai untuk unsur itu di
    train. Ini yang dilakukan chemist TANPA ML. Kalau MACE tidak mengalahkan
    ini, angka Jaccard langkah 5 tidak bermakna. Uji berpasangan (Wilcoxon +
    bootstrap CI) terhadap MACE, bukan sekadar beda rata-rata.

(B) Cek confound anion pada set synthesizability. Kalau aturan satu-bit
    "mengandung O -> negatif" menyamai akurasi model, maka head_synth tidak
    belajar synthesizability melainkan menebak anion — artefak konstruksi
    dataset (positif kurasi = sulfida/selenida nyata; negatif OQMD hipotetis
    = mayoritas oksida).
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Composition
from scipy.stats import wilcoxon

from matsynth.knn_prec import jaccard, retrieve_and_score
from matsynth.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
CH_EMB = ROOT / "data" / "processed" / "embeddings_chalcogenide.pt"
ARGY_EMB = ROOT / "data" / "processed" / "embeddings_argyrodite.pt"
K = 3
N_BOOT = 20000


def elements_of(formula: str) -> list[str]:
    return sorted({e.symbol for e in Composition(formula)})


def build_element_map(train_prec_sets: list[frozenset]) -> dict[str, str]:
    """unsur -> precursor yang paling sering memuat unsur itu di train."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for prec_set in train_prec_sets:
        for p in prec_set:
            try:
                for e in elements_of(p):
                    counts[e][p] += 1
            except Exception:
                continue
    return {e: c.most_common(1)[0][0] for e, c in counts.items()}


def predict_naive(formula: str, elem_map: dict[str, str]) -> frozenset:
    return frozenset(elem_map[e] for e in elements_of(formula) if e in elem_map)


# ---------------------------------------------------------------- diagnostik A
def diagnostic_a() -> dict:
    ch = torch.load(CH_EMB, weights_only=False)
    mids = sorted(ch.keys())
    Z = np.stack([ch[m]["z"].numpy() for m in mids])
    cf = [ch[m]["formula"] for m in mids]
    cfam = [ch[m]["family"] for m in mids]
    cprec = [frozenset(ch[m]["precursors"]) for m in mids]
    cgt = [[frozenset(s) for s in ch[m]["all_precursor_sets"]] for m in mids]
    assert Z.shape[0] == len(mids)

    naive_j, mace_j = {}, {}
    for fam in sorted(set(cfam)):
        q_idx = [i for i, f in enumerate(cfam) if f == fam]
        d_idx = [i for i, f in enumerate(cfam) if f != fam]
        if not d_idx:
            continue
        emap = build_element_map([cprec[i] for i in d_idx])
        for qi in q_idx:
            naive_j[cf[qi]] = max(jaccard(predict_naive(cf[qi], emap), g)
                                  for g in cgt[qi])
        rs = retrieve_and_score(Z[d_idx], [cprec[i] for i in d_idx],
                                Z[q_idx], [cgt[i] for i in q_idx], k=K)
        for qi, r in zip(q_idx, rs):
            mace_j[cf[qi]] = r["jaccard_top1"]

    keys = sorted(set(naive_j) & set(mace_j))
    a = np.array([mace_j[k] for k in keys])
    b = np.array([naive_j[k] for k in keys])
    d = a - b
    assert len(keys) == len(mids), "sebagian query hilang saat pemasangan"

    nz = d[np.abs(d) > 1e-9]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(N_BOOT)])
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    # seberapa trivial resep GT
    triv = []
    for i, formula in enumerate(cf):
        tel = set(elements_of(formula))
        for g in cgt[i]:
            if g:
                triv.append(np.mean([set(elements_of(pp)) <= (tel | {"O", "C", "H", "N"})
                                     for pp in g]))

    out = {
        "n_paired_queries": len(keys),
        "mace_jaccard_mean": float(a.mean()),
        "naive_element_heuristic_jaccard_mean": float(b.mean()),
        "mean_difference_mace_minus_naive": float(d.mean()),
        "wins_mace": int((d > 1e-9).sum()),
        "ties": int((np.abs(d) <= 1e-9).sum()),
        "wins_naive": int((d < -1e-9).sum()),
        "wilcoxon_p": p,
        "bootstrap_ci95_difference": [lo, hi],
        "significant_at_95": bool(lo > 0 or hi < 0),
        "fraction_of_gt_precursors_that_are_trivial": float(np.mean(triv)),
        "n_recipes_checked": len(triv),
    }
    print("=" * 68)
    print("DIAGNOSTIK A — retrieval precursor vs heuristik unsur (LOFO, n=99)")
    print(f"  MACE   J = {out['mace_jaccard_mean']:.4f}")
    print(f"  naive  J = {out['naive_element_heuristic_jaccard_mean']:.4f}")
    print(f"  selisih  = {out['mean_difference_mace_minus_naive']:+.4f}  "
          f"Wilcoxon p = {p:.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]")
    print(f"  menang {out['wins_mace']} / seri {out['ties']} / kalah {out['wins_naive']}")
    print(f"  -> {'SIGNIFIKAN' if out['significant_at_95'] else 'TIDAK SIGNIFIKAN'}")
    print(f"  fraksi precursor GT yang trivial (hanya unsur target +O/C/H/N) = "
          f"{out['fraction_of_gt_precursors_that_are_trivial']:.3f}")
    return out


# ---------------------------------------------------------------- diagnostik B
def diagnostic_b() -> dict:
    d = torch.load(ARGY_EMB, weights_only=False)
    rows = []
    for m in sorted(d.keys()):
        els = {e.symbol for e in Composition(d[m]["formula"])}
        rows.append({"formula": d[m]["formula"], "family": d[m]["family"],
                     "label": int(d[m]["label"]), "has_O": "O" in els,
                     "source": d[m].get("source", "?")})

    xtab = {
        "has_O_and_positive": sum(r["has_O"] and r["label"] == 1 for r in rows),
        "has_O_and_negative": sum(r["has_O"] and r["label"] == 0 for r in rows),
        "no_O_and_positive": sum(not r["has_O"] and r["label"] == 1 for r in rows),
        "no_O_and_negative": sum(not r["has_O"] and r["label"] == 0 for r in rows),
    }
    acc_all = float(np.mean([(0 if r["has_O"] else 1) == r["label"] for r in rows]))

    mace_acc = {"AB6C7": 0.875, "AB6C8": 0.5087719298245613, "ABC5D6": 1.0}
    per_family = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        a = float(np.mean([(0 if r["has_O"] else 1) == r["label"] for r in sub]))
        per_family[fam] = {
            "n": len(sub),
            "oxygen_rule_accuracy": a,
            "mace_accuracy": mace_acc[fam],
            "oxygen_rule_matches_or_beats_mace": bool(a >= mace_acc[fam] - 1e-9),
        }

    out = {
        "crosstab_label_vs_oxygen": xtab,
        "oxygen_rule_accuracy_whole_set": acc_all,
        "per_family": per_family,
        "confound_present": bool(acc_all >= 0.85),
        "interpretation": (
            "Positif (kurasi literatur) hampir semuanya sulfida/selenida; negatif "
            "(OQMD hipotetis) hampir semuanya oksida. Aturan satu-bit 'ada O -> "
            "negatif' menyamai/mengalahkan MACE di SEMUA family. Akurasi head_synth "
            "karenanya tidak dapat diklaim sebagai bukti model belajar "
            "synthesizability — ia konsisten dengan model yang hanya mendeteksi anion."
        ),
    }
    print()
    print("=" * 68)
    print("DIAGNOSTIK B — confound anion pada head_synth (n=37)")
    print(f"  tabel silang: {xtab}")
    print(f"  aturan 'ada O -> negatif', seluruh set: acc = {acc_all:.3f}")
    for fam, v in per_family.items():
        print(f"    {fam:<8} n={v['n']:>2}  aturan-O = {v['oxygen_rule_accuracy']:.3f}"
              f"   MACE = {v['mace_accuracy']:.3f}")
    print(f"  -> confound {'ADA' if out['confound_present'] else 'tidak terdeteksi'}")
    return out


def main() -> None:
    a = diagnostic_a()
    b = diagnostic_b()
    save_json("step8_sanity_baselines.json", {
        "purpose": "floor check untuk menentukan apakah langkah 4-6 bermakna",
        "diagnostic_a_retrieval_vs_naive_element_heuristic": a,
        "diagnostic_b_anion_confound_in_synth_labels": b,
        "verdict": (
            "Kedua head gagal floor check. Retrieval tidak berbeda signifikan dari "
            "heuristik unsur (p=%.2f). head_synth confounded oleh anion. Tidak ada "
            "bukti encoder geometrik membantu pada eksperimen sebagaimana dirancang."
            % a["wilcoxon_p"]
        ),
    })


if __name__ == "__main__":
    main()
