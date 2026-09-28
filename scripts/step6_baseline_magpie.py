"""Langkah 6: BASELINE Magpie/matminer — floor komposisi yang harus dikalahkan.

Mengulang persis protokol langkah 4 (head_synth LOFO) dan langkah 5 (retrieval
Eval A + Eval B) tetapi vektor fitur = Magpie composition vector (fungsi
komposisi murni), bukan z MACE. Split, seed, arsitektur head, dan k identik.
Hasil dibandingkan side-by-side per family dengan hasil MACE.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matsynth.families import assert_no_leakage
from matsynth.knn_prec import retrieve_and_score
from matsynth.magpie_embed import MagpieEmbedder
from matsynth.utils import RESULTS_DIR, save_json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_synth_head import SEEDS, load_embeddings, lofo_eval  # noqa: E402
from step5_precursor_knn import (ARGY_EMB_PATH, CH_EMB_PATH, K,  # noqa: E402
                                 summarize)


def main() -> None:
    magpie = MagpieEmbedder()
    print(f"dim magpie = {magpie.dim}")

    # ---------- head_synth LOFO, fitur Magpie ----------
    mids, _Zmace, y, fams, formulas = load_embeddings()
    Zm = np.stack([magpie.embed_formula(f) for f in formulas])
    assert Zm.shape == (len(mids), magpie.dim)
    print("head_synth baseline Magpie (LOFO):")
    per_family_synth = lofo_eval(Zm, y, fams, formulas, seeds=SEEDS)

    # ---------- retrieval, fitur Magpie ----------
    ch_emb = torch.load(CH_EMB_PATH, weights_only=False)
    cmids = sorted(ch_emb.keys())
    cf = [ch_emb[m]["formula"] for m in cmids]
    cfam = [ch_emb[m]["family"] for m in cmids]
    cprec = [frozenset(ch_emb[m]["precursors"]) for m in cmids]
    cgt = [[frozenset(s) for s in ch_emb[m]["all_precursor_sets"]] for m in cmids]
    Zc = np.stack([magpie.embed_formula(f) for f in cf])

    argy = torch.load(ARGY_EMB_PATH, weights_only=False)
    q_mids = sorted(m for m, v in argy.items()
                    if v["source"] == "kononova" and v["precursors"])
    qf = [argy[m]["formula"] for m in q_mids]
    qfam = [argy[m]["family"] for m in q_mids]
    q_gt = [[frozenset(argy[m]["precursors"])] for m in q_mids]
    Zq = np.stack([magpie.embed_formula(f) for f in qf])

    db_idx = [i for i, f in enumerate(cfam) if f not in set(qfam)]
    assert_no_leakage([cf[i] for i in db_idx], qf)
    resA = retrieve_and_score(Zc[db_idx], [cprec[i] for i in db_idx], Zq, q_gt, k=K)
    for r, f in zip(resA, qf):
        r["query_formula"] = f
        r["neighbor_formulas"] = [cf[db_idx[j]] for j in r.pop("neighbor_ids")]
    evalA_per_family = summarize(resA, qfam)

    resB, groupB = [], []
    for fam in sorted(set(cfam)):
        q_idx = [i for i, f in enumerate(cfam) if f == fam]
        d_idx = [i for i, f in enumerate(cfam) if f != fam]
        if not d_idx:
            continue
        assert_no_leakage([cf[i] for i in d_idx], [cf[i] for i in q_idx])
        rs = retrieve_and_score(Zc[d_idx], [cprec[i] for i in d_idx],
                                Zc[q_idx], [cgt[i] for i in q_idx], k=K)
        for r in rs:
            r.pop("neighbor_ids")
        resB.extend(rs)
        groupB.extend([fam] * len(rs))
    evalB_per_family = summarize(resB, groupB)

    # ---------- perbandingan side-by-side vs MACE ----------
    mace4 = json.loads((RESULTS_DIR / "step4_synth_mace.json").read_text(encoding="utf-8"))
    mace5 = json.loads((RESULTS_DIR / "step5_precursor_mace.json").read_text(encoding="utf-8"))
    comparison = {"head_synth_per_family": {}, "retrieval_per_family": {}}
    for fam, m in per_family_synth.items():
        comparison["head_synth_per_family"][fam] = {
            "magpie": {"accuracy": m["accuracy"], "f1": m["f1"]},
            "mace": {"accuracy": mace4["per_family"][fam]["accuracy"],
                     "f1": mace4["per_family"][fam]["f1"]},
        }
    for fam, m in evalB_per_family.items():
        comparison["retrieval_per_family"][fam] = {
            "magpie_jaccard_top1": m["jaccard_top1_mean"],
            "mace_jaccard_top1":
                mace5["evalB_chalcogenide_lofo"]["per_family"][fam]["jaccard_top1_mean"],
        }

    print("\nperbandingan head_synth (acc):")
    for fam, c in comparison["head_synth_per_family"].items():
        print(f"  {fam:<8} magpie={c['magpie']['accuracy']:.3f} "
              f"mace={c['mace']['accuracy']:.3f}")

    save_json("step6_baseline_magpie.json", {
        "features": "matminer ElementProperty magpie",
        "dim": magpie.dim,
        "head_synth_per_family": per_family_synth,
        "retrieval_evalA_argyrodite": {"per_family": evalA_per_family,
                                       "per_query": resA},
        "retrieval_evalB_per_family": evalB_per_family,
        "comparison_vs_mace": comparison,
    })


if __name__ == "__main__":
    main()
