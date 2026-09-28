"""Langkah 5: head_prec = kNN retrieval precursor di ruang z MACE.

Kendala data: ketiga resep argyrodite Kononova ada di satu family (AB6C8),
sehingga database retrieval untuk query argyrodite HARUS dari luar family itu
(split by family tetap tegak). Database = record Kononova chalcogenida
(S/Se/Te, 2-4 unsur, ber-mp_id) yang ter-resolve — di-featurize dengan encoder
yang sama.

Eval A (primer, argyrodite): query = resep argyrodite Kononova; db = record
chalcogenida dari family LAIN (assert_no_leakage). Eval B (suplemen, statistik
lebih bermakna): leave-one-family-out di seluruh set chalcogenida.
Skor: Jaccard(prediksi top-1, GT terbaik); GT bisa multi-resep per formula.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Composition, Structure

from matsynth.families import assert_no_leakage, family_key
from matsynth.knn_prec import retrieve_and_score
from matsynth.kononova import load_records
from matsynth.mace_embed import MaceEmbedder, precompute_embeddings
from matsynth.resolve import resolve_by_mp_id, resolve_structure
from matsynth.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
CH_SET_PATH = PROCESSED / "chalcogenide_set.json"
CH_EMB_PATH = PROCESSED / "embeddings_chalcogenide.pt"
ARGY_EMB_PATH = PROCESSED / "embeddings_argyrodite.pt"
CHALC = {"S", "Se", "Te"}
K = 3


def build_chalcogenide_set() -> dict[str, dict]:
    records, _ = load_records(validate_schema=False)
    by_formula: dict[str, dict] = {}
    for r in records:
        comp = Composition(r.target_formula)
        els = {e.symbol for e in comp}
        if not (els & CHALC) or not (2 <= len(els) <= 4) or not r.target_mp_id:
            continue
        d = by_formula.setdefault(r.target_formula, {"recipes": [], "mp_ids": []})
        d["recipes"].append(tuple(r.precursor_formulas))
        if r.target_mp_id not in d["mp_ids"]:
            d["mp_ids"].append(r.target_mp_id)

    items: dict[str, dict] = {}
    n_miss = 0
    for formula in sorted(by_formula):
        info = by_formula[formula]
        rs = None
        for mp_id in info["mp_ids"]:
            rs = resolve_by_mp_id(mp_id)
            if rs is not None:
                break
        if rs is None:
            rs = resolve_structure(formula)
        if rs is None:
            n_miss += 1
            continue
        counts = Counter(info["recipes"])
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        items[rs.material_id] = {
            "structure": rs.structure.as_dict(),
            "formula": formula,
            "family": family_key(formula),
            "precursors": sorted(best),
            "all_precursor_sets": [sorted(set(rcp)) for rcp in sorted(set(info["recipes"]))],
            "provider": rs.provider,
        }
    print(f"[set-chalc] resolved={len(items)} miss={n_miss}")
    assert items
    return items


def summarize(per_query: list[dict], group_of: list[str]) -> dict[str, dict]:
    by_g: dict[str, list[dict]] = defaultdict(list)
    for g, r in zip(group_of, per_query):
        by_g[g].append(r)
    out = {}
    for g in sorted(by_g):
        rs = by_g[g]
        out[g] = {
            "n_queries": len(rs),
            "jaccard_top1_mean": float(np.mean([r["jaccard_top1"] for r in rs])),
            "jaccard_best_of_topk_mean": float(np.mean([r["jaccard_best_of_topk"] for r in rs])),
            "exact_match_rate": float(np.mean([r["exact_match_top1"] for r in rs])),
        }
    return out


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if CH_SET_PATH.exists():
        ch_items = json.loads(CH_SET_PATH.read_text(encoding="utf-8"))
        print(f"[set-chalc] pakai set tersimpan ({len(ch_items)} item)")
    else:
        ch_items = build_chalcogenide_set()
        CH_SET_PATH.write_text(json.dumps(ch_items), encoding="utf-8")

    if CH_EMB_PATH.exists():
        ch_emb = torch.load(CH_EMB_PATH, weights_only=False)
        print(f"[embed-chalc] pakai embeddings tersimpan ({len(ch_emb)})")
    else:
        embedder = MaceEmbedder(model_size="small", device="cpu")
        structures = {m: Structure.from_dict(v["structure"]) for m, v in ch_items.items()}
        meta = {m: {k: v[k] for k in ("formula", "family", "precursors",
                                      "all_precursor_sets")}
                for m, v in ch_items.items()}
        ch_emb = precompute_embeddings(structures, CH_EMB_PATH, embedder, extra_meta=meta)

    mids = sorted(ch_emb.keys())
    Z = np.stack([ch_emb[m]["z"].numpy() for m in mids])
    fams = [ch_emb[m]["family"] for m in mids]
    formulas = [ch_emb[m]["formula"] for m in mids]
    prec = [frozenset(ch_emb[m]["precursors"]) for m in mids]
    gt_sets = [[frozenset(s) for s in ch_emb[m]["all_precursor_sets"]] for m in mids]
    assert Z.shape[0] == len(mids) and np.isfinite(Z).all()

    # ---------- Eval A: query argyrodite Kononova ----------
    argy = torch.load(ARGY_EMB_PATH, weights_only=False)
    q_mids = sorted(m for m, v in argy.items()
                    if v["source"] == "kononova" and v["precursors"])
    assert q_mids, "tidak ada query argyrodite dengan precursor GT"
    Zq = np.stack([argy[m]["z"].numpy() for m in q_mids])
    q_formulas = [argy[m]["formula"] for m in q_mids]
    q_fams = [argy[m]["family"] for m in q_mids]
    q_gt = [[frozenset(argy[m]["precursors"])] for m in q_mids]
    assert Zq.shape[1] == Z.shape[1], "dimensi embedding query != db"

    db_mask = [f not in set(q_fams) for f in fams]
    db_idx = [i for i, keep in enumerate(db_mask) if keep]
    assert_no_leakage([formulas[i] for i in db_idx], q_formulas)
    resA = retrieve_and_score(Z[db_idx], [prec[i] for i in db_idx], Zq, q_gt, k=K)
    for m, r in zip(q_mids, resA):
        r["query_material_id"] = m
        r["query_formula"] = q_formulas[q_mids.index(m)]
        r["neighbor_formulas"] = [formulas[db_idx[j]] for j in r.pop("neighbor_ids")]
    print("\nEval A (query argyrodite, db chalcogenida non-argyrodite):")
    for r in resA:
        print(f"  {r['query_formula']:<12} J_top1={r['jaccard_top1']:.3f} "
              f"tetangga={r['neighbor_formulas'][0]} pred={r['predicted_precursors']}")
    evalA_per_family = summarize(resA, q_fams)

    # ---------- Eval B: LOFO di set chalcogenida ----------
    resB, groupB = [], []
    for fam in sorted(set(fams)):
        q_idx = [i for i, f in enumerate(fams) if f == fam]
        d_idx = [i for i, f in enumerate(fams) if f != fam]
        if not d_idx:
            continue
        assert_no_leakage([formulas[i] for i in d_idx], [formulas[i] for i in q_idx])
        rs = retrieve_and_score(Z[d_idx], [prec[i] for i in d_idx],
                                Z[q_idx], [gt_sets[i] for i in q_idx], k=K)
        for qi, r in zip(q_idx, rs):
            r["query_formula"] = formulas[qi]
            r["neighbor_formulas"] = [formulas[d_idx[j]] for j in r.pop("neighbor_ids")]
        resB.extend(rs)
        groupB.extend([fam] * len(rs))
    evalB_per_family = summarize(resB, groupB)
    j_all = [r["jaccard_top1"] for r in resB]
    print(f"\nEval B (LOFO chalcogenida): n={len(resB)} "
          f"J_top1 makro-family={np.mean([v['jaccard_top1_mean'] for v in evalB_per_family.values()]):.3f} "
          f"mikro={np.mean(j_all):.3f}")

    save_json("step5_precursor_mace.json", {
        "encoder": "mace_invariant_last_layer_meanpool",
        "k": K,
        "db_n_items": len(mids),
        "db_n_families": len(set(fams)),
        "evalA_argyrodite": {
            "protocol": "query=resep argyrodite Kononova; db=chalcogenida family lain",
            "per_family": evalA_per_family,
            "per_query": resA,
        },
        "evalB_chalcogenide_lofo": {
            "protocol": "leave-one-family-out seluruh set chalcogenida",
            "per_family": evalB_per_family,
            "jaccard_top1_micro_mean": float(np.mean(j_all)),
        },
    })


if __name__ == "__main__":
    main()
