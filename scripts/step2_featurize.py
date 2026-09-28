"""Langkah 2 (Tahap A): assembly set argyrodite + featurisasi MACE -> .pt.

Set terdiri dari:
  - positif Kononova (resep argyrodite ter-resolve; menyimpan precursor set),
  - positif kurasi literatur (struktur via OPTIMADE; tanpa precursor),
  - negatif skema CSLLM: entry OQMD hipotetis (tanpa ICSD id) dekat hull
    (_oqmd_stability <= 0.1 eV/atom) pada grid substitusi argyrodite,
    dikecualikan semua formula yang diketahui tersintesis.

Output:
  data/processed/argyrodite_set.json  (struktur + metadata, dipakai langkah 3)
  data/processed/embeddings_argyrodite.pt  ({material_id: {z, label, ...}})
  results/step2_featurize.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Composition, Structure

from matsynth.families import (CURATED_SYNTHESIZED_ARGYRODITES,
                               argyrodite_candidate_formulas, family_key,
                               is_argyrodite)
from matsynth.kononova import load_records
from matsynth.mace_embed import MaceEmbedder, precompute_embeddings
from matsynth.resolve import (fetch_oqmd_hypothetical, resolve_by_mp_id,
                              resolve_structure)
from matsynth.utils import save_json

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
SET_PATH = PROCESSED / "argyrodite_set.json"
EMB_PATH = PROCESSED / "embeddings_argyrodite.pt"


def assemble_set() -> dict[str, dict]:
    records, _ = load_records(validate_schema=False)
    known_synthesized = {r.target_formula for r in records}

    items: dict[str, dict] = {}

    def add(rs, label: int, source: str, precursors=None) -> None:
        formula = rs.structure.composition.reduced_formula
        mid = rs.material_id
        if mid in items:
            return
        if any(v["formula"] == formula for v in items.values()):
            return  # satu struktur per formula supaya tidak dobel label
        items[mid] = {
            "structure": rs.structure.as_dict(),
            "formula": formula,
            "family": family_key(formula),
            "label": label,
            "source": source,
            "precursors": sorted(precursors) if precursors else None,
            "provider": rs.provider,
            "meta": rs.meta,
        }

    # 1) Positif Kononova (dengan precursor set = union resep per formula terbanyak;
    #    kalau ada >1 resep, ambil yang paling sering muncul, deterministik).
    argy_records = [r for r in records if is_argyrodite(r.target_formula)]
    by_formula: dict[str, list] = {}
    for r in argy_records:
        by_formula.setdefault(r.target_formula, []).append(r)
    for formula, recs in sorted(by_formula.items()):
        counts = Counter(r.precursor_formulas for r in recs)
        best_prec = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rs = None
        for r in recs:
            if r.target_mp_id:
                rs = resolve_by_mp_id(r.target_mp_id)
                if rs is not None:
                    break
        if rs is None:
            rs = resolve_structure(formula)
        if rs is None:
            print(f"  [drop] positif Kononova tak ter-resolve: {formula}")
            continue
        add(rs, label=1, source="kononova", precursors=best_prec)

    # 2) Positif kurasi literatur.
    for f in CURATED_SYNTHESIZED_ARGYRODITES:
        formula = Composition(f).reduced_formula
        assert is_argyrodite(formula), f"kurasi bukan pola argyrodite: {f}"
        if any(v["formula"] == formula for v in items.values()):
            continue
        rs = resolve_structure(formula)
        if rs is None:
            print(f"  [skip] kurasi tak ter-resolve di MP/OQMD: {formula}")
            continue
        add(rs, label=1, source="curated")

    # 3) Negatif: OQMD hipotetis stabil-tapi-tak-tersintesis.
    exclude = known_synthesized | {Composition(f).reduced_formula
                                   for f in CURATED_SYNTHESIZED_ARGYRODITES}
    negatives = fetch_oqmd_hypothetical(
        argyrodite_candidate_formulas(), exclude_formulas=exclude,
        stability_max=0.1)
    for rs in negatives:
        add(rs, label=0, source="oqmd_hypothetical")

    assert items, "set argyrodite kosong"
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concat-last-two", action="store_true",
                    help="concat fitur invariant 2 interaction layer terakhir")
    ap.add_argument("--model-size", default="small", choices=["small", "medium"])
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    if SET_PATH.exists():
        print(f"[set] pakai set tersimpan: {SET_PATH}")
        items = json.loads(SET_PATH.read_text(encoding="utf-8"))
    else:
        items = assemble_set()
        SET_PATH.write_text(json.dumps(items), encoding="utf-8")
        print(f"[set] disimpan: {SET_PATH}")

    per_family = Counter((v["family"], v["label"]) for v in items.values())
    print(f"total item: {len(items)}")
    for (fam, lab), n in sorted(per_family.items()):
        print(f"  family={fam:<8} label={lab} n={n}")

    embedder = MaceEmbedder(model_size=args.model_size, device="cpu",
                            concat_last_two=args.concat_last_two)
    structures = {mid: Structure.from_dict(v["structure"]) for mid, v in items.items()}
    meta = {mid: {k: v[k] for k in ("formula", "family", "label", "source", "precursors")}
            for mid, v in items.items()}
    out = precompute_embeddings(structures, EMB_PATH, embedder, extra_meta=meta)

    d = next(iter(out.values()))["z"].shape[0]
    save_json("step2_featurize.json", {
        "n_items": len(items),
        "embedding_dim": d,
        "concat_last_two": args.concat_last_two,
        "model_size": args.model_size,
        "per_family_label_counts": {f"{fam}|label={lab}": n
                                    for (fam, lab), n in sorted(per_family.items())},
        "items": {mid: {k: v[k] for k in ("formula", "family", "label", "source")}
                  for mid, v in items.items()},
        "embeddings_path": str(EMB_PATH),
    })


if __name__ == "__main__":
    main()
