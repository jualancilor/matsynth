"""Langkah 1: loader Kononova + assert skema bertipe + resolusi struktur.

Memvalidasi skema SEMUA record, menormalisasi, memfilter keluarga argyrodite,
lalu me-resolve struktur target argyrodite (mp_id bawaan dataset dulu, fallback
pencarian formula MP -> OQMD). Resolusi korpus penuh (~7k formula unik) ditunda —
scope tahap ini hanya butuh keluarga argyrodite (langkah 2-5).
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matsynth.families import family_key, is_argyrodite
from matsynth.kononova import load_records
from matsynth.resolve import resolve_by_mp_id, resolve_structure
from matsynth.utils import save_json


def main() -> None:
    records, drops = load_records(validate_schema=True)
    print(f"record valid ternormalisasi : {len(records)}")
    print(f"drop: {dict(drops)}")

    argy = [r for r in records if is_argyrodite(r.target_formula)]
    argy_formulas = sorted({r.target_formula for r in argy})
    print(f"\nrecord argyrodite: {len(argy)} | formula unik: {len(argy_formulas)}")
    for f in argy_formulas:
        recs = [r for r in argy if r.target_formula == f]
        mp_ids = sorted({r.target_mp_id for r in recs if r.target_mp_id})
        print(f"  {f:<12} family={family_key(f):<8} n_resep={len(recs)} mp_id={mp_ids}")

    resolved, missed = {}, []
    for f in argy_formulas:
        rs = None
        for r in argy:
            if r.target_formula == f and r.target_mp_id:
                rs = resolve_by_mp_id(r.target_mp_id)
                if rs is not None:
                    break
        if rs is None:
            rs = resolve_structure(f)
        if rs is None:
            missed.append(f)
            print(f"  [miss] {f}")
        else:
            resolved[f] = rs
            print(f"  [ok]   {f} -> {rs.material_id} n_sites={len(rs.structure)}")

    save_json("step1_load.json", {
        "n_raw": len(records) + sum(drops.values()),
        "n_valid_normalized": len(records),
        "drops": dict(drops),
        "n_argyrodite_records": len(argy),
        "argyrodite_formulas": argy_formulas,
        "argyrodite_families": sorted({family_key(f) for f in argy_formulas}),
        "resolved": {f: {"material_id": rs.material_id, "n_sites": len(rs.structure),
                         "provider": rs.provider}
                     for f, rs in resolved.items()},
        "unresolved_dropped": missed,
        "note": "resolusi korpus penuh ditunda; scope tahap ini = keluarga argyrodite",
    })
    assert resolved, "tidak ada struktur argyrodite yang ter-resolve"


if __name__ == "__main__":
    main()
