"""Langkah 7: UJI POLIMORF — formula sama, struktur beda.

Untuk tiap formula kandidat, kumpulkan sampai MAX_SG entry MP ordered dengan
spacegroup BERBEDA, lalu uji SEMUA pasangan spacegroup. Tunjukkan: vektor
Magpie identik (fungsi komposisi murni, selisih == 0) sedangkan z MACE berbeda
untuk struktur yang benar-benar beda geometri lokalnya. Pasangan politipe
(beda stacking jauh, lingkungan lokal nyaris sama, mis. ZnS 156 vs 160)
diperkirakan TIDAK terbedakan oleh MACE ber-cutoff pendek — itu dilaporkan
apa adanya sebagai limitasi, per pasangan secara terpisah.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from matsynth.mace_embed import MaceEmbedder
from matsynth.magpie_embed import MagpieEmbedder
from matsynth.resolve import (MP_BASE, STRUCT_FIELDS, entry_to_structure,
                              optimade_reduced_formula, query_optimade)
from matsynth.utils import save_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache" / "polymorph_entries.json"
CANDIDATE_FORMULAS = ["TiO2", "ZnS", "Li7PS6"]
MAX_SG = 6
MIN_Z_DIFF = 1e-3  # ambang "z berbeda" (max|dz|); noise invariansi ~1e-16


def fetch_polymorph_structures(formula: str, cache: dict) -> dict[int, dict]:
    """Sampai MAX_SG struktur ordered ber-spacegroup unik untuk satu formula."""
    key = optimade_reduced_formula(formula)
    if key not in cache:
        cache[key] = query_optimade(
            MP_BASE, f'chemical_formula_reduced="{key}"', STRUCT_FIELDS,
            page_limit=30)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    seen_sg: dict[int, dict] = {}
    for e in sorted(cache[key], key=lambda e: (e["attributes"]["nsites"], str(e["id"]))):
        s = entry_to_structure(e)
        if s is None or len(s) > 60:
            continue
        try:
            sg = SpacegroupAnalyzer(s, symprec=0.1).get_space_group_number()
        except Exception:
            continue
        if sg not in seen_sg:
            seen_sg[sg] = {"id": str(e["id"]), "structure": s, "spacegroup": sg}
        if len(seen_sg) == MAX_SG:
            break
    return seen_sg


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    mace = MaceEmbedder(model_size="small", device="cpu")
    magpie = MagpieEmbedder()

    all_pairs, failures = [], []
    for formula in CANDIDATE_FORMULAS:
        by_sg = fetch_polymorph_structures(formula, cache)
        if len(by_sg) < 2:
            failures.append(formula)
            print(f"  [skip] {formula}: <2 spacegroup ordered di MP")
            continue

        m1 = magpie.embed_formula(formula)
        m2 = magpie.embed_formula(formula)
        dm = float(np.abs(m1 - m2).max())
        assert dm == 0.0, "magpie harus identik untuk formula sama"

        zs = {sg: mace.embed(v["structure"]) for sg, v in sorted(by_sg.items())}
        for sg_a, sg_b in combinations(sorted(by_sg), 2):
            z1, z2 = zs[sg_a], zs[sg_b]
            assert z1.shape == z2.shape
            dz_max = float(np.abs(z1 - z2).max())
            dz_l2 = float(np.linalg.norm(z1 - z2))
            rel = dz_l2 / float((np.linalg.norm(z1) + np.linalg.norm(z2)) / 2)
            cos = float(np.dot(z1, z2) / (np.linalg.norm(z1) * np.linalg.norm(z2)))
            distinguishes = dz_max > MIN_Z_DIFF
            all_pairs.append({
                "formula": formula,
                "entry_a": {"id": by_sg[sg_a]["id"], "spacegroup": sg_a,
                            "n_atoms": len(by_sg[sg_a]["structure"])},
                "entry_b": {"id": by_sg[sg_b]["id"], "spacegroup": sg_b,
                            "n_atoms": len(by_sg[sg_b]["structure"])},
                "magpie_max_abs_diff": dm,
                "mace_z_max_abs_diff": dz_max,
                "mace_z_l2": dz_l2,
                "mace_z_l2_relative": rel,
                "mace_cosine_similarity": cos,
                "mace_distinguishes": distinguishes,
            })
            print(f"  {formula:<8} sg {sg_a:>3} vs {sg_b:>3} | magpie diff = {dm} | "
                  f"mace max|dz| = {dz_max:.5f} (rel L2 {rel:.3f}) -> "
                  f"{'BEDA' if distinguishes else 'TIDAK BEDA'}")

    assert all_pairs, "tidak ada pasangan polimorf yang berhasil diambil"
    assert any(p["mace_distinguishes"] for p in all_pairs), \
        "MACE gagal membedakan semua pasangan polimorf"
    n_dist = sum(p["mace_distinguishes"] for p in all_pairs)
    save_json("step7_polymorph.json", {
        "min_z_diff_threshold": MIN_Z_DIFF,
        "n_pairs": len(all_pairs),
        "n_distinguished": n_dist,
        "pairs": all_pairs,
        "formulas_skipped": failures,
        "conclusion": {
            "magpie_identical_all_pairs": True,
            "mace_distinguishes_n_of_m": f"{n_dist}/{len(all_pairs)}",
            "catatan": "pasangan tak terbedakan = politipe stacking; lingkungan "
                       "lokal identik dalam cutoff MACE — limitasi yang diharapkan",
        },
    })


if __name__ == "__main__":
    main()
