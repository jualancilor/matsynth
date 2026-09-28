"""Langkah 3 (GATE): uji invariansi rotasi + translasi.

10 struktur diambil dari set argyrodite, dirotasi random (SO(3)) + translasi
fraksional random (wrap PBC), lalu embedding dihitung ulang. Gate:
max |z - z_rot| < 1e-4 untuk SEMUA struktur. Kalau gagal -> exit code 2 dan
laporan; pipeline TIDAK boleh lanjut.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymatgen.core import Lattice, Structure
from scipy.spatial.transform import Rotation

from matsynth.mace_embed import MaceEmbedder
from matsynth.utils import save_json

SET_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "argyrodite_set.json"
TOL = 1e-4
N_STRUCT = 10
SEED = 7


def rotate_translate(s: Structure, rng: np.random.Generator) -> Structure:
    """Rotasi rigid seluruh kristal + translasi fraksional (wrap)."""
    R = Rotation.random(random_state=int(rng.integers(2**31 - 1))).as_matrix()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10) and np.linalg.det(R) > 0
    L_rot = s.lattice.matrix @ R.T          # baris a,b,c dirotasi
    t = rng.random(3)
    new_frac = (s.frac_coords + t) % 1.0
    s2 = Structure(Lattice(L_rot), s.species, new_frac, coords_are_cartesian=False)
    assert len(s2) == len(s)
    return s2


def main() -> None:
    items = json.loads(SET_PATH.read_text(encoding="utf-8"))
    mids = sorted(items.keys())
    rng = np.random.default_rng(SEED)
    chosen = list(rng.choice(mids, size=min(N_STRUCT, len(mids)), replace=False))

    embedder = MaceEmbedder(model_size="small", device="cpu")
    rows, worst = [], 0.0
    for mid in chosen:
        s = Structure.from_dict(items[mid]["structure"])
        z = embedder.embed(s)
        z_rot = embedder.embed(rotate_translate(s, rng))
        assert z.shape == z_rot.shape
        diff = float(np.abs(z - z_rot).max())
        worst = max(worst, diff)
        ok = diff < TOL
        rows.append({"material_id": mid, "formula": items[mid]["formula"],
                     "n_atoms": len(s), "max_abs_diff": diff, "pass": ok})
        print(f"  {items[mid]['formula']:<12} max|z-z_rot| = {diff:.3e}  "
              f"{'OK' if ok else 'GAGAL'}")

    passed = all(r["pass"] for r in rows)
    save_json("step3_invariance.json", {
        "tolerance": TOL, "n_structures": len(rows),
        "worst_max_abs_diff": worst, "all_pass": passed, "per_structure": rows,
    })
    if not passed:
        print(f"\nGATE GAGAL: max|z - z_rot| = {worst:.3e} >= {TOL}. "
              "Embedding TIDAK rotation-invariant — pipeline dihentikan.")
        sys.exit(2)
    print(f"\nGATE LOLOS: worst max|z - z_rot| = {worst:.3e} < {TOL} "
          f"pada {len(rows)} struktur.")


if __name__ == "__main__":
    main()
