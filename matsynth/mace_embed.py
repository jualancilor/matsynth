"""Tahap A: embedding graph-level invariant dari MACE pretrained (frozen).

Ekstraksi: fitur node invariant (l=0) per layer via calc.get_descriptors
(invariants_only=True) yang layout-nya [layer0_inv | layer1_inv | ...] per atom
(mace.modules.utils.extract_invariant). "Interaction block terakhir" = blok k
kolom terakhir; flag concat_last_two mengambil 2 blok terakhir.
Pool: mean over atoms -> z. Rotation-invariant karena hanya kanal skalar l=0.

MACE tidak pernah dipanggil di training loop — hasil disimpan ke .pt (Tahap B
hanya membaca file itu).
"""

from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


class MaceEmbedder:
    def __init__(self, model_size: str = "small", device: str = "cpu",
                 concat_last_two: bool = False):
        from mace.calculators import mace_mp
        self.calc = mace_mp(model=model_size, device=device,
                            default_dtype="float64", dispersion=False)
        model = self.calc.models[0]
        model.eval()
        for p in model.parameters():  # frozen — tidak di-fine-tune
            p.requires_grad_(False)
        self.num_layers = int(model.num_interactions)
        assert self.num_layers >= 2, f"butuh >=2 interaction layers, dapat {self.num_layers}"
        self.concat_last_two = concat_last_two

    def per_atom_invariants(self, structure: Structure) -> np.ndarray:
        assert structure.is_ordered, f"struktur disordered: {structure.composition}"
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Tanpa torch.no_grad(): forward MACE menghitung forces via autograd
        # terhadap posisi (bukan parameter — parameter tetap frozen).
        d = self.calc.get_descriptors(atoms, invariants_only=True, num_layers=-1)
        d = np.asarray(d)
        assert d.ndim == 2, f"descriptor harus 2D, dapat shape {d.shape}"
        assert d.shape[0] == len(structure), \
            f"baris descriptor {d.shape[0]} != jumlah atom {len(structure)}"
        assert d.shape[1] % self.num_layers == 0, \
            f"lebar {d.shape[1]} tidak habis dibagi {self.num_layers} layer"
        k = d.shape[1] // self.num_layers  # fitur invariant per layer
        take = 2 if self.concat_last_two else 1
        out = d[:, -take * k:]
        assert out.shape == (len(structure), take * k)
        assert np.isfinite(out).all(), "descriptor mengandung NaN/inf"
        return out

    @property
    def dim(self) -> int:
        raise AttributeError("dim diketahui setelah embed pertama")

    def embed(self, structure: Structure) -> np.ndarray:
        """Struktur -> z [d], mean pool over atoms, float64."""
        per_atom = self.per_atom_invariants(structure)
        z = per_atom.mean(axis=0)
        assert z.ndim == 1 and z.shape[0] == per_atom.shape[1]
        assert np.isfinite(z).all()
        return z


def precompute_embeddings(items: dict[str, Structure], out_path: Path,
                          embedder: MaceEmbedder,
                          extra_meta: dict[str, dict] | None = None) -> dict:
    """{material_id: Structure} -> simpan {material_id: {z, ...meta}} ke .pt."""
    out: dict[str, dict] = {}
    d_ref = None
    for mid, s in items.items():
        z = embedder.embed(s)
        if d_ref is None:
            d_ref = z.shape[0]
        assert z.shape[0] == d_ref, f"{mid}: dim {z.shape[0]} != {d_ref}"
        entry = {"z": torch.from_numpy(z.copy())}
        if extra_meta and mid in extra_meta:
            entry.update(extra_meta[mid])
        out[mid] = entry
        print(f"  [embed] {mid}: {s.composition.reduced_formula} "
              f"n_atoms={len(s)} d={z.shape[0]}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return out
