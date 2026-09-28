"""Langkah 6: baseline komposisi Magpie (matminer) — floor yang harus dikalahkan z MACE.

Vektor Magpie murni fungsi komposisi: dua polimorf beda struktur dengan formula
sama PASTI memberi vektor identik. Itu yang diuji di langkah 7.
"""

import numpy as np
from pymatgen.core import Composition


class MagpieEmbedder:
    def __init__(self):
        from matminer.featurizers.composition import ElementProperty
        self.fe = ElementProperty.from_preset("magpie")
        self._dim = len(self.fe.feature_labels())

    @property
    def dim(self) -> int:
        return self._dim

    def embed_formula(self, formula: str) -> np.ndarray:
        comp = Composition(formula)
        v = np.asarray(self.fe.featurize(comp), dtype=np.float64)
        assert v.ndim == 1 and v.shape[0] == self._dim, f"magpie shape {v.shape}"
        assert np.isfinite(v).all(), f"magpie NaN/inf untuk {formula}"
        return v
