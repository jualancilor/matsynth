"""Definisi material family, deteksi argyrodite, split by family + assert leakage.

Family key = anonymized reduced formula pymatgen (mis. Li6PS5Cl -> "ABC5D6").
Split train/test dilakukan pada level family: seluruh anggota satu family masuk
split yang sama, sehingga test set tidak punya "tetangga dekat" komposisional
di train. assert_no_leakage memverifikasi family DAN reduced formula disjoint.
"""

import random

from pymatgen.core import Composition

CATIONS = {"Li", "Na", "Cu", "Ag"}
CHALCOGENS = {"O", "S", "Se", "Te"}
PNICTOGENS = {"P", "As", "Sb"}
TETRELS = {"Si", "Ge", "Sn"}
HALOGENS = {"F", "Cl", "Br", "I"}


def family_key(formula: str) -> str:
    return Composition(formula).reduced_composition.anonymized_formula


def _int_amounts(formula: str) -> dict[str, int] | None:
    comp = Composition(formula)
    int_formula, _ = comp.get_integer_formula_and_factor()
    icomp = Composition(int_formula)
    out = {}
    for el, amt in icomp.items():
        n = int(round(amt))
        if abs(amt - n) > 1e-6:
            return None
        out[el.symbol] = n
    return out


def is_argyrodite(formula: str) -> bool:
    """Argyrodite: A6BX5Y (Li6PS5Cl), A7BX6 (Cu7PS6), A8BX6 (Ag8GeS6).

    Cek berbasis peran kimia, bukan cuma pola anonim, supaya mis. oksida
    ABC5D6 acak tidak ikut terjaring.
    """
    amts = _int_amounts(formula)
    if amts is None:
        return False
    by_n: dict[int, list[str]] = {}
    for el, n in amts.items():
        by_n.setdefault(n, []).append(el)

    counts = sorted(amts.values())
    if counts == [1, 1, 5, 6]:  # A6 B X5 Y
        ones = by_n[1]
        return (len(by_n.get(6, [])) == 1 and by_n[6][0] in CATIONS
                and len(by_n.get(5, [])) == 1 and by_n[5][0] in CHALCOGENS
                and len(ones) == 2
                and any(e in PNICTOGENS for e in ones)
                and any(e in HALOGENS for e in ones))
    if counts == [1, 6, 7]:  # A7 B X6
        return (by_n[7][0] in CATIONS and by_n[6][0] in CHALCOGENS
                and by_n[1][0] in PNICTOGENS)
    if counts == [1, 6, 8]:  # A8 B X6
        return (by_n[8][0] in CATIONS and by_n[6][0] in CHALCOGENS
                and by_n[1][0] in TETRELS)
    return False


def split_by_family(formulas: list[str], test_frac: float = 0.35, seed: int = 42
                    ) -> tuple[set[str], set[str]]:
    """Return (train_families, test_families) — deterministik, level family."""
    fams = sorted({family_key(f) for f in formulas})
    assert len(fams) >= 2, f"butuh >=2 family untuk split, dapat {fams}"
    rng = random.Random(seed)
    rng.shuffle(fams)
    n_test = max(1, round(len(fams) * test_frac))
    n_test = min(n_test, len(fams) - 1)  # train tidak boleh kosong
    test = set(fams[:n_test])
    train = set(fams[n_test:])
    return train, test


def assert_no_leakage(train_formulas: list[str], test_formulas: list[str]) -> None:
    train_fams = {family_key(f) for f in train_formulas}
    test_fams = {family_key(f) for f in test_formulas}
    overlap_fam = train_fams & test_fams
    assert not overlap_fam, f"LEAKAGE: family di train dan test: {overlap_fam}"
    train_set = {Composition(f).reduced_formula for f in train_formulas}
    test_set = {Composition(f).reduced_formula for f in test_formulas}
    overlap_f = train_set & test_set
    assert not overlap_f, f"LEAKAGE: formula di train dan test: {overlap_f}"


# Argyrodite tersintesis yang mapan di literatur (era <= 2019, konsisten dengan
# cakupan Kononova). Dipakai untuk menambal positif karena korpus Kononova 2019
# hampir tidak memuat resep argyrodite (hanya 3 record). Rujukan utama:
# Deiseroth 2008 (Li6PS5X), Kong/Deiseroth 2010 (seri Se), mineral
# argyrodite/canfieldite (Ag8GeS6, Ag8SnS6) dan analognya.
CURATED_SYNTHESIZED_ARGYRODITES = [
    "Li6PS5Cl", "Li6PS5Br", "Li6PS5I",          # Deiseroth 2008
    "Li6PSe5Cl", "Li6PSe5Br", "Li6PSe5I",       # Kong 2010
    "Li7PS6", "Li7PSe6",                        # Kong/Deiseroth
    "Ag8GeS6", "Ag8SnS6", "Ag8SiS6",            # mineral + analog
    "Ag8GeSe6", "Ag8SnSe6", "Ag8GeTe6", "Ag8SiSe6",
    "Cu7PS6", "Ag7PS6",                         # argyrodite tembaga/perak-fosfor
    "Cu8GeS6", "Cu8GeSe6", "Cu8SiS6", "Cu8SiSe6",
]


def argyrodite_candidate_formulas() -> list[str]:
    """Grid substitusi charge-balanced untuk mining negatif OQMD."""
    out = []
    for a in sorted(CATIONS):
        for x in sorted(CHALCOGENS):
            for b in sorted(PNICTOGENS):
                for y in sorted(HALOGENS):
                    out.append(f"{a}6{b}{x}5{y}")   # A+6 B5+ X2-5 Y-1
                out.append(f"{a}7{b}{x}6")          # A+7 B5+ X2-6
            for b in sorted(TETRELS):
                out.append(f"{a}8{b}{x}6")          # A+8 B4+ X2-6
    assert len(out) == len(set(out))
    return out
