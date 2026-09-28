"""Langkah 1: loader dataset Kononova (Sci. Data 2019) + normalisasi.

Tiap record mentah di-assert terhadap skema bertipe (schema.py), lalu dinormalisasi
menjadi SynthesisRecord (target reduced formula + precursor set). Record yang tidak
bisa dinormalisasi dibuang dengan alasan tercatat (counter), bukan diam-diam.
"""

import json
from collections import Counter
from pathlib import Path

from pymatgen.core import Composition, Element

from .schema import SynthesisRecord, assert_record_schema

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "solid-state_dataset_2019-06-27_upd.json"


def _parse_amount(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _composition_from_material(mat: dict) -> Composition | None:
    """Material dict Kononova -> pymatgen Composition, atau None kalau tak layak.

    Syarat: tepat satu composition (bukan campuran/komposit), tanpa variabel
    amount/elemen yang belum tersubstitusi, semua simbol elemen valid, semua
    amount numerik positif.
    """
    if mat.get("elements_vars"):
        return None
    comps = mat.get("composition", [])
    if len(comps) != 1:
        return None
    c = comps[0]
    amt_outer = _parse_amount(c.get("amount", "1"))
    if amt_outer is None:
        return None
    el_amt: dict[str, float] = {}
    for el, amt_s in c["elements"].items():
        amt = _parse_amount(amt_s)
        if amt is None:
            return None
        try:
            Element(el)
        except (ValueError, KeyError):
            return None
        el_amt[el] = el_amt.get(el, 0.0) + amt * amt_outer
    if not el_amt:
        return None
    return Composition(el_amt)


def load_raw(path: Path = RAW_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    # File figshare membungkus list reaksi dalam dict {"reactions": [...], ...}
    # atau berupa list langsung — dukung keduanya, assert hasil akhirnya list.
    if isinstance(payload, dict):
        recs = payload.get("reactions", payload.get("data"))
    else:
        recs = payload
    assert isinstance(recs, list) and recs, "dataset harus list record yang tidak kosong"
    return recs


def load_records(path: Path = RAW_PATH, validate_schema: bool = True
                 ) -> tuple[list[SynthesisRecord], Counter]:
    """Load + assert skema + normalisasi. Return (records, drop_counter)."""
    raw = load_raw(path)
    drops: Counter = Counter()
    out: list[SynthesisRecord] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for i, rec in enumerate(raw):
        if validate_schema:
            assert_record_schema(rec, i)

        # File 2019 tidak punya field synthesis_type (semuanya solid-state).
        synthesis_type = rec.get("synthesis_type", "solid-state")
        if synthesis_type != "solid-state":
            drops["bukan_solid_state"] += 1
            continue

        tgt_comp = _composition_from_material(rec["target"])
        if tgt_comp is None:
            drops["target_tak_ternormalisasi"] += 1
            continue
        if len(tgt_comp) < 2:
            drops["target_unsur_tunggal"] += 1
            continue

        prec_formulas: list[str] = []
        ok = True
        for p in rec["precursors"]:
            pc = _composition_from_material(p)
            if pc is None:
                ok = False
                break
            prec_formulas.append(pc.reduced_formula)
        if not ok or not prec_formulas:
            drops["precursor_tak_ternormalisasi"] += 1
            continue

        sr = SynthesisRecord(
            doi=rec["doi"],
            target_formula=tgt_comp.reduced_formula,
            precursor_formulas=tuple(sorted(set(prec_formulas))),
            synthesis_type=synthesis_type,
            target_mp_id=rec["target"].get("mp_id"),
        )
        key = (sr.target_formula, sr.precursor_formulas)
        if key in seen:
            drops["duplikat_target_precursor"] += 1
            continue
        seen.add(key)
        out.append(sr)

    assert out, "tidak ada record valid setelah normalisasi"
    return out, drops
