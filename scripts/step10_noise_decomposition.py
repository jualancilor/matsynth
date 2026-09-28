"""Langkah 10: bongkar isi error oracle 117 °C — polimorf, rute, atau noise?

Pertanyaan asli: apakah sebaran suhu dalam formula yang sama nyambung dengan
POLIMORF yang dibuat, atau acak antar-lab?

Temuan ketersediaan label (bagian A): korpus Kononova praktis tidak menyimpan
identitas polimorf per resep — file 2019 tanpa field phase (0 record), mp_id
di-assign per formula (tidak pernah bervariasi dalam satu formula), string
target hampir tanpa marker polimorf; file 2020 punya field phase tapi terisi
di 264/31.782 record → hanya 1 formula yang memenuhi syarat uji (>=2 fasa x
>=2 resep). Pertanyaan polimorf TIDAK TERJAWAB dari korpus ini.

Yang bisa diukur (bagian B, file 2019, konsisten langkah 9): dekomposisi
sebaran within-formula dengan proksi rute:
  B1  LOO lintas-paper, kondisi (formula)            — angka ~117 °C versi lintas-lab
  B2  LOO lintas-paper, kondisi (formula, precursor) — rute difiksasi, lab tetap beda
  B3  LOO dalam-paper,  kondisi (formula, doi)       — floor reprodusibilitas intra-paper
B1 vs B2 dibandingkan BERPASANGAN pada subset resep yang evaluable di keduanya.
Kalau B2 << B1: sebaran itu route-systematic -> ceiling 117 °C LUNAK (artefak
input formula-saja), dan pintu untuk model berinput rute/struktur masih terbuka.
Kalau B2 ~ B1: sebaran tak dijelaskan rute -> noise beneran.
"""

import json
import lzma
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import wilcoxon

from matsynth.kononova import _composition_from_material, load_raw
from matsynth.utils import save_json
from step9_temperature_triviality import extract_temperature

ROOT = Path(__file__).resolve().parents[1]
PATH_2020 = ROOT / "data" / "raw" / "solid-state_dataset_20200713.json.xz"
N_BOOT = 20000


def load_rows_2019() -> list[dict]:
    rows = []
    for rec in load_raw():
        t = extract_temperature(rec)
        if t is None:
            continue
        comp = _composition_from_material(rec["target"])
        if comp is None or len(comp) < 2:
            continue
        precs = []
        ok = True
        for p in rec["precursors"]:
            pc = _composition_from_material(p)
            if pc is None:
                ok = False
                break
            precs.append(pc.reduced_formula)
        rows.append({
            "formula": comp.reduced_formula, "t": float(t), "doi": rec["doi"],
            "route": frozenset(precs) if ok and precs else None,
        })
    assert len(rows) > 1000
    return rows


# ------------------------------------------------- bagian A: label polimorf
def phase_label_availability() -> dict:
    with lzma.open(PATH_2020, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    recs = payload["reactions"] if isinstance(payload, dict) else payload
    n_phase = sum(1 for r in recs if (r.get("target") or {}).get("phase"))

    by_f: dict[str, list] = defaultdict(list)
    n_temp = 0
    for rec in recs:
        t = extract_temperature(rec)
        if t is None:
            continue
        comp = _composition_from_material(rec["target"])
        if comp is None or len(comp) < 2:
            continue
        n_temp += 1
        ph = (rec["target"].get("phase") or "").strip().lower() or None
        by_f[comp.reduced_formula].append((float(t), ph))

    usable = {}
    for f, v in by_f.items():
        if len(v) < 2:
            continue
        groups = defaultdict(list)
        for t, ph in v:
            if ph:
                groups[ph].append(t)
        big = {p: ts for p, ts in groups.items() if len(ts) >= 2}
        if len(big) >= 2:
            usable[f] = {p: {"n": len(ts), "median_c": float(np.median(ts))}
                         for p, ts in big.items()}

    out = {
        "korpus_2019": "field phase tidak ada; mp_id konstan per formula; "
                       "string target tanpa marker polimorf yang usable (0 formula)",
        "korpus_2020_n_records": len(recs),
        "korpus_2020_n_phase_filled": n_phase,
        "korpus_2020_n_recipes_with_temp": n_temp,
        "formulas_usable_for_polymorph_test": usable,
        "n_formulas_usable": len(usable),
        "conclusion": "pertanyaan 'sebaran suhu nyambung polimorf atau tidak' "
                      "TIDAK TERJAWAB dari korpus text-mined ini — label polimorf "
                      "praktis tidak pernah terekstrak. Butuh data lab.",
    }
    print("=" * 68)
    print("BAGIAN A — ketersediaan label polimorf")
    print(f"  2019: phase tidak ada; mp_id tak bervariasi; marker teks ~0")
    print(f"  2020: phase terisi {n_phase}/{len(recs)} record")
    print(f"  formula usable utk uji polimorf: {len(usable)}")
    for f, d in usable.items():
        parts = "; ".join(f"{p}: med {v['median_c']:.0f} °C (n={v['n']})"
                          for p, v in sorted(d.items()))
        print(f"    {f}: {parts}   <- anekdot, bukan uji")
    return out


# ------------------------------------------------- bagian B: dekomposisi rute
def loo_cross_paper(rows: list[dict], key_fn) -> dict[int, float]:
    """Prediksi tiap resep = median resep lain SE-GRUP dari PAPER BERBEDA."""
    groups: dict[object, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        k = key_fn(r)
        if k is not None:
            groups[k].append(i)
    preds = {}
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [rows[j]["t"] for j in idx
                      if j != i and rows[j]["doi"] != rows[i]["doi"]]
            if others:
                preds[i] = float(np.median(others))
    return preds


def paired_stats(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    d = err_b - err_a  # negatif = B (rute) lebih baik
    nz = d[np.abs(d) > 1e-9]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(N_BOOT)])
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"mean_diff": float(d.mean()), "wilcoxon_p": p,
            "bootstrap_ci95": [lo, hi], "significant": bool(hi < 0)}


def route_decomposition(rows: list[dict]) -> dict:
    pred_f = loo_cross_paper(rows, lambda r: r["formula"])
    pred_fr = loo_cross_paper(
        rows, lambda r: (r["formula"], r["route"]) if r["route"] else None)

    common = sorted(set(pred_f) & set(pred_fr))
    assert common, "tidak ada resep yang evaluable di kedua kondisi"
    err_f = np.array([abs(rows[i]["t"] - pred_f[i]) for i in common])
    err_fr = np.array([abs(rows[i]["t"] - pred_fr[i]) for i in common])
    stats = paired_stats(err_f, err_fr)
    reduction = 100.0 * (err_f.mean() - err_fr.mean()) / err_f.mean()

    # B3: floor reprodusibilitas dalam paper yang sama
    groups_fd: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups_fd[(r["formula"], r["doi"])].append(i)
    intra = []
    for idx in groups_fd.values():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [rows[j]["t"] for j in idx if j != i]
            intra.append(abs(rows[i]["t"] - float(np.median(others))))
    intra_mae = float(np.mean(intra))

    out = {
        "n_recipes_total": len(rows),
        "n_paired_recipes_B1_B2": len(common),
        "B1_formula_cross_paper_mae": float(err_f.mean()),
        "B2_formula_route_cross_paper_mae": float(err_fr.mean()),
        "route_error_reduction_percent": float(reduction),
        "paired_stats_B2_vs_B1": stats,
        "B3_within_paper_mae": intra_mae,
        "B3_n": len(intra),
    }
    print()
    print("=" * 68)
    print("BAGIAN B — dekomposisi sebaran within-formula (2019, lintas-paper)")
    print(f"  n berpasangan = {len(common)}")
    print(f"  B1 kondisi (formula)            MAE = {err_f.mean():6.1f} °C")
    print(f"  B2 kondisi (formula, precursor) MAE = {err_fr.mean():6.1f} °C")
    print(f"     pengurangan {reduction:.1f}%  Wilcoxon p = {stats['wilcoxon_p']:.2g}"
          f"  CI95 [{stats['bootstrap_ci95'][0]:+.1f}, "
          f"{stats['bootstrap_ci95'][1]:+.1f}]"
          f"  -> {'SIGNIFIKAN' if stats['significant'] else 'TIDAK signifikan'}")
    print(f"  B3 dalam-paper (formula, doi)   MAE = {intra_mae:6.1f} °C "
          f"(n={len(intra)}; floor reprodusibilitas)")
    return out


def main() -> None:
    part_a = phase_label_availability()
    rows = load_rows_2019()
    part_b = route_decomposition(rows)

    sig = part_b["paired_stats_B2_vs_B1"]["significant"]
    red = part_b["route_error_reduction_percent"]
    if sig and red > 15:
        verdict = (
            "Sebaran within-formula sebagian besar SISTEMATIK-RUTE, bukan noise "
            "acak: memfiksasi precursor memangkas error lintas-lab "
            f"{red:.0f}%. Ceiling 117 °C LUNAK — artefak input formula-saja. "
            "Atribusi spesifik ke polimorf tetap tak terjawab (label tidak ada "
            "di korpus); jawaban finalnya butuh data lab."
        )
    elif sig:
        verdict = (
            f"Rute menjelaskan sebagian kecil sebaran ({red:.0f}%, signifikan). "
            "Mayoritas sisa adalah noise antar-lab yang tak terjelaskan input "
            "yang tersedia; ruang struktur di bawah 117 °C kecil."
        )
    else:
        verdict = (
            "Memfiksasi rute TIDAK memangkas error lintas-lab: sebaran "
            "within-formula adalah noise beneran. Jatah struktur hanya ~50 °C "
            "(167->117) dan matsynth-suhu mati secara empiris. Hasil negatif "
            "bersih, tetap layak dilaporkan."
        )
    print()
    print("=" * 68)
    print("VONIS:", verdict)

    save_json("step10_noise_decomposition.json", {
        "question": "error oracle 117 °C isinya apa: polimorf, rute, atau noise?",
        "part_a_polymorph_label_availability": part_a,
        "part_b_route_decomposition": part_b,
        "verdict": verdict,
        "catatan_pembanding": (
            "Angka MAE di sini tidak sebanding dengan langkah 9 (target & subset "
            "beda: LOO lintas-paper pada resep multi-recipe, bukan GroupKFold "
            "seluruh korpus). Jangan disandingkan lintas-eksperimen."
        ),
    })


if __name__ == "__main__":
    main()
