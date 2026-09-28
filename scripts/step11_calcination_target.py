"""Langkah 11: ulangi eksperimen suhu dgn target KALSINASI, bukan maksimum.

Hipotesis yang diuji (lanjutan langkah 10): "noise" 117-120 °C bukan noise,
melainkan kerusakan definisi target — max(semua suhu pemanasan) mencampur
kalsinasi (pembentuk fasa) dgn sintering/annealing/seri-suhu dalam satu paper.

Definisi target baru: suhu KALSINASI = max suhu pada operasi ber-token
calcin*/fired/firing/decarbonated (varian pre-* dikecualikan — itu tahap awal).

Protokol: subset = resep yang PUNYA suhu kalsinasi; kedua target (T_calc,
T_max) dievaluasi pada subset yang SAMA, berpasangan per resep:
  - floor konstan (median train) + Magpie+RF, GroupKFold by family
  - oracle per-formula LOO lintas-paper (B1 langkah 10)
  - floor reprodusibilitas dalam-paper (B3 langkah 10)
Kalau oracle/within-paper runtuh di T_calc: definisi target lama merusak
eksperimen. Kalau bertahan: noise terkonfirmasi di level kalsinasi juga.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from matsynth.families import family_key
from matsynth.kononova import _composition_from_material, load_raw
from matsynth.magpie_embed import MagpieEmbedder
from matsynth.utils import save_json
from step9_temperature_triviality import T_MAX, T_MIN, extract_temperature

CALC_RE = re.compile(r"^(calcin\w*|fir(e|ed|ing)|decarbonated)$")
N_BOOT = 20000


def extract_calc_temperature(rec: dict) -> float | None:
    temps: list[float] = []
    for op in rec.get("operations") or []:
        tok = (op.get("string") or "").strip().lower()
        if not CALC_RE.match(tok):
            continue
        cond = op.get("conditions") or {}
        for ht in cond.get("heating_temperature") or []:
            if ht.get("units") not in ("C", "°C"):
                continue
            vals = ht.get("values") or []
            lo, hi = ht.get("min_value"), ht.get("max_value")
            if vals:
                temps.extend(float(v) for v in vals)
            elif lo is not None and hi is not None:
                temps.append((float(lo) + float(hi)) / 2.0)
            elif hi is not None:
                temps.append(float(hi))
            elif lo is not None:
                temps.append(float(lo))
    temps = [t for t in temps if T_MIN <= t <= T_MAX]
    return max(temps) if temps else None


def build_subset() -> list[dict]:
    rows = []
    for rec in load_raw():
        t_calc = extract_calc_temperature(rec)
        if t_calc is None:
            continue
        t_max = extract_temperature(rec)
        assert t_max is not None and t_max >= t_calc
        comp = _composition_from_material(rec["target"])
        if comp is None or len(comp) < 2:
            continue
        f = comp.reduced_formula
        rows.append({"formula": f, "family": family_key(f), "doi": rec["doi"],
                     "t_calc": float(t_calc), "t_max": float(t_max)})
    assert len(rows) > 1000
    return rows


def paired(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """b - a; negatif = b lebih baik."""
    d = err_b - err_a
    nz = d[np.abs(d) > 1e-9]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(N_BOOT)])
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"mean_diff": float(d.mean()), "wilcoxon_p": p,
            "bootstrap_ci95": [lo, hi], "significant": bool(hi < 0 or lo > 0)}


def gkf_metrics(rows: list[dict], target: str, X: np.ndarray) -> dict:
    y = np.array([r[target] for r in rows])
    fams = np.array([r["family"] for r in rows])
    gkf = GroupKFold(n_splits=5)
    e_const, e_rf = [], []
    for tr, te in gkf.split(X, y, groups=fams):
        assert not (set(fams[tr]) & set(fams[te]))
        med = float(np.median(y[tr]))
        rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        e_const.extend(np.abs(y[te] - med))
        e_rf.extend(np.abs(y[te] - rf.predict(X[te])))
    return {"mae_constant": float(np.mean(e_const)),
            "mae_magpie_rf": float(np.mean(e_rf))}


def oracle_errors(rows: list[dict], target: str) -> dict[int, float]:
    """LOO lintas-paper per formula (identik B1 langkah 10)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["formula"]].append(i)
    errs = {}
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [rows[j][target] for j in idx
                      if j != i and rows[j]["doi"] != rows[i]["doi"]]
            if others:
                errs[i] = abs(rows[i][target] - float(np.median(others)))
    return errs


def within_paper_errors(rows: list[dict], target: str) -> dict[int, float]:
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["formula"], r["doi"])].append(i)
    errs = {}
    for idx in groups.values():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [rows[j][target] for j in idx if j != i]
            errs[i] = abs(rows[i][target] - float(np.median(others)))
    return errs


def main() -> None:
    rows = build_subset()
    n = len(rows)
    tc = np.array([r["t_calc"] for r in rows])
    tm = np.array([r["t_max"] for r in rows])
    frac_eq = float(np.mean(tc == tm))
    print(f"subset: {n} resep ber-suhu kalsinasi "
          f"(dari 8518 bersuhu; T_calc == T_max di {frac_eq:.1%})")
    print(f"T_calc: median {np.median(tc):.0f}  IQR "
          f"{np.percentile(tc, 25):.0f}-{np.percentile(tc, 75):.0f} °C")

    magpie = MagpieEmbedder()
    uniq = {f: magpie.embed_formula(f) for f in {r["formula"] for r in rows}}
    X = np.stack([uniq[r["formula"]] for r in rows])

    res = {}
    for target in ("t_max", "t_calc"):
        res[target] = gkf_metrics(rows, target, X)

    o_max = oracle_errors(rows, "t_max")
    o_calc = oracle_errors(rows, "t_calc")
    common_o = sorted(set(o_max) & set(o_calc))
    eo_max = np.array([o_max[i] for i in common_o])
    eo_calc = np.array([o_calc[i] for i in common_o])
    st_oracle = paired(eo_max, eo_calc)

    w_max = within_paper_errors(rows, "t_max")
    w_calc = within_paper_errors(rows, "t_calc")
    common_w = sorted(set(w_max) & set(w_calc))
    ew_max = np.array([w_max[i] for i in common_w])
    ew_calc = np.array([w_calc[i] for i in common_w])
    st_within = paired(ew_max, ew_calc)

    print("\nmetrik pada SUBSET YANG SAMA, target T_max vs T_calc:")
    print(f"{'':32}{'T_max':>9}{'T_calc':>9}")
    print(f"  floor konstan (GKF family)    {res['t_max']['mae_constant']:9.1f}"
          f"{res['t_calc']['mae_constant']:9.1f}")
    print(f"  Magpie+RF     (GKF family)    {res['t_max']['mae_magpie_rf']:9.1f}"
          f"{res['t_calc']['mae_magpie_rf']:9.1f}")
    print(f"  oracle formula lintas-paper   {eo_max.mean():9.1f}"
          f"{eo_calc.mean():9.1f}   (n={len(common_o)}, "
          f"p={st_oracle['wilcoxon_p']:.2g})")
    print(f"  dalam-paper                   {ew_max.mean():9.1f}"
          f"{ew_calc.mean():9.1f}   (n={len(common_w)}, "
          f"p={st_within['wilcoxon_p']:.2g})")

    gap_max = res["t_max"]["mae_magpie_rf"] - eo_max.mean()
    gap_calc = res["t_calc"]["mae_magpie_rf"] - eo_calc.mean()
    oracle_drop = 100.0 * (eo_max.mean() - eo_calc.mean()) / eo_max.mean()
    within_drop = 100.0 * (ew_max.mean() - ew_calc.mean()) / ew_max.mean()
    print(f"\n  ruang struktur (RF - oracle): T_max {gap_max:.1f} °C -> "
          f"T_calc {gap_calc:.1f} °C")
    print(f"  penurunan oracle {oracle_drop:+.1f}%  dalam-paper {within_drop:+.1f}%")

    if st_oracle["significant"] and oracle_drop > 15:
        verdict = ("Definisi target lama MERUSAK eksperimen: noise oracle runtuh "
                   f"{oracle_drop:.0f}% saat target = suhu kalsinasi. Angka "
                   "langkah 9 harus dibaca ulang dengan target ini.")
    elif st_oracle["significant"] and oracle_drop > 0:
        verdict = (f"Target kalsinasi memangkas noise oracle {oracle_drop:.0f}% "
                   "(signifikan tapi kecil). Definisi target menyumbang sebagian "
                   "noise; mayoritas sebaran antar-lab tetap tak terjelaskan.")
    else:
        verdict = ("Noise TERKONFIRMASI di level kalsinasi: mengganti definisi "
                   "target tidak menolong. Digabung langkah 10 (rute juga tidak "
                   "menjelaskan), sebaran within-formula adalah noise beneran — "
                   "matsynth-suhu mati secara empiris. Hasil negatif bersih.")
    print(f"\nVONIS: {verdict}")

    save_json("step11_calcination_target.json", {
        "question": "apakah definisi target max(suhu) yang merusak eksperimen?",
        "n_subset": n,
        "fraction_t_calc_equals_t_max": frac_eq,
        "t_calc_distribution": {"median": float(np.median(tc)),
                                "p25": float(np.percentile(tc, 25)),
                                "p75": float(np.percentile(tc, 75))},
        "metrics_same_subset": {
            "t_max": {**res["t_max"], "oracle_cross_paper": float(eo_max.mean()),
                      "within_paper": float(ew_max.mean())},
            "t_calc": {**res["t_calc"], "oracle_cross_paper": float(eo_calc.mean()),
                       "within_paper": float(ew_calc.mean())},
        },
        "paired_oracle_tcalc_vs_tmax": st_oracle,
        "paired_within_paper_tcalc_vs_tmax": st_within,
        "structure_headroom_rf_minus_oracle_c": {"t_max": float(gap_max),
                                                 "t_calc": float(gap_calc)},
        "verdict": verdict,
        "catatan": "angka tidak sebanding dgn langkah 9 (subset beda: hanya "
                   "resep ber-token kalsinasi); perbandingan valid hanya "
                   "T_max vs T_calc di dalam tabel ini",
    })


if __name__ == "__main__":
    main()
