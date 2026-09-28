"""Langkah 9: apakah suhu sintesis punya sinyal, atau trivial seperti precursor?

Pelajaran langkah 8: jangan bandingkan model dengan floor yang salah. Di sini
dipasang TIGA floor sekaligus sebelum menyimpulkan ada-tidaknya sinyal:

  FLOOR 1 — prediksi konstan. MAE dari menebak satu angka yang sama untuk semua
    material. CATATAN: untuk MAE, konstanta optimal adalah MEDIAN train, bukan
    rata-rata (median meminimalkan L1; rata-rata meminimalkan L2). Dipakai
    median; rata-rata dilaporkan sebagai pembanding.
  FLOOR 2 — noise intrinsik. Untuk formula yang sama muncul di >=2 paper,
    berapa sebaran suhunya? Ini error yang TIDAK BISA dihilangkan model apapun,
    karena input identik memetakan ke target berbeda. Ini CEILING performa.
  FLOOR 3 — model komposisi (Magpie + RandomForest). Kalau tidak mengalahkan
    FLOOR 1 secara signifikan, tidak ada sinyal yang bisa dipanen komposisi.

Protokol split: GroupKFold by material family (family = anonymized formula),
konsisten dengan langkah 4-6. Tanpa ini, formula sama bocor train->test dan
angkanya bohong.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from matsynth.families import family_key
from matsynth.kononova import _composition_from_material, load_raw
from matsynth.magpie_embed import MagpieEmbedder
from matsynth.utils import save_json

T_MIN, T_MAX = 50.0, 3000.0  # rentang fisis masuk akal untuk solid-state (C)
N_SPLITS = 5
N_BOOT = 20000


def extract_temperature(rec: dict) -> float | None:
    """Suhu pemanasan TERTINGGI dalam satu resep (konvensi: sintering/calcination).

    Tiap entry heating_temperature punya values / (min_value, max_value).
    Prioritas: values eksplisit -> midpoint rentang -> max_value tunggal.
    """
    temps: list[float] = []
    for op in rec.get("operations") or []:
        cond = op.get("conditions") or {}
        for ht in cond.get("heating_temperature") or []:
            if ht.get("units") not in ("C", "°C"):  # 2019 pakai "C", 2020 "°C"
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


def build_dataset() -> tuple[list[str], np.ndarray, list[str]]:
    raw = load_raw()
    formulas, temps = [], []
    n_no_temp = n_bad_target = 0
    for rec in raw:
        t = extract_temperature(rec)
        if t is None:
            n_no_temp += 1
            continue
        comp = _composition_from_material(rec["target"])
        if comp is None or len(comp) < 2:
            n_bad_target += 1
            continue
        formulas.append(comp.reduced_formula)
        temps.append(t)
    print(f"[data] resep dgn suhu valid = {len(temps)} "
          f"(tanpa suhu {n_no_temp}, target tak ternormalisasi {n_bad_target})")
    assert len(temps) > 1000, "terlalu sedikit data suhu"
    fams = [family_key(f) for f in formulas]
    return formulas, np.asarray(temps, dtype=float), fams


def intrinsic_noise(formulas: list[str], y: np.ndarray) -> dict:
    """FLOOR 2: sebaran suhu untuk formula IDENTIK di paper berbeda."""
    by_f: dict[str, list[float]] = defaultdict(list)
    for f, t in zip(formulas, y):
        by_f[f].append(t)
    multi = {f: v for f, v in by_f.items() if len(v) >= 2}
    # MAE yang dicapai oracle yang tahu formula persis & menebak median-nya
    errs = [abs(t - float(np.median(v))) for f, v in multi.items() for t in v]
    spreads = [float(np.std(v, ddof=1)) for v in multi.values()]
    return {
        "n_formulas_with_multiple_recipes": len(multi),
        "n_recipes_covered": int(sum(len(v) for v in multi.values())),
        "oracle_per_formula_median_mae": float(np.mean(errs)),
        "mean_within_formula_std": float(np.mean(spreads)),
        "median_within_formula_std": float(np.median(spreads)),
        "example_spreads": {
            f: {"n": len(v), "min": float(min(v)), "max": float(max(v)),
                "std": float(np.std(v, ddof=1))}
            for f, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:5]
        },
    }


def main() -> None:
    formulas, y, fams = build_dataset()
    magpie = MagpieEmbedder()
    X = np.stack([magpie.embed_formula(f) for f in formulas])
    assert X.shape[0] == len(y) == len(fams)
    assert np.isfinite(X).all() and np.isfinite(y).all()

    print("\n" + "=" * 70)
    print("DISTRIBUSI SUHU (C)")
    pct = {f"p{p}": float(np.percentile(y, p)) for p in (5, 25, 50, 75, 95)}
    print(f"  n={len(y)}  mean={y.mean():.1f}  median={np.median(y):.1f}  "
          f"std={y.std(ddof=1):.1f}")
    print(f"  p5={pct['p5']:.0f}  p25={pct['p25']:.0f}  p50={pct['p50']:.0f}  "
          f"p75={pct['p75']:.0f}  p95={pct['p95']:.0f}")
    iqr = pct["p75"] - pct["p25"]
    print(f"  IQR = {iqr:.0f} C   rentang p5-p95 = {pct['p95'] - pct['p5']:.0f} C")

    # ---------------- GroupKFold by family: konstan vs Magpie+RF ----------------
    gkf = GroupKFold(n_splits=N_SPLITS)
    err_const, err_mean, err_rf = [], [], []
    for tr, te in gkf.split(X, y, groups=fams):
        assert not (set(np.asarray(fams)[tr]) & set(np.asarray(fams)[te])), \
            "kebocoran family antar fold"
        med = float(np.median(y[tr]))
        avg = float(np.mean(y[tr]))
        rf = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[te])
        err_const.extend(np.abs(y[te] - med))
        err_mean.extend(np.abs(y[te] - avg))
        err_rf.extend(np.abs(y[te] - pred))

    err_const = np.asarray(err_const)
    err_mean = np.asarray(err_mean)
    err_rf = np.asarray(err_rf)
    assert len(err_const) == len(err_rf) == len(y)

    d = err_rf - err_const  # negatif = RF lebih baik
    nz = d[np.abs(d) > 1e-9]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    reduction = 100.0 * (err_const.mean() - err_rf.mean()) / err_const.mean()

    noise = intrinsic_noise(formulas, y)

    print("\n" + "=" * 70)
    print(f"MAE, GroupKFold by family (k={N_SPLITS}, n={len(y)})")
    print(f"  FLOOR 1a konstan = MEDIAN train   : {err_const.mean():7.1f} C")
    print(f"  FLOOR 1b konstan = RATA2 train    : {err_mean.mean():7.1f} C")
    print(f"  FLOOR 3  Magpie + RandomForest    : {err_rf.mean():7.1f} C")
    print(f"  -> pengurangan error vs floor konstan: {reduction:.1f}%")
    print(f"  -> selisih rata2 (RF - konstan) = {d.mean():+.1f} C")
    print(f"     Wilcoxon p = {p:.3g}   bootstrap CI95 [{lo:+.1f}, {hi:+.1f}]")
    sig = bool(hi < 0)
    print(f"  -> {'SIGNIFIKAN lebih baik' if sig else 'TIDAK signifikan'}")

    print("\n" + "=" * 70)
    print("FLOOR 2 — noise intrinsik (formula identik, paper berbeda)")
    print(f"  formula dgn >=2 resep: {noise['n_formulas_with_multiple_recipes']} "
          f"({noise['n_recipes_covered']} resep)")
    print(f"  MAE oracle yg tahu formula persis : "
          f"{noise['oracle_per_formula_median_mae']:.1f} C")
    print(f"  rata2 std dalam formula yang sama  : "
          f"{noise['mean_within_formula_std']:.1f} C")
    print("  contoh sebaran:")
    for f, v in noise["example_spreads"].items():
        print(f"    {f:<14} n={v['n']:>3}  {v['min']:.0f}-{v['max']:.0f} C  "
              f"std={v['std']:.0f}")

    verdict = (
        "ADA SINYAL: RF mengalahkan floor konstan secara signifikan"
        if sig and reduction > 10 else
        "SINYAL LEMAH/TIDAK ADA: RF tidak cukup mengalahkan prediksi konstan"
    )
    print("\n" + "=" * 70)
    print("VONIS:", verdict)

    save_json("step9_temperature_triviality.json", {
        "question": "apakah suhu sintesis punya variasi bermakna atau trivial?",
        "n_recipes": int(len(y)),
        "temperature_distribution_celsius": {
            "mean": float(y.mean()), "median": float(np.median(y)),
            "std": float(y.std(ddof=1)), "min": float(y.min()),
            "max": float(y.max()), "percentiles": pct, "iqr": float(iqr),
        },
        "protocol": f"GroupKFold by family, k={N_SPLITS}, assert no family leakage",
        "floor1_constant_prediction": {
            "mae_median_train": float(err_const.mean()),
            "mae_mean_train": float(err_mean.mean()),
            "note": "median adalah konstanta optimal untuk MAE, bukan rata-rata",
        },
        "floor2_intrinsic_noise": noise,
        "floor3_magpie_randomforest": {
            "mae": float(err_rf.mean()),
            "error_reduction_vs_constant_percent": float(reduction),
            "mean_difference_rf_minus_constant": float(d.mean()),
            "wilcoxon_p": p,
            "bootstrap_ci95_difference": [lo, hi],
            "significantly_better": sig,
        },
        "verdict": verdict,
    })


if __name__ == "__main__":
    main()
