"""Figure presentasi: temuan yang layak ditunjukkan, dibaca dari results/*.json.

Lima gambar (results/figures/*.png, 300 dpi):
  fig1  gate invariansi        — bukti implementasi benar (12 orde di bawah toleransi)
  fig2  uji polimorf           — struktur membedakan di mana komposisi buta (15/16 vs 0/16)
  fig3  confound anion         — head_synth batal: aturan satu-bit menyamai model
  fig4  retrieval floor check  — tidak beda dari heuristik trivial; 98,8% resep trivial
  fig5  suhu                   — sinyal nyata: tiga floor + ruang tersisa utk struktur

Warna konsisten per ENTITAS di seluruh gambar (color follows the entity):
  MACE=biru, Magpie=aqua, baseline trivial=kuning, oracle/ceiling=violet.
Palet = reference palette dataviz (tervalidasi CVD); teks selalu memakai
token tinta, bukan warna seri.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# ---------------------------------------------------------------- palet & gaya
INK = "#0b0b0b"      # tinta primer (judul, nilai)
INK2 = "#52514e"     # tinta sekunder (subjudul)
MUTED = "#898781"    # sumbu, footnote
GRID = "#e1e0d9"     # hairline grid
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

BLUE = "#2a78d6"     # entitas: MACE
AQUA = "#1baf7a"     # entitas: Magpie
YELLOW = "#eda100"   # entitas: baseline trivial (aturan-O / heuristik / konstan)
VIOLET = "#4a3aa7"   # entitas: oracle / ceiling
DEEMPH = "#b5b3ac"   # mark de-emphasis (konteks)

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 300,
})


def style_ax(ax, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def header(fig, title: str, subtitle: str, footnote: str) -> None:
    fig.text(0.045, 0.955, title, ha="left", va="top",
             fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.045, 0.895, subtitle, ha="left", va="top",
             fontsize=9.5, color=INK2)
    fig.text(0.045, 0.022, footnote, ha="left", va="bottom",
             fontsize=8, color=MUTED)


def sub_formula(s: str) -> str:
    """Na8SnO6 -> Na$_8$SnO$_6$ (subskrip mathtext)."""
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", s)


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


# ------------------------------------------------------------------- figure 1
def fig1_invariance() -> None:
    d = load("step3_invariance.json")
    rows = sorted(d["per_structure"], key=lambda r: r["max_abs_diff"])
    names = [sub_formula(r["formula"]) for r in rows]
    vals = [max(r["max_abs_diff"], 1e-18) for r in rows]
    tol = d["tolerance"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.16, right=0.955, top=0.79, bottom=0.15)
    style_ax(ax, grid_axis="x")

    y = np.arange(len(rows))
    ax.scatter(vals, y, s=64, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=2)
    ax.set_yticks(y, names, fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 1e-2)
    ax.axvline(tol, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(tol * 0.55, 0.2, "toleransi gate = 10⁻⁴",
            fontsize=9, color=INK2, va="center", ha="right")
    ax.annotate("celah 12 orde besaran",
                xy=(3e-16, 4.5), xytext=(3e-11, 4.5),
                fontsize=9.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="<-", color=MUTED, lw=1.1))
    ax.set_xlabel("max |z − z_rot|  (skala log)", fontsize=9.5, color=MUTED)

    header(fig,
           "Embedding benar-benar rotation-invariant",
           "Rotasi SO(3) + translasi acak pada 10 struktur; error terburuk 1,9×10⁻¹⁶ "
           "(presisi mesin float64).\n12 orde besaran di bawah toleransi gate 10⁻⁴ "
           "— GATE LOLOS.",
           "Sumber: results/step3_invariance.json · encoder MACE-MP-0 small (frozen), "
           "fitur invariant l=0 layer terakhir, mean-pool, d=128")
    fig.savefig(FIGDIR / "fig1_invariance_gate.png")
    plt.close(fig)


# ------------------------------------------------------------------- figure 2
def fig2_polymorph() -> None:
    d = load("step7_polymorph.json")
    pairs = sorted(d["pairs"], key=lambda p: p["mace_z_max_abs_diff"])
    labels, vals, colors = [], [], []
    for p in pairs:
        sg_a, sg_b = p["entry_a"]["spacegroup"], p["entry_b"]["spacegroup"]
        labels.append(f"{sub_formula(p['formula'])}  ·  sg {sg_a} vs {sg_b}")
        vals.append(p["mace_z_max_abs_diff"])
        colors.append(BLUE if p["mace_distinguishes"] else DEEMPH)
    thr = d["min_z_diff_threshold"]

    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.subplots_adjust(left=0.24, right=0.955, top=0.825, bottom=0.115)
    style_ax(ax, grid_axis="x")

    y = np.arange(len(pairs))
    ax.barh(y, vals, height=0.55, color=colors, zorder=3)
    ax.set_yticks(y, labels, fontsize=8.6, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 1.2)
    ax.axvline(thr, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(thr, len(pairs) - 0.15, f" ambang beda = {thr:g}",
            fontsize=9, color=INK2, va="top", ha="left")
    ax.text(vals[0] * 1.35, 0,
            "politipe stacking — lingkungan lokal identik dalam cutoff MACE",
            fontsize=8.6, color=INK2, va="center")
    ax.set_xlabel("max |Δz| antar polimorf, formula sama  (skala log)",
                  fontsize=9.5, color=MUTED)

    header(fig,
           "Struktur membedakan polimorf di mana komposisi buta total",
           "MACE membedakan 15/16 pasangan formula-sama-struktur-beda; Magpie: selisih "
           "0 persis di 16/16 pasangan\n(nol tak tampil di skala log) — buta struktur "
           "secara matematis. Catatan: demonstrasi kapabilitas, bukan bukti prediktif.",
           "Sumber: results/step7_polymorph.json · entry MP OPTIMADE ordered, "
           "spacegroup via pymatgen (symprec 0.1)")
    fig.savefig(FIGDIR / "fig2_polymorph.png")
    plt.close(fig)


# ------------------------------------------------------------------- figure 3
def fig3_confound() -> None:
    s4 = load("step4_synth_mace.json")["per_family"]
    s6 = load("step6_baseline_magpie.json")["head_synth_per_family"]
    s8 = load("step8_sanity_baselines.json")
    orule = s8["diagnostic_b_anion_confound_in_synth_labels"]["per_family"]

    fams = ["ABC5D6", "AB6C7", "AB6C8"]
    fam_label = {
        "ABC5D6": "ABC5D6\n(Li$_6$PS$_5$Cl, dst.)",
        "AB6C7": "AB6C7\n(Li$_7$PS$_6$, dst.)",
        "AB6C8": "AB6C8\n(Ag$_8$GeS$_6$, dst.)",
    }
    mace = [s4[f]["accuracy"] for f in fams]
    magp = [s6[f]["accuracy"] for f in fams]
    orl = [orule[f]["oxygen_rule_accuracy"] for f in fams]
    # floor kelas mayoritas per fold test
    major = [max(s4[f]["n_test_pos"], s4[f]["n_test_neg"]) / s4[f]["n_test"]
             for f in fams]
    ns = [s4[f]["n_test"] for f in fams]

    fig, ax = plt.subplots(figsize=(9, 5.1))
    fig.subplots_adjust(left=0.09, right=0.965, top=0.76, bottom=0.145)
    style_ax(ax, grid_axis="y")

    x = np.arange(len(fams))
    w = 0.24
    series = [("MACE (struktur)", mace, BLUE, -w),
              ("Magpie (komposisi)", magp, AQUA, 0.0),
              ("aturan “ada O → negatif”", orl, YELLOW, w)]
    for name, vals, color, off in series:
        bars = ax.bar(x + off, vals, width=w - 0.03, color=color, label=name,
                      zorder=3)
        for b, v, mj in zip(bars, vals, major):
            label = f"{v:.2f}".replace(".", ",")
            cx = b.get_x() + b.get_width() / 2
            if 0.0 < mj - v < 0.09:  # label di luar akan menabrak garis floor
                ax.text(cx, v - 0.03, label, ha="center", va="top",
                        fontsize=8.8, color=INK, zorder=5)
            else:
                ax.text(cx, v + 0.018, label, ha="center", va="bottom",
                        fontsize=8.8, color=INK)
    for xi, mj in zip(x, major):
        ax.hlines(mj, xi - 0.42, xi + 0.42, color=INK2, linewidth=1.2,
                  linestyle=(0, (4, 3)), zorder=4)
    ax.hlines([], [], [], label="_nolegend_")

    ax.set_xticks(x, [f"{fam_label[f]}\nn = {n}" for f, n in zip(fams, ns)],
                  fontsize=9, color=INK2)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("akurasi (test = family OOD penuh)", fontsize=9.5, color=MUTED)

    from matplotlib.lines import Line2D
    handles, labels_ = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=INK2, linewidth=1.2,
                          linestyle=(0, (4, 3))))
    labels_.append("floor kelas mayoritas")
    ax.legend(handles, labels_, loc="upper center", frameon=False, fontsize=8.5,
              ncols=4, bbox_to_anchor=(0.5, 1.10), labelcolor=INK2,
              columnspacing=1.2, handlelength=1.6)

    header(fig,
           "head_synth batal: aturan satu-bit menyamai atau mengalahkan model",
           "Positif = sulfida/selenida nyata; negatif = oksida hipotetis OQMD → "
           "distribusi kimia berbeda.\nAturan “ada O → negatif” berakurasi 0,92 — "
           "model belajar proses sampling, bukan synthesizability.",
           "Sumber: results/step4·6·8_*.json · leave-one-family-out, 3 seed, "
           "assert tanpa kebocoran family/formula · n total = 37 (19 pos / 18 neg)")
    fig.savefig(FIGDIR / "fig3_confound_anion.png")
    plt.close(fig)


# ------------------------------------------------------------------- figure 4
def fig4_retrieval() -> None:
    s8 = load("step8_sanity_baselines.json")
    a = s8["diagnostic_a_retrieval_vs_naive_element_heuristic"]

    # Magpie mikro (LOFO identik) — dihitung ulang agar sebanding apel-ke-apel
    import torch
    from matsynth.knn_prec import retrieve_and_score
    from matsynth.magpie_embed import MagpieEmbedder
    ch = torch.load(ROOT / "data" / "processed" / "embeddings_chalcogenide.pt",
                    weights_only=False)
    mids = sorted(ch.keys())
    cf = [ch[m]["formula"] for m in mids]
    cfam = [ch[m]["family"] for m in mids]
    cprec = [frozenset(ch[m]["precursors"]) for m in mids]
    cgt = [[frozenset(s) for s in ch[m]["all_precursor_sets"]] for m in mids]
    Zm = np.stack([MagpieEmbedder().embed_formula(f) for f in cf])
    js = []
    for fam in sorted(set(cfam)):
        q = [i for i, f in enumerate(cfam) if f == fam]
        db = [i for i, f in enumerate(cfam) if f != fam]
        if not db:
            continue
        rs = retrieve_and_score(Zm[db], [cprec[i] for i in db],
                                Zm[q], [cgt[i] for i in q], k=3)
        js.extend(r["jaccard_top1"] for r in rs)
    magpie_micro = float(np.mean(js))

    rows = [("heuristik unsur (tanpa ML)",
             a["naive_element_heuristic_jaccard_mean"], YELLOW),
            ("MACE (struktur)", a["mace_jaccard_mean"], BLUE),
            ("Magpie (komposisi)", magpie_micro, AQUA)]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.4), width_ratios=[1.75, 1.0])
    fig.subplots_adjust(left=0.24, right=0.96, top=0.775, bottom=0.16,
                        wspace=0.34)
    style_ax(ax, grid_axis="x")

    y = np.arange(len(rows))[::-1]
    for yi, (name, v, c) in zip(y, rows):
        ax.barh(yi, v, height=0.5, color=c, zorder=3)
        ax.text(v + 0.006, yi, f"{v:.3f}".replace(".", ","),
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5, color=INK2)
    ax.set_xlim(0, 0.46)
    ax.set_xlabel("Jaccard top-1 (LOFO 50 family, n = 99 query)",
                  fontsize=9.5, color=MUTED)
    ax.text(0.36, y[0] - 0.52,
            f"MACE vs heuristik: menang {a['wins_mace']} / seri {a['ties']} / "
            f"kalah {a['wins_naive']}\nWilcoxon p = "
            f"{a['wilcoxon_p']:.2f} · CI95 [{a['bootstrap_ci95_difference'][0]:+.3f}, "
            f"{a['bootstrap_ci95_difference'][1]:+.3f}]".replace(".", ","),
            fontsize=8.8, color=INK2, va="top", ha="center")

    # panel kanan: kenapa — hero number
    ax2.axis("off")
    triv = a["fraction_of_gt_precursors_that_are_trivial"]
    ax2.text(0.5, 0.72, f"{triv * 100:.1f}%".replace(".", ","),
             ha="center", va="center", fontsize=44, fontweight="bold",
             color=INK, transform=ax2.transAxes)
    ax2.text(0.5, 0.40,
             "precursor ground-truth hanya memuat\nunsur target sendiri (+O/C/H/N)",
             ha="center", va="center", fontsize=9.5, color=INK2,
             transform=ax2.transAxes)
    ax2.text(0.5, 0.13,
             "→ tugasnya “tebak reagen standar per unsur”;\n"
             "tidak ada ruang bagi informasi struktur untuk menang",
             ha="center", va="center", fontsize=8.8, color=MUTED,
             transform=ax2.transAxes)

    header(fig,
           "Retrieval precursor tidak mengalahkan heuristik kimia trivial",
           "kNN di ruang embedding vs “precursor tersering per unsur” — tidak berbeda "
           "signifikan (p = 0,92).\nMagpie sebelumnya adalah floor yang salah.",
           "Sumber: results/step8_sanity_baselines.json · Magpie mikro dihitung ulang "
           "dgn protokol LOFO identik · GT multi-resep: skor = Jaccard terbaik")
    fig.savefig(FIGDIR / "fig4_retrieval_floor.png")
    plt.close(fig)


# ------------------------------------------------------------------- figure 5
def fig5_temperature() -> None:
    s9 = load("step9_temperature_triviality.json")
    from step9_temperature_triviality import build_dataset
    _, ytemp, _ = build_dataset()

    mae_const = s9["floor1_constant_prediction"]["mae_median_train"]
    mae_rf = s9["floor3_magpie_randomforest"]["mae"]
    mae_oracle = s9["floor2_intrinsic_noise"]["oracle_per_formula_median_mae"]
    red = s9["floor3_magpie_randomforest"]["error_reduction_vs_constant_percent"]
    frac_band = float(np.mean((ytemp >= 700) & (ytemp <= 900)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.9),
                                  width_ratios=[1.25, 1.0])
    fig.subplots_adjust(left=0.075, right=0.965, top=0.72, bottom=0.15,
                        wspace=0.30)

    # ---- panel kiri: distribusi suhu
    style_ax(ax, grid_axis="y")
    ax.hist(ytemp, bins=np.arange(0, 3001, 50), color=BLUE, zorder=3)
    ax.axvspan(700, 900, color=GRID, alpha=0.55, zorder=1)
    ax.annotate("praduga “semua di 700–900 °C”:\nhanya "
                f"{frac_band * 100:.1f}% resep".replace(".", ","),
                xy=(760, 480), xytext=(300, 620), fontsize=8.8, color=INK2,
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    med = float(np.median(ytemp))
    ax.axvline(med, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(med + 35, 672, f"median {med:.0f} °C", fontsize=8.8, color=INK2,
            va="top")
    ax.set_xlim(0, 2300)
    ax.set_ylim(0, 700)
    ax.set_xlabel("suhu pemanasan maksimum per resep (°C)", fontsize=9.5,
                  color=MUTED)
    ax.set_ylabel("jumlah resep", fontsize=9.5, color=MUTED)
    ax.set_title(f"Sebaran suhu, n = {len(ytemp):,} resep — p5–p95 membentang "
                 "550–1521 °C", fontsize=9.5, color=INK2, loc="left", pad=8)

    # ---- panel kanan: tangga MAE
    style_ax(ax2, grid_axis="x")
    rows = [("floor: prediksi konstan\n(median train, 1100 °C)", mae_const, YELLOW),
            ("Magpie + Random Forest", mae_rf, AQUA),
            ("ceiling: oracle yang tahu\nformula persis (noise intrinsik)",
             mae_oracle, VIOLET)]
    y = np.arange(len(rows))[::-1]
    for yi, (name, v, c) in zip(y, rows):
        ax2.barh(yi, v, height=0.5, color=c, zorder=3)
        ax2.text(v + 6, yi, f"{v:.0f} °C", va="center", fontsize=10,
                 color=INK, fontweight="bold")
    ax2.set_yticks(y, [r[0] for r in rows], fontsize=9, color=INK2)
    ax2.set_xlim(0, 320)
    ax2.set_xlabel("MAE (°C), GroupKFold by family, k = 5", fontsize=9.5,
                   color=MUTED)
    ax2.annotate("", xy=(mae_rf, y[1] + 0.44), xytext=(mae_const, y[1] + 0.44),
                 arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
    ax2.text((mae_rf + mae_const) / 2, y[1] + 0.56,
             f"−{red:.0f}% (p ≈ 0)", ha="center", fontsize=8.8, color=INK2)
    ax2.set_title("Komposisi memanen 63% sinyal;\nsisa utk struktur ≈ 50 °C",
                  fontsize=9.5, color=INK2, loc="left", pad=8)

    header(fig,
           "Suhu sintesis PUNYA sinyal — kebalikan dari precursor",
           "Magpie+RF memangkas MAE 33% dari floor konstan (Wilcoxon p ≈ 0, CI95 "
           "[−87, −80] °C).\nTapi oracle per-formula tetap salah 117 °C: noise "
           "intrinsik (rute berbeda + artefak text-mining) membatasi semua model.",
           "Sumber: results/step9_temperature_triviality.json · Kononova 2019, 8.518 "
           "resep bersuhu valid · suhu = pemanasan maksimum per resep, 50–3000 °C")
    fig.savefig(FIGDIR / "fig5_temperature_signal.png")
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig1_invariance()
    print("[fig] fig1_invariance_gate.png")
    fig2_polymorph()
    print("[fig] fig2_polymorph.png")
    fig3_confound()
    print("[fig] fig3_confound_anion.png")
    fig4_retrieval()
    print("[fig] fig4_retrieval_floor.png")
    fig5_temperature()
    print("[fig] fig5_temperature_signal.png")
    print(f"[selesai] {FIGDIR}")


if __name__ == "__main__":
    main()
