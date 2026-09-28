# matsynth

Prediktor synthesizability + precursor berbasis struktur: reimplementasi konsep
CSLLM (arXiv 2407.07016) dengan encoder MACE equivariant GNN (frozen, pretrained
MACE-MP-0 small) menggantikan LLM text-serialized. Scope tahap ini: HEAD 1
(synthesizability) + HEAD 2 (precursor retrieval kNN). Head kondisi sintesis
ditunda (out of scope).


## Arsitektur

- **Tahap A** (`matsynth/mace_embed.py`, `scripts/step2_featurize.py`):
  fitur node invariant (l=0) dari interaction block terakhir MACE (flag
  `--concat-last-two` untuk 2 layer terakhir), mean-pool atom → z [128],
  disimpan ke `.pt`. MACE tidak pernah dipanggil di training loop.
- **Tahap B** (`matsynth/heads.py`, `matsynth/knn_prec.py`): trunk MLP [d→h] +
  head_synth [h→1] BCEWithLogits; head_prec = kNN cosine di ruang z (top-1
  precursor set, non-parametrik).


## Data

- Resep: Kononova et al., Sci. Data 2019 (`solid-state_dataset_2019-06-27_upd.json`,
  figshare id 17412674, MD5 diverifikasi). Skema di-assert bertipe per record.
  File mentah (~92 MB) tidak ikut di repo: unduh dari figshare ke `data/raw/`
  (`solid-state_dataset_2019-06-27_upd.json`, dan `solid-state_dataset_20200713.json.xz`
  untuk langkah 10). Cache struktur (`data/cache/`) dan embedding (`data/processed/`)
  sudah disertakan.
- Struktur: `mp_id` bawaan dataset → Materials Project OPTIMADE; fallback
  pencarian formula MP → OQMD (keduanya tanpa API key). Cache di `data/cache/`.
- Negatif (skema CSLLM): entry OQMD hipotetis (`_oqmd_icsd_id == null`) dekat
  hull (`_oqmd_stability <= 0.1 eV/atom`) pada grid substitusi argyrodite
  charge-balanced, dikecualikan semua formula yang diketahui tersintesis.
- Positif argyrodite: korpus Kononova 2019 hanya memuat 3 resep argyrodite
  (Ag8GeTe6, Ag8SiS6, Li8SiO6) — ditambal daftar kurasi literatur
  (`families.CURATED_SYNTHESIZED_ARGYRODITES`; Deiseroth 2008, Kong 2010, dst).
- Split **by material family** (family = anonymized formula pymatgen);
  `assert_no_leakage` menjamin family dan formula disjoint antara train/test.

## Menjalankan

```
.venv\Scripts\python.exe scripts\step1_load.py        # loader + resolusi
.venv\Scripts\python.exe scripts\step2_featurize.py   # Tahap A (argyrodite)
.venv\Scripts\python.exe scripts\step3_invariance.py  # GATE invariansi
.venv\Scripts\python.exe scripts\step4_synth_head.py  # head_synth LOFO
.venv\Scripts\python.exe scripts\step5_precursor_knn.py
.venv\Scripts\python.exe scripts\step6_baseline_magpie.py
.venv\Scripts\python.exe scripts\step7_polymorph.py
.venv\Scripts\python.exe scripts\step8_sanity_baselines.py  # floor check — kritis
.venv\Scripts\python.exe scripts\step9_temperature_triviality.py  # suhu: 3 floor
.venv\Scripts\python.exe scripts\step10_noise_decomposition.py  # isi 117 °C
.venv\Scripts\python.exe scripts\step11_calcination_target.py   # target kalsinasi
.venv\Scripts\python.exe scripts\make_figures.py            # figure → results/figures/
```

Hasil per langkah → `results/stepN_*.json`, metrik selalu per family (tidak
ada agregat tunggal yang menyembunyikan kegagalan per family).

## Hasil — baca [FINDINGS.md](FINDINGS.md) untuk laporan lengkap

**Hipotesis TIDAK terbukti.** Dua floor check di langkah 8 membatalkan klaim
performa langkah 4-6. Ringkasnya:

Yang valid:
- Gate invariansi (langkah 3): worst max|z − z_rot| = **1.9e-16** — LOLOS.
- Uji polimorf (langkah 7): Magpie buta total (selisih **0 persis** di 16/16
  pasangan), z MACE membedakan **15/16**. Gagal hanya pada politipe stacking
  ZnS sg156 vs sg160 (lingkungan lokal identik dalam cutoff MACE).
  Ini demonstrasi kapabilitas, BUKAN bukti prediktif.

Yang batal:
- **head_synth ter-confound anion.** Positif = sulfida nyata, negatif = oksida
  hipotetis OQMD → distribusi kimia berbeda. Aturan satu-bit "mengandung O →
  negatif" mencapai acc 0.919 dan **menyamai/mengalahkan MACE di ketiga family**
  (AB6C7 0.875 vs 0.875; AB6C8 **0.895 vs 0.509**; ABC5D6 1.000 vs 1.000).
  Model belajar proses sampling, bukan synthesizability.
- **Retrieval tidak mengalahkan heuristik trivial.** Magpie adalah floor yang
  salah. Heuristik "precursor tersering per unsur": J=0.3496 vs MACE 0.3488,
  Wilcoxon **p=0.92**, CI95 [−0.067, +0.066]. Sebab: **98,8%** precursor GT
  hanya memuat unsur target (+O/C/H/N) — tugasnya memang trivial.

Perbaikan prioritas: (1) negatif dari distribusi kimia yang sama dengan positif,
(2) metrik yang menskor hanya bagian non-trivial resep, (3) uji polimorf
prediktif (formula sama, satu tersintesis satu tidak).

## Keputusan desain yang tercatat

- Resolusi korpus penuh (~7k formula unik) ditunda; tahap ini hanya me-resolve
  keluarga argyrodite + database retrieval chalcogenida (99 formula).
- Ketiga resep argyrodite Kononova satu family (AB6C8) → database retrieval
  langkah 5 memakai record chalcogenida (S/Se/Te) non-argyrodite; Eval B
  (LOFO seluruh chalcogenida) menambah kekuatan statistik.
- Label negatif mengikuti pengetahuan era 2019 (konsisten cakupan Kononova);
  senyawa yang baru tersintesis setelah itu bisa berlabel negatif.
