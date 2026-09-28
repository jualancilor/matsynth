# matsynth — Laporan Temuan Lengkap

**Status: pipeline selesai (8/8 langkah), hipotesis TIDAK terbukti.**
Dokumen ini self-contained — semua angka, metode, dan kegagalan ada di sini.
Run terakhir: 2026-07-15. Semua hasil mentah di `results/step*.json`.

---

## 1. Ringkasan eksekutif

Saya membangun ulang konsep CSLLM (arXiv 2407.07016) dengan satu perubahan:
mengganti encoder LLM berbasis teks dengan **MACE** (equivariant GNN) sebagai
feature extractor struktur. Hipotesis: representasi geometrik menggeneralisasi
lebih baik daripada teks/komposisi pada struktur out-of-distribution dan polimorf.

**Infrastrukturnya jalan dan terverifikasi. Hipotesisnya tidak terbukti.**

Dua floor check (langkah 8) membatalkan kedua klaim utama:

| Klaim awal | Status setelah floor check |
|---|---|
| "head_synth: MACE (1.000/0.875) > Magpie (0.900/0.792)" | **Batal.** Label ter-*confound* dengan anion. Aturan satu-bit "mengandung O → negatif" mencapai akurasi 0.919 dan **menyamai atau mengalahkan MACE di ketiga family**. |
| "retrieval: MACE (0.380) > Magpie (0.305)" | **Batal.** Magpie adalah floor yang salah. Heuristik kimia trivial mencapai J=0.350 vs MACE 0.349. Uji berpasangan: **p = 0.92**, CI95 [−0.067, +0.066]. Tidak ada beda. |

Yang tersisa sebagai hasil valid: gate invariansi (bukti implementasi benar) dan
uji polimorf (bukti kapabilitas, bukan bukti prediktif).

**Kesimpulan jujur: eksperimen sebagaimana dirancang tidak mampu menjawab
pertanyaan yang diajukan.** Penyebabnya bukan bug — pipeline-nya benar. Penyebabnya
adalah konstruksi dataset dan pilihan metrik. Keduanya bisa diperbaiki, dan
diagnosisnya justru merupakan kontribusi paling berguna dari iterasi ini.

---

## 2. Konteks & hipotesis

CSLLM memprediksi synthesizability material dengan men-serialisasi struktur
kristal menjadi teks lalu men-fine-tune LLM. Kelemahan konseptual: serialisasi
teks tidak menjamin invariansi rotasi, dan representasi berbasis komposisi tidak
bisa membedakan polimorf sama sekali (TiO₂ rutile dan anatase punya string
komposisi identik tapi sifat sintesis berbeda total).

**Hipotesis:** encoder equivariant yang bekerja langsung pada geometri 3D akan
(a) invariant terhadap rotasi secara arsitektural, (b) membedakan polimorf, dan
karena itu (c) menggeneralisasi lebih baik ke material OOD.

Scope iterasi ini: HEAD 1 (synthesizability) + HEAD 2 (retrieval precursor).
Head kondisi sintesis (suhu/atmosfer) sengaja tidak diimplementasikan.

---

## 3. Metode

### 3.1 Arsitektur dua tahap

**Tahap A — precompute embedding (sekali jalan, encoder frozen):**
- Checkpoint: **MACE-MP-0 small** (`2023-12-10-mace-128-L0_energy_epoch-249`),
  pretrained pada Materials Project, **frozen** (tidak pernah di-fine-tune).
- Untuk tiap struktur: ekstrak fitur node **invariant (l=0)** dari interaction
  block terakhir → `[n_atoms, 128]`.
- Mean-pool atas atom → `z ∈ R¹²⁸`, graph-level.
- Simpan `{material_id: z}` ke `.pt`. MACE **tidak pernah dipanggil di training
  loop** — ini memisahkan biaya encoder dari eksperimen head.
- Flag opsional `--concat-last-two` (konkatenasi 2 layer terakhir → d=256).

**Tahap B — training head (ringan):**
- `head_synth`: MLP trunk [128→64] + [64→1], sigmoid, BCEWithLogitsLoss,
  3 seed, 800 epoch.
- `head_prec`: **kNN cosine** di ruang z (k=3), ambil precursor set dari tetangga
  terdekat. Non-parametrik — sengaja, agar yang diuji adalah **kualitas ruang
  embedding**, bukan kapasitas decoder.

### 3.2 Alasan pilihan arsitektur

| Pilihan | Alasan |
|---|---|
| MACE, bukan LLM teks | Invariansi rotasi arsitektural (terverifikasi 1.9e-16), bukan sekadar diharapkan dari augmentasi data. |
| Frozen, tidak fine-tune | n=37. Fine-tune 4.7M parameter pada 37 titik = overfit dijamin. Frozen encoder + head kecil adalah satu-satunya pilihan yang jujur. |
| Fitur invariant l=0 saja | Target (synthesizability, precursor) adalah skalar — tidak butuh komponen vektor/tensor. |
| kNN, bukan decoder parametrik | Menguji ruang embedding secara langsung tanpa mencampur kapasitas model. |
| Mean-pool | Paling sederhana, permutation-invariant, tidak menambah parameter. |

### 3.3 Data

**Sumber resep:** Kononova et al., *Sci. Data* 6:203 (2019), text-mined solid-state
synthesis. File `solid-state_dataset_2019-06-27_upd.json` (figshare 17412674,
MD5 diverifikasi). 19.488 record mentah → 6.034 setelah normalisasi.
Drop: 9.105 target tak ternormalisasi, 956 precursor tak ternormalisasi,
3.393 duplikat.

**Resolusi struktur:** OPTIMADE **tanpa API key** — Materials Project
(`optimade.materialsproject.org/v1`), fallback OQMD (`oqmd.org/optimade`).
Jalur utama: field `mp_id` bawaan dataset (terisi di 5.798/19.488 record).

**⚠ Temuan data yang menentukan segalanya:** korpus Kononova **nyaris tidak
memuat argyrodite**. Hanya **3 resep** (Ag₈GeTe₆, Li₈SiO₆, Si(Ag₄S₃)₂), dan
ketiganya di **satu family yang sama (AB6C8)**. Li₆PS₅Cl/Br/I — argyrodite paling
penting secara teknologi (solid electrolyte) — **tidak ada sama sekali**.

Konsekuensi: set positif harus ditambal manual dari daftar kurasi literatur
(16 argyrodite: Deiseroth 2008, Kong 2010, dll.).

**Konstruksi label synthesizability (n=37):**
- **Positif (19):** 3 dari Kononova + 16 dari kurasi literatur manual.
- **Negatif (18):** entry OQMD **hipotetis** — `_oqmd_icsd_id == null` (belum
  pernah dibuat) DAN `_oqmd_stability ≤ 0.1 eV/atom` (stabil secara DFT), dari
  grid substitusi argyrodite charge-balanced. Mengikuti skema CSLLM.

**Split:** leave-one-family-out (LOFO), family = anonymized formula pymatgen.
`assert_no_leakage` memverifikasi family dan formula disjoint antara train/test
di setiap fold. Tidak ada agregat global tunggal — semua metrik per family.

---

## 4. Hasil per langkah

### Langkah 1–2: loader + featurisasi
6.034 record ternormalisasi, skema di-assert bertipe per record. 37 struktur
argyrodite → z ∈ R¹²⁸. Shape assertion di setiap boundary tensor.

### Langkah 3: GATE INVARIANSI — **LOLOS** ✅
Rotasi SO(3) random + translasi fraksional pada 10 struktur:

```
worst max|z − z_rot| = 1.9e-16   (toleransi 1e-4)
```

Ini **bukan temuan ilmiah** — ini bukti bahwa implementasinya benar. Tapi ini
gate yang wajib dan banyak implementasi gagal di sini.

### Langkah 4–6: head_synth dan retrieval vs Magpie

head_synth, LOFO, akurasi rata-rata 3 seed:

| family test (OOD) | n_test | pos/neg | MACE | Magpie |
|---|---|---|---|---|
| ABC5D6 (A₆BX₅Y) | 10 | 3/7 | 1.000 | 0.900 |
| AB6C7 (A₇BX₆) | 8 | 4/4 | 0.875 | 0.792 |
| AB6C8 (A₈BX₆) | 19 | 12/7 | 0.509 | 0.579 |

Retrieval precursor (Jaccard top-1):

| eval | MACE | Magpie |
|---|---|---|
| Eval A: query argyrodite (n=3, db lintas family) | 0.150 | 0.067 |
| Eval B: LOFO chalcogenida (n=99, 50 family), makro-family | 0.380 | 0.305 |
| Eval B, mikro | 0.349 | — |

Terlihat seperti kemenangan MACE di 2/3 family + kedua eval retrieval.
**Langkah 8 menunjukkan interpretasi ini salah.**

### Langkah 7: uji polimorf

16 pasangan formula-sama-struktur-beda:

| formula | pasangan | Magpie | MACE max\|dz\| | terbedakan? |
|---|---|---|---|---|
| TiO₂ | 15 pasangan spacegroup (15/141/152/166/189/225) | **0.0 semua** | 0.012 – 0.280 | **15/15** ✅ |
| ZnS | sg156 vs sg160 | 0.0 | 0.00024 | ❌ |

Magpie memberi selisih **nol persis** di 16/16 pasangan — secara matematis buta
terhadap struktur. MACE membedakan 15/16. Kegagalan ZnS informatif: itu politipe
*stacking* — lingkungan lokal identik dalam radius cutoff MACE. Relevan karena
politipe nyata di material riil (SiC, ZnS, GaN).

**Status logis: ini demonstrasi kapabilitas, bukan temuan.** Membuktikan z
membedakan polimorf ≠ membuktikan perbedaan itu prediktif untuk synthesizability.

---

## 5. Langkah 8: dua floor check yang membatalkan klaim

Ini bagian terpenting dokumen ini.

### 5.1 Diagnostik A — retrieval tidak mengalahkan heuristik trivial

Baseline: **"untuk tiap unsur di target, pakai precursor yang paling sering
dipakai untuk unsur itu di train set."** Ini yang dilakukan chemist tanpa ML
sama sekali. Protokol LOFO identik.

```
n = 99 query berpasangan
MACE                  J = 0.3488
heuristik unsur       J = 0.3496
selisih               = −0.0008
menang 35 / seri 27 / kalah 37
Wilcoxon signed-rank  p = 0.9150
bootstrap CI95        [−0.0668, +0.0662]
→ TIDAK SIGNIFIKAN
```

**Magpie adalah floor yang salah.** Floor yang benar — pengetahuan kimia dasar —
menyamai MACE persis.

**Diagnosis kenapa:** **98,8%** precursor di ground truth hanya memuat unsur
target sendiri (+O/C/H/N). Artinya tugas "prediksi precursor" sebagaimana diukur
Jaccard hampir seluruhnya adalah *"tebak bentuk reagen standar tiap unsur"* —
Li→Li₂CO₃, Si→SiO₂, Ag→Ag₂S. **Tugas itu tidak butuh informasi struktur.**
Encoder geometrik secanggih apapun tidak punya ruang untuk menang. Metriknya
jenuh oleh bagian trivial.

### 5.2 Diagnostik B — head_synth ter-confound anion

Tabel silang label vs kandungan oksigen (n=37):

| | label=1 (dibuat) | label=0 (hipotetis) |
|---|---|---|
| **mengandung O** | 1 | 16 |
| **tanpa O (S/Se/Te)** | 18 | 2 |

Aturan **satu bit** "mengandung O → negatif":

| family | n | aturan-O | MACE |
|---|---|---|---|
| AB6C7 | 8 | 0.875 | 0.875 (**seri**) |
| AB6C8 | 19 | **0.895** | 0.509 (**aturan trivial menang telak**) |
| ABC5D6 | 10 | 1.000 | 1.000 (**seri**) |
| seluruh set | 37 | **0.919** | — |

**Aturan satu-bit menyamai atau mengalahkan MACE di ketiga family.**

**Diagnosis:** ini artefak konstruksi dataset, bukan bug kode. Positif kurasi
saya adalah argyrodite sulfida/selenida **nyata** (Li₆PS₅Cl, Cu₈GeS₆, …). Negatif
OQMD hipotetis kebetulan **mayoritas oksida** — karena grid substitusi
charge-balanced over-generate oksida yang memang tidak eksis. Jadi positif dan
negatif **diambil dari distribusi kimia yang berbeda**.

Model tidak belajar synthesizability. Model belajar **proses sampling saya**.

Ini failure mode klasik di ML-for-science: *negative set drawn from a different
distribution than the positive set.* Classifier apapun akan "berhasil" — dan
keberhasilannya tidak berarti apa-apa.

---

### 5.3 Kontras: suhu sintesis (langkah 9) — floor check yang LOLOS

Setelah dua kegagalan di atas, pertanyaan wajarnya: apakah semua target sintesis
trivial? **Tidak.** Suhu pemanasan maksimum per resep (8.518 resep bersuhu valid)
diuji dengan tiga floor, GroupKFold by family:

| | MAE |
|---|---|
| Floor: prediksi konstan (median train, 1100 °C) | 250,1 °C |
| **Magpie + Random Forest** | **166,6 °C** |
| Ceiling: oracle yang tahu formula persis (noise intrinsik) | 116,9 °C |

- Distribusinya lebar: p5–p95 = 550–1521 °C. Praduga "semua solid-state di
  700–900 °C" salah — hanya 22,5% resep di rentang itu; "tebak 800" memberi
  MAE 339 °C, lebih buruk dari konstanta optimal.
- Magpie+RF memangkas error **33,4%** dari floor konstan (Wilcoxon p ≈ 0,
  CI95 [−87, −80] °C). **Sinyal nyata** — kebalikan dari precursor.
- Tapi oracle per-formula tetap salah **117 °C**: formula identik muncul di
  paper berbeda dengan suhu berbeda (rute sintesis berbeda + artefak
  text-mining; LiFePO₄ tercatat 60–1250 °C). Komposisi sudah memanen ~63%
  sinyal yang tersedia; sisa ruang untuk encoder struktur ≈ 50 °C.
- Pertanyaan praktis untuk eksperimentalis: apakah MAE 167 °C berguna di
  furnace? Signifikan secara statistik ≠ berguna secara praktis.
- ⚠ Angka MAE di bagian ini hanya sebanding di dalam eksperimennya sendiri.
  Jangan menyandingkannya dengan angka dari dataset, subset, atau definisi
  target lain (termasuk run lama pada subset argyrodite atau klaim literatur)
  — data dan target berbeda membuat perbandingannya tidak bermakna.

Hasil mentah: `results/step9_temperature_triviality.json`.

### 5.4 Bongkar isi 117 °C (langkah 10–11): polimorf, rute, atau noise?

Ceiling 117 °C hanya "lunak" kalau isinya sistematik terhadap sesuatu yang
bisa dijadikan input. Tiga kandidat diuji, ketiganya gugur:

**Kandidat 1 — polimorf (langkah 10A): TIDAK TERJAWAB dari korpus, dan itu
temuan.** Korpus Kononova praktis tidak pernah merekam polimorf yang dibuat:
file 2019 tanpa field `phase` (0 record), `mp_id` di-assign per formula (tidak
pernah bervariasi antar-resep formula yang sama → nol informasi polimorf),
string target hampir tanpa marker (745/833 "marker" adalah δ dari notasi
defisiensi oksigen). File 2020 (31.782 reaksi) punya field `phase` tapi terisi
di 264 record → **hanya 1 formula** memenuhi syarat uji (NaFeO₂: α med 475 °C
n=4 vs O3 med 650 °C n=2 — arah yang benar, tapi anekdot, bukan uji).
Pertanyaan "sebaran suhu nyambung polimorf atau tidak" hanya bisa dijawab
dengan data lab yang mencatat fasa produk.

**Kandidat 2 — rute/precursor (langkah 10B): GUGUR.** LOO lintas-paper pada
resep berpasangan (n=2.526–3.502): kondisi (formula) MAE 150,4 °C vs kondisi
(formula + precursor set persis) MAE 152,0 °C — memfiksasi rute **tidak
memangkas apa-apa** (−1,1%, Wilcoxon p=0,54, CI95 [−1,0, +4,4]). Sebaran
antar-lab bukan sistematik-rute.

**Kandidat 3 — definisi target (langkah 11): GUGUR.** Kecurigaan: max(semua
suhu) mencampur kalsinasi dengan sintering/seri-suhu. Diulang dengan target =
suhu kalsinasi (token calcin\*/fired/firing, varian pre-\* dikecualikan;
n=4.470, subset sama untuk kedua target): oracle 123,2 → 121,8 °C (−1,1%;
p=0,012 — signifikan tapi praktis nol), dalam-paper malah 104 → 113 °C (ns).
Ruang struktur (RF − oracle) menciut 16,1 → 14,3 °C. Definisi target bukan
masalahnya.

**Vonis gabungan:** sebaran suhu within-formula adalah **noise beneran** —
tidak terjelaskan oleh rute, definisi target, maupun (sejauh korpus bisa
bicara) polimorf. Jatah encoder struktur di task suhu tinggal belasan–50 °C
di atas plafon yang tidak bisa ditembus input apa pun yang tersedia.
**matsynth-suhu mati secara empiris pada korpus ini** — hasil negatif yang
bersih dan layak dilaporkan. Sinyal komposisi (§5.3) tetap berdiri; yang mati
adalah klaim nilai tambah struktur di atasnya. Satu-satunya jalan hidup:
data lab yang mencatat fasa produk + suhu (kembali ke permintaan data §8.5).

Hasil mentah: `results/step10_noise_decomposition.json`,
`results/step11_calcination_target.json`.

## 6. Apa yang tetap kokoh

Jangan buang semuanya. Yang valid:

1. **Infrastruktur dua tahap jalan.** Loader bertipe, resolusi struktur keyless
   (MP+OQMD, cache disk), featurisasi, split by family dengan assert anti-leakage,
   pelaporan per-family. Semuanya reusable.
2. **Gate invariansi lolos di 1.9e-16.** Encoder benar-benar invariant. Ini
   properti yang tidak dimiliki pendekatan teks CSLLM.
3. **z membedakan polimorf di mana komposisi buta total** (15/16 vs 0/16).
   Justifikasi arsitektural hipotesis tetap berdiri — yang belum terbukti adalah
   apakah kemampuan itu *berguna*.
4. **Kuirk teknis terdokumentasi:** `mace.get_descriptors(invariants_only=True)`
   tidak boleh dibungkus `torch.no_grad()` (forward menghitung gaya via autograd
   posisi).

---

## 7. Challenge & limitasi

**Fundamental (membatalkan klaim):**
- **Negatif adalah proksi, bukan negatif asli.** "Stabil di DFT tapi belum dibuat"
  ≠ "tidak bisa dibuat". Bisa jadi hanya belum ada yang mencoba. CSLLM pun punya
  masalah ini — kami mewarisi kelemahannya.
- **Confound anion** (§5.2) membuat semua metrik head_synth tidak dapat ditafsirkan.
- **Metrik retrieval jenuh** oleh 98,8% resep trivial (§5.1).

**Statistik:**
- **n=37** untuk head_synth. Tidak ada klaim yang bisa berdiri di n segini.
- **n=3** untuk Eval A retrieval. Praktis anekdot.
- 16/19 positif dari kurasi manual saya → risiko bias seleksi saya sendiri.

**Cakupan data:**
- Korpus Kononova nyaris kosong untuk argyrodite (3 resep, 1 family).
- Label negatif mengikuti pengetahuan era 2019; material yang tersintesis setelah
  itu bisa salah label negatif.
- Li₆PSe₅Cl/Br/I tidak ada di MP maupun OQMD (miss genuin).

**Teknis:**
- MACE cutoff pendek → buta terhadap politipe stacking (ZnS).
- Struktur dari database adalah geometri **DFT-relaxed**, bukan struktur riil
  hasil eksperimen pada suhu sintesis.
- CPU only, tanpa GPU.

---

## 8. Yang perlu dikerjakan (urutan prioritas)

1. **Perbaiki set negatif — ini blocker.** Negatif harus diambil dari
   **distribusi kimia yang sama** dengan positif. Konkretnya: negatif harus
   mencakup sulfida/selenida hipotetis, bukan didominasi oksida. Uji ulang
   diagnostik B sampai aturan-O turun ke ~chance (≈0.5). **Sebelum ini beres,
   semua angka head_synth tidak bermakna.**
2. **Ganti metrik retrieval.** Skor hanya bagian **non-trivial**: pemilihan sumber
   khalkogen (Ag₂Te vs Ag₂S vs Te elemental), flux, reagen di luar unsur target.
   Di situlah kimia sesungguhnya — dan di situlah struktur mungkin berarti.
3. **Ubah uji polimorf jadi uji prediktif.** Sekarang: "z berbeda". Seharusnya:
   pasangan formula sama, satu polimorf tersintesis satu tidak — apakah head_synth
   membedakan dengan benar? **Ini klaim inti hipotesis dan sama sekali belum diuji.**
   Paling murah, paling menentukan.
4. **Naikkan n.** Resolver korpus penuh sudah siap di `matsynth/resolve.py`,
   sengaja ditunda. Sekarang alasannya untuk dibuka.
5. **Negatif asli dari lab.** Catatan sintesis yang **gagal** — tidak ada di
   dataset publik manapun karena jurnal tidak menerbitkan kegagalan. Ini satu-
   satunya cara keluar dari masalah proksi di poin 1.

---

## 9. Pertanyaan terbuka untuk didiskusikan

1. **Apakah "prediksi precursor" memang masalah nyata?** Kalau 98,8% resep trivial,
   AI tidak menambah nilai di sana. Mungkin masalah sesungguhnya adalah suhu,
   atmosfer, dan apakah fasanya terbentuk sama sekali — yang justru sengaja
   di-*out-of-scope*-kan di iterasi ini. Apakah scope-nya salah sejak awal?
2. **Bisakah synthesizability dipelajari tanpa negatif asli?** Semua kerja di
   bidang ini (termasuk CSLLM) memakai proksi "stabil tapi belum dibuat". Apakah
   ini secara fundamental cacat? Alternatif: positive-unlabeled learning?
3. **Apakah MACE-MP-0 encoder yang tepat?** Ia dilatih untuk energi/gaya, bukan
   untuk sintesis. Fitur yang berguna untuk memprediksi gaya belum tentu berguna
   untuk memprediksi apakah sesuatu bisa dibuat di furnace.
4. **Struktur DFT-relaxed vs struktur nyata.** Kita meng-encode geometri hasil
   relaksasi DFT pada 0 K. Sintesis terjadi di 500–900 °C. Seberapa relevan?
5. **Apakah argyrodite pilihan family yang tepat?** Korpus hampir kosong untuk
   keluarga ini. Family lain mungkin punya data 100× lebih banyak.

---

## 10. Reproduksi

```
.venv\Scripts\python.exe scripts\step1_load.py            # loader + resolusi
.venv\Scripts\python.exe scripts\step2_featurize.py       # Tahap A
.venv\Scripts\python.exe scripts\step3_invariance.py      # GATE
.venv\Scripts\python.exe scripts\step4_synth_head.py      # head_synth
.venv\Scripts\python.exe scripts\step5_precursor_knn.py   # head_prec
.venv\Scripts\python.exe scripts\step6_baseline_magpie.py # floor komposisi
.venv\Scripts\python.exe scripts\step7_polymorph.py       # uji polimorf
.venv\Scripts\python.exe scripts\step8_sanity_baselines.py # FLOOR CHECK ← kritis
.venv\Scripts\python.exe scripts\step9_temperature_triviality.py  # suhu: 3 floor
.venv\Scripts\python.exe scripts\step10_noise_decomposition.py # isi 117 °C: rute/polimorf
.venv\Scripts\python.exe scripts\step11_calcination_target.py  # target kalsinasi
.venv\Scripts\python.exe scripts\make_figures.py           # figure presentasi
```

Figure presentasi (`results/figures/`, 300 dpi, siap slide):

| file | temuan |
|---|---|
| `fig1_invariance_gate.png` | gate invariansi lolos, celah 12 orde besaran |
| `fig2_polymorph.png` | MACE membedakan polimorf 15/16; Magpie 0/16 |
| `fig3_confound_anion.png` | head_synth batal — aturan "ada O" menyamai model |
| `fig4_retrieval_floor.png` | retrieval = heuristik trivial; hero 98,8% |
| `fig5_temperature_signal.png` | suhu punya sinyal: 3 floor + histogram distribusi |

Env: Windows 10, tanpa GPU. venv Python 3.12 (via uv), torch 2.12.1+cpu,
mace-torch 0.3.16, pymatgen, matminer, scipy.
Hasil mentah: `results/step1..step8_*.json`.

---

## 11. Satu paragraf untuk dibaca dosen

Saya membangun pipeline lengkap yang mengganti encoder teks CSLLM dengan MACE
equivariant GNN, dan memverifikasi bahwa encoder-nya benar (invariansi rotasi
1.9e-16) serta bisa membedakan polimorf di mana representasi komposisi buta total
(15/16 pasangan vs 0/16). Namun setelah menambahkan dua *floor check* yang
seharusnya saya lakukan sejak awal, kedua klaim performa saya batal: retrieval
precursor tidak berbeda signifikan dari heuristik kimia trivial (p=0.92), karena
98,8% resep memang trivial; dan head_synth ter-*confound* — aturan satu-bit
"mengandung oksigen" menyamai atau mengalahkan model saya di semua family, karena
set negatif saya (oksida hipotetis dari OQMD) diambil dari distribusi kimia yang
berbeda dari set positif (sulfida nyata). Model belajar proses sampling saya,
bukan synthesizability. Ini bukan bug — ini kesalahan desain eksperimen yang
sekarang saya tahu cara perbaikinya, dan yang paling saya butuhkan untuk
memperbaikinya adalah data yang tidak ada di literatur mana pun: **catatan
sintesis yang gagal**.
