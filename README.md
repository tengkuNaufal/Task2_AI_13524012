# Task #2 Seleksi Laboratorium Inteligensi Buatan

**Local Search** untuk penjadwalan mata kuliah, dan **DTL / Logistic Regression / SVM**
*from scratch* untuk Kaggle *Loan Approval Prediction*.

Tengku Naufal Saqib · NIM **13524012** · STEI ITB

Dokumen lengkap (spesifikasi + write-up): [`docs/Task2_AI_13524012.pdf`](docs/Task2_AI_13524012.pdf)

---

## 1. Ringkasan

| Bagian | Isi | Hasil utama |
|---|---|---|
| **Local Search** | Penjadwalan 36 kelas ke 25 slot × 11 ruang. 4 varian Hill-Climbing, Simulated Annealing, Genetic Algorithm | f(X) turun **2015 → 16**, jadwal **feasible** (0 pelanggaran keras). SA paling efisien: kualitas terbaik dengan evaluasi 11× lebih sedikit daripada Random Restart |
| **DTL, LR, SVM** | CART, Logistic Regression, dan SVM ditulis dari nol dengan numpy | **CART 0,9160** akurasi 5-fold CV, identik dengan scikit-learn sampai empat desimal. LR 0,8973 · SVM 0,8998 |

---

## 2. Prasyarat

Python **3.10+** beserta:

```bash
pip install numpy pandas matplotlib scikit-learn pillow
```

- `scikit-learn` **hanya** dipakai sebagai pembanding, tidak pernah di jalur *from scratch*.
- `pillow` diperlukan untuk menyimpan animasi GIF Local Search.
- Untuk membuka notebook: `pip install jupyter`.

---

## 3. Struktur Repository

```
Task2_AI_13524012/
├── src/
│   ├── local_search/                  # PoC Local Search
│   │   ├── problem.py                 # state, constraints, objective, successor function
│   │   ├── algorithms.py              # 4 varian HC, Simulated Annealing, Genetic Algorithm
│   │   ├── visualize.py               # render jadwal, grafik, heatmap, animasi GIF
│   │   └── main.py                    # CLI
│   └── dtl_lr_svm/                    # implementasi DTL, LR, SVM
│       ├── preprocessing.py           # pembersihan, rekayasa fitur, encoding, scaling, RFF
│       ├── dtl.py                     # CART + cost-complexity pruning
│       ├── logistic_regression.py     # batch GD / mini-batch SGD / Adam
│       ├── svm.py                     # LinearSVM (Pegasos, sub-gradien) + KernelSVM (SMO)
│       ├── metrics.py                 # metrik & stratified K-fold CV dari nol
│       ├── plots.py                   # gambar pohon, kontur galat, ROC, matriks konfusi
│       └── main.py                    # runner eksperimen
├── notebooks/
│   ├── local_search/                  # notebook eksperimen Local Search
│   └── dtl_lr_svm/                    # notebook eksperimen DTL, LR, SVM
├── data/                              # train.csv, test.csv, sample_submission.csv
├── results/
│   ├── local_search/                  # grafik, jadwal terbaik, animasi, ringkasan JSON
│   └── dtl_lr_svm/                    # gambar, submission_*.csv, results.json
├── docs/
│   └── Task2_AI_13524012.pdf          # spesifikasi Local Search + write-up DTL/LR/SVM
└── README.md
```

---

## 4. Cara Menjalankan

Seluruh perintah dijalankan dari direktori **`src/`**.

### 4.1 Local Search

```bash
cd src
python -m local_search.main --algo all --seed 42 --plot            # keenam algoritma + grafik
python -m local_search.main --algo hc-steepest --seed 42           # satu algoritma
python -m local_search.main --algo sa --sa-t0 60 --sa-alpha 0.9997
python -m local_search.main --algo ga --pop 160 --mutation 0.5
python -m local_search.main --algo all --seed 42 --plot --animate  # + animasi GIF
```

| Opsi | Fungsi |
|---|---|
| `--algo` | `hc-steepest`, `hc-sideways`, `hc-stochastic`, `hc-random-restart`, `sa`, `ga`, atau `all` |
| `--seed N` | Seed RNG; state awal dan keputusan acak sepenuhnya reprodusibel |
| `--instance FILE.json` | Memuat instansi lain (default: instansi bawaan STEI ITB) |
| `--out DIR` | Direktori keluaran (default `results/local_search`) |
| `--max-iter`, `--max-sideways`, `--stochastic-iter`, `--restarts` | Parameter varian Hill-Climbing |
| `--sa-t0`, `--sa-alpha`, `--sa-tmin`, `--sa-iter` | Parameter Simulated Annealing |
| `--pop`, `--gen`, `--mutation`, `--crossover-rate`, `--elitism`, `--crossover` | Parameter Genetic Algorithm |
| `--plot` | Simpan grafik objective, diagnostik SA/GA, dan heatmap |
| `--animate` | Simpan animasi GIF perubahan state antar iterasi |
| `--quiet` | Tekan cetak jadwal lengkap |
| `--save-instance FILE` | Simpan instansi bawaan ke JSON lalu keluar |

### 4.2 DTL, LR, dan SVM

```bash
cd src
python -m dtl_lr_svm.main --task all       # CV + banding sklearn + gambar + submission
python -m dtl_lr_svm.main --task cv        # hanya validasi silang
python -m dtl_lr_svm.main --task plots     # hanya membuat gambar
python -m dtl_lr_svm.main --task smo       # verifikasi SVM dual (SMO) pada subsampel
python -m dtl_lr_svm.main --task submit    # hanya melatih ulang & membuat submission
```

Opsi lain: `--data DIR`, `--out DIR`, `--folds N`, `--seed N`, `--no-rff`,
`--submit-model {dtl,lr,svm}`.

### 4.3 Notebook

```bash
jupyter notebook notebooks/local_search/eksperimen_local_search.ipynb
jupyter notebook notebooks/dtl_lr_svm/eksperimen_dtl_lr_svm.ipynb
```

Keduanya sudah tersimpan **lengkap dengan seluruh keluaran dan gambarnya**, jadi dapat
dibaca tanpa dijalankan ulang. Notebook menemukan `src/` secara otomatis, asalkan
dijalankan dari direktorinya masing-masing.

---

## 5. Bagian I: Local Search

### 5.1 Rumusan masalah

Menjadwalkan **36 kelas mata kuliah** ke **25 slot waktu** (5 hari × 5 sesi) dan
**11 ruangan** (8 ruang kuliah + 3 laboratorium), untuk 10 dosen dan 8 kelompok mahasiswa.

**State**, vektor panjang tetap dengan satu gen per mata kuliah:

```
X = [ (t_0, r_0), (t_1, r_1), ..., (t_35, r_35) ]      t ∈ {0..24},  r ∈ {0..10}
```

Setiap state adalah jadwal utuh (*complete-state formulation*). Ukuran ruang pencarian
275³⁶ ≈ 1,2 × 10⁸⁷.

**Objective**, satu fungsi biaya yang diminimalkan:

```
f(X) = Σ w_h · v_h(X)  +  Σ w_s · v_s(X)          f(X) ≥ 0, makin kecil makin baik
```

| | Batasan | Bobot |
|---|---|---|
| **H1** | Bentrok ruangan | 100 |
| **H2** | Bentrok dosen | 100 |
| **H3** | Bentrok kelompok mahasiswa | 100 |
| **H4** | Kapasitas ruangan terlampaui | 60 |
| **H5** | Praktikum tidak di laboratorium | 80 |
| **S1** | Dosen dijadwalkan saat tidak bersedia | 10 |
| **S2** | Kelas pada sesi terakhir (15.00 sampai 17.00) | 3 |
| **S3** | Dosen mengajar > 2 kelas per hari | 5 |
| **S4** | Kelompok mahasiswa kuliah > 3 sesi per hari | 4 |
| **S5** | Jam kosong di jadwal harian mahasiswa | 2 |
| **S6** | Jam kosong di jadwal harian dosen | 2 |
| **S7** | Kursi menganggur (per kelipatan 10) | 1 |
| **S8** | Ruang lab dipakai kuliah teori | 4 |

Jadwal **feasible** bila H1 sampai H5 seluruhnya nol.

**Successor function**, tiga jenis *move* dengan total **1.854 tetangga** per iterasi:

| Move | Deskripsi | Jumlah |
|---|---|---|
| `change-time` | Pindahkan satu kelas ke slot waktu lain | 36 × 24 = 864 |
| `change-room` | Pindahkan satu kelas ke ruangan lain | 36 × 10 = 360 |
| `swap` | Tukar (slot, ruang) dua kelas | 630 |

**Initial state** dibangkitkan sepenuhnya random, tanpa perbaikan apa pun.

### 5.2 Hasil (seed 42, state awal identik, f = 2015)

| Algoritma | f akhir | Feasible | Iterasi | Evaluasi f(X) | Detik |
|---|---|---|---|---|---|
| Hill-Climbing (Steepest-Ascent) | 18 | ya | 41 | 77.866 | 3,48 |
| Hill-Climbing (Sideways Move) | **16** | ya | 83 | 155.734 | 7,14 |
| Hill-Climbing (Stochastic) | 75 | tidak | 30.000 | 30.001 | 1,40 |
| Hill-Climbing (Random Restart, 5×) | **16** | ya | 216 | 409.728 | 18,35 |
| Simulated Annealing | **16** | ya | 36.669 | 36.670 | 1,71 |
| Genetic Algorithm | 31 | ya | 400 | 32.080 | 1,55 |

Atas **10 state awal berbeda**:

| Algoritma | f rata-rata | Simpangan baku | % feasible |
|---|---|---|---|
| Hill-Climbing (Sideways) | **16,0** | 0,00 | 100% |
| Simulated Annealing | **16,0** | 0,00 | 100% |
| Hill-Climbing (Random Restart) | 16,2 | 0,63 | 100% |
| Genetic Algorithm | 21,0 | 3,30 | 100% |
| Hill-Climbing (Steepest-Ascent) | 45,7 | 41,57 | 60% |
| Hill-Climbing (Stochastic) | 64,2 | 46,84 | 40% |

Steepest-Ascent murni hanya menghasilkan jadwal feasible pada 6 dari 10 percobaan;
ketergantungannya pada state awal sangat besar. Sideways Move dan Simulated Annealing
sama-sama mencapai f = 16 di seluruh seed, tetapi SA melakukannya empat kali lebih cepat.

Berkas hasil: `results/local_search/` (grafik objective, diagnostik SA/GA, heatmap,
animasi GIF, jadwal terbaik, ringkasan JSON).

---

## 6. Bagian II: DTL, LR, dan SVM

Dataset: **28.800** baris latih, **7.200** baris uji, 11 atribut, target biner
`loan_status` (22,2% positif). Setelah pembersihan outlier, 12 fitur turunan, dan
encoding → **26 fitur**.

### 6.1 Hasil 5-fold stratified cross-validation

| Model | Akurasi | F1 | ROC-AUC |
|---|---|---|---|
| **DTL (from scratch)** | **0,9160 ± 0,0013** | 0,8001 | 0,9640 |
| DTL (sklearn) | 0,9160 ± 0,0014 | 0,7990 | 0,9640 |
| LR + RFF (from scratch) | 0,9047 ± 0,0032 | 0,7715 | 0,9576 |
| **SVM (from scratch, Pegasos)** | 0,8998 ± 0,0028 | 0,7727 | 0,9562 |
| LR (sklearn) | 0,8997 ± 0,0021 | 0,7736 | 0,9568 |
| SVM + RFF (from scratch) | 0,8995 ± 0,0044 | 0,7512 | 0,9477 |
| SVM (sklearn, LinearSVC) | 0,8993 ± 0,0022 | 0,7735 | 0,9565 |
| **LR (from scratch, Adam)** | 0,8973 ± 0,0025 | 0,7700 | 0,9550 |

Ketiga implementasi *from scratch* setara dengan scikit-learn (selisih < 0,001), sehingga
perbedaan performa antar algoritma murni berasal dari sifat algoritmanya.

### 6.2 Yang diimplementasikan

**Decision Tree Learning (CART).** Dipilih karena 11 dari 13 atribut kontinu; ID3 tidak
menanganinya dan koreksi bias kardinalitas C4.5 tidak relevan di sini. Pencarian split
memakai jumlah kumulatif atas nilai terurut sehingga kompleksitasnya O(d·n log n) per
simpul. Tersedia kriteria Gini dan entropi, bobot kelas, serta *cost-complexity pruning*.
Konfigurasi akhir: `max_depth=8`, `min_samples_leaf=5`, `criterion="entropy"`,
`ccp_alpha=5e-4` (32 daun).

**Logistic Regression.** Cross-entropy berbobot kelas + L2, sigmoid stabil secara numerik.
Tiga optimizer: batch GD, mini-batch SGD, dan **Adam** (Kingma & Ba, 2015).

**SVM.** Formulasi primal (hinge + L2) dengan optimizer **Pegasos** (Shalev-Shwartz dkk.,
2011) dan sub-gradien, serta formulasi dual dengan **SMO** (Platt, 1998). Kernel RBF pada
data penuh didekati lewat **Random Fourier Features** (Rahimi & Recht, 2007).

**Metrik & validasi.** Akurasi, presisi, recall, F1, ROC-AUC (Mann-Whitney U), matriks
konfusi, dan stratified K-fold, seluruhnya ditulis dari nol.

### 6.3 Submission

`results/dtl_lr_svm/submission_dtl.csv` (model terbaik), `submission_lr.csv`, dan
`submission_svm.csv`. Masing-masing keluaran **satu model tunggal**, tidak ada *ensemble*,
sesuai aturan kompetisi.

Skor leaderboard untuk `submission_dtl.csv`: **0,86013**.

---

## 7. Bonus yang Diimplementasikan

### 7.1 Local Search

**Seluruh varian Hill-Climbing, bukan hanya satu.** Keempatnya ada di
`src/local_search/algorithms.py` dan dapat dijalankan lewat `--algo`:

| Varian | Fungsi | Aturan pemilihan langkah |
|---|---|---|
| Basic (Steepest-Ascent) | `hill_climbing_steepest` | Evaluasi seluruh 1.854 tetangga, pindah ke f terkecil |
| Sideways Move | `hill_climbing_sideways` | Boleh melangkah mendatar (f sama) dengan kuota 40 langkah berturut-turut |
| Stochastic | `hill_climbing_stochastic` | Undi satu tetangga acak, terima hanya bila lebih baik |
| Random Restart | `hill_climbing_random_restart` | Jalankan Steepest-Ascent dari 5 state awal acak, ambil terbaik |

**Visualisasi dan animasi proses pencarian.** Dibangun oleh `src/local_search/visualize.py`,
dihasilkan dengan `--plot` dan `--animate`:

| Berkas di `results/local_search/` | Isi |
|---|---|
| `animasi_pencarian.gif` | Animasi perubahan state antar iterasi: okupansi ruangan berubah seiring pencarian, berdampingan dengan penanda posisi iterasi pada kurva f(X) |
| `objective_all_seed42.png` | Nilai objective terhadap iterasi untuk keenam algoritma sekaligus. Dua panel: sumbu-x logaritmik untuk seluruh lintasan, dan perbesaran 300 iterasi pertama |
| `sa_diagnostics.png` | Tiga panel Simulated Annealing: f(X), suhu, dan peluang penerimaan exp(-dE/T) untuk setiap langkah yang memperburuk |
| `ga_diagnostics.png` | Perkembangan populasi GA tiap generasi: f terbaik, rata-rata, dan terburuk |
| `heatmap_awal.png`, `heatmap_akhir.png` | Okupansi ruangan sebelum dan sesudah pencarian; sel bertumpuk berarti bentrok |

### 7.2 DTL, LR, dan SVM

**Algoritma optimasi tambahan di luar materi kelas.** Empat algoritma, masing-masing dengan
penjelasan cara kerja pada docstring modulnya dan pada write-up Bagian 3 sampai 5:

| Algoritma | Dipakai untuk | Kode | Referensi |
|---|---|---|---|
| **Adam** (Adaptive Moment Estimation) | Logistic Regression | `logistic_regression.py` | D. P. Kingma dan J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015. https://arxiv.org/abs/1412.6980 |
| **Pegasos** (primal estimated sub-gradient solver) | SVM | `svm.py` | S. Shalev-Shwartz dkk., "Pegasos: primal estimated sub-gradient solver for SVM," *Mathematical Programming*, vol. 127, no. 1, hlm. 3 sampai 30, 2011. doi:10.1007/s10107-010-0420-4 |
| **Random Fourier Features** | Aproksimasi kernel RBF untuk SVM dan LR | `preprocessing.py` | A. Rahimi dan B. Recht, "Random Features for Large-Scale Kernel Machines," NeurIPS 2007, hlm. 1177 sampai 1184 |
| **Cost-complexity pruning** | Decision Tree | `dtl.py` | L. Breiman dkk., *Classification and Regression Trees*, Wadsworth, 1984, Bab 3 |

Ringkas cara kerjanya:

- **Adam** memelihara rata-rata bergerak momen pertama `m` (arah gradien) dan momen kedua `v`
  (skala kuadrat gradien), keduanya dikoreksi bias, lalu melangkah sebesar
  `alpha * m_hat / (sqrt(v_hat) + eps)`. Karena tiap parameter memperoleh laju belajar
  efektifnya sendiri, Adam tidak terhambat fitur dengan skala gradien kecil dan mencapai
  galat yang sama dengan batch GD dalam epoch jauh lebih sedikit.
- **Pegasos** memakai laju belajar yang meluruh sendiri `eta_t = 1/(lambda*t)` lalu
  memproyeksikan `w` ke bola berjari-jari `1/sqrt(lambda)`, tempat solusi optimum dijamin
  berada. Jumlah iterasi untuk mencapai galat eps adalah `O(1/(lambda*eps))`, tidak
  bergantung pada jumlah data.
- **Random Fourier Features** mendekati kernel Gaussian sebagai hasil kali dalam pada ruang
  eksplisit berdimensi D lewat `z(x) = sqrt(2/D) * cos(Wx + b)` dengan `W ~ N(0, 2*gamma)`
  dan `b ~ U(0, 2*pi)`, sehingga biaya turun dari `O(n^2)` menjadi `O(n*D)`. Matriks Gram
  penuh 28.800 x 28.800 akan memerlukan sekitar 6,6 GB, jadi kernel non-linier mustahil
  dijalankan tanpa aproksimasi ini.
- **Cost-complexity pruning** memangkas simpul secara *weakest-link* bila
  `R(t) + alpha <= R(T_t) + alpha*|T_t|`, dengan risiko dinormalisasi terhadap jumlah sampel
  mengikuti konvensi scikit-learn. Pohon berkedalaman 12 dengan 229 daun terpangkas menjadi
  20 daun dan akurasinya justru naik dari 0,9113 ke 0,9153.

Sebagai pelengkap, formulasi dual SVM juga diimplementasikan dengan **SMO** (J. C. Platt,
"Sequential Minimal Optimization," Microsoft Research, Tech. Rep. MSR-TR-98-14, 1998) di
`svm.py`, dipakai untuk memverifikasi bahwa versi dual bekerja pada subsampel.

**Gambar percabangan Decision Tree.** `results/dtl_lr_svm/decision_tree.png`, dibuat oleh
`plots.py:plot_tree` langsung dari objek `Node` hasil implementasi sendiri, bukan dari
`sklearn.tree.plot_tree`. Warna simpul menyatakan proporsi kelas positif. Struktur teksnya
juga tersedia di `results/dtl_lr_svm/tree_structure.txt`.

**Visualisasi proses training Logistic Regression.**
`results/dtl_lr_svm/lr_loss_contour.png` menampilkan kontur fungsi galat beserta lintasan
parameter selama training untuk tiga optimizer (batch GD, mini-batch SGD, Adam). Supaya
konturnya benar-benar fungsi galat yang dioptimasi dan bukan proyeksi dari ruang 26 dimensi,
model dilatih ulang hanya dengan dua fitur terkuat dan tanpa intercept, sehingga ruang
parameternya memang berdimensi dua. Kurva galat per epoch ada di `lr_loss_curves.png`.

---

## 8. Reprodusibilitas

Seluruh keacakan dikendalikan lewat seed eksplisit (`random.Random` untuk Local Search,
`numpy.random.default_rng` untuk bagian ML). Menjalankan perintah yang sama dengan seed
yang sama menghasilkan angka yang identik dengan yang dilaporkan di atas dan di dalam PDF.

