# UAS Kecerdasan Buatan — Prediksi Diabetes

Proyek ini membangun model klasifikasi untuk memprediksi risiko diabetes menggunakan dua algoritma *machine learning*: **Decision Tree** dan **Random Forest**, berdasarkan dataset Pima Indians Diabetes.

## Struktur Repository

```
UAS-KecerdasanBuatan/
├── README.md
├── Laporan_uas.md       (laporan lengkap)
├── uas_model.ipynb       (notebook implementasi)
└── data/
    ├── dataset           (file diabetes.csv)
    └── Jurnal             (referensi jurnal ilmiah)
```

## Ringkasan Proyek

- **Dataset:** Pima Indians Diabetes Dataset (768 baris, 9 kolom) — sumber: Kaggle/UCI Machine Learning Repository
- **Target:** Klasifikasi biner — Diabetes (1) vs Tidak Diabetes (0)
- **Algoritma:** Decision Tree Classifier dan Random Forest Classifier
- **Hasil terbaik:** Random Forest Classifier (Accuracy 77%, Precision 70%, F1-Score 0.44)

## Langkah Pengerjaan

**1. Persiapan data**
Dataset diunduh dari Kaggle (`diabetes.csv`), lalu dimuat ke Google Colab menggunakan pustaka Pandas.

**2. Exploratory Data Analysis (EDA)**
- Memeriksa struktur data (`df.info()`, `df.describe()`)
- Memeriksa nilai kosong/tidak wajar (misalnya Glucose atau BMI bernilai 0)
- Membuat visualisasi distribusi data dan heatmap korelasi antar fitur
- Mendeteksi ketidakseimbangan kelas (*imbalanced classes*) pada kolom target

**3. Data Preparation**
- Membersihkan data yang hilang/tidak wajar (`dropna()`, `drop_duplicates()`)
- Menstandardisasi fitur numerik menggunakan `StandardScaler`
- Membagi data menjadi data latih dan data uji (80:20) dengan `train_test_split` (`random_state=42`, `stratify=y`)

**4. Modeling**
- Melatih model **Decision Tree Classifier**
- Melatih model **Random Forest Classifier**
- Memvisualisasikan struktur pohon keputusan (Decision Tree)

**5. Evaluation**
- Menghitung Confusion Matrix untuk kedua model
- Menghitung metrik Accuracy, Precision, Recall, dan F1-Score
- Membandingkan performa kedua model dan menentukan model terbaik

**6. Kesimpulan**
Menyusun ringkasan hasil, kelebihan/keterbatasan tiap model, dan rekomendasi pengembangan lebih lanjut (dituliskan lengkap di `Laporan_uas.md`).

## Cara Menjalankan

1. Buka `uas_model.ipynb` di Google Colab
2. Upload file dataset (`diabetes.csv`) ke sesi Colab
3. Jalankan seluruh cell secara berurutan (**Runtime → Run all**)
4. Hasil EDA, model, dan evaluasi akan tampil di output notebook

## Referensi

Referensi ilmiah lengkap (format APA) tercantum di Bab 9 pada `Laporan_uas.md`.

## Kontributor

- [Nama Anggota 1]
- [Nama Anggota 2] *(opsional)*
