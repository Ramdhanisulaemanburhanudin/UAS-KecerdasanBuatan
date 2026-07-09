# Laporan Analisis Prediksi Diabetes Menggunakan Machine Learning

## 1. Judul Proyek

**Prediksi Penyakit Diabetes Menggunakan Algoritma Decision Tree dan Random Forest**

**Nama Kelompok:**
- [Nama Anggota 1]
- [Nama Anggota 2] *(opsional, maksimal 2 orang)*

**Domain Proyek (Latar Belakang):**
Diabetes melitus merupakan salah satu penyakit kronis dengan prevalensi yang terus meningkat di seluruh dunia. Menurut World Health Organization (WHO), diabetes menjadi salah satu penyebab utama kematian akibat penyakit tidak menular, dan jumlah penderitanya diperkirakan mencapai ratusan juta jiwa secara global. Keterlambatan diagnosis diabetes dapat menyebabkan komplikasi serius seperti gangguan jantung, ginjal, dan saraf. Oleh karena itu, dibutuhkan sebuah sistem prediksi berbasis data yang dapat membantu mendeteksi risiko diabetes sejak dini berdasarkan indikator kesehatan seseorang, sehingga penanganan medis dapat dilakukan lebih cepat.

Proyek ini memanfaatkan teknik *machine learning*, khususnya algoritma **Decision Tree** dan **Random Forest**, untuk membangun model klasifikasi yang mampu memprediksi apakah seseorang berisiko menderita diabetes berdasarkan data diagnostik seperti kadar glukosa, tekanan darah, BMI, dan riwayat kehamilan.

---

## 2. Business Understanding

**Permasalahan Dunia Nyata dan Literatur Review:**
Diagnosis diabetes secara konvensional membutuhkan pemeriksaan laboratorium yang memakan waktu dan tidak selalu dapat diakses oleh semua kalangan masyarakat, terutama di daerah dengan fasilitas kesehatan terbatas. Penelitian sebelumnya menunjukkan bahwa algoritma *machine learning* seperti Random Forest dan Decision Tree mampu memberikan akurasi yang kompetitif dalam memprediksi diabetes berdasarkan data diagnostik non-invasif (Apriliah & Haryati, 2021; Aditya et al., 2024).

**Tujuan Proyek:**
Membangun model klasifikasi yang dapat memprediksi kemungkinan seseorang menderita diabetes berdasarkan delapan indikator kesehatan, serta membandingkan performa dua algoritma untuk menentukan model terbaik.

**Siapa User/Pengguna Sistem:**
- Tenaga medis (dokter, perawat) sebagai alat bantu skrining awal
- Peneliti atau mahasiswa di bidang kesehatan dan data science
- Masyarakat umum sebagai alat edukasi kesadaran risiko diabetes

**Solusi dan Manfaat Implementasi AI:**
Dengan model prediksi berbasis AI, proses skrining awal risiko diabetes dapat dilakukan lebih cepat dan efisien tanpa memerlukan pemeriksaan laboratorium yang kompleks. Model ini juga dapat menjadi dasar pengembangan sistem pendukung keputusan klinis (*clinical decision support system*) di masa depan.

---

## 3. Data Understanding

**Sumber Data:**
Dataset yang digunakan adalah **Pima Indians Diabetes Dataset**, yang bersumber dari Kaggle (diunduh dengan nama file `diabetes (1).csv`), berasal dari National Institute of Diabetes and Digestive and Kidney Diseases (Smith et al., 1988).

**Ukuran dan Format Data:**
Dataset terdiri dari **768 baris data (entries)** dan **9 kolom**, dalam format `.csv`. Seluruh data bertipe numerik (7 kolom `int64` dan 2 kolom `float64`), tanpa data kategorik.

**Deskripsi Setiap Fitur (Atribut):**

| Fitur | Deskripsi |
|---|---|
| Pregnancies | Jumlah kehamilan yang pernah dialami pasien |
| Glucose | Konsentrasi glukosa plasma pada tes toleransi glukosa oral 2 jam (mg/dL) |
| BloodPressure | Tekanan darah diastolik (mm Hg) |
| SkinThickness | Ketebalan lipatan kulit triceps (mm) |
| Insulin | Kadar insulin serum 2 jam (mu U/ml) |
| BMI | Indeks massa tubuh (berat dalam kg / (tinggi dalam m)²) |
| DiabetesPedigreeFunction | Skor yang merepresentasikan riwayat/silsilah keturunan diabetes |
| Age | Usia pasien (tahun) |
| Outcome | Target klasifikasi biner: 1 = Diabetes, 0 = Tidak Diabetes |

**Tipe Data dan Target Klasifikasi:**
Ini merupakan kasus **klasifikasi biner (binary classification)**, dengan kolom `Outcome` sebagai target/label yang akan diprediksi (0 = tidak diabetes, 1 = diabetes).

---

## 4. Exploratory Data Analysis (EDA)

**Insight Awal dari Struktur Data:**
Pemeriksaan awal menggunakan `df.info()` menunjukkan bahwa dataset berisi 768 baris tanpa ada nilai *null* eksplisit (*Non-Null Count* = 768 pada seluruh kolom). Namun demikian, ditemukan bahwa beberapa fitur memiliki nilai 0 yang secara medis tidak wajar (misalnya Glucose atau BloodPressure bernilai 0), yang sesungguhnya merepresentasikan data hilang (*missing value* tersembunyi).

**Visualisasi Distribusi Data:**
Distribusi masing-masing fitur numerik divisualisasikan menggunakan histogram untuk melihat sebaran nilai (misalnya Glucose cenderung berdistribusi normal di sekitar 120, sementara Insulin dan SkinThickness memiliki banyak nilai 0/kosong).

**Analisis Korelasi Antar Fitur:**
Heatmap korelasi menunjukkan bahwa fitur **Glucose** memiliki korelasi paling kuat terhadap `Outcome`, diikuti oleh **BMI** dan **Age**. Hal ini konsisten dengan pengetahuan medis bahwa kadar gula darah merupakan indikator utama diabetes.

**Deteksi Data Tidak Seimbang (Imbalanced Classes):**
Distribusi kelas target menunjukkan proporsi yang tidak seimbang, dengan jumlah pasien **tidak diabetes (Outcome = 0)** jauh lebih banyak dibandingkan pasien **diabetes (Outcome = 1)**. Hal ini penting diperhatikan karena dapat menyebabkan model bias terhadap kelas mayoritas, sehingga metrik seperti *Recall* dan *F1-Score* menjadi lebih relevan dibandingkan hanya *Accuracy*.

---

## 5. Data Preparation

Pada tahap persiapan data, dataset diabetes yang diunduh (`diabetes (1).csv`) pertama-tama dimuat ke dalam lingkungan Python menggunakan pustaka Pandas. Setelah pemuatan, dilakukan inspeksi awal terhadap data, termasuk melihat beberapa baris pertama (`df.head()`) dan informasi umum dataset (`df.info()`) untuk memahami struktur dan tipe data setiap kolom.

Pengecekan terhadap *missing values* (nilai yang hilang) dan data duplikat adalah langkah krusial. Dalam kasus ini, jika ditemukan, baris yang mengandung *missing values* atau duplikasi akan dihapus (`dropna()` dan `drop_duplicates()`). Tahap ini penting untuk memastikan kualitas data dan mencegah bias pada model.

Selanjutnya, fitur-fitur numerik pada dataset dinormalisasi/distandardisasi menggunakan `StandardScaler`. Standardisasi mengubah data sedemikian rupa sehingga memiliki rata-rata nol dan standar deviasi satu. Hal ini penting untuk algoritma pembelajaran mesin yang sensitif terhadap skala fitur, seperti yang berbasis jarak, agar setiap fitur memberikan kontribusi yang setara pada proses pembelajaran model.

Terakhir, dataset dibagi menjadi set pelatihan (*training set*) dan set pengujian (*testing set*) dengan proporsi 80:20. Pembagian ini dilakukan menggunakan `train_test_split` dengan `random_state=42` untuk memastikan hasil yang konsisten dan dapat direproduksi, serta `stratify=y` untuk menjaga proporsi kelas target yang sama di kedua set (penting untuk data yang mungkin tidak seimbang).

---

## 6. Modeling

Dalam studi kasus prediksi diabetes ini, dua algoritma klasifikasi berbasis pohon dipilih dan diimplementasikan: **Decision Tree Classifier** dan **Random Forest Classifier**.

### 6.1 Model 1 — Decision Tree Classifier

**Cara Kerja:**
Decision Tree adalah algoritma *supervised learning* yang bekerja dengan memecah dataset menjadi subset yang lebih kecil secara rekursif (disebut *recursive binary splitting*), sambil membangun struktur pohon keputusan. Setiap **node internal** merepresentasikan sebuah uji terhadap satu atribut/fitur (misalnya "Glucose ≤ 0.318?"), setiap **cabang** merepresentasikan hasil dari uji tersebut (ya/tidak), dan setiap **node daun (leaf node)** merepresentasikan label kelas akhir (Diabetes / Non-Diabetes).

Pemilihan fitur dan titik pemisahan (*split point*) terbaik pada setiap node dilakukan dengan mengukur tingkat *impurity* menggunakan **Gini Index**: `Gini = 1 - sum(p_i^2)`, di mana p_i adalah proporsi sampel kelas i pada node tersebut. Algoritma memilih split yang menghasilkan penurunan Gini Index terbesar (paling memisahkan kelas secara bersih).

**Parameter yang digunakan:** `random_state=42` untuk reprodusibilitas hasil.

**Hasil Implementasi:**
Berdasarkan visualisasi pohon yang dihasilkan, model memilih **Glucose** sebagai fitur pada *root node* (Gini = 0.454, 614 sampel, distribusi kelas [430, 184]), yang berarti Glucose merupakan fitur paling signifikan dalam menentukan klasifikasi diabetes pada model ini. Dari root node, pohon kemudian bercabang berdasarkan ambang batas Glucose, dan berlanjut ke fitur-fitur lain seperti BMI, Age, DiabetesPedigreeFunction, dan SkinThickness pada level-level berikutnya, membentuk struktur pohon yang cukup dalam.

**Alasan pemilihan:** Decision Tree dipilih karena interpretasinya yang mudah (jalur keputusan dapat divisualisasikan dan ditelusuri secara langsung), tidak memerlukan asumsi distribusi data, serta mampu menangani hubungan non-linear antar fitur. Dalam konteks medis, kemampuan menelusuri jalur keputusan ini bernilai karena tenaga medis dapat memahami "alasan" di balik sebuah prediksi.

### 6.2 Model 2 — Random Forest Classifier

**Cara Kerja:**
Random Forest adalah metode *ensemble learning* yang membangun sejumlah besar Decision Tree secara paralel, masing-masing dilatih pada:
1. **Subsampel data** yang berbeda, diambil secara acak dengan pengembalian dari data latih (teknik *bootstrap aggregating / bagging*)
2. **Subset fitur** acak pada setiap split (bukan seluruh fitur), untuk menjaga agar antar-pohon tidak terlalu mirip satu sama lain

Setiap pohon menghasilkan prediksinya sendiri, kemudian **hasil akhir ditentukan lewat voting mayoritas** (untuk klasifikasi) dari seluruh pohon dalam forest. Pendekatan ini secara matematis menurunkan *variance* model tanpa menaikkan *bias* secara signifikan, sehingga hasil prediksi menjadi lebih stabil dibandingkan satu Decision Tree tunggal.

**Parameter yang digunakan:** `random_state=42` untuk reprodusibilitas hasil (jumlah pohon/`n_estimators` menggunakan nilai default scikit-learn, yaitu 100 pohon).

**Alasan pemilihan:** Random Forest dipilih sebagai model pembanding karena secara teoritis mampu mengatasi kelemahan utama Decision Tree tunggal, yaitu *overfitting* dan ketidakstabilan terhadap perubahan kecil pada data. Dengan menggabungkan banyak pohon, Random Forest umumnya menghasilkan generalisasi yang lebih baik pada data baru (data uji), yang sangat penting untuk aplikasi medis di mana keandalan prediksi menjadi prioritas.

Kedua model dilatih menggunakan set pelatihan yang telah distandardisasi (`X_train` dan `y_train`) dan kemudian digunakan untuk membuat prediksi pada set pengujian (`X_test`).

**Visualisasi Model:**
Pohon keputusan dari model Decision Tree divisualisasikan (lihat lampiran) untuk menunjukkan jalur pengambilan keputusan, dimulai dari fitur **Glucose** sebagai *root node*. Berbeda dengan Decision Tree, struktur internal Random Forest (100 pohon) tidak divisualisasikan secara individual karena kompleksitasnya, namun performa gabungannya dievaluasi secara agregat pada Bab 7.

### 6.3 Perbandingan Karakteristik Kedua Model

| Aspek | Decision Tree | Random Forest |
|---|---|---|
| Struktur | 1 pohon tunggal | Kumpulan (ensemble) 100 pohon |
| Cara prediksi | Mengikuti satu jalur keputusan | Voting mayoritas dari seluruh pohon |
| Interpretasi | Sangat mudah divisualisasikan & ditelusuri | Sulit ditelusuri karena gabungan banyak pohon |
| Risiko overfitting | Tinggi, terutama jika pohon dalam | Lebih rendah karena efek averaging antar pohon |
| Waktu komputasi | Cepat | Lebih lambat (melatih banyak pohon) |
| Stabilitas terhadap perubahan data | Rendah (sensitif) | Tinggi (lebih stabil) |

---

## 7. Evaluation & Visualization

Evaluasi model adalah tahapan penting untuk mengukur seberapa baik model yang telah dilatih dalam membuat prediksi. Metrik evaluasi yang digunakan adalah **Confusion Matrix**, **Accuracy**, **Precision**, **Recall**, dan **F1-Score**.

**Confusion Matrix** adalah tabel yang merangkum kinerja model klasifikasi pada sekumpulan data uji. Ini menunjukkan jumlah *True Positives* (TP), *True Negatives* (TN), *False Positives* (FP), dan *False Negatives* (FN). Untuk kedua model, Confusion Matrix divisualisasikan menggunakan *heatmap* dari Seaborn, memberikan representasi visual yang jelas tentang jumlah prediksi yang benar dan salah.

**Hasil Evaluasi:**

**Decision Tree Classifier** — Confusion Matrix:

| | Prediksi Negatif | Prediksi Positif |
|---|---|---|
| **Aktual Negatif** | 87 | 23 |
| **Aktual Positif** | 32 | 12 |

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.73 | 0.79 | 0.76 | 110 |
| 1 | 0.34 | 0.27 | 0.30 | 44 |
| **accuracy** | | | **0.64** | 154 |
| macro avg | 0.54 | 0.53 | 0.53 | 154 |
| weighted avg | 0.62 | 0.64 | 0.63 | 154 |

**Random Forest Classifier** — Confusion Matrix:

| | Prediksi Negatif | Prediksi Positif |
|---|---|---|
| **Aktual Negatif** | 104 | 6 |
| **Aktual Positif** | 30 | 14 |

| | precision | recall | f1-score | support |
|---|---|---|---|---|
| 0 | 0.78 | 0.95 | 0.86 | 110 |
| 1 | 0.70 | 0.32 | 0.44 | 44 |
| **accuracy** | | | **0.77** | 154 |
| macro avg | 0.74 | 0.63 | 0.65 | 154 |
| weighted avg | 0.76 | 0.77 | 0.74 | 154 |

**Perbandingan Metrik Utama:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 0.6429 | 0.3429 | 0.2727 | 0.3040 |
| Random Forest | 0.7662 | 0.7000 | 0.3182 | 0.4375 |

**Analisis Perbandingan Performa (Fokus pada Recall/F1-Score):**
Dalam konteks medis, seperti prediksi diabetes, metrik Recall dan F1-Score memiliki kepentingan yang sangat krusial. Recall (juga dikenal sebagai sensitivitas) mengukur proporsi aktual pasien positif yang teridentifikasi dengan benar oleh model (TP / (TP + FN)). *False Negatives* (FN) — kasus di mana seseorang sebenarnya menderita diabetes tetapi diprediksi sehat — sangat berbahaya karena dapat menunda diagnosis dan pengobatan yang diperlukan, berpotensi menyebabkan komplikasi serius.

F1-Score adalah rata-rata harmonik dari Precision dan Recall, yang memberikan keseimbangan antara kedua metrik tersebut. Ini sangat berguna ketika distribusi kelas tidak seimbang dan kita membutuhkan metrik yang menghukum prediksi yang salah secara seimbang untuk kelas positif dan negatif.

Dari hasil evaluasi, dapat diamati bahwa:
- **Decision Tree** cenderung memiliki Recall yang bervariasi tergantung kedalaman pohon dan bagaimana *overfitting* ditangani. Jika pohon terlalu kompleks, ia mungkin memiliki Recall yang tinggi pada data pelatihan tetapi rendah pada data uji.
- **Random Forest** secara konsisten menunjukkan performa yang lebih baik dalam hal Accuracy, Precision, dan F1-Score dibandingkan dengan Decision Tree tunggal (Accuracy 0.77 vs 0.64; F1-Score 0.44 vs 0.30). Hal ini karena Random Forest mengurangi bias dan varians dengan menggabungkan prediksi dari banyak pohon keputusan yang berbeda.

**Model terbaik dan alasannya:** Berdasarkan seluruh metrik evaluasi, **Random Forest Classifier dipilih sebagai model terbaik**, karena secara konsisten unggul pada Accuracy (0.77 vs 0.64), Precision (0.70 vs 0.34), dan F1-Score (0.44 vs 0.30) dibandingkan Decision Tree. Meskipun nilai Recall kelas positif pada Random Forest (0.32) belum terlalu tinggi, model ini tetap lebih seimbang dan lebih dapat diandalkan dibandingkan Decision Tree yang cenderung *overfitting* dan kurang stabil.

**Rincian per model:**

*Decision Tree Classifier* — dari 44 pasien yang benar-benar diabetes pada data uji, model hanya berhasil mengenali 12 dengan benar (Recall 0.27), sementara 32 pasien diabetes justru diprediksi sehat (*False Negative*). Precision kelas diabetes juga rendah (0.34), artinya dari semua prediksi "diabetes" yang dikeluarkan model, sebagian besar ternyata salah (*False Positive* tinggi, 23 kasus). Ini mengindikasikan model Decision Tree pada konfigurasi ini kurang optimal dan cenderung membuat keputusan yang kurang presisi pada kelas minoritas.

*Random Forest Classifier* — dari 44 pasien diabetes, model berhasil mengenali 14 dengan benar (Recall 0.32, sedikit lebih baik dari Decision Tree), dengan jumlah *False Positive* yang jauh lebih rendah (hanya 6 kasus, dibanding 23 pada Decision Tree). Precision kelas diabetes meningkat signifikan menjadi 0.70, artinya ketika model ini memprediksi "diabetes", prediksinya jauh lebih dapat dipercaya dibandingkan Decision Tree. Pada kelas non-diabetes, Random Forest juga unggul telak (Recall 0.95 vs 0.79), menunjukkan model ini sangat baik mengenali pasien yang benar-benar sehat.

**Catatan kritis:** Meskipun Random Forest lebih unggul secara keseluruhan, Recall kelas diabetes pada kedua model masih tergolong rendah (di bawah 0.35). Ini berarti sebagian besar pasien diabetes pada data uji belum berhasil terdeteksi oleh kedua model — sebuah keterbatasan penting yang perlu ditindaklanjuti (lihat Bab 8, Rekomendasi Perbaikan), mengingat *False Negative* pada kasus medis memiliki konsekuensi yang serius.

---

## 8. Kesimpulan dan Rekomendasi

**Ringkasan Hasil Modeling dan Evaluasi:**
Berdasarkan analisis dan implementasi dua model klasifikasi untuk prediksi diabetes, dapat disimpulkan bahwa Random Forest Classifier secara umum menunjukkan performa yang lebih superior dibandingkan Decision Tree Classifier tunggal, terutama dalam hal Accuracy, Precision, dan F1-Score. Robustitas dan kemampuan Random Forest untuk mengatasi *overfitting* menjadikannya pilihan yang lebih aman dan efektif untuk tugas prediksi medis di mana *False Negatives* sangat berisiko.

**Apakah Tujuan Proyek Tercapai?**
Tujuan proyek untuk membangun model klasifikasi prediksi diabetes dan membandingkan dua algoritma **telah tercapai**. Model Random Forest berhasil mencapai akurasi 77% pada data uji, meskipun performa pada kelas minoritas (pasien diabetes) masih dapat ditingkatkan.

**Kelebihan dan Keterbatasan Model:**

*Decision Tree Classifier:*
- Kelebihan: Mudah diinterpretasikan, dapat memvisualisasikan jalur keputusan, memerlukan sedikit persiapan data (tidak perlu *scaling* fitur). Sangat berguna untuk mendapatkan insight awal tentang faktor-faktor penentu.
- Keterbatasan: Rentan terhadap *overfitting* jika tidak dikontrol dengan baik (misalnya dengan membatasi kedalaman pohon), bisa menjadi tidak stabil (perubahan kecil pada data dapat menghasilkan pohon yang sangat berbeda).

*Random Forest Classifier:*
- Kelebihan: Umumnya memberikan akurasi yang lebih tinggi, sangat efektif dalam mengatasi *overfitting* melalui *ensemble learning*, dapat menangani data dengan banyak fitur, kurang rentan terhadap *noise* dalam data.
- Keterbatasan: Kurang *interpretable* dibandingkan Decision Tree tunggal karena merupakan gabungan banyak pohon, membutuhkan sumber daya komputasi yang lebih besar.

**Rekomendasi Perbaikan:**
1. **Hyperparameter Tuning:** Menggunakan teknik seperti *Grid Search* atau *Random Search* untuk mengoptimalkan hyperparameter dari kedua model (misalnya `max_depth` untuk Decision Tree, `n_estimators` dan `max_features` untuk Random Forest).
2. **Penanganan Data Imbalanced:** Mengingat ketidakseimbangan kelas yang ditemukan pada tahap EDA, teknik seperti *oversampling* (SMOTE), *undersampling*, atau penggunaan bobot kelas (`class_weight`) dapat diimplementasikan untuk mencegah model bias terhadap kelas mayoritas.
3. **Feature Engineering:** Mencoba membuat fitur baru dari fitur yang sudah ada (misalnya rasio BMI terhadap usia) yang mungkin memberikan informasi tambahan yang berguna bagi model.
4. **Eksplorasi Model Lain:** Menguji algoritma klasifikasi lain seperti Support Vector Machine (SVM), K-Nearest Neighbor (KNN), Logistic Regression, atau model berbasis *boosting* seperti Gradient Boosting atau XGBoost.
5. **Dataset Lebih Besar:** Mengumpulkan atau menggunakan dataset dengan jumlah sampel lebih besar untuk meningkatkan generalisasi model, terutama pada kelas minoritas.
6. **Validasi Silang (Cross-Validation):** Menggunakan teknik validasi silang seperti *k-fold cross-validation* untuk mendapatkan estimasi performa model yang lebih *robust* dan tidak terlalu bergantung pada satu pembagian *train-test* saja.

Dengan menerapkan rekomendasi ini, diharapkan model prediksi diabetes dapat lebih akurat, *robust*, dan dapat diandalkan dalam aplikasi klinis.

---

## 9. Referensi

Aditya, M. F., Pramuntadi, A., Wijaya, D. P., & Wicaksono, Y. (2024). Implementasi Metode Decision Tree pada Prediksi Penyakit Diabetes Melitus Tipe 2. *MALCOM: Indonesian Journal of Machine Learning and Computer Science*, 4(3), 1104–1110. https://doi.org/10.57152/malcom.v4i3.1284

Apriliah, D., & Haryati. (2021). Prediksi Kemungkinan Diabetes pada Tahap Awal Menggunakan Algoritma Klasifikasi Random Forest. *Sistemasi: Jurnal Sistem Informasi*, 10(1), 163–171.

Nurussakinah, N., & Faisal, M. (2023). Klasifikasi Penyakit Diabetes Menggunakan Algoritma Decision Tree. *Jurnal Informatika*, 10(2), 143–149. https://doi.org/10.31294/inf.v10i2.15989

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Symposium on Computer Applications and Medical Care*, 261–265. IEEE Computer Society Press.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

---

## 10. Lampiran (Opsional)

- Dataset mentah: `diabetes (1).csv` (768 baris, 9 kolom)
- Notebook analisis: `uas_model.ipynb`
- Grafik tambahan: Visualisasi pohon keputusan (Decision Tree), Confusion Matrix Decision Tree, Confusion Matrix Random Forest
