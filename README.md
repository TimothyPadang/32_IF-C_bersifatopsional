Prediksi Harga Mobil Bekas di Indonesia (Regresi)

Project Data Science IF-C (UPN). Notebook ini memodelkan harga penawaran mobil bekas dari karakteristik kendaraannya, memakai regresi linear (OLS, Ridge, Lasso).

Tujuan

Menjawab pertanyaan: seberapa baik karakteristik mobil yang tersedia dapat dipakai untuk memperkirakan harga penawaran mobil bekas?

Jenis masalah: supervised learning, regresi
Target: price (Rp)
Metrik evaluasi: MSE, RMSE, MAE, dan R²
Dataset

File used_car.csv berisi daftar iklan mobil bekas di Indonesia: 609 baris, 20 kolom.

Kelompok	Kolom
Identitas	car name, brand, year, mileage (km)
Kategorikal	location, transmission, plate type
Fitur biner (0/1)	rear camera, sun roof, auto retract mirror, electric parking brake, map navigator, vehicle stability control, keyless push start, sports mode, 360 camera view, power sliding door, auto cruise control
Target	price (Rp)
Dikeluarkan	instalment (Rp|Monthly), karena berisiko data leakage (cicilan dihitung dari harga)

Catatan bias sampel:

Sekitar 81% data berasal dari Jabodetabek (495 dari 609 baris).
Sekitar 8,5% lokasinya "Unknown".
Tiga merek teratas (Toyota 171, Honda 145, Daihatsu 82) menguasai 65% data.

Hasil model lebih mewakili pasar Jabodetabek dan merek populer.

Struktur Notebook
Bagian	Isi
1–2. Data Understanding	Load data, data dictionary, cek kategorikal dan biner, analisis representativitas dan bias
2. EDA	Distribusi harga, tahun, dan mileage. Harga per merek dan transmisi. Korelasi, fitur biner vs harga
3. Data Quality Audit	Missing value, duplikasi, format mileage, whitespace, rentang nilai, outlier IQR, audit leakage
4. Data Preparation	Cleaning, split 80/20, feature engineering car name, preprocessing, seleksi fitur
5. Regression Modeling	OLS, Ridge, Lasso, tuning alpha dengan CV, perbandingan model, residual, koefisien
6. Evaluasi (Progress 4)	Metrik akhir di data testing, train vs test, model utama, analisis error, kesimpulan, keterbatasan
Alur Pemodelan
Cleaning: rapikan spasi dan kapitalisasi kategori. Lokasi Unknown dipertahankan sebagai kategori.
Split: 80% train dan 20% test (122 observasi), random_state=42.
Feature engineering: car name (ratusan nilai unik) dipecah menjadi fitur family dan engine agar One-Hot Encoding tidak meledak.
Preprocessing: One-Hot Encoding untuk kategorikal dan scaling untuk numerik, dalam satu Pipeline. Imputasi (jika perlu) hanya memakai statistik data train.
Seleksi fitur: cicilan dikeluarkan karena leakage, lalu fitur dievaluasi dengan bukti (CV Ridge).
Model: OLS sebagai baseline, serta Ridge dan Lasso dengan alpha dicari lewat 5-fold CV (GridSearchCV).
Pemilihan model utama: berdasarkan CV MSE terkecil di data train. Data test hanya dipakai untuk evaluasi akhir.
Hasil Utama

Model utama adalah Ridge (alpha = 1,0).

Metrik (data test)	Nilai
R²	0,8674
RMSE	Rp26.040.246
MAE	Rp15.408.410
Rasio Test RMSE / Train RMSE	1,365

Dalam bahasa sehari-hari, model menjelaskan sekitar 86,7% variasi harga. Rata-rata selisih prediksi dengan harga aktual sekitar Rp15,4 juta.

Pada split ini OLS punya Test RMSE lebih kecil (Rp23,8 juta). Ridge tetap dipilih karena kriteria pemilihan adalah CV di data train, bukan data test.

Keterbatasan
Jumlah data kecil (609 baris), sehingga hasil sensitif terhadap pembagian data. Std CV MSE sekitar 27% dari rata-ratanya.
Sampel condong ke Jabodetabek dan merek populer.
Harga yang dimodelkan adalah harga penawaran, bukan harga transaksi.
Model linear mungkin kurang menangkap hubungan non-linear, dan error cenderung besar pada mobil mahal atau langka.
Ambang keberhasilan bisnis (mis. R² minimum atau MAE maksimum) belum ditetapkan.
Cara Menjalankan

1. Siapkan folder

project/
├── Prediksi_Harga_Mobil_Bekas_Regresi.ipynb
└── data/
    └── used_car.csv

2. Install dependensi

bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

3. Jalankan notebook

bash
jupyter notebook Prediksi_Harga_Mobil_Bekas_Regresi.ipynb

Jalankan semua sel secara berurutan (Restart & Run All). Bagian 2 dan 4 mencari used_car.csv secara otomatis di folder data/ atau folder induknya.

Teknologi

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn.
