# Laporan Proyek Machine Learning - Abimanyu Rio Yulianto
## Domain Proyek
### Latar Belakang

Dalam industri otomotif, analisis data menjadi semakin penting dalam mengidentifikasi tren pasar, memahami preferensi pelanggan, dan mengambil keputusan yang lebih tepat waktu dan tepat sasaran. Penjualan kendaraan adalah indikator penting bagi kesehatan dan kinerja industri otomotif, serta merupakan refleksi dari permintaan konsumen dan dinamika pasar.

Analisis data penjualan kendaraan dan tren pasar melibatkan pengumpulan, pengolahan, dan analisis data historis dan saat ini untuk mendapatkan wawasan berharga tentang bagaimana pasar bergerak, perilaku pembeli, dan potensi peluang atau tantangan di masa depan. Dengan informasi ini, perusahaan otomotif, dealer, dan pemangku kepentingan lainnya dapat membuat strategi pemasaran yang lebih efektif, mengelola rantai pasokan dengan lebih baik, dan meningkatkan penjualan serta kepuasan pelanggan.

**Beberapa aspek yang dapat dianalisis dalam konteks ini termasuk**:

1. Tren Penjualan: Menganalisis data historis penjualan kendaraan untuk mengidentifikasi tren jangka panjang dan musiman, serta fluktuasi yang mungkin terjadi akibat faktor eksternal seperti perubahan ekonomi, kebijakan pemerintah, atau tren konsumen.
2. Analisis Demografi Pelanggan: Mengevaluasi data demografis pelanggan seperti usia, jenis kelamin, lokasi geografis, dan pendapatan untuk memahami preferensi dan perilaku pembeli yang berbeda.
3. Analisis Produk: Memeriksa penjualan berbagai jenis kendaraan seperti mobil, truk, SUV, atau kendaraan listrik untuk mengidentifikasi produk yang paling diminati dan mengantisipasi permintaan masa depan.
4. Penetrasi Pasar: Menilai pangsa pasar perusahaan atau merek dalam segmen tertentu dan membandingkannya dengan pesaing untuk mengetahui posisi relatifnya dan peluang pengembangan lebih lanjut.
5. Evaluasi Kinerja Dealer: Menganalisis data penjualan dealer individual untuk mengidentifikasi kinerja terbaik, tren penjualan, dan area yang memerlukan perbaikan.
6. Prediksi Harga: Menggunakan teknik analisis prediktif seperti pemodelan regresi atau analisis deret waktu untuk memprediksi permintaan masa depan berdasarkan data historis dan variabel yang relevan.

Dengan menggunakan analisis data yang cermat dan alat analisis yang tepat, perusahaan otomotif dapat mengoptimalkan strategi pemasaran, meningkatkan efisiensi operasional, dan menghasilkan keputusan yang didukung oleh data untuk mencapai keunggulan kompetitif dalam industri yang semakin kompetitif.

Dalam proyek ini perusahaan akan membuat beberapa model Machine Learning yang kemudian di evaluasi untuk membandingkan model mana yang hasil prediksinya paling baik lalu diharapkan dapat memprediksi harga yang sesuai berdasarkan penjualan kendaraan.

### Problem Statements

Berdasarkan latar belakang di atas, rincian masalahnya adalah sebagai berikut:
- Algoritma apa yang cocok untuk memprediksi harga penjualan kendaraan ?
- Bagaimana cara menentukan hasil prediksi suatu Algoritma Machine Learning dapat dikatakan baik?

### Goals

Untuk menjawab pertanyaan di atas, maka akan dijabarkan sebagai berikut:
- Ada banyak algoritma yang dapat menyelesaikan masalah tersebut, namun di proyek ini akan menggunakan algoritma K-Nearest Neighbor (KNN) dan RandomForest.
- Melakukan evaluasi terhadap metrik dari masing-masing algoritma.

### Solution Statements
Solusi yang dapat dilakukan untuk memenuhi goals proyek ini diantaranya sebagai berikut:
- Membuat 2 model Machine Learning yaitu dengan algoritma K-Nearest Neighbor (KNN) dan RandomForest. 

  * Konsep dari algoritma K-Nearest Neighbor (KNN) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. [[1]](https://www.dicoding.com/academies/319/tutorials/18580).  
 ![image](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:07942fa0c26e69b34aa15a9d63678b4320210912094515.png)  
  * Konsep dari algoritma RandomForest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. [[2]](https://www.dicoding.com/academies/319/tutorials/18585)  
   ![image](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:5e086364e59025d11dd0dfd3bc965e7c20210912094833.png)

## Data Understanding
***
Dataset yang digunakan dapat diakses menggunakan [Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)  
Informasi dari dataset dapat dirangkum sebagai berikut:

Tabel 1. Rangkuman informasi Dataset    

| Jenis                  | Keterangan                                                                                                        |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Sumber                 | [Kaggle Dataset: EVehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data)  |
| Lisensi                | MIT(https://www.mit.edu/~amini/LICENSE.md)                                                   |
| Kategori               | Vehicle/Car Sales Trends and Pricing Insights                                                                                                            |
| Jenis & Ukuran berkas  | CSV (88.05 MB)                                                                                                      |  
***

### Variabel-variabel pada Vehicle Sales Data dataset adalah sebagai berikut:
- Year: Tahun pembuatan kendaraan (misalnya, 2015)
- Make: Merek atau pabrikan kendaraan (misalnya Kia, BMW, Volvo)
- Model: Model kendaraan tertentu (misalnya Sorento, Seri 3, S60, Seri 6 Gran Coupe)
- Trim: Penunjukan tambahan untuk versi tertentu atau paket opsi model (misalnya, LX, 328i SULEV, T5, 650i)
- Body: Jenis bodi kendaraan (misalnya SUV, Sedan)
- Transmission: Jenis transmisi pada kendaraan (misalnya otomatis)
- VIN : Vehicle Identification Number, kode unik yang digunakan untuk mengidentifikasi suatu kendaraan bermotor
- State: Negara bagian di mana kendaraan berada atau didaftarkan (misalnya, CA untuk California)
- Condition: Representasi numerik dari kondisi kendaraan (misalnya 5.0)
- Odometer: Jarak tempuh atau jarak yang ditempuh kendaraan
- Color: Warna eksterior kendaraan
- Interior: Warna interior kendaraan
- Seller: Entitas atau perusahaan yang menjual kendaraan (misalnya, Kia Motors America Inc, Financial Services Remarketing)
- MMR: Manheim Market Report, alat penetapan harga yang digunakan dalam industri otomotif
- Selling Price : Harga dimana kendaraan itu dijual
- Sale Date: Tanggal dan waktu kendaraan dijual

### EDA - Missing Value
- Untuk kolom kategori yang memiliki value Null maka akan diisi dengan Other untuk Kolom make, model, trim & color.
- Untuk kolom body, transmission & interior diisi dengan Mode.
- Untuk kolom numerical yang Null akan dihapus.

### EDA - Outliers Handling
- Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. [[3]](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf)
![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/fffe94d4-66e3-4398-a5c8-a3d78fd053a4)
Dari hasil visualisasi di atas dapat disimpulkan bahwa: 
- Sebagian besar sampel Fitur Numerical memiliki Outliers.
- Maka perlu dilakukan IQR Method [[4]](https://stevkarta.medium.com/mendeteksi-univariate-outliers-dengan-metode-iqr-python-3adfad87de82)
  Untuk metode IQR batas bawahnya didefiniskan seperti berikut :

  ![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/f36e2e36-f0e4-44c7-859d-863ad7862a3b)

  Untuk metode IQR batas atasnya didefinisikan seperti berikut :

  ![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/353451a7-fc39-4215-b7f8-ad2f812617bc)

### EDA - Univariate Analysis
![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/cbb4977c-20a8-4a4a-8ea2-7393df59ef52)
Dari hasil visualisasi di atas dapat disimpulkan bahwa: 
- Berdasarkan kategori transmisi Automatic lebih banyak unitnya dari pada Manual.
- Berdasarkan kategori warna Black & White lebih banyak unitnya dari pada warna lain.
- Berdasarkan kategori interior Black lebih banyak unitnya dari pada interior lain.

### EDA - Multivariate Analysis 
![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/cbc97d20-0bcf-4633-b444-c819e8bfc9c3)
![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/5517b84e-3fd3-4f18-b34f-3840e634b79c)

Berdasarkan visualisasi di atas dapat disimpulkan bahwa:
- Variabel mmr berkorelasi positif dengan variabel selling price, skornya yaitu 0.98.

## Data Preparation
Berikut merupakan tahapan-tahapan dalam Data Preparation:
- Melakukan Split Data, dataset yang ada dibagi menjadi 2 bagian yaitu data latih dan data uji dengan rasio 80:20. Proses ini dilakukan dengan menggunakan modul [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) dari library scikit-learn.
- Melakukan Encoding Fitur Kategori dengan menggunakan OneHotEncoder dari library sckit-learn.
- Melakukan standarisasi pada data latih dengan menggunakan StandardScaler dari library sckit-learn.

## Modeling
Setelah melakukan data preparation data yang sudah siap akan digunakan untuk membuat model, kali ini akan dibuat 2 model sebagai perbandingan.
- Membuat model dengan menggunakan algoritma [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html), alasan menggunakan algoritma ini karena K-NN dapat memberikan kinerja yang baik untuk dataset kecil dengan dimensi yang tidak terlalu besar. Ini membuatnya menjadi pilihan yang baik untuk masalah kecil dan sederhana..
- Membuat model dengan menggunakan algoritma [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), alasan menggunakan algoritma ini karena Random Forest adalah model yang relatif mudah diimplementasikan dan tidak memerlukan banyak tuning parameter. Selain itu, ia memiliki kemampuan bawaan untuk menangani fitur yang hilang dan tidak teratur. 


## Evaluation
Proses evaluasi model pada proyek ini menggunakan metrik Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi[[5]](https://www.dicoding.com/academies/319/tutorials/18595). MSE didefinisikan dalam persamaan berikut:
![image](https://cdn-media-1.freecodecamp.org/images/hmZydSW9YegiMVPWq2JBpOpai3CejzQpGkNG)  
Keterangan:  
N = jumlah dataset  
yi = nilai sebenarnya  
y_pred = nilai prediksi  

Hasil evaluasi pada data latih dan data test adalah sebagai berikut:

![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/f1f1475f-ef4d-4539-9323-b05d57d0a0de)

![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/292a7d52-80dc-4c4b-ac8c-8c7f8f078d8d)

Dari gambar diatas bisa dilihat tingkat nilai error yang paling kecil adalah Model Random Forest (RF) walaupun antara dengan K-NN tidak terlalu jauh perbedaannya. Namun tetap model terbaik untuk melakukan prediksi harga jual kendaraan menggunakan model Random Forest.

![image](https://github.com/Abimanyurioy/ML-Terapan---Predictive-Analytics/assets/158838276/9ce0f634-2be2-47a3-a6f8-a0e457c64b51)

Terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati.

## Referensi
[1] https://www.dicoding.com/academies/319/tutorials/18580
[2] https://www.dicoding.com/academies/319/tutorials/18585
[3] https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf
[4] https://stevkarta.medium.com/mendeteksi-univariate-outliers-dengan-metode-iqr-python-3adfad87de82
[5] https://www.dicoding.com/academies/319/tutorials/18595
