# Active Learning: Pembelajaran Aktif untuk Reduksi Biaya Labeling Data

Di dunia nyata, mengumpulkan data mentah sering kali sangat mudah dan murah, sedangkan melabeli data tersebut (misalnya oleh tenaga medis ahli, insinyur annotator, atau pakar hukum) menuntut biaya yang sangat tinggi dan waktu yang lama. **Active Learning (Pembelajaran Aktif)** adalah sebuah paradigma dalam machine learning yang mengatasi masalah ini. 

Alih-alih melatih model pada keseluruhan dataset yang dilabeli di awal (Passive Learning), model Pembelajaran Aktif secara cerdas memilih data mentah/tak berlabel dari suatu kolam (*pool*) yang diyakini akan memberikan informasi paling berguna untuk meningkatkan performanya jika dilabeli oleh evaluator (*oracle* manusia).

---

## Ringkasan

Pembelajaran Aktif bekerja secara interaktif dengan mengajukan pertanyaan kepada manusia. Teknik ini mampu mencapai performa model yang sama (atau bahkan lebih baik) dibandingkan modeling konvensional, namun dengan menggunakan jumlah data berlabel yang jauh lebih sedikit (bisa memangkas kebutuhan data berlabel hingga 60%-90%).

### Kapan Menggunakan Active Learning?
- **Domain Khusus & Mahal**: Klasifikasi citra medis (CT-Scan/MRI), analisis sentimen dokumen hukum, deteksi malware, atau anotasi rekaman audio berdurasi panjang.
- **Batasan Budget Labeling**: Di mana Anda memiliki jutaan baris data tapi hanya sanggup mendanai annotator untuk melabeli beberapa ribu baris saja.
- **Iterative Deployment**: Ketika Anda ingin men-deploy model awal secara cepat dan memperbaruinya secara bertahap berdasarkan data produksi baru.

### Keunggulan
1. **Reduksi Biaya Signifikan**: Menghemat anggaran anotasi data hingga berkali-kali lipat.
2. **Performa Lebih Efisien**: Menghindari anomali noise dari unlabeled data yang kurang penting atau duplikat.
3. **Efisiensi Waktu**: Mempercepat *Time-to-Market* untuk model kecerdasan buatan baru.

---

## Intuisi Matematis

Ada berbagai skenario query dalam Active Learning. Yang paling umum dan praktis adalah **Pool-Based Active Learning**, di mana kita memiliki kumpulan data tak berlabel (*Pool* $\mathcal{U}$) dan sekumpulan kecil data berlabel ($\mathcal{L}$). Model dilatih di $\mathcal{L}$ kemudian mengevaluasi semua data di $\mathcal{U}$ menggunakan **Acquisition Function** (Fungsi Akuisisi) untuk memilih data mana yang harus diberikan kepada pelabel manusia.

Berikut adalah tiga strategi prioritisasi utama berdasarkan **Uncertainty Sampling** (Sampling Ketidakpastian):

### 1. Least Confidence Sampling
Strategi ini memilih sampel yang model rasa paling tidak percaya diri dengan prediksi kelas tertingginya. 
Secara matematis, untuk sampel $x$:

$$x^*_{\text{LC}} = \arg\max_{x} \left( 1 - P(\hat{y} \mid x) \right)$$

Di mana $\hat{y} = \arg\max_{y} P(y \mid x)$ adalah kelas yang diprediksi yang memiliki probabilitas tertinggi.
* **Intuisi**: Jika model memprediksi Kelas A dengan probabilitas $0.51$ dan Kelas B dengan $0.49$, tingkat keyakinan kelas tertingginya sangat rendah ($0.51$). Ini adalah area ambigu yang perlu divalidasi oleh manusia.

### 2. Margin Sampling
Strategi ini mengukur selisih probabilitas antara kelas prediksi pertama ($\hat{y}_1$) dan kedua ($\hat{y}_2$) yang paling dominan.
Secara matematis:

$$M(x) = P(\hat{y}_1 \mid x) - P(\hat{y}_2 \mid x)$$

$$x^*_{\text{Margin}} = \arg\min_{x} M(x)$$

* **Intuisi**: Margin yang sangat sempit berarti model bingung membedakan dua kelas terkuat. Margin lebar berarti model sangat yakin dengan keputusan kelas pertama dibanding kelas kedua. Kita mencari sampel dengan margin terkecil ($\arg\min$).

### 3. Entropy Sampling
Strategi yang memanfaatkan teori informasi Shannon Entropy untuk menilai ketidakpastian di seluruh distribusi probabilitas semua kelas yang didefinisikan secara resmi.
Secara matematis:

$$H(x) = -\sum_{i=1}^{C} P(y_i \mid x) \log P(y_i \mid x)$$

$$x^*_{\text{Entropy}} = \arg\max_{x} H(x)$$

* **Intuisi**: Entropy akan bernilai tinggi jika probabilitas terdistribusi merata ke seluruh kelas (ketidakpastian maksimum) dan bernilai rendah jika probabilitas terpusat pada satu kelas tunggal (kepercayaan penuh).

### 4. Query-By-Committee (QBC)
Pada metode ini, kita melatih beberapa model (*Committee* $\mathcal{c} = \{ \theta^{(1)}, \theta^{(2)}, \dots, \theta^{(C)} \}$) dengan arsitektur berbeda atau subset data berbeda (misal, ensemble bagging). Sampel yang dipilih untuk dilabeli adalah sampel yang memicu perbedaan opini paling tinggi di antara anggota komite (**Vote Entropy** atau **Kullback-Leibler Divergence**).

---

## Alur Kerja Active Learning (Loop)

```
       ┌────────────────────────┐
       │ (1) Latih Model Awal   │ ◄────────────────────────┐
       │  pada Labeled Data (L) │                          │
       └───────────┬────────────┘                          │
                   │                                       │
                   ▼                                       │
       ┌────────────────────────┐                          │
       │ (2) Prediksi Data pada │                          │
       │  Unlabeled Pool (U)    │                          │
       └───────────┬────────────┘                          │ (5) Tambahkan Data
                   │                                       │     Baru ke L
                   ▼                                       │
       ┌────────────────────────┐                          │
       │ (3) Hitung Nilai       │                          │
       │  Acquisition Function  │                          │
       └───────────┬────────────┘                          │
                   │                                       │
                   ▼                                       │
       ┌────────────────────────┐   (Ya)   ┌───────────────┴────────┐
       │ (4) Pilih k-sampel     ├─────────►│ Anotasi oleh Manusia   │
       │ ketidakpastian tertinggi│          │ (Oracle Labeling)      │
       └────────────────────────┘          └────────────────────────┘
```

---

## Implementasi Kode Python

Di dalam `notebook.py`, kita akan membuat simulasi komparasi performa antara:
1. **Passive Learning (Random Sampling)**: Memilih secara acak data dari pool untuk dilabeli.
2. **Active Learning (Uncertainty Entropy)**: Memilih data menggunakan Entropy Sampling.
3. **Active Learning (Margin)**: Memilih data menggunakan Margin Sampling.
4. **Active Learning (Query-By-Committee)**: Memilih data menggunakan perbedaan suara komite model.

Kita akan menggunakan library `modAL`, salah satu framework Active Learning terpopuler berbasis `scikit-learn`.

### Persyaratan Dependencies
Anda dapat menginstall requirements dengan menjalankannya di terminal:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn modAL-python
```

---

## Dataset yang Digunakan
Untuk tutorial ini, kita menggunakan gabungan dataset sintetis non-linier menggunakan `make_classification` dengan beberapa noise tambahan untuk merepresentasikan data nyata yang menantang. Selain itu, dataset ini dibagi menjadi representasi:
* **Initial Labeled Set**: Sekitar 1% - 2% data awal berlabel.
* **Unlabeled Pool**: Sisa data tanpa label yang akan dievaluasi oleh strategi query.
* **Test Set**: Data hold-out untuk mengevaluasi efisiensi perkembangan model di setiap batch iterasi.

---

## Evaluasi Metrik
Model dievaluasi di setiap langkah iterasi penambahan label dengan metrik:
1. **Accuracy Score**: Persentase tebakan kelas yang benar secara keseluruhan.
2. **F1-Score (Macro)**: Menilai stabilitas model pada kelas mayoritas/minoritas.
3. **Accuracy vs Labeled Budget Curve**: Grafik visualisasi lineplot yang menunjukkan seberapa cepat model dengan strategi Active Learning menanjak akurasinya dibanding model pasif dengan jumlah label yang sama.

---

## Real-World Applications

* **Diagnosa Medis (Computer Vision & NLP)**: Klasifikasi gambar patologi langka (misalnya kanker metastasis). Sebagai ganti melabeli 10.000 citra medis yang menyita waktu dokter spesialis saraf, Active learning membantu dokter hanya perlu melakukan anotasi pada 800-1000 gambar tersulit.
* **Deteksi Penipuan Finansial (Fraud Detection)**: Transaksi terindikasi fraud yang membutuhkan konfirmasi manual dari tim investigasi bank.
* **Pembangunan Fitur Pencarian & NLP (Ranker Optimizing)**: Penilaian manual kecocokan antara kueri pengguna dengan hasil pencarian produk yang relevan untuk melatih ulang mesin rekomendasi e-commerce.

---

## Tips & Pitfalls (Jebakan Umum)

1. **Test Set Contamination (Kontaminasi Data Uji)**: Jangan pernah menjalankan query pemilihan data aktif pada data uji (*test set*). Evaluasi performa harus selalu dilakukan pada set data uji terpisah yang distribusinya mencerminkan populasi asli.
2. **Cold Start Problem**: Di awal siklus, model dengan data awal berlabel yang sangat minim (misal < 5 sampel) akan memiliki estimasi probabilitas yang buruk. Pilihan terbaik adalah memulai dengan sampel acak yang sedikit memadai sebelum meluncurkan pemilihan aktif.
3. **Outlier Bias**: Sampling berbasis ketidakpastian sangat rentan memilih data outlier/kebisingan (*noise*) karena model secara konsisten kurang memahami noise tersebut. Kombinasikan dengan teknik evaluasi kepadatan (*density-weighted sampling*) untuk menyeimbangkan representasi data.

---

## Referensi Ilmiah

1. **Settles, Burr (2009).** *Active Learning Literature Survey*. Computer Sciences Technical Report 1648, University of Wisconsin-Madison.
2. **Seung, H. S., Opper, M., & Sompolinsky, H. (1992).** *Query by committee*. In Proceedings of the fifth annual workshop on Computational learning theory (pp. 287-294).
3. **Lewis, David D., and William A. Gale (1994).** *A sequential algorithm for training text classifiers*. Proceedings of the 17th annual international ACM SIGIR conference on Research and development in information retrieval.
