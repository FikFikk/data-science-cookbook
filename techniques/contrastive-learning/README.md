# Contrastive Learning & SimCLR: Belajar Representasi Tanpa Label

> **Kategori**: Self-Supervised Learning, Representation Learning  
> **Tingkat Kesulitan**: Menengah–Lanjut  
> **Framework**: PyTorch  
> **Waktu Baca**: ~25 menit

---

## 📌 Apa Itu Contrastive Learning?

Contrastive Learning adalah pendekatan **self-supervised learning** yang mengajarkan model untuk belajar representasi (embeddings) yang bermakna dari data **tanpa membutuhkan label**. Ide intinya sederhana:

> **"Buat dua versi berbeda dari gambar yang sama → paksa model agar menempatkan keduanya berdekatan di ruang embedding. Buat gambar berbeda → dorong agar berjauhan."**

Ini mirip cara manusia belajar: kita tahu bahwa foto anjing dari sudut berbeda tetap "anjing yang sama", meskipun tidak ada yang memberi tahu kita secara eksplisit.

### Analogi Sehari-hari

Bayangkan kamu belajar mengenali wajah orang tanpa flash card berlabel:
- Foto teman yang sama dengan pencahayaan berbeda → **sama**
- Foto teman yang berbeda → **berbeda**

Model contrastive learning melakukan hal serupa, tapi untuk ribuan gambar sekaligus.

---

## 🧠 Mengapa Ini Penting?

| Masalah | Solusi Contrastive Learning |
|---|---|
| Labeling data mahal & lambat | Belajar tanpa label sama sekali |
| Supervised model butuh ribuan sampel berlabel | Pretrain dulu, fine-tune dengan sedikit label |
| Fitur manual (feature engineering) tidak optimal | Model belajar representasi optimal secara otomatis |
| Transfer learning terbatas domain | Representasi universal yang bisa dipindah ke berbagai task |

---

## 📐 Intuisi Matematis (Tidak Terlalu Formal)

### Ruang Embedding

Model mengubah input **x** (gambar) menjadi vektor di ruang berdimensi tinggi (misal 128 dimensi). Kita ingin:

```
sim(z_i, z_j) tinggi  ← jika x_i dan x_j dari gambar yang sama
sim(z_i, z_k) rendah  ← jika x_i dan x_k dari gambar berbeda
```

Di mana `sim` biasanya adalah **cosine similarity**:

```
sim(u, v) = (u · v) / (||u|| × ||v||)
```

Nilainya antara -1 (sangat berbeda) hingga +1 (identik).

---

### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

Ini adalah loss function yang digunakan SimCLR. Untuk sepasang positif `(i, j)`:

```
L(i,j) = -log [ exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) ]
```

**Penjelasan komponen:**
- `τ` (tau) = **temperature** — mengontrol "ketajaman" distribusi
  - τ kecil: model lebih tegas membedakan positif vs negatif
  - τ besar: model lebih "pemaaf", distribusi lebih mulus
- **Pembilang**: seberapa mirip pasangan positif
- **Penyebut**: total similarity terhadap semua sampel lain dalam batch (negatif)

**Cara membacanya**: Loss rendah ketika model yakin bahwa dari semua kandidat, pasangan positif memang yang paling mirip.

---

### Augmentasi Data sebagai "Jangkar"

SimCLR tidak butuh label karena ia membuat labelnya sendiri lewat augmentasi:

```
Gambar asli x
├── Augmentasi 1 → x_i  (crop + flip + color jitter)
└── Augmentasi 2 → x_j  (crop berbeda + grayscale)

x_i dan x_j adalah "pasangan positif"
Semua gambar lain di batch adalah "pasangan negatif"
```

---

## 🏗️ Arsitektur SimCLR

```
Input Image
    │
    ├──[Augment 1]──► Encoder (ResNet) ──► h_i ──► Projection Head ──► z_i
    │                                                                       │
    └──[Augment 2]──► Encoder (ResNet) ──► h_j ──► Projection Head ──► z_j
                                                                            │
                                                                    NT-Xent Loss
```

**Komponen:**
1. **Augmentation Module**: Random crop, flip, color jitter, grayscale
2. **Encoder (f)**: ResNet-50 atau arsitektur apa pun — menghasilkan representasi `h`
3. **Projection Head (g)**: MLP 2-3 layer — memetakan `h` ke `z` untuk training
4. **Loss**: NT-Xent dihitung atas `z`, bukan `h`
5. **Downstream task**: Gunakan `h` (bukan `z`) untuk fine-tuning!

> **Catatan penting**: Projection head dibuang setelah pretraining. Representasi `h` yang digunakan untuk downstream task.

---

## 🔄 Pipeline Step-by-Step

### Step 1: Data Augmentation

```python
# Dua augmentasi berbeda dari gambar yang sama
augmentasi = [
    RandomResizedCrop(size=96),
    RandomHorizontalFlip(),
    RandomApply([ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=9),
    Normalize(mean, std)
]
```

### Step 2: Forward Pass

```python
# Satu gambar → dua view → dua embedding
z_i = projection_head(encoder(augment1(x)))
z_j = projection_head(encoder(augment2(x)))
```

### Step 3: Hitung Loss

```python
# Untuk setiap pasangan positif (i,j), negatif = semua sampel lain
loss = nt_xent_loss(z_i, z_j, temperature=0.5)
```

### Step 4: Downstream Task

```python
# Setelah pretraining, fine-tune encoder dengan sedikit label
classifier = LinearClassifier(encoder.output_dim, num_classes)
# Fine-tune hanya dengan 1-10% data berlabel
```

---

## 💻 Use Cases di Dunia Nyata

| Domain | Aplikasi |
|---|---|
| **Computer Vision** | Image classification dengan sedikit label, anomaly detection |
| **NLP** | SimCSE untuk sentence embeddings tanpa supervised data |
| **Medical Imaging** | Radiografi/CT scan — labeling dokter sangat mahal |
| **Bioinformatics** | Representasi protein (AlphaFold menggunakan prinsip serupa) |
| **Rekrutmen** | Embedding CV tanpa label manual |
| **E-commerce** | Product similarity search dari gambar |
| **Audio** | Speech recognition pretraining (wav2vec) |

---

## ⚠️ Tips & Pitfalls

### ✅ Tips Sukses

1. **Batch size besar = lebih baik**: Lebih banyak negatif → gradient lebih informatif. SimCLR asli menggunakan batch 4096!
2. **Temperature tuning**: Mulai dengan τ=0.5, lalu tune. Terlalu kecil: collapse; terlalu besar: tidak belajar
3. **Augmentasi adalah kunci**: Pilih augmentasi yang sesuai domain. Untuk teks, augmentasinya berbeda
4. **Projection head**: Gunakan 2-3 layer MLP, bukan linear
5. **Training lama**: SimCLR butuh 200-1000 epoch untuk konvergen

### ❌ Pitfalls Umum

1. **Mode Collapse**: Model menghasilkan vektor yang sama untuk semua input → gradients menghilang
   - Solusi: Pastikan augmentasi beragam, gunakan batch besar
2. **Representational Collapse**: Semua embedding identik
   - Solusi: BN yang benar, temperature yang tepat
3. **Augmentasi terlalu ekstrem**: Model tidak bisa lagi menemukan invariansi
4. **Batch kecil**: Dengan 32-64 sampel, terlalu sedikit negatif
5. **Evaluasi hanya pada pretext task**: Selalu evaluasi pada downstream task!

---

## 📊 Evaluation Metrics

### Linear Evaluation Protocol
```
1. Freeze encoder setelah pretraining
2. Train linear classifier di atasnya menggunakan label
3. Ukur Top-1 accuracy
```

### KNN Evaluation
```python
# No fine-tuning sama sekali!
# Gunakan KNN pada embeddings
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_embeddings, train_labels)
accuracy = knn.score(test_embeddings, test_labels)
```

### Metric yang Digunakan
- **Top-1 & Top-5 Accuracy** (ImageNet benchmark)
- **Embedding uniformity**: seberapa merata distribusi di hypersphere
- **Embedding alignment**: seberapa dekat pasangan positif
- **Transfer accuracy**: performa pada dataset lain setelah fine-tune

---

## 🔬 Perbandingan Metode

| Metode | Tahun | Keunggulan | Kelemahan |
|---|---|---|---|
| **SimCLR** | 2020 | Simple, efektif | Butuh batch besar |
| **MoCo** | 2020 | Memory efficient (queue) | Lebih kompleks |
| **BYOL** | 2020 | Tanpa negatif sama sekali | Rentan collapse |
| **SimSiam** | 2021 | Stop-gradient, no negatives | Tricky hyperparams |
| **DINO** | 2021 | ViT-based, sangat powerful | Butuh GPU besar |
| **VICReg** | 2022 | Stabil, regularisasi eksplisit | ---|

---

## 📚 Referensi

1. **Chen et al. (2020)** — *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR)  
   https://arxiv.org/abs/2002.05709

2. **Chen et al. (2020)** — *Big Self-Supervised Models Are Strong Semi-Supervised Learners* (SimCLRv2)  
   https://arxiv.org/abs/2006.10029

3. **Grill et al. (2020)** — *Bootstrap Your Own Latent* (BYOL)  
   https://arxiv.org/abs/2006.07733

4. **He et al. (2020)** — *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo)  
   https://arxiv.org/abs/1911.05722

5. **Caron et al. (2021)** — *Emerging Properties in Self-Supervised Vision Transformers* (DINO)  
   https://arxiv.org/abs/2104.14294

6. **Bardes et al. (2022)** — *VICReg: Variance-Invariance-Covariance Regularization*  
   https://arxiv.org/abs/2105.04906

---

## 📁 File dalam Folder Ini

- `README.md` — Penjelasan lengkap (file ini)
- `notebook.py` — Implementasi lengkap yang bisa dijalankan

---

*Dibuat oleh Hermes Agent — Data Science Cookbook*
