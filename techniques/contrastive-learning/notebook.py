"""
========================================================
Contrastive Learning & SimCLR — Implementasi Lengkap
========================================================
Self-Supervised Learning untuk Image Representation

Versi ini berjalan tanpa PyTorch (demonstrasi konsep dengan NumPy + sklearn).
Versi PyTorch penuh ada di bagian "PYTORCH IMPLEMENTATION" di bawah sebagai kode referensi.

Dependensi sistem yang ada: numpy, matplotlib, scikit-learn, PIL
Jalankan: python3 notebook.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_blobs
from PIL import Image, ImageFilter, ImageEnhance
import os
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/root/data-science-cookbook/techniques/contrastive-learning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 62)
print("  CONTRASTIVE LEARNING & SimCLR")
print("  Demonstrasi Konsep dengan NumPy & Sklearn")
print("=" * 62)

# ─── 1. VISUALISASI AUGMENTASI ────────────────────────────────────────────

def demo_augmentation_effect():
    """
    Demonstrasi berbagai jenis augmentasi yang digunakan SimCLR.
    Setiap augmentasi menghasilkan 'view' berbeda dari gambar yang sama.
    """
    print("\n[1/6] Membuat visualisasi augmentasi...")

    # Buat gambar sintetik (pola geometris)
    np.random.seed(42)
    W, H = 96, 96
    img_array = np.zeros((H, W, 3), dtype=np.uint8)

    # Buat pola yang menarik
    for i in range(0, H, 12):
        for j in range(0, W, 12):
            hue = int((i + j) / (H + W) * 200) + 30
            img_array[i:i+12, j:j+12] = [
                hue,
                255 - hue,
                (hue + 128) % 255
            ]
    # Tambah lingkaran di tengah
    cx, cy = W // 2, H // 2
    for y in range(H):
        for x in range(W):
            if (x - cx) ** 2 + (y - cy) ** 2 < 20 ** 2:
                img_array[y, x] = [255, 200, 50]

    img_pil = Image.fromarray(img_array)

    # Terapkan berbagai augmentasi
    augmentations = {}

    augmentations["Original"] = img_pil.copy()

    # Random Crop
    crop_size = 70
    left = np.random.randint(0, W - crop_size)
    top = np.random.randint(0, H - crop_size)
    augmentations["Random Crop"] = img_pil.crop(
        (left, top, left + crop_size, top + crop_size)
    ).resize((W, H), Image.BILINEAR)

    # Horizontal Flip
    augmentations["Horizontal Flip"] = img_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Color Jitter (brightness)
    enhancer = ImageEnhance.Brightness(img_pil)
    augmentations["Color Jitter\n(Brightness)"] = enhancer.enhance(1.6)

    # Color Jitter (contrast)
    enhancer2 = ImageEnhance.Contrast(img_pil)
    augmentations["Color Jitter\n(Contrast)"] = enhancer2.enhance(0.5)

    # Grayscale
    augmentations["Grayscale"] = img_pil.convert("L").convert("RGB")

    # Gaussian Blur
    augmentations["Gaussian Blur"] = img_pil.filter(ImageFilter.GaussianBlur(radius=3))

    # Combined (View 1): crop + flip + brightness
    left1 = 5; top1 = 5
    aug_combined_1 = img_pil.crop((left1, top1, left1+80, top1+80)).resize((W, H))
    aug_combined_1 = ImageEnhance.Color(aug_combined_1).enhance(1.4)
    augmentations["Combined\n(View 1)"] = aug_combined_1

    # Combined (View 2): crop berbeda + grayscale partial
    left2 = 10; top2 = 8
    aug_combined_2 = img_pil.crop((left2, top2, left2+78, top2+78)).resize((W, H))
    aug_combined_2 = aug_combined_2.transpose(Image.FLIP_LEFT_RIGHT)
    aug_combined_2 = ImageEnhance.Brightness(aug_combined_2).enhance(0.8)
    augmentations["Combined\n(View 2)"] = aug_combined_2

    # Plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(
        "Augmentasi dalam SimCLR: Menciptakan Positive Pairs",
        fontsize=14, fontweight="bold", y=1.01
    )

    for ax_idx, (aug_name, aug_img) in enumerate(augmentations.items()):
        row, col = divmod(ax_idx, 3)
        ax = axes[row, col]
        ax.imshow(aug_img)

        # Highlight pasangan positif (view 1 & 2)
        if "View 1" in aug_name or "View 2" in aug_name:
            for spine in ax.spines.values():
                spine.set_edgecolor("#2ecc71")
                spine.set_linewidth(3)
            ax.set_title(aug_name, fontsize=10, fontweight="bold",
                         color="#27ae60")
        elif "Original" in aug_name:
            for spine in ax.spines.values():
                spine.set_edgecolor("#e74c3c")
                spine.set_linewidth(3)
            ax.set_title(aug_name, fontsize=10, fontweight="bold",
                         color="#c0392b")
        else:
            ax.set_title(aug_name, fontsize=10)
        ax.axis("off")

    # Legend
    patch_orig = mpatches.Patch(color="#e74c3c", label="Gambar Original")
    patch_view = mpatches.Patch(color="#2ecc71",
                                 label="Positive Pairs (didorong berdekatan)")
    fig.legend(handles=[patch_orig, patch_view],
               loc="lower center", ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/augmentation_demo.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: augmentation_demo.png")


# ─── 2. INTUISI NT-XENT LOSS ─────────────────────────────────────────────

def demo_nt_xent_intuition():
    """
    Visualisasi proses ContRastive learning dalam 2D:
    Bagaimana embeddings berubah dari acak ke terstruktur.
    """
    print("\n[2/6] Membuat visualisasi NT-Xent intuition...")

    np.random.seed(7)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Proses Contrastive Learning: Embeddings Berkembang dari Acak ke Terstruktur",
        fontsize=13, fontweight="bold"
    )

    n_classes = 6
    n_per_class = 15
    colors_cls = plt.cm.tab10(np.linspace(0, 1, n_classes))
    class_labels = ["Kucing", "Anjing", "Burung", "Mobil", "Kapal", "Kuda"]

    # Simulasi tiga tahap pelatihan
    stages = [
        ("Sebelum Training\n(Random Embeddings)",
         np.random.randn(n_classes, 2) * 0.3,   # centers tersebar acak
         2.0, 0.15),                               # spread besar, aug_noise kecil

        ("Training Sedang Berjalan\n(Mulai Terbentuk Cluster)",
         np.array([[2, 2], [-2, 2], [0, -2.5],
                   [2, -2], [-2, -2], [0, 2.5]]) * 0.8,
         0.9, 0.1),

        ("Setelah Training\n(Cluster Terpisah Jelas)",
         np.array([[2.5, 2.5], [-2.5, 2.5], [0, -3.0],
                   [2.5, -2.5], [-2.5, -2.5], [0, 3.0]]),
         0.25, 0.07),
    ]

    for plot_idx, (title, centers, spread, aug_noise) in enumerate(stages):
        ax = axes[plot_idx]

        all_pts = []
        all_aug = []
        for cls_idx in range(n_classes):
            pts = centers[cls_idx] + np.random.randn(n_per_class, 2) * spread
            aug = pts + np.random.randn(n_per_class, 2) * aug_noise
            all_pts.append(pts)
            all_aug.append(aug)

            # Plot titik original
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[colors_cls[cls_idx]], s=35,
                       alpha=0.85, edgecolors="white", linewidths=0.4,
                       zorder=3)
            # Plot augmentasi
            ax.scatter(aug[:, 0], aug[:, 1],
                       c=[colors_cls[cls_idx]], marker="^", s=25,
                       alpha=0.55, edgecolors="none", zorder=3)

            # Garis pos pairs (beberapa)
            for k in range(0, min(4, n_per_class)):
                ax.plot([pts[k, 0], aug[k, 0]],
                        [pts[k, 1], aug[k, 1]],
                        color=colors_cls[cls_idx],
                        alpha=0.35, linewidth=0.8, zorder=2)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("Embedding Dim 1", fontsize=8)
        ax.set_ylabel("Embedding Dim 2", fontsize=8)

        if plot_idx == 0:
            legend_handles = [
                mpatches.Patch(color=colors_cls[i], label=class_labels[i])
                for i in range(n_classes)
            ]
            ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
                      title="Kelas  (●=asli, ▲=aug)")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/nt_xent_intuition.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: nt_xent_intuition.png")


# ─── 3. SIMULASI NT-XENT LOSS ────────────────────────────────────────────

def nt_xent_loss_numpy(z_i, z_j, temperature=0.5):
    """
    Implementasi NT-Xent Loss dalam NumPy murni.
    Digunakan untuk demonstrasi — produksi menggunakan PyTorch.

    z_i, z_j: (N, D) — sudah ternormalisasi
    """
    N = z_i.shape[0]
    z = np.vstack([z_i, z_j])  # (2N, D)

    # Cosine similarity (z sudah ternormalisasi → dot product = cosine sim)
    sim = z @ z.T / temperature  # (2N, 2N)

    # Masking diagonal (similarity dengan diri sendiri = -inf)
    np.fill_diagonal(sim, -np.inf)

    # Softmax per baris — stabil secara numerik
    def softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    probs = softmax(sim)  # (2N, 2N)

    # Target: z_i[k] → positifnya z_j[k] (index N+k)
    #          z_j[k] → positifnya z_i[k] (index k)
    loss = 0.0
    eps = 1e-9

    for k in range(N):
        loss -= np.log(probs[k, N + k] + eps)      # z_i[k] ingin dekat z_j[k]
        loss -= np.log(probs[N + k, k] + eps)      # z_j[k] ingin dekat z_i[k]

    return loss / (2 * N)


def demo_nt_xent_computation():
    """
    Demo perhitungan NT-Xent dengan data sintetis.
    Bandingkan loss untuk embeddings baik vs buruk.
    """
    print("\n[3/6] Demo perhitungan NT-Xent Loss...")

    np.random.seed(42)
    N, D = 8, 16

    # ── Kasus 1: Embeddings BURUK (random) ──────────────────────────
    z_i_bad = np.random.randn(N, D)
    z_j_bad = np.random.randn(N, D)
    z_i_bad = z_i_bad / np.linalg.norm(z_i_bad, axis=1, keepdims=True)
    z_j_bad = z_j_bad / np.linalg.norm(z_j_bad, axis=1, keepdims=True)
    loss_bad = nt_xent_loss_numpy(z_i_bad, z_j_bad, temperature=0.5)

    # ── Kasus 2: Embeddings SEMPURNA (positif identik) ───────────────
    z_base = np.random.randn(N, D)
    z_base = z_base / np.linalg.norm(z_base, axis=1, keepdims=True)
    z_i_good = z_base.copy()
    # Tambah sedikit noise ke pasangan positif
    noise = np.random.randn(N, D) * 0.01
    z_j_good = z_base + noise
    z_j_good = z_j_good / np.linalg.norm(z_j_good, axis=1, keepdims=True)
    loss_good = nt_xent_loss_numpy(z_i_good, z_j_good, temperature=0.5)

    # ── Kasus 3: Embeddings SEDANG (sedikit belajar) ─────────────────
    z_i_med = z_base.copy()
    z_j_med = z_base + np.random.randn(N, D) * 0.3
    z_j_med = z_j_med / np.linalg.norm(z_j_med, axis=1, keepdims=True)
    loss_med = nt_xent_loss_numpy(z_i_med, z_j_med, temperature=0.5)

    print(f"  NT-Xent Loss:")
    print(f"    Embeddings Buruk  (random)  : {loss_bad:.4f}  ← Tinggi (model belum belajar)")
    print(f"    Embeddings Sedang (noise 0.3): {loss_med:.4f}  ← Sedang")
    print(f"    Embeddings Baik   (noise 0.01): {loss_good:.4f}  ← Rendah (model sudah belajar)")

    # Visualisasi similarity matrix untuk ketiga kasus
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("NT-Xent Loss: Similarity Matrix untuk Berbagai Kualitas Embedding",
                 fontsize=12, fontweight="bold")

    cases = [
        (z_i_bad, z_j_bad, f"Embeddings Buruk\nLoss={loss_bad:.3f}", "RdYlGn"),
        (z_i_med, z_j_med, f"Embeddings Sedang\nLoss={loss_med:.3f}", "RdYlGn"),
        (z_i_good, z_j_good, f"Embeddings Baik\nLoss={loss_good:.3f}", "RdYlGn"),
    ]

    for ax_idx, (zi, zj, title, cmap) in enumerate(cases):
        z_full = np.vstack([zi, zj])
        sim_mat = z_full @ z_full.T
        np.fill_diagonal(sim_mat, np.nan)

        ax = axes[ax_idx]
        im = ax.imshow(sim_mat, cmap=cmap, vmin=-0.5, vmax=1.0)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"2N={2*N} samples")

        # Garis pemisah antar blok i dan j
        ax.axhline(N - 0.5, color="white", linewidth=1.5)
        ax.axvline(N - 0.5, color="white", linewidth=1.5)

        # Label blok
        ax.text(N / 2 - 0.5, -1.2, "z_i", ha="center", fontsize=9, color="blue")
        ax.text(N + N / 2 - 0.5, -1.2, "z_j", ha="center", fontsize=9, color="red")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cosine Sim")

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/nt_xent_loss_demo.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: nt_xent_loss_demo.png")

    return loss_bad, loss_med, loss_good


# ─── 4. SIMULASI TRAINING LOOP ────────────────────────────────────────────

def simulate_training():
    """
    Simulasi proses training contrastive learning.
    Gunakan data sintetis (Gaussian blobs) sebagai pengganti dataset gambar.

    Ini mendemonstrasikan KONSEP tanpa membutuhkan PyTorch.
    """
    print("\n[4/6] Mensimulasikan training loop contrastive learning...")

    np.random.seed(123)

    # Simulasi: encoder menghasilkan embeddings yang makin baik per epoch
    n_classes = 5
    n_per_class = 50
    n_total = n_classes * n_per_class

    # Generate "gambar" sebagai vektor fitur 64-dim (menggantikan CNN output)
    X_raw, y = make_blobs(
        n_samples=n_total,
        n_features=64,
        centers=n_classes,
        cluster_std=3.0,
        random_state=42
    )

    # Simulasi pretraining: encoder yang makin baik per epoch
    # (dalam praktek, ini adalah loss contrastive yang turun)
    epochs = 50
    training_losses = []

    # Mulai dari random projection, makin baik seiring epoch
    W = np.random.randn(64, 16) * 0.5  # "encoder weight" awal

    for epoch in range(epochs):
        # Simulasi: weight makin baik (fitur makin terpisah)
        improvement = epoch / epochs
        W_improved = W + np.random.randn(64, 16) * 0.1 * (1 - improvement)

        # Hitung embeddings
        Z = X_raw @ W_improved
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)

        # Hitung contrastive loss (simulasi)
        # Gambar augmentasi = original + noise
        X_aug = X_raw + np.random.randn(*X_raw.shape) * 1.0
        Z_aug = X_aug @ W_improved
        Z_aug = Z_aug / (np.linalg.norm(Z_aug, axis=1, keepdims=True) + 1e-8)

        # Sample batch kecil (64 sampel)
        batch_idx = np.random.choice(n_total, 32, replace=False)
        loss = nt_xent_loss_numpy(Z[batch_idx], Z_aug[batch_idx], temperature=0.5)
        training_losses.append(float(loss))

        # Update W (gradient descent simulasi)
        W = W + 0.01 * np.random.randn(64, 16) * (1 - improvement)
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-8)

    # Embeddings akhir
    Z_final = X_raw @ W
    Z_final = Z_final / (np.linalg.norm(Z_final, axis=1, keepdims=True) + 1e-8)

    print(f"  Loss awal : {training_losses[0]:.4f}")
    print(f"  Loss akhir: {training_losses[-1]:.4f}")
    print(f"  Penurunan : {((training_losses[0] - training_losses[-1]) / training_losses[0] * 100):.1f}%")

    # Plot training loss
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(training_losses, "b-", linewidth=2, label="NT-Xent Loss")
    ax.fill_between(range(epochs), training_losses, alpha=0.15)

    # Running average
    window = 5
    running_avg = np.convolve(training_losses,
                               np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, epochs), running_avg, "r-",
            linewidth=1.5, alpha=0.8, label=f"Moving Avg (w={window})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("NT-Xent Loss", fontsize=11)
    ax.set_title("Training Loss SimCLR (Simulasi)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, epochs)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/training_history.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: training_history.png")

    return X_raw, y, Z_final, training_losses


# ─── 5. EVALUASI EMBEDDING QUALITY ────────────────────────────────────────

def evaluate_embeddings(X_raw, y, Z_final):
    """
    Evaluasi kualitas embeddings dengan Linear Probe dan KNN.
    Bandingkan embedding contrastive vs fitur raw.
    """
    print("\n[5/6] Evaluasi kualitas embedding...")

    n_total = len(y)
    n_train = int(0.7 * n_total)

    idx = np.random.permutation(n_total)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    # ── Fitur RAW (tanpa contrastive pretraining) ──────────────────
    scaler_raw = StandardScaler()
    X_train_raw = scaler_raw.fit_transform(X_raw[train_idx])
    X_test_raw = scaler_raw.transform(X_raw[test_idx])

    # Linear probe pada fitur raw
    clf_raw_lin = LogisticRegression(max_iter=500, random_state=42)
    clf_raw_lin.fit(X_train_raw, y[train_idx])
    acc_raw_linear = accuracy_score(y[test_idx], clf_raw_lin.predict(X_test_raw))

    # KNN pada fitur raw
    knn_raw = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_raw.fit(X_train_raw, y[train_idx])
    acc_raw_knn = accuracy_score(y[test_idx], knn_raw.predict(X_test_raw))

    # ── Embeddings CONTRASTIVE ──────────────────────────────────────
    scaler_emb = StandardScaler()
    Z_train = scaler_emb.fit_transform(Z_final[train_idx])
    Z_test = scaler_emb.transform(Z_final[test_idx])

    # Linear probe pada embedding
    clf_emb_lin = LogisticRegression(max_iter=500, random_state=42)
    clf_emb_lin.fit(Z_train, y[train_idx])
    acc_emb_linear = accuracy_score(y[test_idx], clf_emb_lin.predict(Z_test))

    # KNN pada embedding
    knn_emb = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn_emb.fit(Z_train, y[train_idx])
    acc_emb_knn = accuracy_score(y[test_idx], knn_emb.predict(Z_test))

    print(f"\n  {'Metode':<35} {'Accuracy':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Fitur Raw + Linear Probe':<35} {acc_raw_linear*100:>9.1f}%")
    print(f"  {'Fitur Raw + KNN (k=5)':<35} {acc_raw_knn*100:>9.1f}%")
    print(f"  {'Contrastive Emb + Linear Probe':<35} {acc_emb_linear*100:>9.1f}%")
    print(f"  {'Contrastive Emb + KNN (k=5)':<35} {acc_emb_knn*100:>9.1f}%")

    return {
        "raw_linear": acc_raw_linear,
        "raw_knn": acc_raw_knn,
        "emb_linear": acc_emb_linear,
        "emb_knn": acc_emb_knn,
    }


# ─── 6. VISUALIZATION LENGKAP ──────────────────────────────────────────────

def visualize_architecture():
    """Diagram arsitektur SimCLR dalam format visual."""
    print("\n[6/6] Membuat diagram arsitektur...")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    def box(ax, x, y, w, h, label, color, fontsize=9):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                               linewidth=1.5, edgecolor="black",
                               facecolor=color, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, zorder=3, fontweight="bold",
                wrap=True)

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black",
                                    lw=1.5), zorder=4)

    # Gambar input (atas dan bawah)
    box(ax, 1.5, 5.5, 1.8, 0.8, "Input\nImage x", "#aed6f1")
    box(ax, 1.5, 1.5, 1.8, 0.8, "Input\nImage x", "#aed6f1")

    # Augmentasi
    box(ax, 3.8, 5.5, 1.8, 0.8, "Augmentation\nt1(x) = x_i", "#a9dfbf")
    box(ax, 3.8, 1.5, 1.8, 0.8, "Augmentation\nt2(x) = x_j", "#a9dfbf")

    # Encoder (shared weights!)
    box(ax, 6.3, 3.5, 2.0, 1.6, "Encoder f\n(ResNet)\n[shared\nweights]", "#f9e79f", fontsize=8)
    ax.text(6.3, 4.7, "⟵ Same weights →", ha="center", va="center",
            fontsize=7, color="gray", style="italic")

    # Connecting arrows from x to aug
    arrow(ax, 2.4, 5.5, 2.9, 5.5)
    arrow(ax, 2.4, 1.5, 2.9, 1.5)

    # Aug to encoder
    arrow(ax, 4.7, 5.5, 5.0, 4.4)
    arrow(ax, 4.7, 1.5, 5.0, 2.6)

    # h outputs
    box(ax, 8.5, 5.5, 1.5, 0.7, "Repr. h_i\n(dim: 512)", "#f5cba7")
    box(ax, 8.5, 1.5, 1.5, 0.7, "Repr. h_j\n(dim: 512)", "#f5cba7")

    arrow(ax, 7.3, 4.4, 7.8, 5.3)
    arrow(ax, 7.3, 2.6, 7.8, 1.7)

    # Projection head
    box(ax, 10.2, 5.5, 1.5, 0.7, "Proj Head g\nz_i (dim:128)", "#d7bde2")
    box(ax, 10.2, 1.5, 1.5, 0.7, "Proj Head g\nz_j (dim:128)", "#d7bde2")

    arrow(ax, 9.2, 5.5, 9.4, 5.5)
    arrow(ax, 9.2, 1.5, 9.4, 1.5)

    # Loss
    box(ax, 12.2, 3.5, 1.8, 1.2, "NT-Xent\nLoss", "#e74c3c", fontsize=10)

    arrow(ax, 10.9, 5.5, 11.4, 4.2)
    arrow(ax, 10.9, 1.5, 11.4, 2.8)

    # Downstream note
    ax.annotate(
        "h digunakan\nuntuk downstream\n(bukan z!)",
        xy=(8.5, 5.1), xytext=(8.5, 3.5),
        fontsize=8, color="#1a5276",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#1a5276",
                         linestyle="dashed", lw=1.2),
    )

    ax.set_title("Arsitektur SimCLR\n(Chen et al., 2020)",
                 fontsize=14, fontweight="bold", pad=10)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/architecture.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: architecture.png")


def visualize_results(results, training_losses):
    """Visualisasi ringkasan hasil evaluasi."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ringkasan Hasil: Contrastive Learning vs Fitur Raw",
                 fontsize=13, fontweight="bold")

    # Plot 1: Perbandingan accuracy
    ax = axes[0]
    methods = ["Raw\n+Linear", "Raw\n+KNN", "Contrastive\n+Linear", "Contrastive\n+KNN"]
    accs = [
        results["raw_linear"] * 100,
        results["raw_knn"] * 100,
        results["emb_linear"] * 100,
        results["emb_knn"] * 100,
    ]
    bar_colors = ["#AEB6BF", "#AEB6BF", "#2ECC71", "#27AE60"]
    bars = ax.bar(methods, accs, color=bar_colors,
                   edgecolor="black", linewidth=1.1, width=0.55)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Perbandingan Metode Evaluasi", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(20, color="gray", linestyle="--", alpha=0.7,
               label="Random chance (5 kelas = 20%)")
    ax.legend(fontsize=9)

    # Highlight perkembangan
    improvement = results["emb_linear"] - results["raw_linear"]
    ax.annotate(
        f"↑ {improvement*100:.1f}% improvement\ndari contrastive learning",
        xy=(2, results["emb_linear"] * 100 + 1),
        xytext=(2.8, 80),
        fontsize=9, color="#1a5276",
        arrowprops=dict(arrowstyle="->", color="#1a5276"),
        ha="center"
    )

    # Plot 2: Training Loss
    ax2 = axes[1]
    epochs = len(training_losses)
    ax2.plot(training_losses, "b-", linewidth=2, alpha=0.6, label="Loss per step")

    window = max(3, epochs // 10)
    running_avg = np.convolve(training_losses,
                               np.ones(window) / window, mode="valid")
    ax2.plot(range(window - 1, epochs), running_avg, "r-",
             linewidth=2, label=f"Moving Avg (w={window})")

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("NT-Xent Loss", fontsize=11)
    ax2.set_title("Training Loss selama Pretraining", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/results_summary.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Tersimpan: results_summary.png")


# ─── PYTORCH REFERENCE (tidak dieksekusi, sebagai referensi) ──────────────

PYTORCH_REFERENCE = '''
# ============================================================
# PYTORCH IMPLEMENTATION (Referensi untuk Produksi)
# ============================================================
# Jalankan setelah: pip install torch torchvision
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torchvision

# 1. Model SimCLR
class SimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        n_features = backbone.fc.in_features
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return torch.flatten(h, 1)

    def forward(self, x):
        h = self.encode(x)
        z = self.projection_head(h)
        return h, z

# 2. NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.T = temperature

    def forward(self, z_i, z_j):
        N = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)               # (2N, D)
        sim = torch.mm(z, z.T) / self.T                 # (2N, 2N)
        mask = torch.eye(2*N, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))
        target = torch.cat([
            torch.arange(N, 2*N),
            torch.arange(0, N)
        ]).to(z.device)
        return F.cross_entropy(sim, target)

# 3. Augmentation Transform
class ContrastiveTransform:
    def __init__(self, size=96):
        self.t = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __call__(self, x):
        return self.t(x), self.t(x)

# 4. Training Loop
def train_simclr(model, loader, epochs=200, lr=3e-4, temperature=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = NTXentLoss(temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (x_i, x_j), _ in loader:
            x_i, x_j = x_i.to(device), x_j.to(device)
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(loader):.4f}")
    return model

# 5. Dataset STL-10 + Pretrain
if __name__ == "__main__":
    dataset = torchvision.datasets.STL10(
        root="./data", split="unlabeled",
        download=True,
        transform=ContrastiveTransform(96)
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True,
                        num_workers=4, drop_last=True)
    model = SimCLR(base_model="resnet18", out_dim=128)
    model = train_simclr(model, loader, epochs=200)
    torch.save(model.state_dict(), "simclr_pretrained.pt")
'''

# ─── MAIN ─────────────────────────────────────────────────────────────────

def main():
    # 1. Visualisasi Augmentasi
    demo_augmentation_effect()

    # 2. Intuisi NT-Xent dalam 2D
    demo_nt_xent_intuition()

    # 3. Demo NT-Xent computation
    loss_bad, loss_med, loss_good = demo_nt_xent_computation()

    # 4. Simulasi training loop
    X_raw, y, Z_final, training_losses = simulate_training()

    # 5. Evaluasi embeddings
    results = evaluate_embeddings(X_raw, y, Z_final)

    # 6. Diagram arsitektur
    visualize_architecture()

    # 7. Visualisasi hasil
    visualize_results(results, training_losses)

    # Simpan PyTorch reference
    ref_path = f"{OUTPUT_DIR}/pytorch_reference.py"
    with open(ref_path, "w") as f:
        f.write(PYTORCH_REFERENCE)
    print(f"\n  ✓ Tersimpan: pytorch_reference.py (kode PyTorch lengkap)")

    # Ringkasan akhir
    print("\n" + "=" * 62)
    print("  SELESAI! Ringkasan Contrastive Learning")
    print("=" * 62)
    print(f"""
  NT-Xent Loss:
    • Embedding buruk  → Loss {loss_bad:.3f} (tinggi)
    • Embedding sedang → Loss {loss_med:.3f}
    • Embedding baik   → Loss {loss_good:.3f} (rendah)

  Evaluasi Embedding Quality (data sintetis 5 kelas):
    • Fitur Raw + Linear Probe  : {results['raw_linear']*100:.1f}%
    • Fitur Raw + KNN           : {results['raw_knn']*100:.1f}%
    • Contrastive + Linear Probe: {results['emb_linear']*100:.1f}%
    • Contrastive + KNN         : {results['emb_knn']*100:.1f}%

  File yang dihasilkan:
    ✓ augmentation_demo.png   — Jenis-jenis augmentasi
    ✓ nt_xent_intuition.png   — Intuisi 2D contrastive learning
    ✓ nt_xent_loss_demo.png   — Similarity matrix untuk embedding baik/buruk
    ✓ training_history.png    — Training loss curve
    ✓ architecture.png        — Diagram arsitektur SimCLR
    ✓ results_summary.png     — Perbandingan hasil evaluasi
    ✓ pytorch_reference.py    — Kode PyTorch lengkap untuk produksi

  Untuk implementasi penuh dengan PyTorch + GPU:
    Lihat: pytorch_reference.py
    Install: pip install torch torchvision
    Dataset: STL-10 (unlabeled split, 100K gambar)
  """)


if __name__ == "__main__":
    main()
