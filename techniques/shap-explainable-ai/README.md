# SHAP: Explainable AI dengan Shapley Values

## Ringkasan

SHAP (SHapley Additive exPlanations) adalah metode untuk menjelaskan prediksi model machine learning berbasis teori permainan (Shapley values). SHAP memberikan jawaban atas pertanyaan krusial: **"Seberapa besar kontribusi setiap fitur terhadap prediksi model untuk instance tertentu?"**

Di era regulasi AI (GDPR, AI Act) dan kebutuhan transparansi, SHAP menjadi tool standar industri untuk explainable AI.

## Mengapa SHAP Penting?

### Problem Statement

Model ML modern (Random Forest, XGBoost, Neural Networks) sangat akurat namun **black-box**: sulit dijelaskan kenapa model membuat keputusan tertentu.

**Konsekuensi:**
- Regulator menolak deployment model di sektor regulated (finance, healthcare)
- User tidak trust rekomendasi model
- Data scientist tidak bisa debug model behavior
- Bias tersembunyi tidak terdeteksi

### Solusi: SHAP

SHAP memberikan:
1. **Feature importance** global: fitur mana yang paling berpengaruh di seluruh dataset
2. **Local explanation**: kontribusi setiap fitur untuk 1 prediksi spesifik
3. **Consistency**: properti matematis yang kuat (berbasis Shapley values dari game theory)
4. **Model-agnostic**: bekerja untuk semua model ML

## Intuisi Matematika: Shapley Values

### Analogi: Berbagi Bonus Tim

Bayangkan tim 3 orang (Alice, Bob, Charlie) menyelesaikan project dengan nilai $1000.

**Pertanyaan:** Bagaimana bagi bonus secara *fair* berdasarkan kontribusi masing-masing?

**Naive approach (marginal contribution):**
- Tambahkan Alice → value naik $300
- Tambahkan Bob → value naik $500
- Tambahkan Charlie → value naik $200
- Total = $1000 ✓

**Masalah:** Urutan matters! Jika Bob masuk duluan, kontribusi marginal-nya bisa berbeda (karena synergy dengan Alice).

**Shapley values = rata-rata kontribusi marginal di SEMUA kemungkinan urutan**

Untuk setiap pemain, kita:
1. Coba semua possible coalitions (subset pemain lain)
2. Hitung kontribusi marginal di setiap coalition
3. Rata-rata dengan weighted average

### Formal Definition

Untuk fitur *i*, Shapley value-nya:

```
φᵢ = Σ [|S|! (|F| - |S| - 1)! / |F|!] × [f(S ∪ {i}) - f(S)]
```

Di mana:
- *F* = set semua fitur
- *S* = subset fitur (coalition) yang tidak include *i*
- *f(S)* = expected prediction dengan hanya fitur di *S*
- Summation over all possible subsets *S*

**Properti Kunci:**
1. **Efficiency**: Σφᵢ = f(x) - f(∅) (sum of SHAP values = prediksi - baseline)
2. **Symmetry**: Fitur dengan kontribusi sama → Shapley value sama
3. **Dummy**: Fitur yang tidak berpengaruh → Shapley value = 0
4. **Additivity**: Konsisten untuk ensemble models

### Dari Game Theory ke ML

| Game Theory | Machine Learning |
|-------------|------------------|
| Pemain (players) | Fitur (features) |
| Koalisi (coalitions) | Subset fitur yang di-include |
| Payoff function | Model prediction |
| Shapley value | SHAP value (kontribusi fitur) |

## Implementasi SHAP

### Complexity Challenge

Menghitung exact Shapley values butuh evaluasi **2^n coalitions** (n = jumlah fitur).

Untuk 50 fitur → 2^50 ≈ 10^15 evaluations → **computationally intractable**

### SHAP Solutions

Library `shap` memberikan beberapa explainers yang efficient:

1. **TreeExplainer**: Untuk tree-based models (XGBoost, Random Forest, LightGBM)
   - Polynomial time complexity!
   - Exact Shapley values via algoritma khusus yang exploit tree structure

2. **KernelExplainer**: Model-agnostic
   - Approximate Shapley values via weighted linear regression
   - Slower, tapi works untuk any model

3. **DeepExplainer**: Untuk neural networks
   - Approximate via DeepLIFT algorithm
   - Fast untuk deep learning models

4. **LinearExplainer**: Untuk linear models
   - Exact dan sangat cepat

## Use Cases Real-World

### 1. Credit Scoring (Finance)
**Problem:** Bank menolak aplikasi kredit, nasabah minta penjelasan (hak konsumen per GDPR)

**SHAP Solution:**
```
Prediksi: REJECT (probability default = 0.78)

Top contributing factors:
  Credit score (650)      → +0.45 (push toward reject)
  Income ($35K/year)      → +0.20
  Debt-to-income (60%)    → +0.15
  Employment (2 years)    → -0.10 (slight push toward accept)
  Age (28)                → +0.08
```

**Value:** Transparency + actionable insight buat nasabah ("tingkatkan credit score untuk approval")

### 2. Medical Diagnosis (Healthcare)
**Problem:** Model prediksi sepsis di ICU, dokter butuh justifikasi sebelum treatment

**SHAP Solution:**
- Highlight vital signs yang paling berpengaruh untuk prediction
- Detect surprising patterns (model rely on feature yang medically questionable → investigate bias)
- Build trust dengan clinicians

### 3. E-commerce Recommendation
**Problem:** User diberi product recommendation, kenapa produk X muncul?

**SHAP Solution:**
```
Recommended: Nike Running Shoes

Factors:
  Past purchases (running gear)        → +0.35
  Recent searches (marathon training)  → +0.28
  Similar users bought this            → +0.18
  Price range match                    → +0.12
  Brand preference (Nike)              → +0.07
```

**Value:** Personalisasi yang explainable → user trust

### 4. Fraud Detection
**Problem:** Transaction di-flag sebagai fraud, merchant komplain

**SHAP Solution:**
- Explain kenapa specific transaction flagged
- Identify false positives (fitur yang misleading)
- Improve model dengan insight dari SHAP analysis

## Jenis Visualisasi SHAP

### 1. Waterfall Plot
Untuk **1 prediction**: menunjukkan bagaimana base value berubah step-by-step saat fitur ditambahkan

```
Base value (mean prediction): 0.3
+ Feature A (+0.2) → 0.5
+ Feature B (+0.1) → 0.6
+ Feature C (-0.05) → 0.55
= Final prediction: 0.55
```

### 2. Force Plot
Mirip waterfall, tapi horizontal dengan visual "pushing" left (negative) atau right (positive)

### 3. Summary Plot
Untuk **global importance**: dot plot yang menunjukkan SHAP value distribution untuk setiap fitur

- Y-axis: fitur (sorted by importance)
- X-axis: SHAP value
- Color: nilai fitur (merah = high, biru = low)
- Each dot = 1 instance

**Insight:**
- Feature importance ranking
- Correlation direction (fitur tinggi → prediksi naik/turun?)
- Spread (consistency of impact)

### 4. Dependence Plot
Scatter plot: fitur value vs SHAP value

**Insight:**
- Non-linear relationships
- Interaction effects (color by another feature)
- Threshold effects

### 5. Bar Plot
Simple: average |SHAP value| per fitur → ranking feature importance

## Step-by-Step Implementation

Lihat `notebook.py` atau `notebook.ipynb` untuk full working example dengan:

1. **Dataset:** Adult Income dataset (predict income >50K)
2. **Model:** XGBoost classifier
3. **SHAP explanation:** TreeExplainer
4. **Visualizations:** Summary plot, waterfall, dependence plots
5. **Analysis:** Interpretasi dan insights

## Tips & Best Practices

### ✅ Do's

1. **Start dengan Tree models + TreeExplainer**
   - Exact Shapley values, very fast
   - XGBoost, LightGBM, Random Forest

2. **Use background data wisely**
   - KernelExplainer butuh background dataset untuk estimate E[f(x)]
   - Sample 50-100 representative instances (bukan full dataset → slow)
   - k-means clustering untuk pilih representative samples

3. **Interpret SHAP values in context**
   - SHAP value = deviation from base value (expected prediction)
   - Base value ≠ 0, perhatikan scale
   - Compare dengan domain knowledge

4. **Combine global + local explanations**
   - Summary plot (global) → identify top features
   - Waterfall/force plot (local) → explain specific predictions

5. **Check interaction effects**
   - `shap.dependence_plot()` dengan `interaction_index='auto'`
   - SHAP interaction values: `shap.TreeExplainer.shap_interaction_values()`

### ❌ Don'ts

1. **Jangan assume SHAP = causality**
   - SHAP mengukur **correlation**, bukan **causation**
   - Fitur dengan SHAP value tinggi bisa jadi confounding variable

2. **Jangan gunakan KernelExplainer untuk large datasets**
   - Computational cost: O(n_samples × n_features × n_background)
   - Gunakan TreeExplainer/LinearExplainer jika possible

3. **Jangan interpretasi SHAP tanpa domain knowledge**
   - SHAP bisa expose **spurious correlations** atau **data leakage**
   - Contoh: model rely on patient ID (uniquely identifies outcome) → high SHAP value, tapi tidak meaningful

4. **Jangan over-trust single instance explanation**
   - Local explanations bisa misleading tanpa global context
   - Check consistency di multiple instances

## Common Pitfalls

### 1. Confusing SHAP dengan Feature Importance

**Scikit-learn feature_importances_** (untuk tree models):
- Based on **impurity decrease** (Gini/entropy reduction)
- Global metric only
- Tidak account untuk feature interactions properly

**SHAP:**
- Based on **Shapley values** (game theory)
- Global (summary plot) + local (per prediction)
- Consistent, fair attribution dengan solid math foundation

### 2. Interpretasi SHAP values di Classification vs Regression

**Regression:** SHAP value = kontribusi ke **predicted value** (same unit sebagai target)

**Classification:** SHAP value = kontribusi ke **log-odds** (atau probability, tergantung implementation)
- Perhatikan scale: log-odds bisa misleading
- Gunakan `shap.plots.waterfall(shap_values[i], show=True)` yang handle scaling

### 3. Background Data untuk KernelExplainer

KernelExplainer butuh `background_data` = dataset untuk estimate expected value.

**Pitfall:** Pakai full training set → explainer super lambat

**Fix:** Sample 50-100 instances via k-means:
```python
background = shap.kmeans(X_train, 50)
explainer = shap.KernelExplainer(model.predict, background)
```

## Evaluation Metrics untuk Explanations

Bagaimana tahu explanation-nya "bagus"?

### 1. Consistency Check
- Remove top features dari SHAP explanation → prediction should change significantly
- Remove bottom features → prediction should stay similar

### 2. Fidelity
- Train simple interpretable model (linear) on SHAP values → how well it approximates black-box?
- High R² → SHAP captures model behavior well

### 3. Stability
- Add small noise to input → SHAP values shouldn't change drastically
- Unstable explanations = model overfitting atau data noise

### 4. Human Evaluation
- Domain experts review explanations → do they make sense?
- User study: apakah explanation meningkatkan trust/understanding?

## Advanced Topics

### SHAP Interaction Values

Standard SHAP values = main effects only.

**SHAP interaction values** = pairwise interaction effects.

```python
shap_interaction = explainer.shap_interaction_values(X)
# Shape: (n_samples, n_features, n_features)
# shap_interaction[i, j, k] = interaction effect antara fitur j dan k untuk sample i
```

**Use case:** Detect non-additive effects (e.g., "Age × Income" interaction dalam credit scoring)

### SHAP for Deep Learning

**DeepExplainer:** Approximate Shapley values untuk neural networks

**Challenge:** Deep networks = banyak parameters, complex interactions

**Alternative:**
- **Gradient-based methods:** Integrated Gradients, GradCAM (lebih cepat, tapi properti berbeda)
- **Attention mechanisms:** Untuk transformers, attention weights ≈ importance

Trade-off: speed vs theoretical guarantees (Shapley values = axiomatically "fair")

### SHAP for NLP

**Text classification:**
- Tokenize text → setiap token = feature
- TreeExplainer (untuk tree models) atau DeepExplainer (untuk transformers)
- Visualize: color-code words by SHAP value

**Challenge:** High-dimensional (vocab size besar), interactions antar tokens

### SHAP for Computer Vision

**Image classification:**
- Superpixel segmentation → setiap superpixel = feature
- Compute SHAP values per superpixel
- Visualize: heatmap overlay on image

**Alternative:** GradCAM, LIME (lebih cepat untuk images)

## Alternatif Explainability Methods

| Method | Pros | Cons |
|--------|------|------|
| **SHAP** | Theoretical guarantees (Shapley values), consistent, additive | Computational cost (untuk large models/data) |
| **LIME** | Fast, model-agnostic, local explanations | Unstable, tidak consistent, arbitrary sampling |
| **Permutation Importance** | Simple, intuitive | Global only, tidak account untuk interactions |
| **Partial Dependence Plot (PDP)** | Visualize feature effect | Assumes feature independence, global only |
| **Integrated Gradients** | Gradient-based (fast untuk NN) | Requires differentiable model, local only |
| **Attention Mechanisms** | Native untuk transformers | Model-specific, tidak general-purpose |

**Rekomendasi:**
- **Tree models:** SHAP (TreeExplainer) — exact dan fast
- **Linear models:** Coefficients (built-in) atau SHAP (LinearExplainer)
- **Neural networks:** Start dengan SHAP (DeepExplainer), consider Integrated Gradients jika terlalu lambat
- **Quick prototype:** LIME (faster than SHAP, acceptable for exploration)

## Resources

### Papers

1. **Original SHAP paper:**
   Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions"
   - NIPS 2017
   - https://arxiv.org/abs/1705.07874

2. **TreeExplainer algorithm:**
   Lundberg et al. (2020). "From local explanations to global understanding with explainable AI for trees"
   - Nature Machine Intelligence
   - https://www.nature.com/articles/s42256-019-0138-9

3. **Shapley values in ML:**
   Štrumbelj & Kononenko (2014). "Explaining prediction models and individual predictions with feature contributions"
   - Knowledge and Information Systems

4. **Survey paper:**
   Molnar (2022). "Interpretable Machine Learning" (Book, Chapter 9: Shapley Values)
   - https://christophm.github.io/interpretable-ml-book/shapley.html

### Libraries & Tools

- **shap:** https://github.com/slundberg/shap
  - Official library, well-maintained
  - `pip install shap`

- **Shapash:** https://github.com/MAIF/shapash
  - Enterprise-ready explainability dashboard
  - Built on top of SHAP, user-friendly interface

- **Alibi Explain:** https://github.com/SeldonIO/alibi
  - Multiple explainability methods (SHAP, anchors, counterfactuals)

### Tutorials & Courses

- **SHAP documentation:** https://shap.readthedocs.io/
- **Kaggle Learn:** Intermediate Machine Learning — Model Interpretability
- **Christoph Molnar's book:** Interpretable Machine Learning (free online)

### Datasets untuk Practice

1. **Adult Income:** Predict income >50K (classification)
   - UCI ML Repository
   - Good for fairness analysis (sensitive attributes: race, gender)

2. **Boston Housing:** Predict house prices (regression)
   - Built-in scikit-learn
   - Good for feature importance comparison

3. **COMPAS:** Recidivism prediction (classification)
   - ProPublica dataset
   - Famous for bias analysis

4. **Credit Card Fraud:** Imbalanced classification
   - Kaggle dataset
   - Good untuk business context (explain false positives)

## Kesimpulan

SHAP adalah **gold standard** untuk explainable AI dengan foundation matematis yang kuat (Shapley values). 

**Key Takeaways:**
1. SHAP memberikan fair attribution untuk setiap fitur via game theory
2. Efficient implementations (TreeExplainer) membuat SHAP praktis untuk production
3. Global + local explanations = comprehensive understanding
4. Critical untuk regulatory compliance dan user trust
5. Powerful diagnostic tool untuk data scientists (detect bias, leakage, spurious correlations)

**When to use:**
- Regulated industries (finance, healthcare) — compliance requirement
- High-stakes decisions (credit, hiring, medical) — transparency + fairness
- Model debugging — understanding model behavior
- Stakeholder communication — non-technical audience

**Trade-offs:**
- Computational cost vs interpretability guarantees
- Correlation vs causation (SHAP ≠ causal inference)
- Model-specific vs model-agnostic explainers

SHAP bukan silver bullet, tapi dengan proper understanding dan domain knowledge, SHAP adalah tool yang sangat powerful untuk membuat AI systems yang transparent, trustworthy, dan actionable.

---

**Next Steps:**
1. Run `notebook.py` atau `notebook.ipynb`
2. Experiment dengan dataset lain
3. Explore SHAP interaction values
4. Integrate SHAP ke ML pipeline anda
5. Read original paper untuk deep understanding

Happy explaining! 🎯
