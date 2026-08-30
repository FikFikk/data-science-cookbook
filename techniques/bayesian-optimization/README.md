# Bayesian Optimization: Hyperparameter Tuning Cerdas dengan Gaussian Process

## Ringkasan

Bayesian Optimization adalah teknik optimasi black-box function yang sangat efisien untuk hyperparameter tuning. Berbeda dengan Grid Search atau Random Search yang "bodoh", Bayesian Optimization menggunakan model probabilistik (Gaussian Process) untuk memprediksi kombinasi hyperparameter mana yang paling menjanjikan, sehingga bisa menemukan setting optimal dengan evaluasi jauh lebih sedikit.

**Kapan menggunakan:**
- Hyperparameter tuning model ML yang training-nya lama
- Optimasi black-box function yang mahal dievaluasi
- Budget evaluasi terbatas (misal: hanya bisa 50-100 trials)
- Fungsi objektif yang noisy dan non-convex

**Keunggulan:**
- Sample efficient: butuh jauh lebih sedikit evaluasi dibanding Grid/Random Search
- Menangani trade-off exploration vs exploitation dengan baik
- Bisa menangani continuous, discrete, dan categorical parameters
- Menyediakan uncertainty estimation

## Intuisi Matematis

### Konsep Dasar

Bayesian Optimization bekerja dalam iterasi:
1. **Surrogate Model**: Membangun model probabilistik (biasanya Gaussian Process) dari hasil evaluasi sebelumnya
2. **Acquisition Function**: Menentukan titik mana yang paling "menjanjikan" untuk dievaluasi selanjutnya
3. **Evaluasi**: Jalankan fungsi objektif pada titik terpilih
4. **Update**: Tambahkan hasil ke data, update model, ulangi

### Gaussian Process (GP)

GP adalah distribusi probabilitas atas fungsi. Untuk setiap titik x, GP memberikan:
- **Mean μ(x)**: prediksi nilai fungsi
- **Variance σ²(x)**: tingkat ketidakpastian

Secara matematis:
```
f(x) ~ GP(μ(x), k(x, x'))
```

Di mana k(x, x') adalah kernel function (biasanya RBF/Squared Exponential):
```
k(x, x') = σ² exp(-||x - x'||² / (2l²))
```

**Intuisi**: 
- Di dekat titik yang sudah dievaluasi → variance rendah (lebih yakin)
- Jauh dari titik evaluasi → variance tinggi (tidak yakin)

### Acquisition Functions

Acquisition function α(x) menentukan titik mana yang worth untuk dievaluasi. Trade-off:
- **Exploitation**: pilih x dengan μ(x) tinggi (area yang sudah terlihat bagus)
- **Exploration**: pilih x dengan σ(x) tinggi (area yang belum dieksplorasi)

#### 1. Expected Improvement (EI)

Paling populer. Mengukur expected improvement dibanding nilai terbaik sejauh ini (f*):

```
EI(x) = E[max(f(x) - f*, 0)]
      = (μ(x) - f*) Φ(Z) + σ(x) φ(Z)

di mana Z = (μ(x) - f*) / σ(x)
```

- Φ: CDF dari distribusi normal standar
- φ: PDF dari distribusi normal standar

**Intuisi**: Menyeimbangkan "seberapa bagus prediksinya" dengan "seberapa tidak pasti kita".

#### 2. Upper Confidence Bound (UCB)

```
UCB(x) = μ(x) + κ σ(x)
```

Parameter κ mengontrol exploration-exploitation:
- κ besar → lebih explore
- κ kecil → lebih exploit

#### 3. Probability of Improvement (PI)

```
PI(x) = P(f(x) > f*) = Φ((μ(x) - f*) / σ(x))
```

### Mengapa Bayesian Optimization Efisien?

**Sample Complexity**: Untuk menemukan optimum ε-akurat, BO butuh O(log(1/ε)) evaluasi, sedangkan Grid Search butuh O((1/ε)^d) dengan d = dimensi.

**Contoh konkret**: 
- Grid Search 10 nilai × 5 hyperparameter = 100,000 evaluasi
- Bayesian Optimization = 50-200 evaluasi untuk hasil serupa

## Implementasi Step-by-Step

### 1. Setup dan Dependencies

```python
pip install scikit-optimize numpy pandas matplotlib scikit-learn
```

### 2. Contoh Sederhana: Optimasi Fungsi 1D

```python
import numpy as np
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

# Fungsi target (misalkan kita tidak tahu bentuknya)
def black_box_function(x):
    return -(x[0] ** 2) + 10 * np.sin(x[0])

# Definisikan search space
space = [(-5.0, 5.0)]  # x ∈ [-5, 5]

# Jalankan Bayesian Optimization
result = gp_minimize(
    black_box_function,    # fungsi yang ingin diminimize
    space,                 # search space
    n_calls=20,           # jumlah evaluasi
    random_state=42,
    acq_func='EI'         # acquisition function
)

print(f"Optimum ditemukan di x = {result.x[0]:.4f}")
print(f"Nilai fungsi = {result.fun:.4f}")
```

### 3. Hyperparameter Tuning untuk Model ML

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Generate dummy dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Definisikan search space
search_spaces = {
    'n_estimators': Integer(10, 200),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

# Bayesian Optimization dengan cross-validation
opt = BayesSearchCV(
    RandomForestClassifier(random_state=42),
    search_spaces,
    n_iter=50,           # 50 evaluasi
    cv=5,
    n_jobs=-1,
    random_state=42
)

opt.fit(X, y)

print("Best parameters:", opt.best_params_)
print("Best CV score:", opt.best_score_)
```

### 4. Custom Objective Function

```python
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(params):
    """Custom objective function untuk tuning XGBoost"""
    learning_rate, max_depth, n_estimators, subsample = params
    
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        subsample=subsample,
        random_state=42
    )
    
    # Minimize negative accuracy (karena gp_minimize mencari minimum)
    score = cross_val_score(model, X, y, cv=3, n_jobs=-1).mean()
    return -score  # negative karena kita ingin maximize accuracy

# Define search space
space = [
    Real(0.01, 0.3, name='learning_rate'),
    Integer(3, 10, name='max_depth'),
    Integer(50, 300, name='n_estimators'),
    Real(0.5, 1.0, name='subsample')
]

# Run optimization
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    random_state=42,
    verbose=True
)

print(f"Best parameters: {result.x}")
print(f"Best accuracy: {-result.fun:.4f}")
```

## Dataset dan Preprocessing

### Dataset untuk Demo

Kita akan gunakan beberapa dataset:
1. **Synthetic 1D function**: untuk visualisasi konsep
2. **Breast Cancer**: klasifikasi biner untuk hyperparameter tuning
3. **Fashion MNIST**: dataset lebih kompleks untuk deep learning tuning

```python
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisasi (penting untuk convergence)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Preprocessing Best Practices

1. **Scaling**: Bayesian Optimization sensitif terhadap skala parameter. Gunakan log-scaling untuk parameter yang rentangnya besar:
```python
from skopt.space import Real

# Jangan: Real(0.001, 1000) → terlalu lebar, tidak seimbang
# Lakukan: Real(1e-3, 1e3, prior='log-uniform') → log-scale
```

2. **Search Space Design**: Jangan terlalu lebar atau sempit
```python
# Terlalu lebar → butuh banyak evaluasi
space = [Real(1e-10, 1e10)]  # ❌

# Lebih baik: gunakan domain knowledge
space = [Real(1e-4, 1e-1, prior='log-uniform')]  # ✓
```

3. **Handling Categorical**: Encode categorical parameters dengan benar
```python
from skopt.space import Categorical

space = [
    Categorical(['gini', 'entropy'], name='criterion'),
    Categorical([True, False], name='bootstrap')
]
```

## Evaluation Metrics

### 1. Convergence Plot

Visualisasi bagaimana performa meningkat seiring iterasi:

```python
from skopt.plots import plot_convergence

plot_convergence(result)
plt.title('Convergence Plot')
plt.show()
```

**Interpretasi**:
- Garis merah = minimum yang ditemukan sejauh ini
- Biru = nilai di setiap iterasi
- Flat di akhir = sudah konvergen

### 2. Objective Function Landscape

```python
from skopt.plots import plot_objective

plot_objective(result, dimensions=['learning_rate', 'max_depth'])
plt.show()
```

Menunjukkan bagaimana objective function berubah terhadap parameter.

### 3. Evaluations Plot

```python
from skopt.plots import plot_evaluations

plot_evaluations(result)
plt.show()
```

Menunjukkan nilai parameter yang dievaluasi dan hasil objective-nya.

### 4. Regret Analysis

Regret mengukur seberapa jauh kita dari optimum sebenarnya:

```python
def simple_regret(result):
    """Simple regret: perbedaan antara best found vs true optimum"""
    return np.minimum.accumulate(result.func_vals)

regret = simple_regret(result)
plt.plot(regret)
plt.xlabel('Iteration')
plt.ylabel('Simple Regret')
plt.title('Simple Regret over Iterations')
plt.show()
```

### 5. Comparison dengan Baseline

Bandingkan dengan Random Search:

```python
from skopt.benchmarks import bench1
from skopt import dummy_minimize  # random search

# Bayesian Optimization
res_bo = gp_minimize(bench1, [(-2.0, 2.0)], n_calls=50, random_state=42)

# Random Search
res_rs = dummy_minimize(bench1, [(-2.0, 2.0)], n_calls=50, random_state=42)

print(f"BO best: {res_bo.fun:.4f}")
print(f"Random Search best: {res_rs.fun:.4f}")
```

## Real-World Applications

### 1. Deep Learning Hyperparameter Tuning

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
import tensorflow as tf

def train_neural_network(params):
    learning_rate, batch_size, dropout_rate = params
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        batch_size=int(batch_size),
        epochs=10,
        validation_split=0.2,
        verbose=0
    )
    
    # Return negative validation accuracy
    return -max(history.history['val_accuracy'])

space = [
    Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
    Integer(16, 256, name='batch_size'),
    Real(0.0, 0.5, name='dropout_rate')
]

result = gp_minimize(train_neural_network, space, n_calls=30)
```

### 2. AutoML Pipeline Optimization

```python
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def optimize_pipeline(params):
    n_components, n_estimators, max_depth = params
    
    pipeline = Pipeline([
        ('pca', PCA(n_components=int(n_components))),
        ('clf', RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            random_state=42
        ))
    ])
    
    score = cross_val_score(pipeline, X, y, cv=3).mean()
    return -score

space = [
    Integer(5, 50, name='n_components'),
    Integer(50, 300, name='n_estimators'),
    Integer(3, 20, name='max_depth')
]

result = gp_minimize(optimize_pipeline, space, n_calls=40)
```

### 3. A/B Testing Optimization

Mengoptimalkan parameter kampanye marketing:

```python
def ab_test_objective(params):
    """
    Simulate A/B testing dengan budget constraints
    params: [price, discount_percentage, ad_spend]
    """
    price, discount, ad_spend = params
    
    # Simulate conversion rate (dalam real-world, ini dari actual A/B test)
    base_conversion = 0.05
    conversion_rate = base_conversion * (1 - discount/100) * (1 + ad_spend/1000)
    
    revenue = conversion_rate * price * 1000  # 1000 visitors
    cost = ad_spend
    profit = revenue - cost
    
    return -profit  # maximize profit

space = [
    Real(10, 100, name='price'),
    Real(0, 50, name='discount_percentage'),
    Real(100, 5000, name='ad_spend')
]

result = gp_minimize(ab_test_objective, space, n_calls=50)
```

### 4. Scientific Experiment Design

Mengoptimalkan kondisi eksperimen kimia/biologi:

```python
def experiment_outcome(params):
    """
    Optimize chemical reaction conditions
    params: [temperature, pressure, catalyst_amount]
    """
    temp, pressure, catalyst = params
    
    # Simulate experimental yield (dalam praktik: jalankan eksperimen sungguhan)
    yield_rate = (temp - 300)**2 / 10000 + (pressure - 2)**2 + catalyst * 0.1
    
    return yield_rate  # minimize (want low value for some metric)

space = [
    Real(250, 350, name='temperature_celsius'),
    Real(1, 5, name='pressure_bar'),
    Real(0.1, 2.0, name='catalyst_grams')
]

result = gp_minimize(experiment_outcome, space, n_calls=20)  # 20 eksperimen
```

## Tips dan Best Practices

### 1. Initial Points

Mulai dengan beberapa random evaluations sebelum menggunakan GP:

```python
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    n_initial_points=10,  # 10 random evaluations pertama
    initial_point_generator='lhs'  # Latin Hypercube Sampling
)
```

**Mengapa?** GP butuh data awal untuk membangun model yang reasonable.

### 2. Acquisition Function Choice

- **EI (Expected Improvement)**: default, paling robust
- **UCB (Upper Confidence Bound)**: lebih aggressive exploration
- **PI (Probability of Improvement)**: lebih conservative

```python
# Untuk ruang pencarian yang sangat besar
result = gp_minimize(objective, space, acq_func='UCB', acq_optimizer='lbfgs')
```

### 3. Parallel Evaluation

Jika bisa menjalankan evaluasi parallel:

```python
from skopt import Optimizer

opt = Optimizer(space, base_estimator='GP', acq_func='EI')

# Ask for multiple points at once
x_candidates = opt.ask(n_points=4)  # 4 parallel evaluations

# Evaluate in parallel
y_values = [objective(x) for x in x_candidates]

# Tell optimizer the results
opt.tell(x_candidates, y_values)
```

### 4. Saving and Resuming

```python
from skopt import dump, load

# Save hasil
dump(result, 'optimization_result.pkl')

# Resume optimization
result_loaded = load('optimization_result.pkl')

# Continue dari checkpoint
result_continued = gp_minimize(
    objective,
    space,
    n_calls=20,  # 20 evaluasi tambahan
    x0=result_loaded.x_iters,  # starting points
    y0=result_loaded.func_vals  # known values
)
```

### 5. Handling Noisy Objectives

Jika objective function noisy (misal: stochastic training):

```python
result = gp_minimize(
    objective,
    space,
    n_calls=50,
    noise=0.1,  # expected noise level
    acq_func='EI'
)
```

Atau jalankan multiple runs dan rata-rata:

```python
def robust_objective(params):
    scores = []
    for seed in range(3):  # 3 random seeds
        score = train_model(params, random_state=seed)
        scores.append(score)
    return np.mean(scores)  # rata-rata untuk reduce noise
```

## Common Pitfalls dan Solusinya

### ❌ Pitfall 1: Search Space Terlalu Besar

```python
# Buruk: 10 dimensi dengan range sangat lebar
space = [Real(1e-10, 1e10) for _ in range(10)]
```

**Solusi**: Gunakan domain knowledge untuk mempersempit:
```python
# Lebih baik
space = [
    Real(1e-4, 1e-1, prior='log-uniform', name='learning_rate'),
    Integer(50, 200, name='n_estimators')
]
```

### ❌ Pitfall 2: Terlalu Sedikit Evaluasi

Untuk d dimensi, aturan praktis: minimal 10d evaluasi.

```python
# 5 dimensi → minimal 50 evaluations
result = gp_minimize(objective, space, n_calls=50)
```

### ❌ Pitfall 3: Tidak Memvalidasi dengan Data Holdout

```python
# Buruk: optimize langsung di test set
def bad_objective(params):
    model.fit(X_train, y_train)
    return -model.score(X_test, y_test)  # ❌ OVERFITTING!
```

**Solusi**: Gunakan cross-validation:
```python
def good_objective(params):
    model = build_model(params)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    return -score  # ✓ Lebih robust
```

### ❌ Pitfall 4: Mengabaikan Kernel Choice

Default RBF kernel bagus untuk smooth functions. Untuk non-smooth:

```python
from skopt.learning import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

gpr = GaussianProcessRegressor(
    kernel=Matern(nu=2.5),  # less smooth than RBF
    normalize_y=True
)

result = gp_minimize(objective, space, base_estimator=gpr)
```

### ❌ Pitfall 5: Tidak Memonitor Convergence

Selalu cek apakah sudah konvergen:

```python
plot_convergence(result)

# Jika belum flat, tambah evaluasi
if result.func_vals[-10:].std() > 0.01:
    print("Belum konvergen, butuh lebih banyak evaluasi")
```

## Perbandingan dengan Metode Lain

| Metode | Evaluasi Butuh | Parallelizable | Handling Categorical | Best For |
|--------|---------------|----------------|---------------------|----------|
| **Grid Search** | O((n/k)^d) | ✓✓✓ | ✓ | Small space, discrete |
| **Random Search** | O(n) | ✓✓✓ | ✓ | Large space, baseline |
| **Bayesian Opt** | O(log n) | ✓ | ✓ | Expensive evaluations |
| **Hyperband** | O(n log n) | ✓✓ | ✗ | Iterative algorithms |
| **Optuna (TPE)** | O(log n) | ✓✓ | ✓✓ | Mixed search spaces |

**Kapan menggunakan Bayesian Optimization:**
- ✓ Setiap evaluasi mahal (training lama, A/B test, eksperimen fisik)
- ✓ Budget evaluasi terbatas (< 500 trials)
- ✓ Search space continuous atau mixed
- ✗ Butuh parallelization masif (> 100 workers)
- ✗ Fungsi objektif sangat noisy

## Referensi dan Resources

### Papers

1. **Mockus, J. (1975)** - "On Bayesian methods for seeking the extremum"
   - Paper foundational tentang Bayesian Optimization

2. **Snoek, J., Larochelle, H., & Adams, R. P. (2012)** - "Practical Bayesian Optimization of Machine Learning Algorithms"
   - Aplikasi BO untuk hyperparameter tuning ML
   - Link: https://arxiv.org/abs/1206.2944

3. **Shahriari, B., et al. (2016)** - "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
   - Review paper yang sangat comprehensive
   - Link: https://ieeexplore.ieee.org/document/7352306

4. **Frazier, P. I. (2018)** - "A Tutorial on Bayesian Optimization"
   - Tutorial matematis yang detail tapi readable
   - Link: https://arxiv.org/abs/1807.02811

### Libraries

1. **scikit-optimize (skopt)**
   - Docs: https://scikit-optimize.github.io/
   - Paling user-friendly, terintegrasi dengan sklearn

2. **GPyOpt**
   - Docs: https://sheffieldml.github.io/GPyOpt/
   - Lebih fleksibel, banyak acquisition functions

3. **BoTorch (PyTorch)**
   - Docs: https://botorch.org/
   - State-of-the-art, untuk advanced users

4. **Optuna**
   - Docs: https://optuna.org/
   - Tree-structured Parzen Estimator (TPE), bukan GP
   - Sangat bagus untuk praktisi

### Courses & Tutorials

1. **Cornell CS4780** - Bayesian Optimization lecture
   - Video: https://www.youtube.com/watch?v=c4KKvyWW_Xk

2. **Distill.pub** - "Exploring Bayesian Optimization"
   - Interactive tutorial: https://distill.pub/2020/bayesian-optimization/

3. **Martin Krasser's Blog**
   - Implementasi from scratch dengan visualisasi bagus
   - Link: http://krasserm.github.io/2018/03/21/bayesian-optimization/

### Books

1. **Rasmussen & Williams (2006)** - "Gaussian Processes for Machine Learning"
   - Textbook definitif untuk GP
   - Free: http://gaussianprocess.org/gpml/

2. **Brochu, E., Cora, V. M., & De Freitas, N. (2010)** - "A Tutorial on Bayesian Optimization of Expensive Cost Functions"
   - Tutorial paper yang sangat praktis

### Online Tools

1. **WandB Sweeps**: https://wandb.ai/sweeps
   - Bayesian optimization untuk deep learning, hosted

2. **SigOpt**: https://sigopt.com/
   - Commercial BO service (ada free tier)

## Kesimpulan

Bayesian Optimization adalah senjata ampuh untuk hyperparameter tuning dan optimasi black-box function. Kunci sukses:

1. **Design search space dengan baik**: gunakan domain knowledge
2. **Pilih acquisition function yang sesuai**: EI untuk umum, UCB untuk exploration
3. **Monitor convergence**: jangan berhenti terlalu cepat atau terlalu lama
4. **Validasi dengan proper**: gunakan CV, bukan test set
5. **Compare dengan baseline**: minimal bandingkan dengan Random Search

Untuk kebanyakan kasus praktis dengan budget evaluasi < 200, Bayesian Optimization akan mengalahkan Grid Search dan Random Search dengan signifikan.

---

**Dibuat oleh**: Hermes Agent (Nous Research)  
**Tanggal**: 22 Juni 2026  
**License**: MIT
