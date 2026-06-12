import os
import json
import glob
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.expanduser('~/epistemic-steering')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'benchmark_activations_v2')
OUT_DIR = os.path.join(BASE_DIR, 'data')
PROBE_DIR = os.path.join(OUT_DIR, 'probes')
os.makedirs(PROBE_DIR, exist_ok=True)

benchmarks = ['arc_challenge', 'gsm8k', 'humaneval', 'math', 'mmlu', 'triviaqa']

print('[D3] Loading activations...')
X_list = []
y_list = []
for bench in benchmarks:
    bench_dir = os.path.join(DATA_DIR, bench)
    npy_files = sorted(glob.glob(os.path.join(bench_dir, '*.npy')))
    print(f'  {bench}: {len(npy_files)} samples')
    for npy_path in npy_files:
        json_path = npy_path.replace('.npy', '.json')
        if not os.path.exists(json_path):
            continue
        vec = np.load(npy_path)
        with open(json_path, 'r') as f:
            meta = json.load(f)
        dataset_label = meta.get('dataset', bench)
        X_list.append(vec)
        y_list.append(dataset_label)

X = np.stack(X_list)
y = np.array(y_list)
print(f'[D3] Total samples: {X.shape[0]}, dim: {X.shape[1]}')

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f'[D3] Train: {len(y_train)}, Test: {len(y_test)}')

# --- Linear Gate (LogisticRegressionCV) ---
print('[D3] Training linear gate...')
clf = LogisticRegressionCV(
    Cs=10, cv=5, max_iter=1000,
    random_state=SEED, n_jobs=-1, solver='lbfgs'
)
clf.fit(X_train, y_train)
y_pred_linear = clf.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)
cm_linear = confusion_matrix(y_test, y_pred_linear, labels=clf.classes_)
print(f'[D3] Linear gate accuracy: {acc_linear:.4f}')
print(f'[D3] Best C: {clf.C_}')

# Save gate weights
np.savez(
    os.path.join(PROBE_DIR, 'gate_linear.npz'),
    coef_=clf.coef_,
    intercept_=clf.intercept_,
    classes_=clf.classes_,
    C_=clf.C_
)
print('[D3] Saved gate_linear.npz')

# --- VecStat Baseline (per-domain mean/variance Gaussian KL) ---
print('[D3] Computing VecStat baseline...')
# Compute per-domain mean and variance (diagonal covariance)
domain_params = {}
for domain in np.unique(y_train):
    mask = y_train == domain
    mu = X_train[mask].mean(axis=0)
    var = X_train[mask].var(axis=0) + 1e-6  # regularize
    domain_params[domain] = (mu, var)

# Predict by minimizing negative log-likelihood (Gaussian)
def gaussian_nll(x, mu, var):
    return 0.5 * np.sum(np.log(var) + (x - mu)**2 / var)

y_pred_vec = []
for x in X_test:
    scores = {d: gaussian_nll(x, mu, var) for d, (mu, var) in domain_params.items()}
    y_pred_vec.append(min(scores, key=scores.get))
acc_vec = accuracy_score(y_test, np.array(y_pred_vec))
print(f'[D3] VecStat accuracy: {acc_vec:.4f}')

# --- NormStat Baseline (per-domain norm mean/variance) ---
print('[D3] Computing NormStat baseline...')
norm_train = np.linalg.norm(X_train, axis=1)
norm_test = np.linalg.norm(X_test, axis=1)

norm_params = {}
for domain in np.unique(y_train):
    mask = y_train == domain
    mu_norm = norm_train[mask].mean()
    var_norm = norm_train[mask].var() + 1e-6
    norm_params[domain] = (mu_norm, var_norm)

y_pred_norm = []
for n in norm_test:
    scores = {d: (n - mu)**2 / var for d, (mu, var) in norm_params.items()}
    y_pred_norm.append(min(scores, key=scores.get))
acc_norm = accuracy_score(y_test, np.array(y_pred_norm))
print(f'[D3] NormStat accuracy: {acc_norm:.4f}')

# --- Build results dict ---
per_class_acc = {}
for i, cls in enumerate(clf.classes_):
    mask = y_test == cls
    if mask.sum() > 0:
        per_class_acc[cls] = float((y_pred_linear[mask] == cls).mean())

results = {
    'linear_gate': {
        'accuracy': float(acc_linear),
        'best_C': [float(c) for c in clf.C_],
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm_linear.tolist(),
        'classes': clf.classes_.tolist(),
    },
    'vecstat': {
        'accuracy': float(acc_vec),
    },
    'normstat': {
        'accuracy': float(acc_norm),
    },
    'test_size': len(y_test),
    'train_size': len(y_train),
    'seed': SEED,
}

results_path = os.path.join(OUT_DIR, 'd3_gate_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'[D3] Saved {results_path}')

# --- Confusion matrix plot ---
print('[D3] Generating confusion matrix plot...')
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_linear, cmap='Blues')
ax.set_xticks(np.arange(len(clf.classes_)))
ax.set_yticks(np.arange(len(clf.classes_)))
ax.set_xticklabels(clf.classes_, rotation=45, ha='right')
ax.set_yticklabels(clf.classes_)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'D3 Domain Gate Confusion Matrix\nLinear Gate Acc={acc_linear:.3f} | VecStat={acc_vec:.3f} | NormStat={acc_norm:.3f}')
# Annotate cells
for i in range(len(clf.classes_)):
    for j in range(len(clf.classes_)):
        text = ax.text(j, i, cm_linear[i, j], ha='center', va='center', color='black')
fig.colorbar(im, ax=ax)
fig.tight_layout()
plot_path = os.path.join(OUT_DIR, 'd3_gate_results.png')
fig.savefig(plot_path, dpi=150)
print(f'[D3] Saved {plot_path}')

print('[D3] Done.')
