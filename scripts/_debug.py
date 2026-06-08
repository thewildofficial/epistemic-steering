import numpy as np, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import expit

rng = np.random.RandomState(42)

# simulate arc_challenge-like data: 200 samples, 2560 dims, 90% positive
n, d = 200, 2560
X = rng.randn(n, d).astype(np.float64)
y = np.array([True]*180 + [False]*20)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print('train', Xtr.shape, 'test', Xte.shape, 'train pos', ytr.sum(), 'test pos', yte.sum())

# real probe
model = LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', max_iter=1000, random_state=42)
model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:,1]
print('real AUROC', roc_auc_score(yte, prob))
print('best C', model.C_)

# random labels
for seed in [999, 123, 456, 789, 2024]:
    yr = ytr.copy()
    np.random.RandomState(seed).shuffle(yr)
    model2 = LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', max_iter=1000, random_state=42)
    model2.fit(Xtr, yr)
    prob2 = model2.predict_proba(Xte)[:,1]
    print(f'rand seed={seed} AUROC={roc_auc_score(yte, prob2):.4f}  C={model2.C_}')

# check: _train_and_evaluate_auroc with a fresh model
yr = ytr.copy()
np.random.RandomState(999).shuffle(yr)
model3 = LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring='roc_auc', max_iter=1000, random_state=42)
model3.fit(Xtr, yr)
prob3 = model3.predict_proba(Xte)[:,1]
print('reproduced AUROC', roc_auc_score(yte, prob3))

# check intercept and weights
print('intercept', model3.intercept_[0])
print('coef norm', np.linalg.norm(model3.coef_))
print('prob mean', prob3.mean(), 'std', prob3.std())
print('prob min', prob3.min(), 'max', prob3.max())
