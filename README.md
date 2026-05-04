# Locally Weighted Regression (LWR) Model — from scratch

This repository contains a simple **Locally Weighted Regression (LWR)** implementation written from scratch in Python, plus a small workflow for **feature preprocessing + bandwidth tuning** on a housing dataset.

LWR is a non-parametric regression approach that makes predictions by taking a **weighted average of nearby training targets**, where weights are computed using a kernel function (here: **Gaussian kernel**) and controlled by a **bandwidth** parameter.

---

## What’s inside

### 1) Core model implementation
- **`Main model code/lwr.py`**: a basic LWR class (core idea: distance → kernel weight → weighted average prediction).

### 2) “Added some tunning” (preprocessing + bandwidth search)
This folder contains a more complete runnable setup:
- **`Added some tunning/scaling.py`**
  - Loads `housing.csv`
  - Fills missing values in `total_bedrooms`
  - One-hot encodes `ocean_proximity`
  - Splits into train/test
  - Standard-scales features
- **`Added some tunning/lwrmodel.py`**
  - LWR model class (includes `numpy` import and the same logic as the core model)
- **`Added some tunning/run.py`**
  - Tries multiple bandwidth values (log-spaced)
  - Prints MSE for each bandwidth
  - Reports best bandwidth + best MSE
- **`Added some tunning/mae.png`**
  - An output plot/image included in the repo (name suggests MAE results)

### 3) Notebook
- **`lwr.ipynb`**: an interactive notebook version (exploration / experimentation).

---

## How the model works (high level)

For each test sample `x`:

1. Compute distances to all training samples  
2. Convert distances to weights using a Gaussian kernel:

\[
w_i = \exp\left(\frac{-d_i^2}{2 \cdot bw^2}\right)
\]

3. Predict using a weighted average of training targets:

\[
\hat{y}(x) = \frac{\sum_i w_i y_i}{\sum_i w_i}
\]

The implementation also includes a safe fallback:
- if the weight sum is invalid or too close to zero, it predicts the **global mean** of `y_train`.

---

## Requirements

Typical dependencies used by the scripts/notebook:

- `numpy`
- `pandas`
- `scikit-learn`

Install (example):

```bash
pip install numpy pandas scikit-learn
```

---

## Dataset

The tuning/preprocessing scripts expect a file named:

- `housing.csv`

placed where you run the code (by default, `scaling.py` loads it with `pd.read_csv('housing.csv')`).

The dataset is expected to include (at minimum):
- `median_house_value` (target)
- `ocean_proximity` (categorical feature)
- `total_bedrooms` (has missing values filled with mean)

---

## Usage

### Option A — Run bandwidth tuning (recommended in this repo)

From the `Added some tunning` directory:

```bash
cd "Added some tunning"
python run.py
```

This will:
- preprocess + scale the data (via `scaling.py`)
- fit/predict LWR for each candidate bandwidth
- print MSE per bandwidth and the best one

### Option B — Use the model in your own script

Minimal example (conceptually):

```python
import pandas as pd
import numpy as np
from lwrmodel import local_weight_regression

# X_train, y_train, X_test should be pandas objects (DataFrame/Series)
model = local_weight_regression(bandwidth=0.1)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

---

## Notes / caveats

- This implementation predicts using a **kernel-weighted average** of `y_train`. (It does not solve a locally weighted *linear* regression system; it’s the simpler kernel regression form.)
- Complexity is roughly **O(N_train)** per prediction (it computes distance to every training point).
- Feature scaling matters a lot because distances drive the weights—`scaling.py` uses `StandardScaler` for that reason.

---

## License

See [LICENSE](LICENSE).
