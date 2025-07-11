# Titanic Survival Prediction using XGBoost

This project demonstrates how to use XGBoost to predict survival on the Titanic dataset using **three different approaches**. Each method is implemented in its own script and shows varying levels of control, flexibility, and integration with other libraries.

---

## 🔍 Approaches Summary

We implement the same task using:

1. `using-xgboost-directly.py` – High-level API (`XGBClassifier`)
2. `using-DMatrix-API.py` – Low-level XGBoost API with `xgb.train()`
3. `using-xgboost-pytorch-data-pipeline.py` – Integration with PyTorch data pipelines

---

## 1. `using-xgboost-directly.py`

**Approach**: **High-level Scikit-learn API (`XGBClassifier`)**

- Uses `XGBClassifier` from `xgboost`, which behaves like any scikit-learn model.  
- Simple `.fit()`, `.predict()` interface.  
- Best for fast prototyping, integration with `GridSearchCV`, pipelines, etc.  
- Internally converts data into `DMatrix`.

**Snippet**:
```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)
```

---

## 2. `using-DMatrix-API.py`

**Approach**: **Low-level XGBoost API (`xgb.train`) with DMatrix**

- Manually converts data into `xgb.DMatrix`.  
- Uses `xgb.train()` instead of `XGBClassifier`.  
- Gives fine-grained control over training (e.g., custom objectives, callbacks).  
- More verbose, but better for advanced tuning and debugging.

**Snippet**:
```python
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {"objective": "binary:logistic", "eval_metric": "logloss"}
bst = xgb.train(params, dtrain, num_boost_round=100)
```

---

## 3. `using-xgboost-pytorch-data-pipeline.py`

**Approach**: **Experimental: XGBoost with PyTorch-style Data Pipeline**

- Uses PyTorch’s `DataLoader` or `TensorDataset` to create batches.  
- Feeds PyTorch tensors into `xgb.DMatrix`, either manually or via bridging tools.  
- Useful when combining XGBoost with PyTorch-based models or pipelines.  
- Great for hybrid workflows in research or production.

**Snippet**:
```python
from torch.utils.data import DataLoader, TensorDataset
# Later: convert batches to DMatrix for XGBoost
```

---

## Comparison Table

| Feature                           | `using-xgboost-directly.py` | `using-DMatrix-API.py` | `using-xgboost-pytorch-data-pipeline.py` |
|----------------------------------|---------------------|-------------------------|-------------------------------------------|
| Uses `XGBClassifier`             | ✅ Yes              | ❌ No                  | ⚠️ Possibly internally                     |
| Uses `xgb.train` + `DMatrix`     | ❌ No               | ✅ Yes                 | ✅ Likely                                  |
| PyTorch `DataLoader` support     | ❌ No               | ❌ No                  | ✅ Yes                                     |
| Easy to use                      | ✅ Easiest           | ⚠️ Medium              | ❌ Most complex                           |
| Fine-grained training control    | ❌ Limited           | ✅ Full control         | ✅ Full control + PyTorch synergy         |

---

Each method serves a purpose depending on your goals:  
- Use `xgboost-direct` API for speed and simplicity.  
- Use `DMatrix` API for fine-tuned control.  
- Use PyTorch pipeline for deep integration with neural networks or hybrid models.