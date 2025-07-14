# 🧠 House Price Prediction using MLP (PyTorch)

This project demonstrates how to predict house prices using a Multi-Layer Perceptron (MLP) on tabular data through **three different PyTorch-based implementations**. Each method shows a different level of control, abstraction, and integration with PyTorch tools.

---

## 🔍 Approaches Summary

We implement the same task using:

1. `using-MLP.py` – Manual MLP training with raw PyTorch  
2. `using_ignite.py` – Event-based training using PyTorch Ignite  
3. `using_lightning.py` – High-level modular training using PyTorch Lightning  

---

## 1. `using-MLP.py`

**Approach**: **Manual MLP using raw PyTorch**

- Implements a full training loop manually.  
- Good for learning and debugging.  
- Most flexible, but also most verbose.

**Snippet**:
```python
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## 2. `using_ignite.py`

**Approach**: **Event-driven training with PyTorch Ignite**

- Uses Ignite’s `Engine` and `Events` API.  
- Cleaner training loop with metric tracking.  
- Easier to extend with custom handlers.

**Snippet**:
```python
from ignite.engine import Engine, Events

def train_step(engine, batch):
    model.train()
    X_batch, y_batch = batch
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)
```

---

## 3. `using_lightning.py`

**Approach**: **Modular training with PyTorch Lightning**

- Encapsulates training, validation, and optimization logic in a `LightningModule`.  
- Automatically handles logging, checkpointing, and device placement.  
- Best for clean, production-ready code.

**Snippet**:
```python
import pytorch_lightning as pl
import torch.nn.functional as F

class PricePredictor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```

---

## 📊 Comparison Table

| Feature                          | `using-MLP.py` | `using_ignite.py` | `using_lightning.py` |
|----------------------------------|----------------|-------------------|------------------------|
| Manual training loop             | ✅ Yes         | ❌ No             | ❌ No                 |
| Event-based training             | ❌ No          | ✅ Yes            | ⚠️ Internally handled |
| High-level modular API           | ❌ No          | ⚠️ Partial        | ✅ Yes                |
| Easy to extend/debug             | ✅ Yes         | ✅ Yes            | ⚠️ Less flexible      |
| Logging & checkpointing          | ❌ Manual       | ⚠️ Custom         | ✅ Built-in           |

---

Each method serves a purpose depending on your needs:  
- Use **raw PyTorch** for full control or education.  
- Use **Ignite** for a clean, event-based approach.  
- Use **Lightning** for scalable and production-ready training.
