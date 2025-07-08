import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ignite.engine import Engine, Events
from ignite.metrics import MeanSquaredError
from ignite.handlers import ModelCheckpoint

# -------------------------
# 1. Load and prepare data
# -------------------------

#-- for docker
df = pd.read_csv("resources/AmesHousing.csv") # Ames Housing dataset from Kaggle

#-- for debug
#df = pd.read_csv("../../../../assets/AmesHousing.csv") # Ames Housing dataset from Kaggle

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical),
    ('cat', categorical_transformer, categorical)
])

# Split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Preprocess
X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

X_train_tensor = torch.tensor(X_train_proc.toarray(), dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_proc.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_proc.toarray(), dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# -------------------------
# 2. Define model and optimizer
# -------------------------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MLP(X_train_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -------------------------
# 3. Ignite training engine
# -------------------------
def train_step(engine, batch):
    model.train()
    xb, yb = batch
    optimizer.zero_grad()
    pred = model(xb)
    loss = loss_fn(pred, yb)
    loss.backward()
    optimizer.step()
    return loss.item()

trainer = Engine(train_step)

# -------------------------
# 4. Ignite evaluation engine
# -------------------------
def val_step(engine, batch):
    model.eval()
    with torch.no_grad():
        xb, yb = batch
        pred = model(xb)
        return pred, yb

evaluator = Engine(val_step)
MeanSquaredError(output_transform=lambda x: (x[0], x[1])).attach(evaluator, "mse")

# -------------------------
# 5. Model checkpointing
# -------------------------
checkpoint_handler = ModelCheckpoint(
    dirname="generated",
    filename_prefix="best_mlp_ames",
    n_saved=1,
    create_dir=True,
    score_function=lambda engine: -engine.state.metrics["mse"],  # lower is better
    score_name="val_rmse",
    require_empty=False
)
evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, {"model": model})

# -------------------------
# 6. Log validation RMSE every 10 epochs
# -------------------------
@trainer.on(Events.EPOCH_COMPLETED(every=10))
def log_validation(engine):
    evaluator.run(val_loader)
    val_mse = evaluator.state.metrics["mse"]
    val_rmse = torch.sqrt(torch.tensor(val_mse))
    print(f"Epoch {engine.state.epoch}: Val RMSE = {val_rmse.item():.2f}")

# -------------------------
# 7. Train the model
# -------------------------
trainer.run(train_loader, max_epochs=100)

# -------------------------
# 8. Load best model and evaluate on test set
# -------------------------
best_model_path = checkpoint_handler.last_checkpoint
model.load_state_dict(torch.load(best_model_path))

model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_rmse = torch.sqrt(loss_fn(test_preds, y_test_tensor))
    print(f"Test RMSE: {test_rmse.item():.2f}")