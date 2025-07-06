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

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("../../resources/AmesHousing.csv")  # Ames Housing dataset from Kaggle

# -------------------------
# 2. Split features and target
# -------------------------
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Identify types
categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# -------------------------
# 3. Preprocessing pipeline
# -------------------------
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

# -------------------------
# 4. Train/val/test split
# -------------------------
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# -------------------------
# 5. Preprocess features
# -------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# -------------------------
# 6. Convert to PyTorch tensors
# -------------------------
X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_processed.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -------------------------
# 7. DataLoaders
# -------------------------
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# -------------------------
# 8. Define MLP model
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

# -------------------------
# 9. Initialize model
# -------------------------
input_dim = X_train_tensor.shape[1]
model = MLP(input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# -------------------------
# 10. Training loop
# -------------------------
best_val_rmse = float("inf")
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor)
        val_rmse = torch.sqrt(loss_fn(val_preds, y_val_tensor))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "generated/best_mlp_ames.pt")

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Val RMSE = {val_rmse.item():.2f}")

# -------------------------
# 11. Final Evaluation
# -------------------------
model.load_state_dict(torch.load("generated/best_mlp_ames.pt"))
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_rmse = torch.sqrt(loss_fn(test_preds, y_test_tensor))
    print(f"Test RMSE: {test_rmse.item():.2f}")

