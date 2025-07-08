# lightning_regression.py

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

# 1. Load and prepare the dataset
#-- for docker
df = pd.read_csv("resources/AmesHousing.csv") # Ames Housing dataset from Kaggle

#-- for debug
#df = pd.read_csv("../../../../assets/AmesHousing.csv") # Ames Housing dataset from Kaggle

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

categorical = X.select_dtypes(include="object").columns
numerical = X.select_dtypes(exclude="object").columns

# 2. Preprocessing pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numerical),
    ('cat', categorical_pipeline, categorical)
])

# 3. Train/Val/Test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# 4. Transform features
X_train_proc = preprocessor.fit_transform(X_train).toarray()
X_val_proc = preprocessor.transform(X_val).toarray()
X_test_proc = preprocessor.transform(X_test).toarray()

# 5. Convert to tensors
X_train_tensor = torch.tensor(X_train_proc, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_proc, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
val_ds = TensorDataset(X_val_tensor, y_val_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

# 6. Define the LightningModule
class MLPRegressor(LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(loss)
        self.log("val_rmse", rmse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(loss)
        self.log("test_rmse", rmse)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 7. Train the model
seed_everything(42)
model = MLPRegressor(X_train_tensor.shape[1])
checkpoint_cb = ModelCheckpoint(monitor="val_rmse", mode="min", save_top_k=1)

trainer = Trainer(max_epochs=100, callbacks=[checkpoint_cb])
trainer.fit(model, train_loader, val_loader)

# 8. Load best model & evaluate
best_model = MLPRegressor(X_train_tensor.shape[1])
best_model.load_state_dict(torch.load(checkpoint_cb.best_model_path)["state_dict"])
trainer.test(best_model, dataloaders=test_loader)
