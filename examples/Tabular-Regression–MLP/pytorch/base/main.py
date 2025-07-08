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

#-- for docker
#df = pd.read_csv("resources/AmesHousing.csv") # Ames Housing dataset from Kaggle

#-- for debug
df = pd.read_csv("../../../../assets/AmesHousing.csv") # Ames Housing dataset from Kaggle

# -------------------------
# 2. Split features and target
# -------------------------

#-- x = all input features, like LotArea, OverallQual, YearBuilt, Neighborhood
#-- "remove SalePrice column and store every other column"
X = df.drop("SalePrice", axis=1)
#-- y = target you want to predict. i.e the SalePrice
#-- "store only SalePrice"
y = df["SalePrice"]

# Identify types
#-- Store all the numerical columns from the table
numerical = X.select_dtypes(exclude="object").columns
#-- Store all the text-like columns from the table
categorical = X.select_dtypes(include="object").columns


# -------------------------
# 3. Preprocessing pipeline
# -------------------------

#-- Pipeline tells how to handle missing data. numeric_transformer stores this stratergy
numeric_transformer = Pipeline([
    #-- Fills in missing numeric values using median
    ('imputer', SimpleImputer(strategy="median")),
    #-- scales the values to have mean = 0 and std = 1 (helps NN training)
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    #-- Fills missing text data using most common value
    ('imputer', SimpleImputer(strategy="most_frequent")),
    #-- Converts text (e.g. "ZoneA") to one-hot vectors (binary format)
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

#-- Combines the above two pipelines and store them in the pre processor
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical),
    ('cat', categorical_transformer, categorical)
])

#-- NOTE : The actual handling of missing data does not happen here. This stage only stores the stratergy in which we will handle missing data



# -------------------------
# 4. Train/val/test split
# -------------------------

#-- X_train_full: 80% of features for training+validation
#-- X_test: 20% of features for final testing
#-- y_train_full: 80% of targets for training+validation
#-- y_test: 20% of targets for final testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#-- X_train: 64% of original data (80% of 80%) for actual training
#-- X_val: 16% of original data for validation
#-- y_train: 64% of original targets
#-- y_val: 16% of original targets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)



# -------------------------
# 5. Preprocess features
# -------------------------

#-- fit_transform -> "learn statistics from this data"
#-- Learn median values to fill in missing numbers
X_train_processed = preprocessor.fit_transform(X_train)

#-- transform - "apply that knowledge to the data"
#-- Apply the same preprocessing (already fitted on training data) to the validation set
X_val_processed = preprocessor.transform(X_val)
#-- Apply the trained preprocessing to the test set
X_test_processed = preprocessor.transform(X_test)

# -------------------------
# 6. Convert to PyTorch tensors
# -------------------------

#-- Convert the NumPy (or sparse) arrays into PyTorch Tensors.
#-- toarray() is used because OneHotEncoder returns sparse matrix by default.

X_train_tensor = torch.tensor(X_train_processed.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_processed.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test_processed.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# -------------------------
# 7. DataLoaders
# -------------------------

#-- Create a training DataLoader by wrapping the input and target tensors into a TensorDataset.
#-- The DataLoader splits the data into mini-batches of 64 and shuffles them every epoch to help the model
#-- generalize better and reduce overfitting.
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
#-- Create a validation DataLoader without shuffling (since we don't train on validation data).
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
#-- Create a test DataLoader (no shuffling)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64)

# -------------------------
# 8. Define MLP model
# -------------------------

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        #-- First fully connected (dense) layer: input → 128 hidden units
        self.fc1 = nn.Linear(input_dim, 128)

        #-- Second fully connected layer: 128 hidden units → 64 hidden units
        self.fc2 = nn.Linear(128, 64)

        #-- Final output layer: 64 hidden units → 1 output (since it's regression)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        #-- Apply ReLU activation after first layer to introduce non-linearity
        x = torch.relu(self.fc1(x))

        #-- Apply ReLU after second layer
        x = torch.relu(self.fc2(x))

        #-- Final layer without activation (regression: raw output for predicting price)
        return self.fc3(x)

# -------------------------
# 9. Initialize model
# -------------------------

#-- Get the number of input features from the training tensor. This becomes the input dimension for the first layer
#-- of the MLP
input_dim = X_train_tensor.shape[1]
model = MLP(input_dim)
#-- Adam is an adaptive learning rate optimizer, good default for most problems
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#-- Since this is a regression task (predicting house prices), we use Mean Squared Error (MSE)
loss_fn = nn.MSELoss()

# -------------------------
# 10. Training loop
# -------------------------

#-- sets the initial best RMSE to infinity
best_val_rmse = float("inf")
for epoch in range(100):
    #-- Tells PyTorch we're in training mode (important for dropout, batchnorm etc.
    model.train()
    for xb, yb in train_loader:
        #--  Forward pass
        pred = model(xb)
        #-- Computes MSE loss between prediction and actual value
        loss = loss_fn(pred, yb)
        #-- Clears any previous gradient values before backpropagation
        optimizer.zero_grad()
        #-- Compute the gradient
        loss.backward()
        #-- Updates weights using gradients
        optimizer.step()

    # Validation
    #--  Switches to evaluation mode
    model.eval()
    #-- Disables gradient tracking. Saves memory and computation since we're not updating weights here
    with torch.no_grad():
        #-- Get predictions for the full validation set
        val_preds = model(X_val_tensor)
        #-- Compute Root Mean Squared Error between predictions and actual values
        val_rmse = torch.sqrt(loss_fn(val_preds, y_val_tensor))
        #-- If current RMSE is better than the best one, update the best RMSE and store the model in a file
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "generated/best_mlp_ames.pt")

    #-- Every 10 epochs, prints how well the model is doing
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Val RMSE = {val_rmse.item():.2f}")

# -------------------------
# 11. Final Evaluation
# -------------------------

#-- This is the final evaluation step on the test dataset, using the best model we saved earlier
model.load_state_dict(torch.load("generated/best_mlp_ames.pt"))
model.eval()
with torch.no_grad():
    #-- Run Inference on Test Data
    test_preds = model(X_test_tensor)
    #-- Compute RMSE
    test_rmse = torch.sqrt(loss_fn(test_preds, y_test_tensor))
    print(f"Test RMSE: {test_rmse.item():.2f}")

