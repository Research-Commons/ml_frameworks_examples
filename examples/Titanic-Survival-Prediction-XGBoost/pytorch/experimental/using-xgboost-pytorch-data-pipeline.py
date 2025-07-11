import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# 1. Load Data

#-- for docker
df = pd.read_csv("resources/Titanic-Dataset.csv") # Titanic-Dataset from Kaggle

#-- for debug
#df = pd.read_csv("../../../../assets/Titanic-Dataset.csv") # Titanic-Dataset from Kaggle

# 2. Preprocessing (no inplace=True to avoid chained assignment warning)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Encode categorical features
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Optional feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(['SibSp', 'Parch'], axis=1)

# 3. Define Features & Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# 4. Train-test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. PyTorch Dataset
class TitanicDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TitanicDataset(X_train, y_train)
test_dataset = TitanicDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# 6. Fetch one full batch for XGBoost training
for batch_X, batch_y in train_loader:
    dtrain = xgb.DMatrix(batch_X.numpy(), label=batch_y.numpy())

for batch_X, batch_y in test_loader:
    dtest = xgb.DMatrix(batch_X.numpy(), label=batch_y.numpy())

# 7. Define XGBoost Parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0
}

# 8. Train Model
num_round = 100
bst = xgb.train(params, dtrain, num_boost_round=num_round)

# 9. Predict
y_prob = bst.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

# 10. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
