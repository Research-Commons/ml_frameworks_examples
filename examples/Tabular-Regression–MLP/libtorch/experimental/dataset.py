import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Preprocess the Ames Housing dataset for numerical modeling.

Steps:
1. Load CSV
2. Drop rows with missing target ('SalePrice')
3. Drop 'Id', separate target
4. One-hot encode categoricals
5. Fill missing values (median)
6. Normalize features (z-score)
7. Rejoin target
8. Split into train/test
9. Save as CSVs
"""

os.makedirs("generated", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("resources/AmesHousing.csv")

# 2. Drop rows with missing target
df = df.dropna(subset=["SalePrice"])

# 3. Separate features and target
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# 4. One-hot encode categorical columns
X_encoded = pd.get_dummies(X)

# 5. Fill missing values with column medians
X_filled = X_encoded.fillna(X_encoded.median(numeric_only=True))

# 6. Standardize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_filled.columns)

# 7. Combine features and target
final_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# 8. Train/test split
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

# 9. Save output CSVs
train_df.to_csv("generated/train_processed.csv", index=False)
test_df.to_csv("generated/test_processed.csv", index=False)

# Print result summary
print(f"[+] Train shape: {train_df.shape}")
print(f"[+] Test shape:  {test_df.shape}")
