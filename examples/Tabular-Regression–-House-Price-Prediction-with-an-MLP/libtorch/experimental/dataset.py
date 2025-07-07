import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
This script preprocesses the Ames Housing dataset for use in a machine learning model.

Steps:
1. Load the Kaggle House Prices (Ames Housing) training data CSV.
2. Drop any rows where the target variable ('SalePrice') is missing.
3. Split the data into features (X) and target (y), removing the 'Id' column.
4. Convert all categorical features into one-hot encoded numeric columns.
5. Fill any remaining missing values in the features with the median value of the respective column.
6. Standardize (z-score normalize) all features so they have mean=0 and std=1.
7. Combine the normalized features and the original (non-normalized) target column into one dataframe.
8. Split the combined dataframe into training and test sets (80/20 split).
9. Save the training set to 'train_processed.csv' and the test set to 'test_processed.csv'.
10. Print the shapes of the resulting train and test datasets.

The resulting CSV files are fully numeric, normalized, and ready for use with a model (e.g., in LibTorch).
"""

# Download dataset: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
df = pd.read_csv("../resources/AmesHousing.csv")

# Drop rows with missing target
df = df.dropna(subset=["SalePrice"])

# Separate features & target
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

# Encode categoricals
X = pd.get_dummies(X)

# Fill missing with median
X = X.fillna(X.median())

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Final dataframe
final_df = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)

# Split
train_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)

train_df.to_csv("train_processed.csv", index=False)
test_df.to_csv("test_processed.csv", index=False)

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
