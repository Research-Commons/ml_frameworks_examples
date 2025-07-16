import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("../../../../assets/AmesHousing.csv")

# Drop non-informative columns (e.g., ID if present)
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Target column
target = 'SalePrice'
if target not in df.columns:
    raise ValueError("Expected 'SalePrice' column not found in dataset.")

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Fill missing values
# - Numeric: fill with median
# - Categorical: fill with mode
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_encoded), columns=X_encoded.columns)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save to CSV files (no headers, no index)
X_train.to_csv("X_train.csv", index=False, header=False)
X_test.to_csv("X_test.csv", index=False, header=False)
y_train.to_csv("y_train.csv", index=False, header=False)
y_test.to_csv("y_test.csv", index=False, header=False)

print("✅ Preprocessing complete. Files saved:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")
