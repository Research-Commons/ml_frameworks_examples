import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

#---------------------------------------------------------------
# DMatrix: A special XGBoost-optimized data format for training and inference.
# params: Defines the objective, evaluation metric, and booster hyperparameters.
# bst: The trained booster model (Booster object), not a scikit-learn estimator.
# y_prob: Raw predicted probabilities (from logistic function).
# y_pred: Final class predictions (0 or 1) using a threshold of 0.5.
#---------------------------------------------------------------

# 1. Load Data

#-- for docker
df = pd.read_csv("resources/Titanic-Dataset.csv")

#-- for debug
#df = pd.read_csv("../../../../assets/Titanic-Dataset.csv")

# 2. Preprocessing (no chained inplace assignment)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unnecessary columns
df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Encode categorical variables
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

# 5. Convert to DMatrix (XGBoost's internal format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 6. Define Parameters
params = {
    "objective": "binary:logistic",   # binary classification
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1,                       # learning rate
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0
}

# 7. Train Model
num_round = 100
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round
)

# 8. Predict
y_prob = bst.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)

# 9. Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
