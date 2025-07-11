import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

#---------------------------------------------------------------
# Under the hood, XGBClassifier:
# - Converts Pandas DataFrame into optimized DMatrix format.
# - Uses objective='binary:logistic' for classification.
# - Trains an ensemble of boosted decision trees.
# - Optimizes using gradient descent on 'logloss'.
#---------------------------------------------------------------

# 1. Load Data

#-- for docker
df = pd.read_csv("resources/Titanic-Dataset.csv")

#-- for debug
#df = pd.read_csv("../../../../assets/Titanic-Dataset.csv")

# 2. Preprocessing (no chained inplace)
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

# 5. Define and Train XGBoost Model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
