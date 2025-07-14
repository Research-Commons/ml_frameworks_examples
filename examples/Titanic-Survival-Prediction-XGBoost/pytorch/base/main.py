import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

#---- Set seeds for reproducibility (so results don’t change every run)
random.seed(42)
np.random.seed(42)

def load_data(path):
    #-- Load CSV as pandas DataFrame
    return pd.read_csv(path)

def preprocess_data(df):
    #-- Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    #-- Fill missing Embarked with most common value
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    #-- Drop irrelevant columns , we don't really need these columns to find whether passenger survived or not
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    #-- Encode categorical columns to numbers (Sex, Embarked)
    for col in ['Sex', 'Embarked']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    #-- Feature engineering: create new column FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    #-- Drop SibSp and Parch after creating FamilySize
    df = df.drop(['SibSp', 'Parch'], axis=1)

    #-- Separate features and target
    X = df.drop('Survived', axis=1)  #-- input features
    y = df['Survived']               #-- label (0 or 1)
    return X, y

#------------------------------
#-- Load and preprocess dataset
#------------------------------

#-- for docker run
df = load_data("resources/Titanic-Dataset.csv")

#-- for local debugging
#df = load_data("../../../../assets/Titanic-Dataset.csv")

X, y = preprocess_data(df)

#-- Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#------------------------------
#-- Define XGBoost model with hyperparameters
#------------------------------
model = XGBClassifier(
    #-- number of decision trees
    n_estimators=100,
    #-- tree depth
    max_depth=4,
    #-- shrinkage (how much to move after each tree)
    learning_rate=0.1,
    #-- use 80% rows per tree
    subsample=0.8,
    #-- use 80% features per tree
    colsample_bytree=0.8,
    #-- use log(loss) for finding the first prediction
    eval_metric='logloss',
    # -- for reproducibility
    random_state=42
)
#------------------------------
#-- Train the model on training data and evaluate on test set
#------------------------------

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
#------------------------------
#-- Predict class labels and probabilities on test set
#------------------------------

#-- predicted classes
y_pred = model.predict(X_test)
#-- probability of class 1
y_prob = model.predict_proba(X_test)[:, 1]

#------------------------------
#-- Evaluate results
#------------------------------

#-- how many correct out of total
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# -- how good the model ranks positives above negatives
print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob):.4f}")
#-- precision, recall, F1 etc.
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#-- Plot which columns affect the output the most
plot_importance(model)
plt.show()
