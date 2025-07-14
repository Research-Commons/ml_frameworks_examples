import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("resources/Titanic-Dataset.csv")

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unused columns
df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

# Encode categorical
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

# Feature engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop(columns=['SibSp', 'Parch'])

# Define features & target
X = df.drop(columns='Survived')
y = df['Survived']

df_full = pd.concat([y, X], axis=1)

train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

train_df.to_csv("train_cleaned.csv", index=False)
test_df.to_csv("test_cleaned.csv", index=False)

print("[INFO] Cleaned datasets saved ✅")
