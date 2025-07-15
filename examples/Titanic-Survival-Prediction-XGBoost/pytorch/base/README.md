# Titanic Survival Prediction using XGBoost

## Problem Statement

The task is to **predict whether a passenger survived or not** based on features from the Titanic dataset.  
The dataset includes demographic and travel-related information such as **age**, **sex**, **ticket class**, **fare**, and **family details**.

This is a **binary classification** problem (output is either 0 or 1), and the goal is to build a model that performs well on unseen data.

---

## Solution Overview

This implementation uses **XGBoost** (Extreme Gradient Boosting) – a powerful and scalable tree-based model – to train on the dataset and predict survival.


> **Note** : Xgboost has an internal tensor like format called DMatrix, and has its own ways to use gradients. Hence, 
> this use case **does not require pytorch** to work.

### **Pandas (`pd`)**

> **Handles tabular data loading & manipulation**

- Loads the dataset from a `.csv` file
- Fills missing values (`Age`, `Embarked`)
- Drops irrelevant columns (`Name`, `Ticket`, `Cabin`, `PassengerId`)
- Adds new features (`FamilySize`)
- Encodes categorical columns (`Sex`, `Embarked`)
- Splits features (`X`) and target (`y`)

---

### **Scikit-learn (`sklearn`)**

> **Handles preprocessing, splitting, and evaluation**

- `train_test_split` – Splits dataset into training and testing subsets
- `LabelEncoder` – Converts categorical columns to integers
- `accuracy_score`, `roc_auc_score`, `classification_report` – Evaluate model predictions using common classification metrics

---

### **XGBoost (`xgboost`)**

> **Handles model training and feature importance analysis**

- `XGBClassifier` – Trains a boosted ensemble of decision trees using gradient descent
- Automatically optimizes with respect to **log loss** for binary classification
- Allows customization of tree depth, learning rate, subsampling, and column sampling
- Supports visualization of **feature importances** using `plot_importance()`

---

## Training Pipeline

### Step-by-step breakdown of the code:

1. **Load Dataset**  
   Uses `pandas.read_csv()` to read the Titanic dataset.

2. **Preprocess the Data**
   - Fill missing values (`Age` with median, `Embarked` with mode)
   - Drop unused columns like `Name`, `Cabin`, etc.
   - Encode `Sex` and `Embarked` into numeric values
   - Create new feature `FamilySize = SibSp + Parch + 1`
   - Drop `SibSp` and `Parch` after combining
   - Separate input (`X`) and target (`y`)

3. **Split Data**  
   80% used for training, 20% for testing.

4. **Train Model**  
   The model is trained with:
   - `n_estimators=100` trees
   - `max_depth=4` (controls complexity)
   - `learning_rate=0.1` (shrinkage after each round)
   - `subsample=0.8` (sample 80% of rows per tree)
   - `colsample_bytree=0.8` (sample 80% of features per tree)

5. **Evaluate Performance**
   - **Accuracy**: Proportion of correct predictions
   - **AUC-ROC**: Measures how well the model ranks positives over negatives
   - **Classification Report**: Includes precision, recall, F1-score

6. **Visualize Feature Importance**
   - Highlights which features had the most impact on survival prediction

---

## Why use **XGBoost** for this task?

| Reason                      | Benefit                                                                                   |
|----------------------------|--------------------------------------------------------------------------------------------|
| **Gradient Boosting**      | Learns from errors of previous trees, reducing bias and variance                          |
| **Built-in regularization**| Helps prevent overfitting compared to traditional decision trees                          |
| **Speed & scalability**    | Efficient with large datasets and supports parallel computation                           |
| **Handles missing values** | Can deal with NaNs natively during training                                               |
| **Feature importance**     | Automatically ranks features based on contribution to prediction                          |

---

## Why use **AUC-ROC** for evaluation?

| Metric       | Purpose                                                                 |
|--------------|-------------------------------------------------------------------------|
| **Accuracy** | Good when classes are balanced (survivors vs non-survivors)            |
| **AUC-ROC**  | Better when you care about **ranking** and **class separation**        |
| **F1-score** | Useful if you care about **precision-recall tradeoff**                 |

---

## Output Example

```bash
Accuracy: 0.8324
AUC-ROC:  0.8737

Classification Report:
              precision    recall  f1-score   support

           0       0.85      0.88      0.87       105
           1       0.81      0.77      0.79        74

    accuracy                           0.83       179
   macro avg       0.83      0.82      0.83       179
weighted avg       0.83      0.83      0.83       179
