# House Price Prediction with PyTorch

## Problem Statement

The task is to **predict house prices** based on structured tabular data from the Ames Housing dataset. The dataset contains several numerical and categorical features (e.g., square footage, number of bedrooms, neighborhood, etc.) and the target variable is the house sale price — a continuous value.
This is a **regression problem**, and the aim is to minimize prediction error on unseen data.

## Solution Overview

### Brief breakdown of what each library handles in this use case

### **Pandas (`pd`)**

> **Handles tabular data loading & manipulation**

- Loads the CSV file (`read_csv`)
- Separates features (`X`) and target (`y`)
- Helps identify **categorical** vs **numerical** columns
- Converts data into NumPy arrays for further processing

---

### **Scikit-learn (`sklearn`)**

> **Handles preprocessing & splitting**

- `train_test_split` – Splits data into train/val/test sets
- `SimpleImputer` – Fills in missing values (e.g., median, most frequent)
- `StandardScaler` – Normalizes numerical data
- `OneHotEncoder` – Encodes categorical features into one-hot vectors
- `ColumnTransformer` – Combines multiple preprocessing pipelines
- `Pipeline` – Chains preprocessing steps together
- `fit_transform` / `transform` – Applies preprocessing to datasets

---

### **PyTorch (`torch`)**

> **Handles model, training, and evaluation**

- Converts NumPy data into tensors
- Wraps tensors in `TensorDataset` and `DataLoader` for batching
- Defines the `MLP` neural network model using `nn.Module`
- Performs forward pass, loss computation, backpropagation, and optimizer updates
- Tracks the best validation model and evaluates on test set
- Uses `nn.MSELoss`, `torch.sqrt`, and `optim.Adam` for training

## Why choose **Adam Optimizer** for this use case?

You're training an MLP (Multi-Layer Perceptron) on **tabular regression data** (predicting house prices from the Ames Housing dataset). Adam is a **very popular choice** for this kind of deep learning model.

### Advantages of Adam:

| Feature                        | Benefit                                                                                                                                               |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Adaptive learning rates**   | Adam adapts the learning rate individually for each parameter using estimates of **first** and **second moments** of the gradients.                   |
| **Fast convergence**          | Adam usually converges **faster** than standard optimizers like SGD, which is helpful in training MLPs for regression tasks.                         |
| **Works well with sparse gradients** | Good if you have sparse or one-hot encoded inputs (like here, due to categorical features).                                                       |
| **Low tuning effort**         | Often works well with default parameters (`lr=1e-3`), making it practical for quick prototyping.                                                      |

---

## Why use **RMSE (Root Mean Squared Error)** for this use case?

You're solving a **regression** problem — predicting a **continuous value**: `SalePrice`.

### RMSE is a common and intuitive metric for regression:

| Feature                  | Benefit                                                                                                                                   |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **Interpretable scale**  | RMSE has **the same units** as the target variable (dollars here), so it's easy to understand: “My predictions are off by ~$15,000.”     |
| **Punishes large errors**| Since RMSE squares the errors, **larger mistakes are penalized more heavily**. Useful when big prediction mistakes are more costly.      |
| **Standard metric**      | It's widely used in housing price prediction and other regression tasks — so it's easy to compare models across papers and projects.     |

---

## Why **not** other choices?

- **Why not SGD?**  
  SGD needs **careful learning rate tuning** and usually converges slower on MLPs.

- **Why not MAE (Mean Absolute Error)?**  
  MAE treats all errors equally. RMSE is preferred when **larger errors** are more critical — which is the case when mispredicting high-priced houses could have a bigger impact.