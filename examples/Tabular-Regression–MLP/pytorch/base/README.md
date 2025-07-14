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

## Why is **PyTorch** a good fit for this use case?

You're building a simple MLP model on **tabular data** (structured rows and columns) to predict a **continuous value** (house price). PyTorch is especially beneficial here because:

- **Seamless integration with NumPy and pandas** — Makes it easy to load, preprocess, and batch structured datasets.
- **Simple MLP construction using `nn.Module`** — Building custom feedforward networks is clean and intuitive.
- **Fast prototyping with dynamic computation graph** — You can change your architecture or loss on the fly without rewriting boilerplate.
- **Powerful GPU support** — Easy to switch between CPU and GPU during training with `.to(device)`.
- **Readable training loops** — You have full control over each training step, useful for debugging or experimenting with custom logic.

In short, PyTorch gives you just the right balance of **low-level flexibility** and **high-level convenience** for experimenting with neural networks on tabular regression tasks.

## Output
Epoch 0:   Val RMSE = 190,975.06  
Epoch 10:  Val RMSE = 153,368.25  
Epoch 20:  Val RMSE = 42,215.84  
Epoch 30:  Val RMSE = 31,461.06  
Epoch 40:  Val RMSE = 28,710.35  
Epoch 50:  Val RMSE = 27,201.25  
Epoch 60:  Val RMSE = 26,197.74  
Epoch 70:  Val RMSE = 25,405.48  
Epoch 80:  Val RMSE = 24,775.47  
Epoch 90:  Val RMSE = 24,260.08  

Test RMSE: 31,909.95