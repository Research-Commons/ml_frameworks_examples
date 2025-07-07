# House Price Prediction with LibTorch

## Problem Statement

The task is to **predict house prices** based on structured tabular data from the Ames Housing dataset. The dataset contains several numerical and categorical features (e.g., square footage, number of bedrooms, neighborhood, etc.) and the target variable is the house sale price — a continuous value.
This is a **regression problem**, and the aim is to minimize prediction error on unseen data.

Traditionally, for tabular data, tree-based methods such as Gradient Boosted Trees dominate. Here, however, the objective is to demonstrate how a **deep learning approach using a Multi-Layer Perceptron (MLP)** can be applied to tabular regression tasks using **LibTorch**, the C++ API for PyTorch.

## Solution Overview

We implement and train an MLP in **C++** with LibTorch, taking preprocessed features as input and predicting the sale price.

The overall pipeline is:

1. **Preprocess data (in Python)**

   * Download the Kaggle Ames Housing dataset (`train.csv`).
   * Encode categorical variables using one-hot encoding.
   * Fill missing values with median for numerical features.
   * Standardize (z-score normalization) all features.
   * Save processed train and test CSV files (`train_processed.csv`, `test_processed.csv`), with the last column being the target (SalePrice).

2. **Load data (in C++)**

   * Use a custom CSV loader to read the preprocessed CSVs into `torch::Tensor` objects.
   * The final tensors are:

     * `train_data` (features) and `train_targets` (targets) for training.
     * `test_data` and `test_targets` for evaluation.

3. **Model**

   * A standard fully-connected MLP:

     * Input layer: number of features (dynamically determined).
     * Hidden layers: 128 and 64 neurons, ReLU activations.
     * Output layer: single neuron for the predicted price.

4. **Training**

   * Use MSE loss and Adam optimizer.
   * Train for a specified number of epochs and learning rate.
   * Batch size: 32.
   * After each epoch, print the average training loss and a progress bar.

5. **Evaluation**

   * Predict on the test set.
   * Compute **Root Mean Squared Error (RMSE)**, which indicates the average deviation of predictions from actual prices, in the same units as the target.

## API and Code Structure

### Components

* `MLP`:
  A `torch::nn::Module` subclass implementing the forward pass of the model. Defined with three fully connected layers.

* `HousePriceDataset`:
  A custom dataset class derived from `torch::data::datasets::Dataset`. Wraps feature and target tensors and returns examples in the format LibTorch’s `DataLoader` expects.

* `load_csv`:
  Utility function to read a preprocessed CSV into two tensors: one for features and one for targets.

* `compute_rmse`:
  Computes the RMSE from predicted and actual values.

* `print_progress`:
  Prints a clean two-line progress report: epoch number and average loss, followed by a progress bar.


## Why LibTorch

### Advantages

* **Close to PyTorch API**:
  LibTorch mirrors the PyTorch Python API’s core abstractions (`Tensor`, `Module`, `DataLoader`), making it familiar and easy to adopt for anyone with PyTorch experience.

* **Performance**:
  Being native C++, LibTorch offers the performance benefits of compiled code with access to the same backend as PyTorch.

* **Production-ready**:
  Useful for embedding deep learning models in production C++ applications without Python overhead.

* **Integration**:
  Can be linked directly into other C++ systems, making it suitable for latency-sensitive or resource-constrained environments.

### Caveats

* **Fewer utilities than Python**:
  LibTorch does not include data preprocessing, visualization, or dataset utilities present in Python. These have to be implemented manually or done beforehand (e.g., using Python).

* **Smaller ecosystem**:
  Many high-level libraries built around PyTorch (like Lightning) do not have C++ counterparts.

* **Error reporting**:
  Errors are often cryptic, particularly shape mismatches, and debugging requires familiarity with tensor dimensions.

* **Documentation and community**:
  While improving, the documentation and community support for LibTorch are smaller compared to PyTorch.


## How to Run

### Prerequisites

* Build and install LibTorch: [https://pytorch.org/cppdocs/installing.html](https://pytorch.org/cppdocs/installing.html)
* Preprocess the dataset in Python and save `train_processed.csv` and `test_processed.csv` in the same directory.

### Build

```bash
cmake -S . -B build 
cmake --build build -j 8
```

### Run

```bash
# lr is learning rate
./house_price_mlp epochs=100 lr=1e-3 threads=4 
```

Even if you do not provide the above arguments, there are default fallbacks.

## Output

During training, the console shows:

```
Epoch [1/50] Avg Loss: ...
[=====>                     ] 12%
...
Epoch [50/50] Avg Loss: ...
[===========================] 100%
>> Training Completed.
>> Evaluating on test set...
>> Test RMSE: 42100.0
```

## Notes on Extensibility

* The model architecture can easily be modified by changing the `MLP` struct.
* Alternative optimizers, activation functions, or loss functions can be plugged in with minimal changes.
* Could be extended to support k-fold cross-validation, early stopping, or logging frameworks.

## Conclusion

This implementation demonstrates how to solve a classic tabular regression problem using deep learning in C++ with LibTorch. The API is intuitive and flexible for core model-building tasks, but the lack of preprocessing and visualization tools means extra effort is required for a complete pipeline. Nonetheless, for production environments where Python is not feasible, LibTorch provides a solid alternative with familiar abstractions and competitive performance.
