# Tabular Regression with tiny-dnn

This project demonstrates tabular regression using the tiny-dnn C++ neural network library on the Ames Housing dataset.

## Quick Start

The Docker container automatically handles data preprocessing and model training:

```bash
docker build -t test-tinydnn .
docker run --rm -it test-tinydnn
```

## What happens inside the container:

1. **Data Preprocessing**: The `preprocess.py` script automatically:
   - Loads the Ames Housing dataset
   - Handles missing values
   - One-hot encodes categorical features
   - Normalizes numerical features
   - Splits data into train/test sets
   - Saves CSV files for the C++ program

2. **Model Training**: The C++ program (`main.cpp`) then:
   - Loads the preprocessed CSV files
   - Builds a 3-layer MLP (128 → 64 → 1 neurons)
   - Trains the model using Adagrad optimizer
   - Evaluates performance (MSE, RMSE, MAE)

## Manual Execution (Alternative)

If you prefer to run steps manually:

```bash
# 1. Install Python dependencies
pip3 install pandas scikit-learn

# 2. Run preprocessing
python3 preprocess.py

# 3. Build C++ program
mkdir build && cd build
cmake .. && make

# 4. Run the trained model
./test_tinydnn
```