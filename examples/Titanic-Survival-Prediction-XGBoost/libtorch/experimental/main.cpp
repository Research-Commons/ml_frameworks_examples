#include "CSVLoader.hpp"
#include <torch/torch.h>
#include <xgboost/c_api.h>

#include <iostream>

/// TitanicDataset: wraps features & labels into a LibTorch Dataset
struct TitanicDataset : torch::data::Dataset<TitanicDataset> {
  torch::Tensor features_;
  torch::Tensor labels_;

  TitanicDataset(torch::Tensor features, torch::Tensor labels)
      : features_(features), labels_(labels) {}

  /// Return a single example (features and label)
  torch::data::Example<> get(size_t idx) override {
    return {features_[idx], labels_[idx]};
  }

  /// Return total number of samples
  torch::optional<size_t> size() const override { return features_.size(0); }
};

int main() {
  std::cout << "[INFO] Titanic Survival Prediction - LibTorch + XGBoost\n";

  // === STEP 1: Load preprocessed CSV files into tensors ===
  CSVLoadOptions opts;
  opts.has_header = true;
  opts.target_column = 0; // assuming 'Survived' is the first column

  std::cout << "[INFO] Loading train dataset from CSV...\n";
  auto [X_train, y_train] = load_csv("train_cleaned.csv", opts);

  std::cout << "[INFO] Loading test dataset from CSV...\n";
  auto [X_test, y_test] = load_csv("test_cleaned.csv", opts);

  // === STEP 2: Wrap tensors into LibTorch Dataset & DataLoader ===
  std::cout << "[INFO] Creating LibTorch Datasets & DataLoaders...\n";
  auto train_dataset =
      TitanicDataset(X_train, y_train).map(torch::data::transforms::Stack<>());
  auto test_dataset =
      TitanicDataset(X_test, y_test).map(torch::data::transforms::Stack<>());

  auto train_loader = torch::data::make_data_loader(
      std::move(train_dataset),
      torch::data::DataLoaderOptions().batch_size(X_train.size(0)));

  auto test_loader = torch::data::make_data_loader(
      std::move(test_dataset),
      torch::data::DataLoaderOptions().batch_size(X_test.size(0)));

  torch::Tensor batch_X_train, batch_y_train;
  torch::Tensor batch_X_test, batch_y_test;

  // === STEP 3: Fetch full-batch tensors from DataLoaders ===
  std::cout << "[INFO] Fetching full batch tensors...\n";
  for (auto &batch : *train_loader) {
    batch_X_train = batch.data;
    batch_y_train = batch.target;
  }

  for (auto &batch : *test_loader) {
    batch_X_test = batch.data;
    batch_y_test = batch.target;
  }

  std::cout << "[INFO] Train samples: " << batch_X_train.size(0)
            << ", Features: " << batch_X_train.size(1) << "\n";
  std::cout << "[INFO] Test samples: " << batch_X_test.size(0)
            << ", Features: " << batch_X_test.size(1) << "\n";

  // === STEP 4: Flatten tensors to 1D buffer for XGBoost C API ===
  auto train_flat = batch_X_train.flatten().contiguous();
  auto test_flat = batch_X_test.flatten().contiguous();

  int train_rows = batch_X_train.size(0);
  int test_rows = batch_X_test.size(0);
  int cols = batch_X_train.size(1);

  // === STEP 5: Create XGBoost DMatrix from tensors ===
  std::cout << "[INFO] Creating XGBoost DMatrices...\n";
  DMatrixHandle dtrain, dtest;
  XGDMatrixCreateFromMat(train_flat.data_ptr<float>(), train_rows, cols, -1,
                         &dtrain);
  XGDMatrixSetFloatInfo(dtrain, "label", batch_y_train.data_ptr<float>(),
                        train_rows);

  XGDMatrixCreateFromMat(test_flat.data_ptr<float>(), test_rows, cols, -1,
                         &dtest);
  XGDMatrixSetFloatInfo(dtest, "label", batch_y_test.data_ptr<float>(),
                        test_rows);

  // === STEP 6: Initialize and configure XGBoost Booster ===
  std::cout << "[INFO] Configuring XGBoost Booster parameters...\n";
  BoosterHandle booster;
  XGBoosterCreate(&dtrain, 1, &booster);
  XGBoosterSetParam(booster, "objective", "binary:logistic");
  XGBoosterSetParam(booster, "eval_metric", "logloss");
  XGBoosterSetParam(booster, "max_depth", "4");
  XGBoosterSetParam(booster, "eta", "0.1");
  XGBoosterSetParam(booster, "subsample", "0.8");
  XGBoosterSetParam(booster, "colsample_bytree", "0.8");

  // === STEP 7: Train XGBoost model ===
  std::cout << "[INFO] Training XGBoost model...\n";
  int num_round = 100;
  for (int iter = 0; iter < num_round; ++iter) {
    XGBoosterUpdateOneIter(booster, iter, dtrain);
    if ((iter + 1) % 10 == 0) {
      std::cout << ">> Completed iteration: " << iter + 1 << "/" << num_round
                << "\n";
    }
  }

  // === STEP 8: Run prediction on test set ===
  std::cout << "[INFO] Running prediction on test set...\n";
  bst_ulong out_len;
  const float *out_result;
  XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);

  // === STEP 9: Evaluate accuracy ===
  std::cout << "[INFO] Evaluating predictions...\n";
  int correct = 0;
  for (int i = 0; i < test_rows; ++i) {
    int pred = (out_result[i] > 0.5f) ? 1 : 0;
    int actual = static_cast<int>(batch_y_test[i].item<float>());
    if (pred == actual)
      correct++;
    if (i < 5) { // print first few predictions
      std::cout << "Sample " << i << ": Predicted=" << pred
                << ", Actual=" << actual << "\n";
    }
  }

  float acc = static_cast<float>(correct) / test_rows;
  std::cout << "[INFO] Final Test Accuracy: " << acc << "\n";

  // === STEP 10: Free XGBoost resources ===
  std::cout << "[INFO] Cleaning up XGBoost resources...\n";
  XGDMatrixFree(dtrain);
  XGDMatrixFree(dtest);
  XGBoosterFree(booster);

  std::cout << "[INFO] Finished successfully.\n";

  return 0;
}
