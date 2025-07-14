#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

using i64 = int64_t;

struct CSVLoadOptions {
  int target_column = -1; // -1 means last column
  bool has_header = true;
  bool verbose = true;
};

template <typename scalar_t = float>
std::pair<torch::Tensor, torch::Tensor>
load_csv(const std::string &filename, const CSVLoadOptions &options = {}) {
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  if (options.has_header) {
    std::getline(file, line); // skip header
  }

  std::vector<std::vector<scalar_t>> features_vec;
  std::vector<scalar_t> targets_vec;

  if (options.verbose) {
    std::cout << ">> Loading: " << filename << "\n";
  }

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<scalar_t> row;

    while (std::getline(ss, cell, ',')) {
      row.push_back(static_cast<scalar_t>(std::stof(cell)));
    }

    int target_idx =
        options.target_column >= 0 ? options.target_column : row.size() - 1;
    if (target_idx >= static_cast<int>(row.size())) {
      throw std::runtime_error("target_column index out of bounds");
    }

    targets_vec.push_back(row[target_idx]);
    row.erase(row.begin() + target_idx);

    features_vec.push_back(row);
  }

  const auto num_samples = features_vec.size();
  const auto num_features = features_vec[0].size();

  if (options.verbose) {
    std::cout << ">> Found " << num_samples << " samples with " << num_features
              << " features.\n";
  }

  torch::Tensor features =
      torch::empty({(i64)num_samples, (i64)num_features}, torch::kFloat32);
  torch::Tensor targets = torch::empty({(i64)num_samples, 1}, torch::kFloat32);

  for (size_t i = 0; i < num_samples; ++i) {
    for (size_t j = 0; j < num_features; ++j) {
      features[i][j] = features_vec[i][j];
    }
    targets[i][0] = targets_vec[i];
  }

  return {features, targets};
}
