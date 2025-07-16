#pragma once

#include <fstream>
#include <iostream>
#include <torch/torch.h>
#include <vector>

// Reads a CIFAR-10 binary file (batch) and returns a pair of tensors: {images,
// labels} images: [num_samples, 3, 32, 32] (float32 normalized to [0,1])
// labels: [num_samples] (int64)
inline std::pair<torch::Tensor, torch::Tensor>
load_cifar10_bin(const std::string &file_path) {
  constexpr int kImageRows = 32;
  constexpr int kImageCols = 32;
  constexpr int kImageChannels = 3;
  constexpr int kImageSize = kImageRows * kImageCols * kImageChannels;
  constexpr int kRecordSize = 1 + kImageSize; // label + image

  std::ifstream file(file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Could not open CIFAR-10 file: " + file_path);
  }

  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (file_size % kRecordSize != 0) {
    throw std::runtime_error("Invalid CIFAR-10 file size: " + file_path);
  }

  int num_samples = file_size / kRecordSize;

  std::vector<uint8_t> buffer(file_size);
  file.read(reinterpret_cast<char *>(buffer.data()), file_size);

  torch::Tensor images = torch::empty(
      {num_samples, kImageChannels, kImageRows, kImageCols}, torch::kFloat32);
  torch::Tensor labels = torch::empty(num_samples, torch::kInt64);

  for (int i = 0; i < num_samples; ++i) {
    labels[i] = buffer[i * kRecordSize];
    auto img_data = buffer.data() + i * kRecordSize + 1;

    // CIFAR-10 is stored as R plane, G plane, B plane
    for (int c = 0; c < kImageChannels; ++c) {
      for (int j = 0; j < kImageRows * kImageCols; ++j) {
        int row = j / kImageCols;
        int col = j % kImageCols;
        images[i][c][row][col] =
            img_data[c * kImageRows * kImageCols + j] / 255.0f;
      }
    }
  }

  return {images, labels};
}
