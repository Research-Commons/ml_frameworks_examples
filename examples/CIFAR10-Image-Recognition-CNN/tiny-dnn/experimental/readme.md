# CIFAR-10 Image Recognition CNN with tiny-dnn

A C++ implementation of a Convolutional Neural Network for CIFAR-10 image classification using the tiny-dnn library.

## Overview

This project demonstrates image classification on the CIFAR-10 dataset using a CNN implemented in C++ with the tiny-dnn library. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Network Architecture

```
Input (32x32x3) 
    ↓
Conv Layer (5x5, 32 filters) → ReLU → MaxPool (2x2) 
    ↓ (16x16x32)
Conv Layer (5x5, 64 filters) → ReLU → MaxPool (2x2)
    ↓ (8x8x64)
Fully Connected (4096 → 256) → ReLU
    ↓
Fully Connected (256 → 10) → Softmax
    ↓
Output (10 classes)
```

## Prerequisites

### Local Build
- GCC 11+ with C++14 support
- CMake 3.10+
- Python 3.x
- Git

### Docker Build
- Docker installed

## Quick Start with Docker

Build and run the container:

```bash
docker build -t cifar10-cnn .
docker run cifar10-cnn
```

The Docker container will:

1. Download and prepare the CIFAR-10 dataset
2. Build the tiny-dnn library
3. Compile the C++ project
4. Execute the training and testing