import pickle
import numpy as np
import os
import csv

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_to_csv(data, labels, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(len(data)):
            # Normalize pixel values to [0, 1]
            row = list(data[i] / 255.0) + [labels[i]]
            writer.writerow(row)

def load_batches(batch_files):
    data = []
    labels = []
    for file in batch_files:
        batch = unpickle(file)
        data.append(batch[b'data'])
        labels.extend(batch[b'labels'])
    data = np.vstack(data)
    return data, labels

base_path = 'cifar-10-batches-py'

# Prepare train data
train_files = [os.path.join(base_path, f"data_batch_{i}") for i in range(1, 6)]
train_data, train_labels = load_batches(train_files)
save_to_csv(train_data, train_labels, 'train.csv')

# Prepare test data
test_batch = os.path.join(base_path, "test_batch")
test = unpickle(test_batch)
test_data = test[b'data']
test_labels = test[b'labels']
save_to_csv(test_data, test_labels, 'test.csv')
