#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tiny_dnn/tiny_dnn.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

void load_csv(const std::string &filename,
              std::vector<vec_t> &images,
              std::vector<label_t> &labels,
              int max_samples = -1) {  // Add parameter to limit samples
    std::ifstream file(filename);
    std::string line;
    int count = 0;

    while (std::getline(file, line) && (max_samples == -1 || count < max_samples)) {
        std::stringstream ss(line);
        std::string val;
        vec_t image;
        for (int i = 0; i < 3072; ++i) {
            std::getline(ss, val, ',');
            image.push_back(std::stof(val));
        }
        images.push_back(image);
        std::getline(ss, val, ',');
        labels.push_back(static_cast<label_t>(std::stoi(val)));
        count++;
    }
}

int main() {
    // Load CIFAR-10 CSV data
    std::vector<vec_t> train_images, test_images;
    std::vector<label_t> train_labels, test_labels;

    std::cout << "Loading training data...\n";
    load_csv("train.csv", train_images, train_labels, 5000);  // Only 5000 samples
    std::cout << "Loading test data...\n";
    load_csv("test.csv", test_images, test_labels, 1000);     // Only 1000 samples

    std::cout << "Loaded " << train_images.size() << " training samples\n";
    std::cout << "Loaded " << test_images.size() << " test samples\n";

    // Define the CNN
    network<sequential> net;

    net << tiny_dnn::convolutional_layer(32, 32, 5, 3, 32, padding::same) << relu()
        << tiny_dnn::max_pooling_layer(32, 32, 32, 2)               // 16x16x32
        << tiny_dnn::convolutional_layer(16, 16, 5, 32, 64, padding::same) << relu()
        << tiny_dnn::max_pooling_layer(16, 16, 64, 2)               // 8x8x64
        << tiny_dnn::fully_connected_layer(8 * 8 * 64, 256) << relu()
        << tiny_dnn::fully_connected_layer(256, 10) << softmax();

    // Define optimizer
    adam optimizer;

    // Train
    std::cout << "Training...\n";
    progress_display disp(train_images.size());
    timer t;

    net.train<cross_entropy>(optimizer, train_images, train_labels,
                             /*batch_size*/ 128,
                             /*epochs*/ 4,
                             []() {},  // on_batch_enumerate
                             [&]() {   // on_epoch_enumerate
                                 std::cout << t.elapsed() << "s elapsed." << std::endl;
                                 t.restart();
                                 disp.restart(train_images.size());
                             });

    // Evaluate
    std::cout << "Testing...\n";
    result res = net.test(test_images, test_labels);
    std::cout << "Test accuracy: " << res.accuracy() << std::endl;

    return 0;
}
