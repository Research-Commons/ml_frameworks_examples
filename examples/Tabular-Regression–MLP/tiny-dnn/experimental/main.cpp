#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <tiny_dnn/tiny_dnn.h>
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

network<sequential> build_mlp(size_t input_dim) {
    network<sequential> net;
    net << fully_connected_layer(input_dim, 128) << relu()
        << fully_connected_layer(128, 64) << relu()
        << fully_connected_layer(64, 1);  // Output: house price
    return net;
}

std::vector<vec_t> load_features(const std::string &filename) {
    std::vector<vec_t> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        vec_t sample;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            if (!value.empty()) {
                sample.push_back(std::stof(value));
            }
        }
        data.push_back(sample);
    }
    return data;
}

std::vector<vec_t> load_labels(const std::string &filename) {
    std::vector<vec_t> labels;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty()) {
            vec_t label(1); // Only 1 output value
            label[0] = std::stof(line);
            labels.push_back(label);
        }
    }
    return labels;
}

std::vector<vec_t> predict(network<sequential> &net, const std::vector<vec_t> &X) {
    std::vector<vec_t> predictions;
    for (const auto &sample : X) {
        predictions.push_back(net.predict(sample));
    }
    return predictions;
}

void evaluate(const std::vector<vec_t> &predictions, const std::vector<vec_t> &targets) {
    float mse = 0.0f;
    float mae = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        float pred = predictions[i][0];
        float actual = targets[i][0];
        float error = pred - actual;

        mse += error * error;
        mae += std::abs(error);
    }

    mse /= predictions.size();
    mae /= predictions.size();

    float rmse = std::sqrt(mse);

    std::cout << "MSE  = " << mse << "\n";
    std::cout << "RMSE = " << rmse << "\n";
    std::cout << "MAE  = " << mae << "\n";
}

int main() {
    auto train_X = load_features("X_train.csv");
    auto train_y = load_labels("y_train.csv");

    auto test_X = load_features("X_test.csv");
    auto test_y = load_labels("y_test.csv");

    auto net = build_mlp(train_X[0].size());

    // Train
    adagrad optimizer;
    net.train<mse>(optimizer, train_X, train_y, 32, 50);

    // Evaluate on train data
    auto train_preds = predict(net, train_X);
    std::cout << "Train Evaluation:\n";
    evaluate(train_preds, train_y);

    // Evaluate on test data
    auto test_preds = predict(net, test_X);
    std::cout << "Test Evaluation:\n";
    evaluate(test_preds, test_y);

    return 0;
}