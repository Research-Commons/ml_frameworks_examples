// Just an example to test the Dockerfile and tiny-dnn setup

#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

int main() {
    using namespace tiny_dnn;
    using namespace tiny_dnn::layers;
    using namespace tiny_dnn::activation;

    // Create a very simple network
    network<sequential> net;
    net << fully_connected_layer(2, 2) << activation::tanh();

    // Input: 2 features
    vec_t input = {0.5, -0.3};

    // Run prediction
    vec_t output = net.predict(input);

    // Print output
    std::cout << "Output:" << std::endl;
    for (float_t val : output)
        std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}