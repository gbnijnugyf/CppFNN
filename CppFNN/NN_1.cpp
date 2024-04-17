#include "NN_1.h"
#include <cmath>
#include <random>
#include <iostream>
using namespace std;

NN_1::NN_1(vector<int> layers, double learningRate) : layers(layers), learningRate(learningRate) {
    initializeNetwork();
}

void NN_1::initializeNetwork() {
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < layers.size(); ++i) {
        Layer layer;
        layer.neurons = vector<double>(layers[i]);
        layer.biases = vector<double>(layers[i]);

        if (i > 0) {
            layer.weights = vector<vector<double>>(layers[i], vector<double>(layers[i - 1]));

            for (int j = 0; j < layers[i]; ++j) {
                for (int k = 0; k < layers[i - 1]; ++k) {
                    layer.weights[j][k] = distribution(generator);
                }

                layer.biases[j] = distribution(generator);
            }
        }

        network.push_back(layer);
    }
}

//Sigmoid
double NN_1::activationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double NN_1::activationFunctionDerivative(double x) {
    double fx = activationFunction(x);
    return fx * (1 - fx);
}
//ReLU
//double NN_1::activationFunction(double x) {
//    return max(0.0, x);
//}
//double NN_1::activationFunctionDerivative(double x) {
//    return x > 0 ? 1.0 : 0.0;
//}


vector<double> NN_1::feedforward(vector<double> input) {
    network[0].neurons = input;

    for (int i = 1; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < layers[i - 1]; ++k) {
                sum += network[i - 1].neurons[k] * network[i].weights[j][k];
            }
            sum += network[i].biases[j];
            network[i].neurons[j] = activationFunction(sum);
        }
    }

    return network.back().neurons;
}


void NN_1::backpropagation(vector<double> expected) {
    for (int i = layers.size() - 1; i > 0; --i) {
        for (int j = 0; j < layers[i]; ++j) {
            double error = 0.0;
            if (i == layers.size() - 1) {
                error = expected[j] - network[i].neurons[j];
            }
            else {
                for (int k = 0; k < layers[i + 1]; ++k) {
                    error += network[i + 1].weights[k][j] * network[i + 1].neurons[k];
                }
            }
            network[i].neurons[j] *= activationFunctionDerivative(network[i].neurons[j]) * error;
        }
    }
}


void NN_1::updateWeightsAndBiases() {
    for (int i = 1; i < layers.size(); ++i) {
        for (int j = 0; j < layers[i]; ++j) {
            for (int k = 0; k < layers[i - 1]; ++k) {
                network[i].weights[j][k] += learningRate * network[i].neurons[j] * network[i - 1].neurons[k];
            }
            network[i].biases[j] += learningRate * network[i].neurons[j];
        }
    }
}
// NN_1.cpp
// ...

void NN_1::train(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        double totalLoss = 0.0;

        for (int j = 0; j < inputs.size(); ++j) {
            vector<double> output = feedforward(inputs[j]);
            backpropagation(expectedOutputs[j]);
            updateWeightsAndBiases();

            totalLoss += calculateLoss(expectedOutputs[j], output);
        }

        cout << "Epoch " << i + 1 << ". Loss: " << totalLoss << endl;

        // 输出每层的权重和偏置
        //for (int layer = 1; layer < layers.size(); ++layer) {
        //    cout << "Weights for layer " << layer << ":\n";
        //    for (const auto& weights : network[layer].weights) {
        //        for (double weight : weights) {
        //            cout << weight << ' ';
        //        }
        //        cout << '\n';
        //    }
        //    cout << "Biases for layer " << layer << ":\n";
        //    for (double bias : network[layer].biases) {
        //        cout << bias << ' ';
        //    }
        //    cout << '\n';
        //}
    }
    //double accuracy = calculateAccuracy(inputs, expectedOutputs);
    //cout << "Accuracy: " << accuracy << endl;
}

double NN_1::calculateLoss(vector<double> expected, vector<double> output) {
    double loss = 0.0;
    for (int i = 0; i < expected.size(); ++i) {
        loss += 0.5 * pow(expected[i] - output[i], 2);
    }
    return loss;
}

double NN_1::calculateAccuracy(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs) {
    int correctCount = 0;
    for (int i = 0; i < inputs.size(); ++i) {
        vector<double> output = feedforward(inputs[i]);
        int predicted = max_element(output.begin(), output.end()) - output.begin();
        int actual = max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()) - expectedOutputs[i].begin();
        if (predicted == actual) {
            ++correctCount;
        }
    }
    return static_cast<double>(correctCount) / inputs.size();
}

