#pragma once
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <vector>
using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);

    vector<double> forward(vector<double> inputs);
    void backward(vector<double> inputs, vector<double> outputs, double learningRate);

private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    vector<double> hiddenWeights;
    vector<double> hiddenBiases;
    vector<double> outputWeights;
    vector<double> outputBiases;
    vector<double> hiddenLayer;
    vector<double> outputLayer;
};

#endif


