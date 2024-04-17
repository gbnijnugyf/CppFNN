#include "NeuralNetwork.h"
#include <cmath>
#include <vector>
using namespace std;

// Sigmoid函数
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// Sigmoid函数的导数
double sigmoid_derivative(double x) {
	return x * (1.0 - x);
}

// 初始化权重和偏差为随机值
NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) :
	inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes) {
	// 初始化权重和偏差
	for (int i = 0; i < hiddenNodes; ++i) {
		hiddenWeights.push_back((double)rand() / RAND_MAX);
		hiddenBiases.push_back((double)rand() / RAND_MAX);
	}
	for (int i = 0; i < outputNodes; ++i) {
		outputWeights.push_back((double)rand() / RAND_MAX);
		outputBiases.push_back((double)rand() / RAND_MAX);
	}
}

// 前向传播
vector<double> NeuralNetwork::forward(vector<double> inputs) {
	// 计算隐藏层的值
	hiddenLayer.clear();
	for (int i = 0; i < hiddenNodes; ++i) {
		double value = 0.0;
		for (int j = 0; j < inputNodes; ++j) {
			value += inputs[j] * hiddenWeights[i];
		}
		value += hiddenBiases[i];
		hiddenLayer.push_back(sigmoid(value));
	}

	// 计算输出层的值
	outputLayer.clear();
	for (int i = 0; i < outputNodes; ++i) {
		double value = 0.0;
		for (int j = 0; j < hiddenNodes; ++j) {
			value += hiddenLayer[j] * outputWeights[i];
		}
		value += outputBiases[i];
		outputLayer.push_back(sigmoid(value));
	}

	return outputLayer;
}

// 反向传播
void NeuralNetwork::backward(vector<double> inputs, vector<double> outputs, double learningRate) {
	// 计算输出层的错误并调整权重和偏差
	vector<double> outputErrors;
	for (int i = 0; i < outputNodes; ++i) {
		double error = outputs[i] - outputLayer[i];
		outputErrors.push_back(error);
		outputWeights[i] += learningRate * error * sigmoid_derivative(outputLayer[i]);
		outputBiases[i] += learningRate * error;
	}

	// 计算隐藏层的错误并调整权重和偏差
	vector<double> hiddenErrors;
	for (int i = 0; i < hiddenNodes; ++i) {
		double error = 0.0;
		for (int j = 0; j < outputNodes; ++j) {
			error += outputErrors[j] * outputWeights[j];
		}
		hiddenErrors.push_back(error);
		hiddenWeights[i] += learningRate * error * sigmoid_derivative(hiddenLayer[i]);
		hiddenBiases[i] += learningRate * error;
	}
}
