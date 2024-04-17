#include "NeuralNetwork.h"
#include <cmath>
#include <vector>
using namespace std;

// Sigmoid����
double sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

// Sigmoid�����ĵ���
double sigmoid_derivative(double x) {
	return x * (1.0 - x);
}

// ��ʼ��Ȩ�غ�ƫ��Ϊ���ֵ
NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) :
	inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes) {
	// ��ʼ��Ȩ�غ�ƫ��
	for (int i = 0; i < hiddenNodes; ++i) {
		hiddenWeights.push_back((double)rand() / RAND_MAX);
		hiddenBiases.push_back((double)rand() / RAND_MAX);
	}
	for (int i = 0; i < outputNodes; ++i) {
		outputWeights.push_back((double)rand() / RAND_MAX);
		outputBiases.push_back((double)rand() / RAND_MAX);
	}
}

// ǰ�򴫲�
vector<double> NeuralNetwork::forward(vector<double> inputs) {
	// �������ز��ֵ
	hiddenLayer.clear();
	for (int i = 0; i < hiddenNodes; ++i) {
		double value = 0.0;
		for (int j = 0; j < inputNodes; ++j) {
			value += inputs[j] * hiddenWeights[i];
		}
		value += hiddenBiases[i];
		hiddenLayer.push_back(sigmoid(value));
	}

	// ����������ֵ
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

// ���򴫲�
void NeuralNetwork::backward(vector<double> inputs, vector<double> outputs, double learningRate) {
	// ���������Ĵ��󲢵���Ȩ�غ�ƫ��
	vector<double> outputErrors;
	for (int i = 0; i < outputNodes; ++i) {
		double error = outputs[i] - outputLayer[i];
		outputErrors.push_back(error);
		outputWeights[i] += learningRate * error * sigmoid_derivative(outputLayer[i]);
		outputBiases[i] += learningRate * error;
	}

	// �������ز�Ĵ��󲢵���Ȩ�غ�ƫ��
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
