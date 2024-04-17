#pragma once
#ifndef NN_1_H
#define NN_1_H
#include <vector>
using namespace std;

// ���ڶ����������еĲ�
struct Layer {
    vector<double> neurons;  // ������Ԫ��ֵ
    vector<vector<double>> weights;  // ����һ�㵽�ò��Ȩ��
    vector<double> biases;  // �ò��ƫ��ֵ
};

class NN_1 {
private:
    vector<int> layers;  // ������Ԫ����������
    vector<Layer> network;  // ������Ľṹ
    double learningRate;  // ѧϰ��

public:
    // ���캯������ʼ��������Ľṹ��ѧϰ��
    NN_1(vector<int> layers, double learningRate);
    // ��ʼ���������Ȩ�غ�ƫ��
    void initializeNetwork();
    // �����
    double activationFunction(double x);
    // ������ĵ���
    double activationFunctionDerivative(double x);
    // ǰ�򴫲�
    vector<double> feedforward(vector<double> input);
    // ���򴫲�
    void backpropagation(vector<double> expected);
    // ����Ȩ�غ�ƫ��
    void updateWeightsAndBiases();
    // ѵ��������
    void train(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs, int iterations);
    // ������ʧ
    double calculateLoss(vector<double> expected, vector<double> output);
    // ����׼ȷ��
    double calculateAccuracy(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs);
};


#endif
