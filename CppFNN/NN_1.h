#pragma once
#ifndef NN_1_H
#define NN_1_H
#include <vector>
using namespace std;

// 用于定义神经网络中的层
struct Layer {
    vector<double> neurons;  // 层中神经元的值
    vector<vector<double>> weights;  // 从上一层到该层的权重
    vector<double> biases;  // 该层的偏置值
};

class NN_1 {
private:
    vector<int> layers;  // 各层神经元数量的向量
    vector<Layer> network;  // 神经网络的结构
    double learningRate;  // 学习率

public:
    // 构造函数，初始化神经网络的结构和学习率
    NN_1(vector<int> layers, double learningRate);
    // 初始化神经网络的权重和偏置
    void initializeNetwork();
    // 激活函数
    double activationFunction(double x);
    // 激活函数的导数
    double activationFunctionDerivative(double x);
    // 前向传播
    vector<double> feedforward(vector<double> input);
    // 反向传播
    void backpropagation(vector<double> expected);
    // 更新权重和偏置
    void updateWeightsAndBiases();
    // 训练神经网络
    void train(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs, int iterations);
    // 计算损失
    double calculateLoss(vector<double> expected, vector<double> output);
    // 计算准确率
    double calculateAccuracy(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs);
};


#endif
