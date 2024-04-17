#include "NeuralNetwork.h"
#include "NN_1.h"
#include <iostream>

using namespace std;


// 生成数据集
void generate_dataset(int dataset_size, vector<vector<double>>& inputs, vector<vector<double>>& targets) {
	srand(time(0)); // 设置随机种子
	for (int i = 0; i < dataset_size; ++i) {
		//double a = (double)rand() / RAND_MAX;
		//double b = (double)rand() / RAND_MAX;
		//inputs.push_back({ a, b });
		//targets.push_back({ a , b });
		int a = rand() % 2;
		int b = rand() % 2;
		inputs.push_back({ static_cast<double>(a), static_cast<double>(b) });
		targets.push_back({ static_cast<double>(a ^ b) });
	}
}


//int main() {
//    // 创建一个神经网络，输入层有 2 个节点，隐藏层有 3 个节点，输出层有 1 个节点
//    NeuralNetwork nn(2, 3, 1);
//
//    // 学习率
//    double learningRate = 0.1;
//    int epochs = 5000;
//
//    // 训练数据
//    vector<vector<double>> inputs = { {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0} };
//    vector<vector<double>> targets = { {0.0}, {1.0}, {1.0}, {2.0} };
//
//    // 训练神经网络
//    for (int i = 0; i < epochs; ++i) {
//        //double loss = 0.;
//        for (int j = 0; j < inputs.size(); ++j) {
//            nn.forward(inputs[j]);
//            nn.backward(inputs[j], targets[j], learningRate);
//            // 计算损失值
//            //for (int j = 0; j < targets[i].size(); ++j) {
//            //    double error = outputs[i][j] - pred[j];
//            //    totalLoss += error * error;
//            //}
//        }
//
//        // 打印损失值
//        //cout << "Epoch " << i << ", Loss: " << totalLoss << endl;
//    }
//
//    // 使用训练好的神经网络进行预测
//    for (const auto& input : inputs) {
//        vector<double> output = nn.forward(input);
//        cout << "Input: (" << input[0] << ", " << input[1] << "), ";
//        cout << "Output: " << output[0] << endl;
//    }
//
//    return 0;
//}
//int main() {
//    // 创建数据集
//    vector<vector<double>> inputs;
//    vector<vector<double>> targets;
//    generate_dataset(1000, inputs, targets);
//
//    // 划分训练集和测试集
//    int train_size = 800; // 训练集大小
//    vector<vector<double>> train_inputs(inputs.begin(), inputs.begin() + train_size);
//    vector<vector<double>> train_targets(targets.begin(), targets.begin() + train_size);
//    vector<vector<double>> test_inputs(inputs.begin() + train_size, inputs.end());
//    vector<vector<double>> test_targets(targets.begin() + train_size, targets.end());
//
//    // 创建一个神经网络，输入层有 2 个节点，隐藏层有 3 个节点，输出层有 1 个节点
//    NeuralNetwork nn(2, 3, 1);
//
//    // 学习率
//    double learningRate = 0.001;
//
//    // 训练神经网络
//    for (int i = 0; i < 5000; ++i) {
//        for (int j = 0; j < train_inputs.size(); ++j) {
//            nn.forward(train_inputs[j]);
//            nn.backward(train_inputs[j], train_targets[j], learningRate);
//        }
//    }
//
//    // 使用训练好的神经网络进行预测
//    for (int i = 0; i < test_inputs.size(); ++i) {
//        vector<double> output = nn.forward(test_inputs[i]);
//        cout << "Input: (" << test_inputs[i][0] << ", " << test_inputs[i][1] << "), ";
//        cout << "Output: " << output[0] << ", Expected: " << test_targets[i][0] << endl;
//    }
//
//    return 0;
//}



// 生成一组随机的AND门数据
pair<vector<double>, double> generateAndData() {
	int a = rand() % 2;
	int b = rand() % 2;
	return { {static_cast<double>(a), static_cast<double>(b)}, static_cast<double>(a&b) };
}

// 生成n组随机的AND门数据
pair<vector<vector<double>>, vector<double>> generateAndDataset(int n) {
	srand(time(0));
	vector<vector<double>> inputs;
	vector<double> outputs;
	for (int i = 0; i < n; ++i) {
		auto data = generateAndData();
		inputs.push_back(data.first);
		outputs.push_back(data.second);
	}
	return { inputs, outputs };
}

int main() {

	// 创建数据集
	vector<vector<double>> inputs;
	vector<vector<double>> targets;
	int train_size = 100; // 训练集大小
	int b = 0.8;//划分比例
	generate_dataset(train_size, inputs, targets);

	// 划分训练集和测试集
	vector<vector<double>> train_inputs(inputs.begin(), inputs.begin() + train_size);
	vector<vector<double>> train_targets(targets.begin(), targets.begin() + train_size);



	vector<int> layers = { 2, 3,3,1 };  // 一个包含2个输入神经元，3个隐藏神经元和1个输出神经元的神经网络
	double learningRate = 0.05; //学习率

	NN_1 nn(layers, learningRate);

	//vector<vector<double>> inputs = {
	//	{0.1, 0.2},
	//	{0.3, 0.4},
	//	{0.5, 0.6},
	//	{0.7, 0.8}
	//};

	//vector<vector<double>> expectedOutputs = {
	//	{0.3},
	//	{0.7},
	//	{1.1},
	//	{1.5}
	//};


	 //打印训练数据
	//for (size_t i = 0; i < inputs.size(); ++i) {
	//	std::cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " - Output: " << targets[i][0] << '\n';
	//}

	nn.train(train_inputs, train_targets, 10000);

	generate_dataset(train_size*b, inputs, targets);
	vector<vector<double>> test_inputs(inputs.begin(), inputs.end());
	vector<vector<double>> test_targets(targets.begin(), targets.end());

	vector<vector<double>> inputs_ = { { 0, 1},{0,0},{1,0},{1,1},{1,1} };

	int i = 0;
	for (auto& input : test_inputs) {
		vector<double> output = nn.feedforward(input);

		// 打印输出
		for (double value : output) {
			cout << value << "		";
			cout << "expected:" << test_targets[i++][0] << ' ';
		}
		cout << '\n';
	}

	return 0;
}
