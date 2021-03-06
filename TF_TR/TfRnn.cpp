#include "TfRnn.h"

TfRnn::TfRnn(string iniPath, string group) :TfBase(iniPath, group)
{

}

TfRnn::~TfRnn()
{}

vector<float> TfRnn::rnnProcess(vector<vector<float>>& input)
{
	vector<float> ret;
	if (input.size() == 0)
		return ret;
	if (input[0].size() == 0)
		return ret;
	//要将vector转为tensor才可以进行运行
	tensorflow::Tensor rnnInput(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ 1, (long long)input.size(), (long long)input[0].size() }));
	//然后将input拷贝到rnnInput里面才可以运行
	for (int i = 0;i<input.size();i++)
	{
		float* ptr = rnnInput.flat<float>().data() + i * input[0].size();
		std::memcpy(ptr, input[i].data(), input[0].size() * sizeof(float));
	}
	//然后可以跑了
	vector<tensorflow::Tensor> outTensors;
	output(rnnInput, outTensors);
	ret = resultOutput(outTensors[0]);
	return ret;
}

vector<float> TfRnn::resultOutput(tensorflow::Tensor& tensor)
{
	vector<float> score;
	if (tensor.dims() != 2)
	{
		cout << "rnn model output tensor dims should be 2...\n";
		return score;
	}
	auto scoreTensor = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(1); i++)
	{
		score.emplace_back(scoreTensor(0, i));
	}
	return score;
}
