#include "TfModel2.h"

TfModel2::TfModel2(std::string iniPath):TfBase(iniPath, "TfModel2"), Model2(iniPath)
{
	
}

void TfModel2::processInBatch(std::vector<cv::Mat> &imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<float> tempScore = resultOutput(tempTensors[0]);
	auto tensorValue = tempTensors[1].tensor<float, 2>();
	int tempSize = imgs.size();
	float* buffer_start = tempTensors[1].flat<float>().data();
	float* buffer_end = tempTensors[1].flat<float>().data() + tempSize * 2048;
	for (int i = 0; i < tempScore.size(); i++)
	{
		model2Result result;
		result.score = tempScore[i];
		result.tensor.insert(result.tensor.end(), buffer_start + i * 2048, buffer_start + (i + 1) * 2048);
		m_results.emplace_back(result);
	}
}

vector<float> TfModel2::resultOutput(const tensorflow::Tensor& tensor)
{
	vector<float> scores;
	if (tensor.dims() != 2)
	{
		cout << "model2 output size should be two...\n";
		return scores;
	}
	auto scoreTensor = tensor.tensor<float, 2>();
	for (int i = 0; i < tensor.dim_size(0); i++)
	{
		float score = (scoreTensor(i, 0));
		scores.emplace_back(score);
	}
	return scores;
}
