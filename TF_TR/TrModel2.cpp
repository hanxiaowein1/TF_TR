#include "TrModel2.h"

TrModel2::TrModel2(std::string iniPath) : TrBase(iniPath, "TrModel2"),
	Model2(iniPath)
{
	//这个时候Model1已经构造完成，开始配置mParam
	//paramConfig({ inputName }, outputNames, { channel, height, width }, batchsize);
	unsigned long long memory = getMemory(iniPath, "TrModel2");
	TrBase::build(memory, inputProp.batchsize);
}

void TrModel2::constructNetwork()
{
	mParser->registerInput(fileProp.inputName.c_str(),
		nvinfer1::Dims3(inputProp.channel, inputProp.height, inputProp.width), 
		nvuffparser::UffInputOrder::kNCHW);
	for (auto& elem : fileProp.outputNames)
	{
		mParser->registerOutput(elem.c_str());
	}
	mParser->parse(fileProp.filepath.c_str(), *mNetwork, nvinfer1::DataType::kFLOAT);
}

vector<float> TrModel2::resultOutput(int size)
{
	vector<float> scores;
	if (!processOutput(size, scores))
	{
		return scores;
	}
	return scores;
}

bool TrModel2::processOutput2(int size, vector<vector<float>>& tensors)
{
	float* output2 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[1]));
	for (int i = 0; i < size; i++)
	{
		vector<float> tmpTensor;
		tmpTensor.insert(tmpTensor.end(), output2 + i * 2048, output2 + (i + 1) * 2048);
		tensors.emplace_back(tmpTensor);
	}
	return true;
}

bool TrModel2::processOutput(int size, vector<float>& scores)
{
	float* output1 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[0]));
	if (size > inputProp.batchsize)
		return false;
	for (int i = 0; i < size; i++)
	{
		float score = output1[i];
		scores.emplace_back(score);
	}
	return true;
}

void TrModel2::processInBatch(std::vector<cv::Mat>& imgs)
{
	infer(imgs);
	vector<float> tempScore = resultOutput(imgs.size());
	vector<vector<float>> tensors;
	processOutput2(imgs.size(), tensors);
	//将结果放到m_results里
	if (tempScore.size() != tensors.size())
	{
		cout << "processInBatch error\n";
		return;
	}
	for (int i = 0; i < tempScore.size(); i++)
	{
		model2Result result;
		result.score = tempScore[i];
		result.tensor = tensors[i];
		m_results.emplace_back(result);
	}
}
