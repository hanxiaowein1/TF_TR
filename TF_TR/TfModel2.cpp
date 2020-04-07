#include "TfModel2.h"

TfModel2::TfModel2(std::string iniPath):TfBase(iniPath, "TfModel2"), Model2(iniPath)
{
	
}

void TfModel2::processInBatch(std::vector<cv::Mat> &imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<model2Result> tempResults = resultOutput(tempTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

vector<model2Result> TfModel2::resultOutput(const vector<tensorflow::Tensor>& tensors)
{
	vector<model2Result> tempResults;
	vector<float> scores = resultOutput(tensors[0]);
	auto tensorValue = tensors[1].tensor<float, 2>();
	int tempSize = scores.size();
	const float* buffer_start = tensors[1].flat<float>().data();
	const float* buffer_end = tensors[1].flat<float>().data() + tempSize * 2048;
	for (int i = 0; i < scores.size(); i++)
	{
		model2Result result;
		result.score = scores[i];
		result.tensor.insert(result.tensor.end(), buffer_start + i * 2048, buffer_start + (i + 1) * 2048);
		tempResults.emplace_back(result);
	}
	return tempResults;
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

void TfModel2::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	//1.先将imgs转为Tensor
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	tensorflow::Tensor dstTensor(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	Mat2Tensor(imgs, dstTensor);
	//2.取得锁
	std::unique_lock<std::mutex> myGuard(queue_lock);
	//3.将dstTensor放到队列里面
	tensorQueue.emplace(std::move(dstTensor));
	//4.解锁
	myGuard.unlock();
	//5.通过条件变量通知另一个等待线程：队列里有数据了！
	tensor_queue_cv.notify_one();
}

bool TfModel2::checkQueueEmpty()
{
	if (tensorQueue.empty())
		return true;
	else
		return false;
}

void TfModel2::processFirstDataInQueue()
{
	tensorflow::Tensor tensorInput = std::move(tensorQueue.front());
	tensorQueue.pop();
	vector<tensorflow::Tensor> outputTensors;
	output(tensorInput, outputTensors);
	vector<model2Result> tempResults = resultOutput(outputTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

