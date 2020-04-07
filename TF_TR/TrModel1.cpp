#include "TrModel1.h"

TrModel1::TrModel1(std::string iniPath):TrBase(iniPath, "TrModel1"),
	Model1(iniPath)
{
	unsigned long long memory = getMemory(iniPath, "TrModel1");
	TrBase::build(memory, inputProp.batchsize);
}

void TrModel1::constructNetwork()
{
	mParser->registerInput(fileProp.inputName.c_str(), 
		nvinfer1::Dims3(inputProp.channel, inputProp.height, inputProp.width),
		nvuffparser::UffInputOrder::kNCHW);
	for (auto& elem : fileProp.outputNames)
	{
		mParser->registerOutput(elem.c_str());
	}
	//在添上两层，一层shuffle layer，一层softmax layer
	mParser->parse(fileProp.filepath.c_str(), *mNetwork, nvinfer1::DataType::kFLOAT);
	ITensor* outputTensor = mNetwork->getOutput(1);
	auto shuffle_layer = mNetwork->addShuffle(*outputTensor);
	Permutation permutation;
	for (int i = 0; i < Dims::MAX_DIMS; i++)
	{
		permutation.order[i] = 0;
	}
	permutation.order[0] = 2;
	permutation.order[1] = 0;
	permutation.order[2] = 1;
	shuffle_layer->setFirstTranspose(permutation);
	auto softmax_layer = mNetwork->addSoftMax(*shuffle_layer->getOutput(0));
	softmax_layer->getOutput(0)->setName("softmax/output");
	fileProp.outputNames.emplace_back("softmax/output");
	mNetwork->markOutput(*softmax_layer->getOutput(0));
}

bool TrModel1::processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems)
{
	float* output1 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[0]));
	float* output2 = static_cast<float*>(mBuffer->getHostBuffer(fileProp.outputNames[2]));
	if (size > inputProp.batchsize)
		return false;
	for (int i = 0; i < size; i++)
	{
		std::pair<float, cv::Mat> pairElem;
		pairElem.first = output1[i];
		cv::Mat temp(16, 16, CV_32FC1, output2 + 512 * i + 256);
		pairElem.second = temp.clone();
		pairElems.emplace_back(std::move(pairElem));
	}
	return true;
}

vector<model1Result> TrModel1::resultOutput(int size)
{
	vector<std::pair<float, cv::Mat>> pairElems;
	vector<model1Result> retResults;
	if (!processOutput(size, pairElems))
		return retResults;
	for (int i = 0; i < size; i++)
	{
		model1Result result;
		result.points = getRegionPoints2(pairElems[i].second, 0.7f);
		result.score = pairElems[i].first;
		retResults.emplace_back(result);
	}
	return retResults;
}

void TrModel1::processInBatch(std::vector<cv::Mat>& imgs)
{
	infer(imgs);
	vector<model1Result> tempResults = resultOutput(imgs.size());
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

void TrModel1::createThreadPool()
{
	int num = idlThrNum;
	for (int size = 0; size < num; ++size)
	{   //初始化线程数量
		pool.emplace_back(
			[this]
			{ // 工作线程函数
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // 获取一个待执行的 task
						std::unique_lock<std::mutex> lock{ this->m_lock };// unique_lock 相比 lock_guard 的好处是：可以随时 unlock() 和 lock()
						this->cv_task.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait 直到有 task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // 取一个 task
						this->tasks.pop();
					}
					idlThrNum--;
					task();
					idlThrNum++;
				}
			}
			);
	}
}

void TrModel1::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	vector<float> neededData(height * width * channel * size);
	transformInMemory(imgs, neededData.data());
	//将其塞到队列里
	std::unique_lock<std::mutex> myGuard(queue_lock);
	tensorQueue.emplace(std::move(neededData));
	myGuard.unlock();
	//通过条件变量通知另一个等待线程：队列里有数据了！
	tensor_queue_cv.notify_one();
}

void TrModel1::processTrModel1(std::vector<cv::Mat>& imgs)
{
	if (imgs.size() == 0)
		return;
	resizeImages(imgs, inputProp.height, inputProp.width);
	m_results.clear();
	std::function<void(std::vector<cv::Mat>&)> mat2tensor_fun = std::bind(&TrModel1::convertMat2NeededDataInBatch, this, std::placeholders::_1);
	auto task = std::make_shared<std::packaged_task<void()>>
		(std::bind(&TrModel1::process2, this, std::ref(imgs), mat2tensor_fun));
	std::unique_lock<std::mutex> myGuard(m_lock);
	tasks.emplace(
		[task]() {
			(*task)();
		}
	);
	myGuard.unlock();
	cv_task.notify_one();

	//开始处理
	int loopTime = std::ceil(float(imgs.size()) / float(inputProp.batchsize));
	//判断队列是否为空
	for (int i = 0; i < loopTime; i++)
	{
		std::unique_lock<std::mutex> myGuard(queue_lock);
		if (!tensorQueue.empty())
		{
			vector<float> neededData = std::move(tensorQueue.front());
			tensorQueue.pop();
			float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(fileProp.inputName));
			std::memcpy(hostInputBuffer, neededData.data(), neededData.size());
			mBuffer->copyInputToDevice();
			int tensorBatch = neededData.size() / 
				(inputProp.height * inputProp.width * inputProp.channel);
			if (!mContext->execute(tensorBatch, mBuffer->getDeviceBindings().data()))
			{
				return;
			}
			mBuffer->copyOutputToHost();
			vector<model1Result> tempResults = resultOutput(tensorBatch);
			m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			//等待
			tensor_queue_cv.wait(myGuard, [this] {
				if (tensorQueue.size() > 0 || stopped.load())
					return true;
				else
					return false;
				});
			if (stopped.load())
				return;
			vector<float> neededData = std::move(tensorQueue.front());
			tensorQueue.pop();
			float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(fileProp.inputName));
			std::memcpy(hostInputBuffer, neededData.data(), neededData.size()*sizeof(float));
			mBuffer->copyInputToDevice();

			int tensorBatch = neededData.size() /
				(inputProp.height * inputProp.width * inputProp.channel);
			if (!mContext->execute(tensorBatch, mBuffer->getDeviceBindings().data()))
			{
				return;
			}
			mBuffer->copyOutputToHost();
			vector<model1Result> tempResults = resultOutput(tensorBatch);
			m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
		}
	}
}


