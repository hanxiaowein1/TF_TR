#include "TfModel1.h"

TfModel1::TfModel1(string iniPath):
	TfBase(iniPath, "TfModel1"), Model1(iniPath)
{
	
}

TfModel1::~TfModel1()
{
	stopped.store(true);
	cv_task.notify_all();
	for (std::thread& thread : pool) {
		if (thread.joinable())
			thread.join();
	}
	//Ȼ���ٰ�processTfModel1ͣ��
	tensor_queue_cv.notify_all();
}

vector<model1Result> TfModel1::resultOutput(vector<tensorflow::Tensor>& tensors)
{
	vector<model1Result> retResults;
	if (tensors.size() != 2)
	{
		cout << "model1Base::output: tensors size should be 2\n";
		return retResults;
	}
	auto scores = tensors[0].tensor<float, 2>();
	for (int i = 0; i < tensors[0].dim_size(0); i++)
	{
		model1Result result;
		cv::Mat dst2;
		TensorToMat(tensors[1].Slice(i, i + 1), &dst2);
		result.points = getRegionPoints2(dst2, 0.7);
		result.score = scores(i, 0);
		retResults.emplace_back(result);
	}
	return retResults;
}

void TfModel1::TensorToMat(tensorflow::Tensor mask, cv::Mat* dst)
{
	float* data = new float[(mask.dim_size(1)) * (mask.dim_size(2))];
	auto output_c = mask.tensor<float, 4>();
	//cout << "data 1 :" << endl;
	for (int j = 0; j < mask.dim_size(1); j++) {
		for (int k = 0; k < mask.dim_size(2); k++) {
			data[j * mask.dim_size(1) + k] = output_c(0, j, k, 1);
		}
	}
	cv::Mat myMat = cv::Mat(mask.dim_size(1), mask.dim_size(2), CV_32FC1, data);
	*dst = myMat.clone();
	delete[]data;
}

void TfModel1::processInBatch(std::vector<cv::Mat>& imgs)
{
	vector<tensorflow::Tensor> tempTensors;
	output(imgs, tempTensors);
	vector<model1Result> tempResults = resultOutput(tempTensors);
	m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
}

void TfModel1::createThreadPool()
{
	int num = idlThrNum;
	for (int size = 0; size < num; ++size)
	{   //��ʼ���߳�����
		pool.emplace_back(
			[this]
			{ // �����̺߳���
				while (!this->stopped.load())
				{
					std::function<void()> task;
					{   // ��ȡһ����ִ�е� task
						std::unique_lock<std::mutex> lock{ this->m_lock };// unique_lock ��� lock_guard �ĺô��ǣ�������ʱ unlock() �� lock()
						this->cv_task.wait(lock,
							[this] {
								return this->stopped.load() || !this->tasks.empty();
							}
						); // wait ֱ���� task
						if (this->stopped.load() && this->tasks.empty())
							return;
						task = std::move(this->tasks.front()); // ȡһ�� task
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

void TfModel1::convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs)
{
	//1.�Ƚ�imgsתΪTensor
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	tensorflow::Tensor dstTensor(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	Mat2Tensor(imgs, dstTensor);
	//2.ȡ����
	std::unique_lock<std::mutex> myGuard(queue_lock);
	//3.��dstTensor�ŵ���������
	tensorQueue.emplace(std::move(dstTensor));
	//4.����
	myGuard.unlock();
	//5.ͨ����������֪ͨ��һ���ȴ��̣߳��������������ˣ�
	tensor_queue_cv.notify_one();
}

//�����ʹ������ˣ��Ȳ���һ��
void TfModel1::processTfModel1(std::vector<cv::Mat>& imgs)
{
	m_results.clear();
	if (imgs.size() == 0)
		return;
	resizeImages(imgs, inputProp.height, inputProp.width);
	std::function<void(std::vector<cv::Mat>&)> mat2tensor_fun = std::bind(&TfModel1::convertMat2NeededDataInBatch,this, std::placeholders::_1);
	auto task = std::make_shared<std::packaged_task<void()>>
		(std::bind(&TfModel1::process2, this, std::ref(imgs), mat2tensor_fun));
	std::unique_lock<std::mutex> myGuard(m_lock);
	tasks.emplace(
		[task]() {
			(*task)();
		}
	);
	m_lock.unlock();
	cv_task.notify_one();

	//Ȼ���tensorQueue��ͣ��ȡԪ�ؽ�������
	//ͨ��imgs/batchsize�ɵõ�ѭ������
	int loopTime = std::ceil(float(imgs.size()) / float(inputProp.batchsize));
	for (int i = 0; i < loopTime; i++)
	{
		//ȡ����
		std::unique_lock<std::mutex> myGuard(queue_lock);
		//�ж϶����Ƿ�Ϊ��
		if (!tensorQueue.empty())
		{
			tensorflow::Tensor tensorInput = std::move(tensorQueue.front());
			tensorQueue.pop();
			vector<tensorflow::Tensor> outputTensors;
			output(tensorInput, outputTensors);
			vector<model1Result> tempResults = resultOutput(outputTensors);
			m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
		}
		else
		{
			//�ȴ�
			tensor_queue_cv.wait(myGuard, [this]{
				if (tensorQueue.size() > 0||stopped.load())
					return true;
				else
					return false;
			});
			if (stopped.load())
				return;
			tensorflow::Tensor tensorInput = std::move(tensorQueue.front());
			tensorQueue.pop();
			vector<tensorflow::Tensor> outputTensors;
			output(tensorInput, outputTensors);
			vector<model1Result> tempResults = resultOutput(outputTensors);
			m_results.insert(m_results.end(), tempResults.begin(), tempResults.end());
		}
	}
}
