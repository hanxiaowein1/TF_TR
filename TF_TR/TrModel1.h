#pragma once
#ifndef _TRMODEL1_H_
#define _TRMODEL1_H_
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include "TrBase.h"
#include "Model1.h"
class TrModel1 : public TrBase, public Model1
{
private:
	std::queue<vector<float>> tensorQueue;//ÿһ��Ԫ�ض�����batchsize��ͼ��ļ�����
	std::mutex queue_lock;
	std::condition_variable tensor_queue_cv;

	using Task = std::function<void()>;
	//�̳߳�
	std::vector<std::thread> pool;//����һ���߳�
	// �������
	std::condition_variable cv_task;
	std::queue<Task> tasks;
	//��task����
	std::mutex m_lock;
	std::atomic<bool> stopped;//ֹͣ�̵߳ı�־
	std::atomic<int> idlThrNum = 1;//�ж��߳��ж�������;
	std::atomic<int> totalThrNum = 1;//�ж��ܹ��ж����߳�;
public:
	TrModel1(std::string iniPath);
	virtual std::string getGroup() { return "TrModel1"; }
	void processTrModel1(std::vector<cv::Mat>& imgs);
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	void createThreadPool();
private:
	virtual void constructNetwork();
	vector<model1Result> resultOutput(int size);
	bool processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);	
};

#endif