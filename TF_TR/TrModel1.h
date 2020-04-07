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
	std::queue<vector<float>> tensorQueue;//每一个元素都是以batchsize个图像的集合体
	std::mutex queue_lock;
	std::condition_variable tensor_queue_cv;

	using Task = std::function<void()>;
	//线程池
	std::vector<std::thread> pool;//开启一个线程
	// 任务队列
	std::condition_variable cv_task;
	std::queue<Task> tasks;
	//对task的锁
	std::mutex m_lock;
	std::atomic<bool> stopped;//停止线程的标志
	std::atomic<int> idlThrNum = 1;//判断线程有多少闲置;
	std::atomic<int> totalThrNum = 1;//判断总共有多少线程;
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