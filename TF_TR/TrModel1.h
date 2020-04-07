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
public:
	TrModel1(std::string iniPath);
	virtual std::string getGroup() { return "TrModel1"; }
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
	virtual void processFirstDataInQueue();
private:
	virtual void constructNetwork();
	vector<model1Result> resultOutput(int size);
	bool processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);	
};

#endif