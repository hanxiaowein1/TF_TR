#pragma once
#ifndef _TRMODEL1_H_
#define _TRMODEL1_H_
#include "TrBase.h"
#include "Model1.h"
class TrModel1 : public TrBase, public Model1
{
public:
	TrModel1(std::string iniPath);
	virtual std::string getGroup() { return "TrModel1"; }
private:
	virtual void constructNetwork();
	vector<model1Result> resultOutput(int size);
	bool processOutput(int size, vector<std::pair<float, cv::Mat>>& pairElems);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);	
};

#endif