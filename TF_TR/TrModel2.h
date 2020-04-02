#pragma once
#ifndef _TRMODEL2_H_
#define _TRMODEL2_H_
#include "TrBase.h"
#include "Model2.h"
class TrModel2 : public TrBase, public Model2
{
public:
	TrModel2(std::string iniPath);
	virtual std::string getGroup() { return "TrModel2"; }
private:
	virtual void constructNetwork();
	bool processOutput(int size, vector<float>& scores);
	bool processOutput2(int size, std::vector<std::vector<float>>& tensors);
	vector<float> resultOutput(int size);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	
};

#endif