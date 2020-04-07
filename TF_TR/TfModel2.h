#pragma once
#ifndef _TFMODEL2_H_
#define _TFMODEL2_H_
#include "Model2.h"
#include "TfBase.h"
class TfModel2 : public TfBase, public Model2
{
public:
	TfModel2(std::string iniPath);
	virtual void processInBatch(std::vector<cv::Mat> &imgs);
	virtual std::string getGroup() { return "TfModel2"; }
	virtual bool checkQueueEmpty();
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual void processFirstDataInQueue();
private:
	vector<float> resultOutput(const tensorflow::Tensor& tensor);
	vector<model2Result> resultOutput(const vector<tensorflow::Tensor>& tensors);
};

#endif