#pragma once
#ifndef _TFMODEL1_H_
#define _TFMODEL1_H_
#include "TfBase.h"
#include "Model1.h"

class TfModel1 : public TfBase, public Model1
{
private:
	vector<model1Result> resultOutput(vector<tensorflow::Tensor>& tensors);
	void TensorToMat(tensorflow::Tensor mask, cv::Mat* dst);
public:
	TfModel1(std::string iniPath);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	virtual std::string getGroup() { return "TfModel1"; }
};

#endif