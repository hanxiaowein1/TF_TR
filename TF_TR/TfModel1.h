#pragma once
#ifndef _TFMODEL1_H_
#define _TFMODEL1_H_
#include <queue>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include "TfBase.h"
#include "Model1.h"

//我可以将这个类写成多线程处理类啊！
//因为从外部从cv::Mat转为tensorflow::Tensor也是时间，而且逻辑也很混乱
//不如直接传入vector<cv::Mat>，然后在类的内部保持一个queue<tensorflow::Tensor>，
//调用多线程将cv::Mat转为tensorflow::Tensor
//那么外部逻辑就跑到了内部来了，更容易修改。
class TfModel1 : public TfBase, public Model1
{
private:
	vector<model1Result> resultOutput(vector<tensorflow::Tensor>& tensors);
	void TensorToMat(tensorflow::Tensor mask, cv::Mat* dst);
		
public:
	TfModel1(std::string iniPath);
	~TfModel1();
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	virtual std::string getGroup() { return "TfModel1"; }

	//先按照自己的思想走
	//1.Mat2Tensor(在父类中已经有实现)
	//2.将batchsize的图像转为Tensor
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
	virtual void processFirstDataInQueue();
};

#endif