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

//�ҿ��Խ������д�ɶ��̴߳����డ��
//��Ϊ���ⲿ��cv::MatתΪtensorflow::TensorҲ��ʱ�䣬�����߼�Ҳ�ܻ���
//����ֱ�Ӵ���vector<cv::Mat>��Ȼ��������ڲ�����һ��queue<tensorflow::Tensor>��
//���ö��߳̽�cv::MatתΪtensorflow::Tensor
//��ô�ⲿ�߼����ܵ����ڲ����ˣ��������޸ġ�
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

	//�Ȱ����Լ���˼����
	//1.Mat2Tensor(�ڸ������Ѿ���ʵ��)
	//2.��batchsize��ͼ��תΪTensor
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	virtual bool checkQueueEmpty();
	virtual void processFirstDataInQueue();
};

#endif