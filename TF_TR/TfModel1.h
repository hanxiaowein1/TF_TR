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
	std::queue<tensorflow::Tensor> tensorQueue;//ÿһ��Ԫ�ض�����batchsize��ͼ��ļ�����
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


private:
	vector<model1Result> resultOutput(vector<tensorflow::Tensor>& tensors);
	void TensorToMat(tensorflow::Tensor mask, cv::Mat* dst);
		
public:
	TfModel1(std::string iniPath);
	virtual void processInBatch(std::vector<cv::Mat>& imgs);
	virtual std::string getGroup() { return "TfModel1"; }
	void createThreadPool();

	//�Ȱ����Լ���˼����
	//1.Mat2Tensor(�ڸ������Ѿ���ʵ��)
	//2.��batchsize��ͼ��תΪTensor
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs);
	void processTfModel1(std::vector<cv::Mat>& imgs);
	
};

#endif