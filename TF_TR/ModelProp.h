#pragma once
#ifndef _MODELPROP_H_
#define _MODELPROP_H_

#include <string>
#include <vector>
#include <functional>
#include <windows.h>
#include "types.h"

class ModelInputProp
{
public:
	int height;
	int width;
	int channel;
	int batchsize;
	double mpp;
public:
	//��ini�ļ��г�ʼ��
	void initByiniFile(std::string iniPath, std::string group);
};

class ModelFileProp
{
public:
	std::string inputName;
	std::vector<std::string> outputNames;
	std::string filepath;
public:
	void initByiniFile(std::string iniPath, std::string group);
};

class ModelProp
{
public:
	ModelInputProp inputProp;
	std::function<void(std::vector<cv::Mat>&)> task;//����������洦��һ��batchsize�ĺ���
							   //��������һ�£����Ҫ����task����ô����Ĳ���һ�������ã����ԣ�����Ҫһ����Ա��������result������task�Ĳ�����Ȼ��ÿ���޸������Ա����
							   //���ҵ���taskѭ������Ҫ�������Ա��������ſ��ԣ���ô����Ҫ�ڶ���task���У�������д���麯��clear()����Model1��Model2����������м���ʵ�ּ���
							   //ʵ�־���clear���Լ����еĽ����perfect�����Ҽ�ֱ�Ǹ��˲ţ��ſ�һ���ع��������õ������������ˣ�66666
public:
	virtual ~ModelProp() {}
	virtual void resizeImages(std::vector<cv::Mat>& imgs, int height, int width);
	virtual void process(std::vector<cv::Mat>& imgs);
	virtual void processInBatch(std::vector<cv::Mat>& imgs) = 0;
	virtual void convertMat2NeededDataInBatch(std::vector<cv::Mat>& imgs) = 0;
	virtual void clearResult() = 0;
};

#endif