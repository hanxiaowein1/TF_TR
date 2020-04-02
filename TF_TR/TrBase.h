#pragma once
#ifndef _TRBASE_H_
#define _TRBASE_H_
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
#include "ModelProp.h"
class TrBase
{
public:
	template <typename T>
	using myUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
	ModelFileProp fileProp;
	//Ϊ�˼򻯴��룬������ͨ�������Ƴ���ȫ������ModelProp����֤
	//samplesCommon::UffSampleParams mParams;
	myUniquePtr<nvinfer1::IBuilder> mBuilder{ nullptr };
	myUniquePtr<nvinfer1::INetworkDefinition> mNetwork{ nullptr };
	myUniquePtr<nvinfer1::IBuilderConfig> mConfig{ nullptr };
	myUniquePtr<nvuffparser::IUffParser> mParser{ nullptr };
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine{ nullptr };
	samplesCommon::BufferManager* mBuffer{ nullptr };
	myUniquePtr<nvinfer1::IExecutionContext> mContext{ nullptr };
public:
	TrBase(std::string iniPath, std::string group);
	virtual ~TrBase() {};
	virtual bool build(unsigned long long memory, int batchsize);
	virtual bool infer(vector<cv::Mat>& imgs);
	virtual void constructNetwork() = 0;
	virtual bool processInput(vector<cv::Mat>& imgs);
	virtual unsigned long long getMemory(std::string iniPath, std::string group);
};


#endif