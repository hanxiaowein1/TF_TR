#pragma once
#ifndef _MODEL2_H_
#define _MODEL2_H_
#include "ModelProp.h"
class Model2 : public ModelProp
{
public:
	std::vector<model2Result> m_results;
public:
	Model2() {}
	Model2(std::string iniPath);
	virtual ~Model2() {};
	virtual std::vector<float> model2Process(std::vector<cv::Mat>& imgs) = 0;
	virtual bool model2Process(std::vector<cv::Mat>& imgs, std::vector<float>& vec) = 0;
	virtual void clearResult();
};

#endif