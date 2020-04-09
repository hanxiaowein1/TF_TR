#pragma once
#ifndef _MODEL1_H_
#define _MODEL1_H_

#include "ModelProp.h"

class Model1
{
public:
	std::vector<model1Result> m_results;
public:
	Model1();
	Model1(std::string iniPath);
	virtual ~Model1() {};
	//�Բ�����model1�ߴ��ͼ�����resize
	
public:
	//�ṩ���е�model1�ĺ���
	virtual std::vector<cv::Point> getRegionPoints2(cv::Mat &mask, float threshold);
	virtual void clearResult();
	
};

#endif