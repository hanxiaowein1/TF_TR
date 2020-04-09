#pragma once
#ifndef _MODEL2_H_
#define _MODEL2_H_
#include "ModelProp.h"
class Model2
{
public:
	std::vector<model2Result> m_results;
public:
	Model2() {}
	Model2(std::string iniPath);
	virtual ~Model2() {};
	virtual void clearResult();
};

#endif