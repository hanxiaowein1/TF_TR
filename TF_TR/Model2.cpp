#include "Model2.h"

Model2::Model2(std::string iniPath)
{
	inputProp.initByiniFile(iniPath, "Model2");
}

void Model2::clearResult()
{
	m_results.clear();
}
