#include "ModelProp.h"

void ModelInputProp::initByiniFile(std::string iniPath, std::string group)
{
	char mpp_v[MAX_PATH];

	height = GetPrivateProfileInt(group.c_str(), "height", -1, iniPath.c_str());
	width = GetPrivateProfileInt(group.c_str(), "width", -1, iniPath.c_str());
	channel = GetPrivateProfileInt(group.c_str(), "channel", -1, iniPath.c_str());
	batchsize = GetPrivateProfileInt(group.c_str(), "batchsize", -1, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "mpp", "default", mpp_v, MAX_PATH, iniPath.c_str());

	mpp = std::stod(std::string(mpp_v));
}

extern std::vector<std::string> split(std::string& s, char delimiter);

void ModelFileProp::initByiniFile(std::string iniPath, std::string group)
{
	char inputName_v[MAX_PATH];
	char outputNames_v[MAX_PATH];
	char path_v[MAX_PATH];

	GetPrivateProfileString(group.c_str(), "input", "default", inputName_v, MAX_PATH, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "output", "default", outputNames_v, MAX_PATH, iniPath.c_str());
	GetPrivateProfileString(group.c_str(), "path", "default", path_v, MAX_PATH, iniPath.c_str());

	inputName = std::string(inputName_v);
	filepath = std::string(path_v);
	std::string compositeOutName = std::string(outputNames_v);

	outputNames = split(compositeOutName, ',');
}

void ModelProp::resizeImages(std::vector<cv::Mat>& imgs, int height, int width)
{
	if (imgs.size() == 0)
		return;
	if (imgs[0].rows != height)
	{
		for (auto& iter : imgs)
		{
			cv::resize(iter, iter, cv::Size(height, width));
		}
	}
}

void ModelProp::process(std::vector<cv::Mat>& imgs)
{
	clearResult();
	resizeImages(imgs, inputProp.height, inputProp.width);
	int start = 0;
	for (int i = 0; i < imgs.size(); i = i + inputProp.batchsize)
	{
		auto iterBegin = imgs.begin() + start;
		std::vector<cv::Mat>::iterator iterEnd = imgs.end();
		if (iterBegin + inputProp.batchsize < iterEnd)
		{
			iterEnd = iterBegin + inputProp.batchsize;
			start = i + inputProp.batchsize;
		}
		std::vector<cv::Mat> tempImgs(iterBegin, iterEnd);
		processInBatch(tempImgs);
	}
}

