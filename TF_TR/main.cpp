#include <iostream>
#include <windows.h>
#include "Model1.h"
#include "TfModel1.h"
#include "TrModel1.h"

int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	string iniPath = "../x64/Release/config.ini";
	TrModel1* trModel1 = new TrModel1(iniPath);
	Model1* model1 = trModel1;
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(512, 512));
	img.convertTo(img, CV_32F);
	img = (img / 255 - 0.5) * 2;
	vector<cv::Mat> imgs;
	imgs.emplace_back(img);
	//model1->process(imgs);
	trModel1->createThreadPool();
	//trModel1->processTrModel1(imgs);
	trModel1->processDataConcurrency(imgs);
	vector<model1Result> results = model1->m_results;
	system("pause");
	return 0;
}