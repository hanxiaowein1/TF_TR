#include <iostream>
#include <windows.h>
#include "Model1.h"
#include "TfModel1.h"
#include "TrModel1.h"
#include "TfModel2.h"
#include "TrModel2.h"
#include "TfRnn.h"
#include "TfModel3.h"
void trModel1Test();

int main()
{
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	string iniPath = "../x64/Release/config.ini";
	TfModel3* tfModel3 = new TfModel3(iniPath, "TfModel3");
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(256, 256));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	tfModel3->createThreadPool();
	tfModel3->processDataConcurrency(imgs);
	vector<model3Result> results = tfModel3->m_results;
	system("pause");
	return 0;
}

void tfRnnTest()
{
	//�ڲ���һ��rnn��
//Ϊ�˲���tfModel2��trModel2�Ĳ�࣬��������tfModel2��trModel2���������rnn�����룬ͬʱ������
	string iniPath = "../x64/Release/config.ini";
	TrModel2* trModel2 = new TrModel2(iniPath, "TrModel2");
	TfModel2* tfModel2 = new TfModel2(iniPath, "TfModel2");
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(256, 256));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	cv::Mat img2 = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img2, img2, cv::Size(256, 256));
	vector<cv::Mat> imgs2;
	for (int i = 0; i < 100; i++)
	{
		imgs2.emplace_back(img2);
	}

	trModel2->createThreadPool();
	trModel2->processDataConcurrency(imgs);
	vector<model2Result> results_tr = trModel2->m_results;

	tfModel2->createThreadPool();
	tfModel2->processDataConcurrency(imgs);
	vector<model2Result> results_tf = tfModel2->m_results;

	//Ȼ��results_tr��results_tf�����뵽rnn�н��в���
	vector<vector<float>> rnn_input_tr;
	for (int i = 0; i < 10; i++)
	{
		rnn_input_tr.emplace_back(results_tr[i].tensor);
	}
	vector<vector<float>> rnn_input_tf;
	for (int i = 0; i < 10; i++)
	{
		rnn_input_tf.emplace_back(results_tf[i].tensor);
	}

	//����rnnʵ��
	TfRnn* tfRnn = new TfRnn(iniPath, "TfRnn1");
	//Ȼ����tfRnn������input
	vector<float> score1 = tfRnn->rnnProcess(rnn_input_tr);
	vector<float> score2 = tfRnn->rnnProcess(rnn_input_tf);
}

void trModel2Test()
{
	string iniPath = "../x64/Release/config.ini";
	TrModel2* trModel2 = new TrModel2(iniPath, "TrModel2");
	//Model1* model1 = trModel1;
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(256, 256));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	//model1->process(imgs);
	trModel2->createThreadPool();
	//trModel1->processTrModel1(imgs);
	trModel2->processDataConcurrency(imgs);
	vector<model2Result> results = trModel2->m_results;
}

void tfModel2Test()
{
	string iniPath = "../x64/Release/config.ini";
	TfModel2* tfModel2 = new TfModel2(iniPath, "TfModel2");
	//Model1* model1 = trModel1;
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(256, 256));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	//model1->process(imgs);
	tfModel2->createThreadPool();
	//trModel1->processTrModel1(imgs);
	tfModel2->processDataConcurrency(imgs);
	vector<model2Result> results = tfModel2->m_results;
}

void tfModel1Test()
{
	string iniPath = "../x64/Release/config.ini";
	TfModel1* tfModel1 = new TfModel1(iniPath, "TfModel1");
	//Model1* model1 = trModel1;
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(512, 512));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	//model1->process(imgs);
	tfModel1->createThreadPool();
	//trModel1->processTrModel1(imgs);
	tfModel1->processDataConcurrency(imgs);
	vector<model1Result> results = tfModel1->m_results;
}

void trModel1Test()
{
	string iniPath = "../x64/Release/config.ini";
	TrModel1* trModel1 = new TrModel1(iniPath, "TrModel1");
	//Model1* model1 = trModel1;
	cv::Mat img = cv::imread("D:\\TEST_DATA\\image_for_test\\12_1.tif");
	cv::resize(img, img, cv::Size(512, 512));
	vector<cv::Mat> imgs;
	for (int i = 0; i < 100; i++)
	{
		imgs.emplace_back(img);
	}

	//model1->process(imgs);
	trModel1->createThreadPool();
	//trModel1->processTrModel1(imgs);
	trModel1->processDataConcurrency(imgs);
	vector<model1Result> results = trModel1->m_results;
}