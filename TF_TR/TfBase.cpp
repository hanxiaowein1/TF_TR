#include "TfBase.h"

TfBase::TfBase(std::string iniPath, std::string group)
{
	fileProp.initByiniFile(iniPath, group);
}

void TfBase::construct()
{
	tensorflow::GraphDef graph_def;
	tensorflow::Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(),
			fileProp.filepath,
			&graph_def);
	if (!load_graph_status.ok()) {
		cout << "[LoadGraph] load graph failed!\n";
		return;
	}

	tensorflow::SessionOptions options;
	//tensorflow::ConfigProto* config = &options.config;
	options.config.mutable_device_count()->insert({ "GPU",1 });
	options.config.mutable_gpu_options()->set_allow_growth(true);
	options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
	m_session.reset(tensorflow::NewSession(options));
	auto status_creat_session = m_session.get()->Create(graph_def);
	std::cout << "create session success\n";
	if (!status_creat_session.ok()) {
		std::cout << "[LoadGraph] creat session failed!\n" << std::endl;
		return;
	}
}

void TfBase::output(tensorflow::Tensor& tensorInput, vector<tensorflow::Tensor>& tensorOutput)
{
	auto status_run = m_session->Run({ { fileProp.inputName,tensorInput } },
		fileProp.outputNames, {}, &tensorOutput);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}
}

void TfBase::output(std::vector<cv::Mat>& imgs, std::vector<tensorflow::Tensor>& Output)
{
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	tensorflow::Tensor tem_tensor_res(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	for (int i = 0; i < size; i++)
	{
		float* ptr = tem_tensor_res.flat<float>().data() + i * height * width * channel;
		cv::Mat tensor_image(height, width, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//תΪfloat���͵�����
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
	output(tem_tensor_res, Output);
}
