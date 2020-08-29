#include <torch/script.h> 
#include <torch/torch.h>
#include <ATen/Aten.h>
#include <torch/cuda.h>

#include <memory>
#include <iostream>
#include <ctime>    

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#include <filesystem>
namespace fs = experimental::filesystem;


int inference(int epoch) {
	std::cout << std::fixed << std::setprecision(4);

	torch::DeviceType device_type;
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Run on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else
	{
		std::cout << "CPU is available! Run on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	//device_type = torch::kCPU;

	//Model Load
	clock_t startTime = clock();
	torch::jit::script::Module module;
	try {
		char path[] = "E:/temp/saved_models/traced_200808_resnext50_fold_0.pt";
		module = torch::jit::load(path);
		if (device_type == torch::kCUDA) {
			module.to(at::kCUDA);
			std::cout << "Module to GPU" << std::endl;
		}
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	module.eval();
	cout << "Model Load " << clock() - startTime << std::endl;
	startTime = clock();

	//Image Load
	std::vector<std::string> img_path;
	std::vector<std::string> img_files;
	img_path.push_back("E:/temp/images/0_Normal/");
	img_path.push_back("E:/temp/images/1_OverProcess/");
	img_path.push_back("E:/temp/images/2_UnderProcess/");
	img_path.push_back("E:/temp/images/3_Defect/");

	for (int i = 0;i < img_path.size();i++) {
		for (auto & p : fs::directory_iterator(img_path[i])) {
			img_files.push_back(p.path().string());
		}
	}

	std::vector<torch::Tensor> vImage;
	for (int i = 0;i < img_files.size();i++) {
		Mat img_bgr = imread(img_files[i], IMREAD_COLOR);
		Mat img;
		cvtColor(img_bgr, img, COLOR_BGR2RGB);
		torch::Tensor tensor_image = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, at::kByte);
		tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
		tensor_image = tensor_image.to(torch::kFloat).div_(255);
		vImage.push_back(tensor_image);
	}
	cout << "Image Load " << clock() - startTime << std::endl;
	startTime = clock();

	// Single File
	int correct = 0;
	for (int i = 0;i < vImage.size();i++) {
		std::vector<torch::jit::IValue> inputs;
		if (device_type == torch::kCUDA) {
			inputs.push_back(vImage[i].to(at::kCUDA));
		}
		else {
			inputs.push_back(vImage[i].to(at::kCPU));
		}
		at::Tensor output = module.forward(inputs).toTensor();
		float od[4];
		if (device_type == torch::kCUDA) {
			for (int j = 0;j < 4;j++) {
				od[j] = output.cpu().data<float>()[j];
			}
		}
		else {
			for (int j = 0;j < 4;j++) {
				od[j] = output.data<float>()[j];
			}
		}
		//Image inference result
		//std::cout << od[0] << ", " << od[1] << ", " << od[2] << ", " << od[3];
		//std::cout << " max " << output.argmax(1).item().toInt();
		//cout << " inference " << clock() - startTime << std::endl;
		//startTime = clock();

		// epco accuracy test
		int total[] = { 512, 607, 1229, 1821 };
		if (i < total[0] && output.argmax(1).item().toInt() == 0) correct++;
		else if (i < total[1] && output.argmax(1).item().toInt() == 1) correct++;
		else if (i < total[2] && output.argmax(1).item().toInt() == 2) correct++;
		else if (i < total[3] && output.argmax(1).item().toInt() == 3) correct++;
	}
	cout << "Accuracy " << correct << std::endl;

	// Multi Files
	//at::Tensor input_ = torch::cat(vImage);
	//std::vector<torch::jit::IValue> input;
	//input.push_back(input_);
	//at::Tensor output = module.forward(input).toTensor();
	//std::cout << output << '\n';
	//cout << "Inference " << clock() - startTime << std::endl;
	cout << "EPOCH " << epoch << " time" << clock() - startTime << std::endl;
	return 0;
}

int main(int argc, const char* argv[]) {
	for (int epoch = 0;epoch < 10;epoch++) {
		inference(epoch);
		std::this_thread::sleep_for(2s);
	}
	return 0;
}