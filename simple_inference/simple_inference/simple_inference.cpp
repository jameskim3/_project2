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

int main(int argc, const char* argv[]) {
	std::cout << std::fixed << std::setprecision(4);

	//Model Load
	clock_t startTime = clock();
	torch::jit::script::Module module;
	try {
		char path[] = "E:/temp/saved_models/traced_200808_resnext50_fold_0.pt";
		module = torch::jit::load(path);
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	torch::Device device(torch::kCPU);
	if (torch::cuda::is_available())
	{
		std::cout << "CUDA is available! Run on GPU." << std::endl;
		device = torch::kCUDA;
	}
	else
	{
		std::cout << "CPU is available! Run on CPU." << std::endl;
	}

	//module.to(at::kCUDA);
	module.eval();
	cout << "Model Load " << clock() - startTime << std::endl;
	startTime = clock();

	//Image Load
	std::vector<std::string> img_path;
	int i = 0;
	img_path.push_back("E:\\temp\\images\\0_Normal\\Train_2.png");
	img_path.push_back("E:\\temp\\images\\1_OverProcess\\Train_1.png");
	img_path.push_back("E:\\temp\\images\\2_UnderProcess\\Train_3.png");
	img_path.push_back("E:\\temp\\images\\3_Defect\\Train_0.png");

	std::vector<torch::Tensor> vImage;
	for (int i = 0;i < 4;i++) {
		Mat img_bgr = imread(img_path[i], IMREAD_COLOR);
		Mat img;
		cvtColor(img_bgr, img, COLOR_BGR2RGB);
		torch::Tensor tensor_image = torch::from_blob(img.data, { 1, img.rows, img.cols, 3 }, at::kByte);
		tensor_image = tensor_image.permute({ 0, 3, 1, 2 });
		tensor_image = tensor_image.to(torch::kFloat).div_(255);
		cout << tensor_image.sizes() << '\n';
		vImage.push_back(tensor_image);
	}

	cout << "Image Load " << clock() - startTime << std::endl;
	startTime = clock();

	// Single File
	for (int i = 0;i < 4;i++) {
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(vImage[i]);
		at::Tensor output = module.forward(inputs).toTensor();
		float* ret = output.data<float>();
		printf("Values: %.4f %.4f %.4f %.4f\n", *(ret + 0), *(ret + 1), *(ret + 2), *(ret + 3));
		printf("max index: %d\n", output.argmax(1).item().toInt());
		cout << "inference " << clock() - startTime << endl;
		startTime = clock();
	}

	// Multi Files
	//at::Tensor input_ = torch::cat(vImage);
	//std::vector<torch::jit::IValue> input;
	//input.push_back(input_);
	//at::Tensor output = module.forward(input).toTensor();
	//std::cout << output << '\n';
	//cout << "Inference " << clock() - startTime << std::endl;
	return 0;
}