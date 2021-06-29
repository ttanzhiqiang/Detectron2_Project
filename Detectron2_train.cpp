#include <Detectron2/Detectron2Includes.h>
#include <string>
#include "assert.h" 
using namespace Detectron2;
using namespace std;

void demo() {
	int selected = 0; // <-- change this number to choose different demo

	static const char* models[] = {
		//"COCO-Detection/faster_rcnn_R_50_FPN_3x/137851257/model_final_f6e8b1.pkl"
		//"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
		"COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
		//"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
		//"COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl",
		//"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl"
	};
	string model = models[selected];
	auto tokens = tokenize(model, '/');

	string configDir = "D:\\libtorch\\detectron2_project\\configs\\";
	Trainer::Options options;
	options.config_file = File::ComposeFilename(configDir, tokens[0] + "\\" + tokens[1] + ".yaml");

	//options.output = "predict";
	options.opts = { {"MODEL.WEIGHTS", YAML::Node("detectron2://" + model) } };
	//try {
		Trainer::start(options);
	//}
	//catch (const std::exception& e) {
	//	const char* msg = e.what();
	//	std::cerr << msg;
	//}
}

int main()
{
	demo();
}