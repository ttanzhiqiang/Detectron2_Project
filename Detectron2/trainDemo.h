#pragma once
#include <Detectron2/Utils/TrainerBase.h>
#include <Detectron2/Structures/Instances.h>
#include <Detectron2/Data/MetadataCatalog.h>
namespace Detectron2
{
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// converted from tools/train_net.py

	class Trainer {
	public:
		struct Options {
			std::string config_file					// path to config file
				= "configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml";
			bool webcam = false;					// Take inputs from webcam
			std::vector<std::string> input;			// A list of space separated input images
													// or a single glob pattern such as 'directory/*.jpg'
			CfgNode::OptionList opts;				// Modify config options using the command-line 'KEY VALUE' pairs
			float confidence_threshold = 0.5; 		// Minimum score for instance predictions to be shown
		};
		static void start(const Options& options);

		static CfgNode setup_cfg(const std::string& config_file, const CfgNode::OptionList& opts,
			float confidence_threshold);

		void run_train();
	public:
		Trainer(const CfgNode& cfg, ColorMode instance_mode = ColorMode::kIMAGE, bool parallel = false);
	private:
		Metadata m_metadata;
		torch::Device m_cpu_device;
		ColorMode m_instance_mode;
		bool m_parallel;
		std::shared_ptr<TrainerBase> m_TrainerBase;
	};
}