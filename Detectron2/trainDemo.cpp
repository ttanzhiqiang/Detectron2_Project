#include "Base.h"
#include "trainDemo.h"
#include <Detectron2/Utils/DefaultTrainer.h>
#include <Detectron2/Data/BuiltinDataset.h>
using namespace Detectron2;

CfgNode Trainer::setup_cfg(const std::string& config_file, const CfgNode::OptionList& opts,
	float confidence_threshold) {
	// load config from file and command-line arguments
	auto cfg = CfgNode::get_cfg();
	cfg.merge_from_file(config_file);
	cfg.merge_from_list(opts);
	// Set score_threshold for builtin models
	cfg["MODEL.RETINANET.SCORE_THRESH_TEST"] = confidence_threshold;
	cfg["MODEL.ROI_HEADS.SCORE_THRESH_TEST"] = confidence_threshold;
	cfg["MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH"] = confidence_threshold;
	cfg.freeze();
	return cfg;
}

void Trainer::start(const Options& options) {
	auto cfg = setup_cfg(options.config_file, options.opts, options.confidence_threshold);
	BuiltinDataset::register_all();
	Trainer m_Trainer(cfg);
	m_Trainer.run_train();
	//m_Trainer.
}

Trainer::Trainer(const CfgNode& cfg, ColorMode instance_mode, bool parallel) :
	m_cpu_device(torch::kCPU), m_instance_mode(instance_mode), m_parallel(parallel)
{
	auto name = CfgNode::parseTuple<std::string>(cfg["DATASETS.TEST"], { "__unused" })[0];
	m_metadata = MetadataCatalog::get(name);
	if (parallel) {
		//int num_gpu = torch::cuda::device_count();
		//m_TrainerBase = make_shared<TrainerBase>(cfg, num_gpu);
	}
	else {
		m_TrainerBase = std::make_shared<DefaultTrainer>(cfg);
	}
}

void Trainer::run_train() {
	//VisImage vis_output;
	m_TrainerBase->train();

}