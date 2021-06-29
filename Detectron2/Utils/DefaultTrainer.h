#pragma once
#include <coco/data.hpp>
#include "TrainerBase.h"
#include "utils_train.hpp"
#include <Detectron2/MetaArch/MetaArch.h>
#include <Detectron2/Data/TransformGen.h>
#include <Detectron2/Data/MetadataCatalog.h>
namespace Detectron2
{
	class DefaultTrainer : public TrainerBase {
	public:
		DefaultTrainer(const CfgNode& cfg);
		virtual void train() override;
		void LoadData(std::vector<Detectron2::ImgData>& img_datas, 
			std::vector<DatasetMapperOutput>& inputs, int& img_data_i);
		float get_lr();
		void warmup_lr();
		void set_lr(float lr);
		torch::Tensor sum_loss(TensorMap& loss_map);
	protected:
		CfgNode m_cfg;
		MetaArch m_model;
		Metadata m_metadata;
		std::shared_ptr<TransformGen> m_transform_gen;
		std::string m_input_format;

	private:
		int batch_size;
		int max_iter;

		float base_lr;
		float base_momentum;
		float base_weight_decay;
		std::shared_ptr<torch::optim::SGD> _optimizer{ nullptr };
		std::shared_ptr<CocoDataset> _dataset{ nullptr };
		ProgressTracker _pg_tracker;
		int _warmup_steps;
		float _warmup_start;
		int total_epochs;
		std::set<int> decay_epochs;
		torch::Device _device = torch::Device(torch::kCPU);
		int train_path_size;
		std::vector<int> decay_step;
	};
}