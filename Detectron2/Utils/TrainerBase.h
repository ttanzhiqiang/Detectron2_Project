#pragma once

#include <Detectron2/Structures/Instances.h>

namespace Detectron2
{
	class TrainerBase {
	public:
		virtual ~TrainerBase() {}

		virtual void train() = 0;
	};
}