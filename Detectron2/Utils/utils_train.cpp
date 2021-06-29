#include <Base.h>
#include <Utils/utils_train.hpp>


namespace Detectron2
{
  bool file_exists(const std::string &path){
    std::ifstream ifs(path);
    return ifs.good();
  }

  void ASSERT(bool cond, const std::string &msg){
    if (!cond){
      throw std::runtime_error(msg);
    }
  }

  float rand(){
    return (std::rand() % RAND_MAX) / (float)RAND_MAX;
  }

  torch::Tensor xyxy2xywhcoco(torch::Tensor xyxy){
    return torch::stack
      ({xyxy.index({Slice(), 0}), xyxy.index({Slice(), 1}),
	  xyxy.index({Slice(), 2}) - xyxy.index({Slice(), 0}),
	  xyxy.index({Slice(), 3}) - xyxy.index({Slice(), 1})}, -1);
  }
  
  torch::Tensor xyxy2xywh(torch::Tensor xyxy){
    return torch::stack
      ({(xyxy.index({Slice(), 0}) + xyxy.index({Slice(), 2})) * 0.5,
	  (xyxy.index({Slice(), 1}) + xyxy.index({Slice(), 3})) * 0.5,
	  xyxy.index({Slice(), 2}) - xyxy.index({Slice(), 0}),
	  xyxy.index({Slice(), 3}) - xyxy.index({Slice(), 1})}, -1);
  }

  torch::Tensor xywh2xyxy(torch::Tensor xywh){
    auto ctr_xy = xywh.index({Slice(), Slice(None, 2)});
    auto half_wh = xywh.index({Slice(), Slice(2, None)}) * 0.5;
    return torch::cat
      ({ctr_xy - half_wh, ctr_xy + half_wh}, -1);
  }

  torch::Tensor bbox_area(torch::Tensor bboxes){
    auto xywh = xyxy2xywh(bboxes);
    return xywh.index({Slice(), 2}) * xywh.index({Slice(), 3});
  }

  torch::Tensor rand_choice(torch::Tensor tsr, int num, int dim){
    ASSERT(dim<tsr.dim() && num<=tsr.size(dim), "invalid input for rand_choice");
    auto chosen_index = torch::randperm(tsr.size(dim),
					torch::TensorOptions()
					.dtype(torch::kLong)
					.device(tsr.device())
					).index({Slice(None, num)});
      return tsr.index_select(dim, chosen_index);
  }

  std::vector<std::vector<int64_t>> get_grid_size(std::vector<torch::Tensor> &feats){
    std::vector<std::vector<int64_t>> sizes;
    for(auto &feat : feats){
      auto vec = feat.sizes().vec();
      sizes.emplace_back(vec.begin()+2, vec.begin()+4);
    }
    return sizes;

  }
  std::vector<torch::Tensor> batch_reshape(std::vector<torch::Tensor> &tensors,
					   const std::vector<int64_t> &size){
    std::vector<torch::Tensor> reshaped;
    for(auto &tsr : tensors){
      decltype(tsr.sizes()) new_size(size); 
      reshaped.push_back(tsr.reshape(new_size));
    }
    return reshaped;
  }
  
  std::vector<torch::Tensor> batch_permute(std::vector<torch::Tensor> &tensors,
                                           const std::vector<int64_t> &size){
    std::vector<torch::Tensor> permute;
    for(auto &tsr : tensors){
      decltype(tsr.sizes()) new_size(size);
      permute.push_back(tsr.permute(size));
    }
    return permute;
  }

  std::vector<torch::Tensor> batch_repeat(std::vector<torch::Tensor> &tensors,
					  const std::vector<int64_t> &size){
    std::vector<torch::Tensor> rpt;
    for(auto & tsr: tensors){
      decltype(tsr.sizes()) new_size(size);
      rpt.push_back(tsr.repeat(size));
    }
    return rpt;
  }

  torch::Tensor restrict_bbox(torch::Tensor bboxes, const std::vector<int64_t> &max_shape){
    auto max_h = max_shape[0], max_w = max_shape[1];
    return torch::stack({
	bboxes.index({Slice(), 0}).clamp(0, max_w),
	bboxes.index({Slice(), 1}).clamp(0, max_h),
	bboxes.index({Slice(), 2}).clamp(0, max_w),
	bboxes.index({Slice(), 3}).clamp(0, max_h),
      }, 1);
  }


  ArgumentParser::ArgumentParser(const std::string &help)
    : help(help)
  { }

  void ArgumentParser::parse(int argc, char **argv){
    if (argc == 1) { print_help(); }
    parsed_opts.clear();
    parsed_args.clear();

    int idx = 1;
    while(idx < argc){
      std::string item(argv[idx]);
      if (starts_with(item, "--")){
	std::string opt = item.substr(2);
	if (opt.size()==0 || starts_with(opt, "-") || registered_opts.find(opt) == registered_opts.end()){
	  throw std::runtime_error("illegal or unregistered option name: " + item);
	}
	idx++;
	while(idx < argc){
	  std::string next_item(argv[idx]);
	  if (starts_with(next_item, "--")){
	    break;
	  }else{
	    parsed_opts[opt].push_back(next_item);
	    idx++;
	  }
	}
      }else{
	parsed_args.push_back(item);
	idx++;
      }
    }
    for(auto &required : required_opts){
      if (parsed_opts.find(required)==parsed_opts.end()){
	throw std::runtime_error("missing required option: " + required);
      }
    }
  }

  ArgumentParser& ArgumentParser::add_argument(const std::string &name,
					       const std::string &help){
    if(arg_name.empty()){
      arg_name = name;
      arg_help = help;
    }else{
      throw std::runtime_error("only support one argument name");
    }
    return *this;
  }

  ArgumentParser &ArgumentParser::add_option(const std::string &name,
					     bool required,
					     const std::string &help){
    if (name.empty() || starts_with(name, "-")){
      throw std::runtime_error
	("add_options error: illegal options name: " + name);
    }
    registered_opts[name]=help;
    if (required) {required_opts.insert(name);}
    return *this;
  }

  bool ArgumentParser::starts_with(const std::string &str, const std::string &starts){
    return starts.size() <= str.size() && str.substr(0, starts.size())==starts;
  }

  void ArgumentParser::print_help(){
    if (help.size()>0){
      std::cout << help << std::endl;
    }
    std::cout << "Usage:\n";
    if (!arg_name.empty()){
      std::cout << arg_name << "\t" << arg_help << std::endl;
    }
    for (auto &pir : registered_opts){
      std::cout << "  --" << pir.first << "\t" << pir.second << std::endl;
    }
    exit(1);
  }


  /*
    ProgressTracker
   */
  std::string ProgressTracker::secs2str(int64_t secs){
    std::string s_str;
    int64_t n=0;
    if ((n = secs / 86400) > 0) { s_str += std::to_string(n)+"d"; secs = secs - n * 86400; } // days
    if ((n = secs / 3600) > 0 ) { s_str += std::to_string(n)+"h"; secs = secs - n * 3600;  } // hours
    if ((n = secs / 60) > 0   ) { s_str += std::to_string(n)+"m"; secs = secs - n * 60;    } // minutes
    if ((n = secs / 1) > 0    ) { s_str += std::to_string(n)+"s"; secs = secs - n * 1;     } // secs
    if (s_str.size()==0) return "0s";
    return s_str;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> ProgressTracker::now(){
    return std::chrono::high_resolution_clock::now();
  }

  std::string ProgressTracker::now_str(){
    std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::string now_str(50, '\0');
    std::strftime(&now_str[0], now_str.size(), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(now_str.c_str());
  }

  double ProgressTracker::elapsed(){ return std::chrono::duration<double>(now() - _start).count(); }

  double ProgressTracker::eta(){
    auto _elapsed = elapsed();
    return _elapsed / (_cur_iter + 1) * (_total_iters - _cur_iter);
  }

  double ProgressTracker::fps(){ return _cur_iter / elapsed(); }

  void ProgressTracker::progress_bar(){
    int bar_size = 50;
    int progress = std::min(int((double)cur_iter() / total_iters() * bar_size + 0.5), bar_size);
    std::cout.flush();
    std::cout << "[" << std::string(progress, '#') << std::string(bar_size - progress, '.')
	      << "]" << ", elapsed: " << secs2str(elapsed()) << ", eta: " << secs2str(eta())
	      << ", fps: " << std::fixed << std::setprecision(2) << fps() <<"\r";
    std::cout.flush();
  }

  void ProgressTracker::track_loss(TensorMap& losses){
    for(auto &loss : losses){
      _tracked_loss[loss.first].push_back(loss.second.item<double>());
    }
  }

  std::map<std::string, double> ProgressTracker::mean_loss(bool clear_history){
    auto report = std::map<std::string, double>();
    for(auto &loss : _tracked_loss){
      if (loss.second.size() == 0){ return report; }
      report[loss.first] = std::accumulate(loss.second.begin(), loss.second.end(), 0.0) / loss.second.size();
    }
    if (clear_history){ for (auto &loss : _tracked_loss){ loss.second.clear(); } }
    return report;
  }

  void ProgressTracker::report_progress(std::ostream &os){
    auto loss_report = mean_loss();
    os << now_str() << ", epoch[" << std::to_string(cur_epoch()) << "]["
       <<  cur_iter() % iters_per_epoch() << "/" << iters_per_epoch() << "], "
       << std::fixed << std::setprecision(5) << lr() << ", ";
    for (auto &loss_key : {"rpn_cls_loss", "rpn_bbox_loss", "rcnn_cls_loss", "rcnn_bbox_loss", "loss"}){
      std::cout << loss_key << ":" << std::fixed << std::setprecision(3) << loss_report[loss_key] << ", ";
    }
    os << "eta: " << secs2str(eta()) << ", fps: " << fps();
    os << std::endl;

  }

}