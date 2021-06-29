#ifndef UTILS_HPP
#define UTILS_HPP

#include <detectron2.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <torch/torch.h>

namespace Detectron2
{
  using namespace torch::indexing;
  
  /*
    This module defines various utilities used oftenly but do not belong to any specific category.
   */
  
  // a simple assert function
  void ASSERT(bool condition, const std::string &msg);
  // test if file exists
  bool file_exists(const std::string &path);
  // return a random float from [0.0, 1.0), resolution is 1/RAND_MAX
  float rand(); 

  // convert between different coordinate format
  // all assume to have shape [n, 4],
  // xy in coco means coordinates of upper-left corner while in others it means center
  template <typename T>
  std::vector<T> xywh2xyxy(std::vector<T> &xywh){
    return std::vector<T>({xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]}); }
  torch::Tensor xyxy2xywh(torch::Tensor xyxy);
  torch::Tensor xywh2xyxy(torch::Tensor xywh);
  torch::Tensor xyxy2xywhcoco(torch::Tensor xyxy);
  torch::Tensor bbox_area(torch::Tensor bboxes);

  /*
    randomly select 'num' from tsr's 'dim' dimension without replacement, which means
    'num' should be <= tsr.size(dim).
    it uses torch.randperm which is sub-optimal but the choice without using numpy
   */
  torch::Tensor rand_choice(torch::Tensor tsr, int num, int dim=0);

  std::vector<std::vector<int64_t>> get_grid_size(std::vector<torch::Tensor> &feats);
  std::vector<torch::Tensor> batch_reshape(std::vector<torch::Tensor> &tensors,
					   const std::vector<int64_t> &size);
  std::vector<torch::Tensor> batch_permute(std::vector<torch::Tensor> &tensors,
					   const std::vector<int64_t> &size);
  std::vector<torch::Tensor> batch_repeat(std::vector<torch::Tensor> &tensors,
					  const std::vector<int64_t> &size);
  torch::Tensor restrict_bbox(torch::Tensor bboxes, const std::vector<int64_t> &max_shape);

  /**
     a very simple argument parser with very limited support.
  */
  struct ArgumentParser
  {
    ArgumentParser(const std::string &help="");
    ArgumentParser& add_option(const std::string &name,
		    bool required = false,
		    const std::string &help="");
    ArgumentParser& add_argument(const std::string &name,
				 const std::string &help);
    void parse(int argc, char **argv);
    void print_help();

    // public members
    std::map<std::string, std::string> registered_opts; // name->help
    std::map<std::string, std::vector<std::string>> parsed_opts;
    std::set<std::string> required_opts; //
    std::string arg_name;
    std::string arg_help;
    std::vector<std::string> parsed_args;
    std::string help;
  private:
    bool starts_with(const std::string &str, const std::string &starts);
  };

  // It helps to track progress of training and and testing.
  // For training, it tracks lr, losses, eta, current iteration and epoch.
  // For testing, it prints a progress bar.
  class ProgressTracker
  {
  public:
    static std::string now_str();
    static std::string secs2str(int64_t secs);
    static std::chrono::time_point<std::chrono::high_resolution_clock> now();
    ProgressTracker() {};
    ProgressTracker(int64_t epochs, int64_t iters_per_epoch)
      : _total_epochs(epochs),
	_iters_per_epoch(iters_per_epoch),
	_lr(0)
    {
      _cur_iter = 0;
      _cur_epoch = 1;
      _total_iters = epochs * iters_per_epoch;
      _start = now();
    }

    void next_iter(){ _cur_iter++; }
    void next_epoch(){ _cur_epoch++; }
    double elapsed();
    double eta();
    double fps();
    double lr(){ return _lr; }

    void track_loss(TensorMap& loss_map);
    void track_lr(double lr){ _lr=lr; }
    std::map<std::string, double> mean_loss(bool clear_history=true); // report mean loss


    void progress_bar(); // print progress bar
    void report_progress(std::ostream &os); 
    
    //
    int64_t total_epochs() { return _total_epochs; }
    int64_t iters_per_epoch() { return _iters_per_epoch; }
    int64_t cur_iter() { return _cur_iter; }
    int64_t cur_epoch() { return _cur_epoch; }
    int64_t total_iters() { return _total_iters; }
  private:
    std::map<std::string, std::vector<double>> _tracked_loss;
    //
    double _lr;
    int64_t _total_epochs;
    int64_t _iters_per_epoch;
    int64_t _cur_iter;
    int64_t _cur_epoch;
    int64_t _total_iters;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  };
    
}

#endif