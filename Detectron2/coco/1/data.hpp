#ifndef DATA_HPP
#define DATA_HPP

#include <torch/torch.h>
#include <coco/json.hpp>
//#include <utils.hpp>
#include <cstdlib>
#include <opencv2/opencv.hpp>

/**
   Here it defines:
     1, CocoAnn: read coco-format annotation, get gt_bboxes and gt_labels of an image
     2, ImgData: main container to hold information of an image that is needed for train/test
     3, CocoDataset: fetch annotation, read image and does transforms
*/

//void ASSERT(bool cond, const std::string& msg) {
//    if (!cond) {
//        throw std::runtime_error(msg);
//    }
//}

namespace Detectron2
{
  using json = nlohmann::json;
  using namespace torch::indexing;
  class CocoAnn
  {
  public:
    CocoAnn() = default;
    CocoAnn(const std::string &json_path);
    CocoAnn(json &json_ann);
    
    std::map<int64_t, std::string> iid2iname; // map from image id to image name
    std::map<std::string, int64_t> iname2iid; // map from image name to image id
    std::map<int64_t, std::string> cid2cname; // map from category id to category name
    std::map<std::string, int64_t> cname2cid; // map from category name to category id
    std::map<int64_t, int64_t> cidx2cid;  // map from category idx to category id
    std::map<int64_t, int64_t> cid2cidx;  // map from category id to category idx
    // map from image id to gt_bboxes and gt_labels
    std::map<int64_t, std::tuple<torch::Tensor, torch::Tensor>> grouped_anns;
    std::vector<int64_t> iids;
    
    void load_data(json &json_ann);
  private:
    void load_category(json &json_ann);
    void load_image(json &json_ann);
    void load_annotation(json &json_ann);
  };

  /**
     put meta information of an image during training and inference
   */
  struct ImgData
  {
    std::string img_dir;
    std::string file_name;
    int64_t img_id;
    cv::Mat img_cv2;
    std::vector<int64_t> ori_shape;  // [h, w, channels]
    std::vector<int64_t> pad_shape;  
    std::vector<int64_t> img_shape;
    torch::Tensor img_tsr;
    torch::Tensor gt_bboxes;
    torch::Tensor gt_labels;
    float scale_factor{-1.0};
    void print(std::ostream &os);
    void to(const torch::TensorOptions &opts);
  };


  class	CocoDataset : public torch::data::datasets::Dataset<CocoDataset, ImgData>
  {
  public:
    CocoDataset() {};
    CocoDataset(const std::string &img_dir, const std::string &ann_path, const int max_size,const std::string random_flip);
    ImgData get(size_t index) override;
    torch::optional<size_t> size() const override;
    CocoAnn coco_ann();
  private:
    void fetch_ann_data();
    void transform(ImgData &img_data);
    CocoAnn _coco_ann;
    std::string _ann_path;
    std::string _img_dir;
    json _json_ann;
    //json _trans_cfgs;
    int _max_size;
    std::string _random_flip;    
  };

  // the following defines some common image transforms like rescale and flip etc.
  cv::Mat rescale_image(cv::Mat img, float scale);
  std::tuple<cv::Mat, float> rescale_image(cv::Mat img,std::vector<float> img_scale);
  cv::Mat flip_image(cv::Mat img, const std::string &dire);
  torch::Tensor flip_bboxes(torch::Tensor bboxes,
			    std::vector<int64_t> img_shape,
			    const std::string &dire);
  cv::Mat normalize_image(cv::Mat img, std::vector<float> mean, std::vector<float> std);
  cv::Mat pad_image(cv::Mat img, int divisor);
}

#endif
