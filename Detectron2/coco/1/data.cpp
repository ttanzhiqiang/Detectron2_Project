#include "Base.h"
#include <Utils/utils_train.hpp>
#include <coco/data.hpp>
#include <Utils/Utils.h>

template <typename T>
std::vector<T> xywh2xyxy(std::vector<T>& xywh) {
    return std::vector<T>({ xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3] });
}

namespace Detectron2
{
  void ImgData::print(std::ostream &os){
    os << "img_dir: " << img_dir << "\n"
       << "file_name: " << file_name << "\n"
       << "img_id: " << img_id << "\n"
       << "ori_shape: " << ori_shape << "\n"
       << "img_shape: " << img_shape << "\n"
       << "pad_shape: " << pad_shape << "\n"
       << "img_tsr: \n";
    if (img_tsr.defined()){
      os << img_tsr.options() << ", shape: " << img_tsr.sizes() << "\n";
    }else{ os << "None\n"; }

    os << "gt_bboxes: \n";
    if (gt_bboxes.defined()){
      os << gt_bboxes << "\n";
    }else{ os << "None\n"; }

    os << "gt_labels: \n";
    if(gt_labels.defined()){
      os << gt_labels << "\n";
    }else{ os << "None\n"; }
    os << "scale_factor: " << scale_factor << "\n";
  }

  void ImgData::to(const torch::TensorOptions &opts){
    img_tsr = img_tsr.to(opts);
    gt_bboxes = gt_bboxes.to(opts);
    gt_labels = gt_labels.to(opts);
  }
  
  CocoAnn::CocoAnn(const std::string &json_path){
    std::ifstream ifs(json_path);
    json anns;
    ifs >> anns;
    load_data(anns);
  }
  CocoAnn::CocoAnn(json &json_ann){
    load_data(json_ann);
  }
  void CocoAnn::load_data(json &json_ann)
  {
    load_category(json_ann);
    load_image(json_ann);
    load_annotation(json_ann);
  }

  void CocoAnn::load_category(json &json_ann){
    cid2cname.clear();
    cname2cid.clear();
    cidx2cid.clear();
    cid2cidx.clear();
    auto json_cates = json_ann["categories"];
    auto num_cates = json_cates.size();
    for (int64_t i=0; i<num_cates; i++){
      auto cur_cate = json_cates[i];
      int64_t cid = cur_cate["id"];
      std::string cname = cur_cate["name"];
      cid2cname[cid] = cname;
      cname2cid[cname] = cid;
      cidx2cid[i] = cid;
      cid2cidx[cid] = i;
    }
  }

  void CocoAnn::load_image(json &json_ann){
    iid2iname.clear();
    iname2iid.clear();
    auto json_imgs = json_ann["images"];
    auto num_imgs = json_imgs.size();
    for(int64_t i=0; i<num_imgs; i++){
      auto cur_img = json_imgs[i];
      int64_t iid = int(cur_img["id"]);
      iids.push_back(iid);
      std::string iname = cur_img["file_name"];
      iid2iname[iid] = iname;
      iname2iid[iname] = iid;
    }
  }

  void CocoAnn::load_annotation(json &json_ann){
    grouped_anns.clear();
    auto json_anns = json_ann["annotations"];
    // use torch::Tensor to hold gt_bboxes and gt_labels
    std::map<int64_t, std::vector<torch::Tensor>> gt_bboxes;  
    std::map<int64_t, std::vector<torch::Tensor>> gt_labels;
    for(auto &ann : json_anns){
      int64_t iid = ann["image_id"];
      std::vector<int32_t> bbox = ann["bbox"];
      bbox = xywh2xyxy(bbox);
      int64_t cid = ann["category_id"];
      auto label = cid2cidx[cid];
      auto bbox_tsr = torch::tensor(bbox);
      auto label_tsr = torch::tensor(label);
      gt_bboxes[iid].push_back(bbox_tsr);
      gt_labels[iid].push_back(label_tsr.view(1));  // can not cat zero-dimension tensor
    }
    for(auto &bbox : gt_bboxes){
      int64_t num = bbox.second.size();
      grouped_anns[bbox.first] = std::make_tuple(torch::cat(bbox.second).view({num, 4}),
                                                 torch::cat(gt_labels[bbox.first]));
    }
  }

  //
  // CocoDataset
  //
  CocoDataset::CocoDataset(const std::string &img_dir,
			   const std::string &ann_path,
			   const int max_size,
               const std::string random_flip)
    : _ann_path(ann_path), _img_dir(img_dir), _max_size(max_size), _random_flip(random_flip)
  { fetch_ann_data(); }

  void CocoDataset::fetch_ann_data(){
    if (_ann_path.size()!=0){
      std::ifstream ifs(_ann_path);
      ifs >> _json_ann;
      _coco_ann.load_data(_json_ann);
    }
  }
  torch::optional<size_t> CocoDataset::size() const {
    return _coco_ann.iids.size();
  }

  // fetch all data of idx-th image
  ImgData CocoDataset::get(size_t idx){
    ImgData idata;
    idata.img_dir = _img_dir;
    auto iid = _coco_ann.iids[idx];
    idata.img_id = iid;
    auto fname = _coco_ann.iid2iname[iid];
    idata.file_name = fname;
    auto img_path = _img_dir + "/" + fname;
    auto img_cv2 = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img_cv2.empty()){
      throw std::runtime_error("can not read image: " + img_path);
    }
    idata.img_cv2 = img_cv2;
    idata.ori_shape = std::vector<int64_t>{img_cv2.rows, img_cv2.cols, 3};
    
    idata.gt_bboxes = std::get<0>(_coco_ann.grouped_anns[iid]);
    idata.gt_labels = std::get<1>(_coco_ann.grouped_anns[iid]);
    // apply tansform
    transform(idata);
    return idata;
  }

  CocoAnn CocoDataset::coco_ann(){
    return _coco_ann;
  }


  cv::Mat rescale_image(cv::Mat img, float scale){
    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_LINEAR);
    return out;
  }

  std::tuple<cv::Mat, float> rescale_image(cv::Mat img, std::vector<float> img_scale){
    auto h = img.rows;
    auto w = img.cols;
    auto max_side = std::max(img_scale[0], img_scale[1]);
    auto min_side = std::min(img_scale[0], img_scale[1]);
    auto scale = std::min(std::min(max_side/h, max_side/w),
			  std::max(min_side/h, min_side/w));
    auto rescaled = rescale_image(img, scale);
    return std::make_tuple(rescaled, scale);
  }

  cv::Mat flip_image(cv::Mat img, const std::string &dire){
    if (dire=="horizontal"){
      cv::flip(img, img, 1);
    }else if (dire=="vertical"){
      cv::flip(img, img, 0);
    }else{
      throw std::runtime_error("unknown flip direction: " + dire);
    }
    return img;
  }

  torch::Tensor flip_bboxes(torch::Tensor bboxes,
			    std::vector<int64_t> img_shape,
			    const std::string &dire){
    auto h = img_shape[0];
    auto w = img_shape[1];
    if (dire == "horizontal"){
      auto flipped_x = w - torch::stack({
	  bboxes.index({Slice(), 2}),
	  bboxes.index({Slice(), 0})}, 1);
      bboxes.index_put_({Slice(), Slice(0,4,2)}, flipped_x);
    } else if (dire == "vertical"){
      auto flipped_y = h - torch::stack({
	  bboxes.index({Slice(), 3}),
	  bboxes.index({Slice(), 1})}, 1);
      bboxes.index_put_({Slice(), Slice(1,4,2)}, flipped_y);
    } else {
      throw std::runtime_error("unknown flip direction: " + dire);
    }
    return bboxes;
  }

  // mean and std are all in order [R, G, B], so may need to reorder color channels first
  cv::Mat normalize_image(cv::Mat img, std::vector<float> mean, std::vector<float> std){
    // convert from BGR to RGB, assume img is from direct result of cv::imread
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    auto scalar_mean = cv::Scalar({mean[0], mean[1], mean[2]});
    auto scalar_std  = cv::Scalar({std[0], std[1], std[2]});
    cv::subtract(img, scalar_mean, img);
    cv::divide(img, scalar_std, img);
    return img;
  }


  cv::Mat normalize_image_new(cv::Mat img, float dsat, float dexp, float dhue) {
      // convert from BGR to RGB, assume img is from direct result of cv::imread
      if (dsat != 1 || dexp != 1 || dhue != 0) {
          if (img.channels() >= 3)
          {
              cv::Mat hsv_src;
              cvtColor(img, hsv_src, cv::COLOR_RGB2HSV);    // RGB to HSV

              std::vector<cv::Mat> hsv;
              cv::split(hsv_src, hsv);

              hsv[1] *= dsat;
              hsv[2] *= dexp;
              hsv[0] += 179 * dhue;

              cv::merge(hsv, hsv_src);

              cvtColor(hsv_src, img, cv::COLOR_HSV2RGB);    // HSV to RGB (the same as previous)
          }
          else
          {
              img *= dexp;
          }
      }
      return img;
  }

  cv::Mat pad_image(cv::Mat img, int divisor){
    //ASSERT(divisor > 0, "divisor must be positive");
    //int tar_h = ((img.rows-1) / divisor + 1) * divisor;
    //int tar_w = ((img.cols-1) / divisor + 1) * divisor;
    cv::copyMakeBorder(img, img, 0, divisor - img.rows, 0, divisor - img.cols,
		       cv::BORDER_CONSTANT, 0);
    return img;
  }


  float random_float()
  {
      unsigned int rnd = 0;
#ifdef WIN32
      rand_s(&rnd);
      return ((float)rnd / (float)UINT_MAX);
#else   // WIN32

      rnd = rand();
#if (RAND_MAX < 65536)
      rnd = rand() * (RAND_MAX + 1) + rnd;
      return((float)rnd / (float)(RAND_MAX * RAND_MAX));
#endif  //(RAND_MAX < 65536)
      return ((float)rnd / (float)RAND_MAX);

#endif  // WIN32
  }

  float rand_uniform_strong(float min, float max)
  {
      if (max < min) {
          float swap = min;
          min = max;
          max = swap;
      }
      return (random_float() * (max - min)) + min;
  }

  unsigned int random_gen()
  {
      unsigned int rnd = 0;
#ifdef WIN32
      rand_s(&rnd);
#else   // WIN32
      rnd = rand();
#if (RAND_MAX < 65536)
      rnd = rand() * (RAND_MAX + 1) + rnd;
#endif  //(RAND_MAX < 65536)
#endif  // WIN32
      return rnd;
  }

  float rand_scale(float s)
  {
      float scale = rand_uniform_strong(1, s);
      if (random_gen() % 2) return scale;
      return 1. / scale;
  }
  /**
     For simplicity, transform is very much fixed, it consists of 
     resize, flip(optional), normalize, pad(optional).
  */
  void CocoDataset::transform(ImgData &img_data){
    cv::Mat img = img_data.img_cv2;
    // resize
    std::vector<float> scale_range;
    scale_range.push_back(_max_size);
    scale_range.push_back(_max_size);
    auto scaled = rescale_image(img, scale_range);
    img = std::get<0>(scaled);
    img_data.scale_factor = std::get<1>(scaled);
    //img_data.scale_factor_h = std::get<2>(scaled);

    //int box_num = img_data.gt_bboxes.size(0);
    //int box_num_size = img_data.gt_bboxes.size(1);
    //for (int i_box_num = 0; i_box_num < box_num; i_box_num++)
    //{
    //    img_data.gt_bboxes.index({ Slice(), 0 + i_box_num * box_num_size }) = img_data.gt_bboxes.index({ Slice(), 0 + i_box_num * box_num_size }) * img_data.scale_factor_w;
    //    img_data.gt_bboxes.index({ Slice(), 1 + i_box_num * box_num_size }) = img_data.gt_bboxes.index({ Slice(), 1 + i_box_num * box_num_size }) * img_data.scale_factor_h;
    //    img_data.gt_bboxes.index({ Slice(), 2 + i_box_num * box_num_size }) = img_data.gt_bboxes.index({ Slice(), 2 + i_box_num * box_num_size }) * img_data.scale_factor_w;
    //    img_data.gt_bboxes.index({ Slice(), 3 + i_box_num * box_num_size }) = img_data.gt_bboxes.index({ Slice(), 3 + i_box_num * box_num_size }) * img_data.scale_factor_h;
    //}
    img_data.gt_bboxes = img_data.gt_bboxes * img_data.scale_factor;
    img_data.img_shape = std::vector<int64_t>({img.rows, img.cols});
    
     //flip
    if (strcmp(_random_flip.c_str(), "horizontal") == 0 ||
        strcmp(_random_flip.c_str(), "vertical") == 0)
    {
        std::vector<int64_t> img_shape({ img.rows, img.cols });
        img = flip_image(img, _random_flip);
        img_data.gt_bboxes = flip_bboxes(img_data.gt_bboxes, img_shape, _random_flip);
    }

    // normalize
    //std::vector<float> mean = _trans_cfgs["img_mean"];
    //std::vector<float> std  = _trans_cfgs["img_std"];
    //img = normalize_image(img, mean, std);

    //float saturation = _trans_cfgs["saturation"];
    //float exposure = _trans_cfgs["exposure"];
    //float hue  = _trans_cfgs["hue"];
    //float dhue = rand_uniform_strong(-hue, hue);
    //float dsat = rand_scale(saturation);
    //float dexp = rand_scale(exposure);
    //img = normalize_image_new(img, dsat, dexp, dhue);

    // pad
    int pad_divisor = _max_size;
    img = pad_image(img, pad_divisor);
    img_data.pad_shape = std::vector<int64_t>({img.rows, img.cols});

    img_data.img_cv2 = img;
    // cv2::mat to torch tensor
    img_data.img_tsr = image_to_tensor(img);
    //img_data.img_tsr = torch::flip(img_data.img_tsr, { -1 });
    img_data.img_tsr = img_data.img_tsr.to(torch::kFloat32).permute({2, 0, 1})/*.unsqueeze(0).contiguous()*/;
  }
}
