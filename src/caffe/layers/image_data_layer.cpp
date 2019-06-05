#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  const int class_num = this->layer_param_.image_data_param().class_num();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label_dim = this->layer_param_.image_data_param().label_dim();
  while (infile >> filename) {
    int* labels = new int[label_dim];
    for(int i = 0;i < label_dim;++i){
        infile >> labels[i];
    }
    lines_.push_back(std::make_pair(filename, labels));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape;
  label_shape.push_back(batch_size);
  label_shape.push_back(label_dim);
  top[1]->Reshape(label_shape);
  this->prefetch_label_.Reshape(label_shape);
  //weight_
  vector<int> weight_shape;
  weight_shape.push_back(class_num);
  top[2]->Reshape(weight_shape);
  this->prefetch_weight_.Reshape(weight_shape);

  Dtype* top_weight_= top[2]->mutable_cpu_data();
  Dtype* prefetch_weight = this->prefetch_weight_.mutable_cpu_data();
  caffe_set(class_num,Dtype(1.0/class_num),top_weight_);
  caffe_copy(class_num,top[2]->cpu_data(),prefetch_weight);
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  LOG(INFO) <<"load batch function";
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    this->prefetch_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
  }

  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  int label_dim = this->layer_param_.image_data_param().label_dim();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    for(int i = 0;i < label_dim;++i){
        prefetch_label[item_id * label_dim + i] = lines_[lines_id_].second[i];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}


INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
//template <typename Dtype>
//void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top) {
//  const int new_height = this->layer_param_.image_data_param().new_height();
//  const int new_width  = this->layer_param_.image_data_param().new_width();
//  const bool is_color  = this->layer_param_.image_data_param().is_color();
//  string root_folder = this->layer_param_.image_data_param().root_folder();
//  const int class_num = this->layer_param_.image_data_param().class_num();
//
//  CHECK((new_height == 0 && new_width == 0) ||
//      (new_height > 0 && new_width > 0)) << "Current implementation requires "
//      "new_height and new_width to be set at the same time.";
//  // Read the file with filenames and labels
//  const string& source = this->layer_param_.image_data_param().source();
//  LOG(INFO) << "Opening file " << source;
//  std::ifstream infile(source.c_str());
//  //string line;
//  string filename;
//  size_t pos;
//  int label_dim = this->layer_param_.image_data_param().label_dim();
//  //int label;
//  /*
//  while (std::getline(infile, line)) {
//    pos = line.find_last_of(' ');
//    label = atoi(line.substr(pos + 1).c_str());
//    lines_.push_back(std::make_pair(line.substr(0, pos), label));
//  }
//
//  CHECK(!lines_.empty()) << "File is empty";
//*/
//  while (infile >> filename) {
//    int* labels = new int[label_dim];
//    for (int i = 0; i< label_dim; ++i){
//      infile >> labels[i];
//    }
//    lines_.push_back(std::make_pair(filename,labels));
//  }
//
//  if (this->layer_param_.image_data_param().shuffle()) {
//    // randomly shuffle data
//    LOG(INFO) << "Shuffling data";
//    const unsigned int prefetch_rng_seed = caffe_rng_rand();
//    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
//    ShuffleImages();
//  } else {
//    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
//        this->layer_param_.image_data_param().rand_skip() == 0) {
//      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
//    }
//  }
//  LOG(INFO) << "A total of " << lines_.size() << " images.";
//
//  lines_id_ = 0;
//  // Check if we would need to randomly skip a few data points
//  if (this->layer_param_.image_data_param().rand_skip()) {
//    unsigned int skip = caffe_rng_rand() %
//        this->layer_param_.image_data_param().rand_skip();
//    LOG(INFO) << "Skipping first " << skip << " data points.";
//    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
//    lines_id_ = skip;
//  }
//  // Read an image, and use it to initialize the top blob.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//                                    new_height, new_width, is_color);
//  const int channels = cv_img.channels();
//  const int height = cv_img.rows;
//  const int width = cv_img.cols;
//  // image
//  const int crop_size = this->layer_param_.transform_param().crop_size();
//  const int batch_size = this->layer_param_.image_data_param().batch_size();
//  if (crop_size > 0) {
//    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
//    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
//    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
//  } else {
//    top[0]->Reshape(batch_size, channels, height, width);
//    this->prefetch_data_.Reshape(batch_size, channels, height, width);
//    this->transformed_data_.Reshape(1, channels, height, width);
//  }
//
//  //CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
//  //// Use data_transformer to infer the expected blob shape from a cv_image.
//  //vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
//  //this->transformed_data_.Reshape(top_shape);
//  //// Reshape prefetch_data and top[0] according to the batch_size.
//  //const int batch_size = this->layer_param_.image_data_param().batch_size();
//  //CHECK_GT(batch_size, 0) << "Positive batch size required";
//  //top_shape[0] = batch_size;
//  //for (int i = 0; i < this->prefetch_.size(); ++i) {
//  //  this->prefetch_[i]->data_.Reshape(top_shape);
//  //}
//  //top[0]->Reshape(top_shape);
//
//  LOG(INFO) << "output data size: " << top[0]->num() << ","
//      << top[0]->channels() << "," << top[0]->height() << ","
//      << top[0]->width();
//  // label
//  vector<int> label_shape(1, batch_size);
//  top[1]->Reshape(label_shape);
//  for (int i = 0; i < this->prefetch_.size(); ++i) {
//    this->prefetch_[i]->label_.Reshape(label_shape);
//  }
//// weight
//  vector<int> weight_shape;
//  weight_shape.push_back(class_num);
//  top[2]->Reshape(weight_shape);
//  this->prefetch_weight_.Reshape(weight_shape);
//
//  Dtype* top_weight_ = top[2]->mutable_cpu_data();
//  Dtype* prefetch_weight = this->prefetch_weight_.mutable_cpu_data();
//  caffe_set(class_num,Dtype(1.0/class_num),top_weight_);
//  caffe_copy(class_num,top[2]->cpu_data(),prefetch_weight);
//
//}
//
//template <typename Dtype>
//void ImageDataLayer<Dtype>::ShuffleImages() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//}
//
//// This function is called on prefetch thread
//template <typename Dtype>
//void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
//  CPUTimer batch_timer;
//  batch_timer.Start();
//  double read_time = 0;
//  double trans_time = 0;
//  CPUTimer timer;
//  CHECK(batch->data_.count());
//  CHECK(this->transformed_data_.count());
//  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
//  const int batch_size = image_data_param.batch_size();
//  const int new_height = image_data_param.new_height();
//  const int new_width = image_data_param.new_width();
//  const bool is_color = image_data_param.is_color();
//  string root_folder = image_data_param.root_folder();
//
//  // Reshape according to the first image of each batch
//  // on single input batches allows for inputs of varying dimension.
//  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//      new_height, new_width, is_color);
//  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
//  // Use data_transformer to infer the expected blob shape from a cv_img.
//  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
//  this->transformed_data_.Reshape(top_shape);
//  // Reshape batch according to the batch_size.
//  top_shape[0] = batch_size;
//  batch->data_.Reshape(top_shape);
//
//  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
//  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
//
//  // datum scales
//  const int lines_size = lines_.size();
//  for (int item_id = 0; item_id < batch_size; ++item_id) {
//    // get a blob
//    timer.Start();
//    CHECK_GT(lines_size, lines_id_);
//    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
//        new_height, new_width, is_color);
//    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
//    read_time += timer.MicroSeconds();
//    timer.Start();
//    // Apply transformations (mirror, crop...) to the image
//    int offset = batch->data_.offset(item_id);
//    this->transformed_data_.set_cpu_data(prefetch_data + offset);
//    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
//    trans_time += timer.MicroSeconds();
//
//    prefetch_label[item_id] = lines_[lines_id_].second;
//    // go to the next iter
//    lines_id_++;
//    if (lines_id_ >= lines_size) {
//      // We have reached the end. Restart from the first.
//      DLOG(INFO) << "Restarting data prefetching from start.";
//      lines_id_ = 0;
//      if (this->layer_param_.image_data_param().shuffle()) {
//        ShuffleImages();
//      }
//    }
//  }
//  batch_timer.Stop();
//  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
//  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
//  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
//}
//
//INSTANTIATE_CLASS(ImageDataLayer);
//REGISTER_LAYER_CLASS(ImageData);
//
//}  
//// namespace caffe
#endif  // USE_OPENCV
