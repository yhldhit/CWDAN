#ifndef CAFFE_ABSVAL_LAYER_HPP_
#define CAFFE_ABSVAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Computes @f$ y = |x| @f$
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the inputs @f$ x @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the computed outputs @f$ y = |x| @f$
 */


template <typename Dtype>
class MMDLossLayer : public NeuronLayer<Dtype> {
  public:
  explicit MMDLossLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param){}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual inline const char* type() const { return "MMDLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  
  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Dtype* beta_;
  //Dtype* weig_;
  Dtype* sum_of_weig_;
  Blob<Dtype> mmd_data_;
  Dtype mmd_lambda_;
  int input_num_;
  int data_dim_;
  int size_of_source_;
  int size_of_target_;
  Dtype gamma_;
  int num_of_kernel_;
  int* source_index_;
  int* target_index_;
  int iter_of_epoch_;
  int now_iter_;
  int now_iter_test_;
  bool fix_gamma_;
  Dtype** Q_;
  Dtype* sum_of_epoch_;
  Dtype* variance_;
  Dtype I_lambda_;
  int all_sample_num_;
  int top_k_;
  Dtype* sum_of_pure_mmd_;
  int method_number_;
  Dtype kernel_mul_;
  int class_num;
  float mmd_lr_;
  float quad_weight_;
  int mmd_lock_;
  int num_class_;
  int test_inter_;
  int total_iter_test_;
  int total_target_num;
  int src_domain_;
  //for evalutation
  Dtype *count_soft;
  Dtype *avg_entropy;
  int *count_hard;
  int *count_tmp;
  int *source_num_batch;
  int *target_num_batch;
  int *source_num_resamp;
  int *target_num_resamp;
  float cross_entropy;
  float entropy_stand;
  float entropy_thresh_;
};



}  // namespace caffe

#endif  // CAFFE_ABSVAL_LAYER_HPP_
