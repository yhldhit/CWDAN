#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  caffe_copy(prefetch_weight_.count(),top[2]->cpu_data(),
        top[2]->mutable_gpu_data());
  // Start a new prefetch thread
  CreatePrefetchThread();
}
//  if (prefetch_current_) {
//    prefetch_free_.push(prefetch_current_);
//  }
//  prefetch_current_ = prefetch_full_.pop("Waiting for data");
//  // Reshape to loaded data.
//  top[0]->ReshapeLike(prefetch_current_->data_);
//  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
//  if (this->output_labels_) {
//    // Reshape to loaded labels.
//    top[1]->ReshapeLike(prefetch_current_->label_);
//    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
//  }
//}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
