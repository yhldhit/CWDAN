#include <algorithm>
#include <cfloat>
#include <vector>
#include <CGAL/basic.h>
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/util/output_matrix.hpp"
#include "caffe/layers/specific_mmd_layer.hpp"
/*
 *specific MMD
 */
typedef CGAL::Quadratic_program_from_iterators
<float **,float*,CGAL::Const_oneset_iterator<CGAL::Comparison_result>,
    bool*, float*,bool*,float*,float**,float*> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

namespace caffe {

template <typename Dtype>
void SpecificMMDLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& top,
const vector<Blob<Dtype>*>& bottom){
}

//void spec_perm_source_and_target(int class_index,int num, int* source_index, int* target_index,int& size_of_source, int& size_of_target, const Dtype* gd_label,const Dtype* predict_label, float thresh_)
template <typename Dtype>
void spec_perm_source_and_target(int class_index,int num, int* source_index, int* target_index,int& size_of_source, int& size_of_target, const Dtype* gd_label,float thresh_)
{
    int source_pos = 0;
    int target_pos = 0;
    for(int i = 0;i < num;++i){
        if(gd_label[i * 2] < 0){
            //source data
            if((gd_label[i*2+1]) == class_index){
                source_index[source_pos++] = i;
            }
        }
        else{
            //target data
            //if(predict_label[2*i] == class_index&&predict_label[2*i+1]<thresh_){
            //if(predict_label[2*i]==class_index)
	    if(gd_label[2*i] == class_index)
            {
                target_index[target_pos++] = i;
            }
        }
    }
    size_of_source = source_pos;
    size_of_target = target_pos;
}

template <typename Dtype>
std::vector<std::pair<Dtype, int> > maxn(int num_of_max, Dtype* mmd, int num_of_kernel){
    std::vector<std::pair<Dtype, int> > temp;
    for(int i = 0; i < num_of_kernel; i++){
        temp.push_back(std::make_pair(mmd[i], i));
    }
    std::partial_sort(
            temp.begin(), temp.begin() + num_of_max, temp.end(), std::greater<std::pair<Dtype, int> >());
    return temp;
}

template <typename Dtype>
void SpecificMMDLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    if(mmd_lambda_ == 0){
        return;
    }
    

    LOG(INFO) <<"mmd_lock_:\t"<<mmd_lock_;
    /*if (bottom.size() == 3)
    {
        Dtype* bottom_label = bottom[2]->mutable_cpu_data();
        const Dtype* top_label = top[0]->cpu_data();
        caffe_copy(bottom[2]->count(0),top_label,bottom_label);

    }*/
    Dtype* bottom_data = bottom[0]->mutable_gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //caffe_gpu_set(bottom[0]->count(0),Dtype(0),bottom_diff);
    
    now_iter_++;
    const int class_num = 10;
    if(mmd_lock_ == 0)
    //assign pseudo labels
    {
        for (int index_class=0;index_class<class_num;index_class++)
        {
            //spec_perm_source_and_target<Dtype>(index_class,input_num_, source_index_, target_index_,size_of_source_, size_of_target_, bottom[1]->cpu_data(),top[0]->cpu_data(),thresh_);
            spec_perm_source_and_target<Dtype>(index_class,input_num_, source_index_, target_index_,size_of_source_, size_of_target_, bottom[1]->cpu_data(),thresh_);
            total_target_num += size_of_target_;
        }
        if(total_target_num >= 1000)
        {
            now_iter_ = 0;
            total_target_num = 0;
            all_sample_num_ = 0;
            mmd_lock_ = 1 - mmd_lock_;
        }
        return;
    }
    
    if(now_iter_ >= 300)
    {
        LOG(INFO)<<"now iter 300";
        mmd_lock_ = 1-mmd_lock_;
        now_iter_ = 0;
        total_target_num = 0;
    }

    //LOG(INFO)<<"bottom number:"<<bottom.size();
    Dtype sum;
    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    LOG(INFO) << "before specific mmd diff " << sum;
    //for each class independently calculate the gradient
    for (int index_class=0;index_class<class_num;index_class++)
    {
        //spec_perm_source_and_target<Dtype>(index_class,input_num_, source_index_, target_index_,size_of_source_, size_of_target_, bottom[1]->cpu_data(),top[0]->cpu_data(),thresh_);
        spec_perm_source_and_target<Dtype>(index_class,input_num_, source_index_, target_index_,size_of_source_, size_of_target_, bottom[1]->cpu_data(),thresh_);
        int class_input_num_ = size_of_target_ + size_of_source_;
        //LOG(INFO)<<"sample number for class:"<<class_input_num_;
        int sample_num;
        if (size_of_source_ <= 1 || size_of_target_ <= 1){
            //return;
            continue;
        }
        if(size_of_source_ > size_of_target_){
            sample_num = size_of_source_;
        }
        else{
            sample_num = size_of_target_;
        }
        int s1,s2,t1,t2;
        srand((unsigned int)time(0));
        Dtype* tempX1 = mmd_data_.mutable_gpu_data();
        Dtype* tempX2 = mmd_data_.mutable_gpu_diff();

        Dtype square_distance;
        Dtype bandwidth = 0;
        for(int i = 0; i < class_input_num_; i++)
        {
            s1 = rand() % class_input_num_;
            s2 = rand() % class_input_num_;
            s2 = (s1 != s2) ? s2 : (s2 + 1) %class_input_num_;

            //LOG(INFO)<<"before s1:"<<s1;
            //LOG(INFO)<<"before s2:"<<s2;
            s1 = (s1<size_of_source_)?source_index_[s1]:target_index_[s1-size_of_source_];
            s2 = (s2<size_of_source_)?source_index_[s2]:target_index_[s2-size_of_source_];   
            //LOG(INFO)<<"s1:"<<s1;
            //LOG(INFO)<<"s2:"<<s2;

            caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s1 * data_dim_, tempX1);
            caffe_gpu_memcpy(sizeof(Dtype) * data_dim_, bottom_data + s2 * data_dim_, tempX2);
            caffe_gpu_sub<Dtype>(data_dim_, tempX1, tempX2, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX2, tempX2, &square_distance);
            bandwidth += square_distance;
        }
        if(fix_gamma_){
            gamma_ = gamma_ < 0 ? (Dtype)class_input_num_ / bandwidth : gamma_;
        }
        else{
            gamma_ = (Dtype)class_input_num_ / bandwidth;
        }
        //LOG(INFO) << "bandwidth " << gamma_;
        Dtype loss = 0;

        Dtype* temp_loss1 = new Dtype[num_of_kernel_];
        Dtype* temp_loss2 = new Dtype[num_of_kernel_];
        Dtype* temp_loss3 = new Dtype[num_of_kernel_];
        Dtype* temp_loss4 = new Dtype[num_of_kernel_];

        all_sample_num_ += sample_num;
        for(int i = 0; i < sample_num; i++){
            //random get sample, insert code
            s1 = rand() % size_of_source_;
            s2 = rand() % size_of_source_;
            s2 = (s1 != s2) ? s2 : (s2 + 1) % size_of_source_;

            t1 = rand() % size_of_target_;
            t2 = rand() % size_of_target_;
            t2 = (t1 != t2) ? t2 : (t2 + 1) % size_of_target_;

            s1 = source_index_[s1];
            s2 = source_index_[s2];
            t1 = target_index_[t1];
            t2 = target_index_[t2];
            //////////////
            Dtype square_sum = 0;
            Dtype factor_for_diff = 0;
            const Dtype* x_s1 = bottom_data + s1 * data_dim_;
            const Dtype* x_s2 = bottom_data + s2 * data_dim_;
            const Dtype* x_t1 = bottom_data + t1 * data_dim_;
            const Dtype* x_t2 = bottom_data + t2 * data_dim_;

            caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_s2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_s1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            Dtype times = pow(kernel_mul_, (Dtype)(num_of_kernel_ / 2));
            Dtype temp_gamma = gamma_ / times;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n);

                sum_of_pure_mmd_[j] += temp_n;
                temp_n = temp_n * beta_[j];
                if(i % 2 == 0){
                    temp_loss1[j] = temp_n;
                }
                else{
                    temp_loss2[j] = temp_n;
                }
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);

            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_s1, x_t2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_s1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            temp_gamma = gamma_ / times;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n) * Dtype(-1);

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + s1 * data_dim_, bottom_diff + s1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);

            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_s2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_s2, x_t1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            temp_gamma = gamma_ / times;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n) * Dtype(-1);

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + s2 * data_dim_, bottom_diff + s2 * data_dim_);

            factor_for_diff = 0;
            caffe_gpu_sub<Dtype>(data_dim_, x_t1, x_t2, tempX1);
            caffe_gpu_sub<Dtype>(data_dim_, x_t2, x_t1, tempX2);
            caffe_gpu_dot<Dtype>(data_dim_, tempX1, tempX1, &square_sum);
            temp_gamma = gamma_ / times;
            for(int j = 0; j < num_of_kernel_; j++){
                Dtype temp_n = (0.0 - temp_gamma) * square_sum;
                temp_n = exp(temp_n);

                sum_of_pure_mmd_[j] += temp_n;
                if(i % 2 == 0){
                    temp_loss1[j] += temp_n;
                }
                else{
                    temp_loss2[j] += temp_n;
                }
                temp_n = temp_n * beta_[j];
                if(i % 2 == 0){
                    temp_loss3[j] = temp_n;
                }
                else{
                    temp_loss4[j] = temp_n;
                }

                loss += temp_n;
                temp_n = (-2) * temp_gamma * temp_n;
                sum_of_epoch_[j] += temp_n;
                factor_for_diff += temp_n;
                temp_gamma = temp_gamma * kernel_mul_;
            }
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX1);
            caffe_gpu_scal(data_dim_, mmd_lambda_ * factor_for_diff / input_num_ * Dtype(32), tempX2);
            caffe_gpu_add(data_dim_, tempX1, bottom_diff + t1 * data_dim_, bottom_diff + t1 * data_dim_);
            caffe_gpu_add(data_dim_, tempX2, bottom_diff + t2 * data_dim_, bottom_diff + t2 * data_dim_);
        }
        delete [] temp_loss1;
        delete [] temp_loss2;
        delete [] temp_loss3;
        delete [] temp_loss4;
    }

    /*LOG(INFO) << num_of_kernel_;*/
    /*for(int i = 0; i < num_of_kernel_; i++){*/
        /*LOG(INFO) << "kernel" << i << ": " << sum_of_epoch_[i];*/
    /*}*/
    caffe_set(num_of_kernel_, Dtype(0), sum_of_epoch_);

    caffe_gpu_asum(input_num_ * data_dim_, bottom[0]->gpu_diff(), &sum);
    LOG(INFO) << "after specific mmd diff sum " << sum;
    LOG(INFO) << "------";
}

INSTANTIATE_LAYER_GPU_FUNCS(SpecificMMDLossLayer);

}


