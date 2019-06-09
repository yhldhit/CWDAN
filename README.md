CWMMD is developed for UDA to deal with class weight bias. This is extension to our WMMD work. Based on Caffe. 
The experimental enviroments:
System: Ubuntu 16.04 + Cuda 8.0

To train a model:
Step1. prepare model for initialization: The bvlc_reference_caffenet and bvlc_googlenet are used as initialzation for Alexnet and GoogleNet, respectively. They can be download from here. The model structure is specified in the relative path model/task_name/train_val.prototxt, e.g., when we transfer from amazon to caltech in office-10+caltech-10 dataset, replace the task_name with amazon_to_caltech.
Step2. prepare data: Since we reach the raw images on the disk when training, all the images file path need to be written into the .txt file, kept in data/task_name/ directory. For example, data/amazon_to_caltech/train(value).txt. To constrcuct such txtfile, a python script is offered in file images/code/data_constructor.py. Following are the main steps:
Step3. fine tune a model: To fine tune the model paramenters, run the shell ./model/amazon2caltech/train.sh You will see the accuracy results once the script is finished. The detailed results could be found in files that are stored in the ./log. And the tuned model will be stored in model/amazon2caltech/
