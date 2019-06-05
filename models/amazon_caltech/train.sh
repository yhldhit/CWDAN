#!/bin/bash
cd $data_dir
if [ ! -e traina2c.bak ]
then
  echo "true"
  cp -f traina2c.txt traina2c.bak
else
  echo "false"
  cp -f traina2c.bak traina2c.txt
fi

cd ../../
./build/tools/caffe train -solver=models/amazon_caltech/solver.prototxt -weights=models/bvlc_reference_caffenet.caffemodel
#-weights=models/icml/amazon_to_dslr/bvlc_reference_caffenet.caffemodel

