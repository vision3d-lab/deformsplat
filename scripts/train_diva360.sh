# bash scripts/train_diva360.sh 0 penguin 0239
# bash scripts/train_diva360.sh 0 wall_e 0222
#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
frame_index=$3
# 0239 0217

data_dir=./data/diva360_processed/${object_name}_${frame_index}/
if [ ! -d $data_dir ]; then
    bash scripts/preprocess_diva360.sh $object_name $frame_index | true
fi

CUDA_VISIBLE_DEVICES=$GPU python simple_trainer.py default \
    --data_dir $data_dir \
    --result_dir ./results/diva360/${object_name}_${frame_index} \
    --data_factor 1 \
    --data_name diva360 \
    --port 8081 \
    --scale_reg 0.1 \
    --random_bkgd \
    --disable_viewer \
    