#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
index_from=$3
index_to=$4
cam_idx=$5
version=$6

data_dir=./data/diva360_processed/${object_name}_${index_to}/
if [ ! -d $data_dir ]; then
    bash scripts/preprocess_diva360.sh $object_name $index_to | true
fi

ckpt=./results/diva360/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt
if [ ! -f $ckpt ]; then
    bash scripts/train_diva360.sh $GPU $object_name $index_from
fi

CUDA_VISIBLE_DEVICES=$GPU python examples/deformsplat_trainer.py default \
    --data_dir $data_dir \
    --result_dir ./results/diva360_finetune/${object_name}_${index_from}_${index_to} \
    --ckpt ./results/diva360/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt \
    --data_factor 1 \
    --data_name diva360 \
    --single_finetune \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name ${object_name}_[$index_from,$index_to] \
    --wandb \
    --wandb_group ${version} \
    --cam_idx $cam_idx \
    --disable_viewer \
    --render_traj_simple \
    --video_path ./output_video/${object_name}.mp4

