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

ckpt=./results/dfa/${object_name}_${index_from}/ckpts/ckpt_best_psnr.pt
if [ ! -f $ckpt ]; then
    bash scripts/train_dfa.sh $GPU $object_name $index_from
fi

CUDA_VISIBLE_DEVICES=$GPU python deform_splat.py default \
    --data_dir data/DFA_processed/${object_name}/${index_to} \
    --result_dir ./results/dfa/${object_name}_finetune \
    --ckpt $ckpt \
    --data_factor 1 \
    --data_name DFA \
    --single_finetune \
    --cam_idx $cam_idx \
    --port 8081 \
    --scale_reg 0.1 \
    --object_name ${object_name}_[$index_from,$index_to]_$cam_idx \
    --wandb \
    --wandb_group ${version} \
    --disable_viewer \
    --render_traj_simple \
    --video_path /data2/wlsgur4011/GESI/output_comparison/dfa_ours/${object_name}.mp4