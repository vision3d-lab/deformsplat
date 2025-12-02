#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=$1
object_name=$2
frame_index=$3

CUDA_VISIBLE_DEVICES=$GPU python simple_trainer.py default \
    --data_dir ./data/DFA_processed/${object_name}/${frame_index} \
    --result_dir ./results/dfa/${object_name}_${frame_index} \
    --data_factor 1 \
    --data_name DFA \
    --port 8081 \
    --scale_reg 0.1 \
    --disable_viewer \
    