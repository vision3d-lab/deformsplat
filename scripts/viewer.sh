#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

CUDA_VISIBLE_DEVICES=5 python simple_viewer.py \
    --ckpt="./results/diva360/blue_car_0142/ckpts/ckpt_best_psnr.pt" \
    --port 8082
