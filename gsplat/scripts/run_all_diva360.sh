#!/bin/bash

SECONDS=0
set -e        # exit when error
set -o xtrace # print command

wandb_group_name=deformsplat
script_name=scripts/deformsplat_diva360.sh

#                   GPU    object_name    frame_from  frame_to cam_idx  wandb_group_name
bash ${script_name} 0     penguin          0217       0239     00      ${wandb_group_name} &&
bash ${script_name} 0     trex             0135       0250     00      ${wandb_group_name} &&
bash ${script_name} 0     red_car          0042       0250     00      ${wandb_group_name} &&
bash ${script_name} 0     tornado          0000       0456     00      ${wandb_group_name} &&
bash ${script_name} 0     truck            0078       0171     00      ${wandb_group_name} &&
bash ${script_name} 0     stirling         0000       0045     00      ${wandb_group_name} &&
bash ${script_name} 0     blue_car         0142       0214     00      ${wandb_group_name} &&
bash ${script_name} 0     dog              0177       0279     00      ${wandb_group_name} &&
bash ${script_name} 0     music_box        0100       0125     00      ${wandb_group_name} &&
bash ${script_name} 0     clock            0000       1500     00      ${wandb_group_name} &&
bash ${script_name} 0     horse            0120       0375     00      ${wandb_group_name} &&
bash ${script_name} 0     bunny            0000       1000     00      ${wandb_group_name} &&
bash ${script_name} 0     wall_e           0222       0286     00      ${wandb_group_name} &&
bash ${script_name} 0     wolf             0357       1953     00      ${wandb_group_name} &&
bash ${script_name} 0     world_globe      0020       0074     00      ${wandb_group_name} &&
bash ${script_name} 0     hour_glass       0100       0200     00      ${wandb_group_name} &&
bash ${script_name} 0     k1_double_punch  0270       0282     01      ${wandb_group_name} &&
bash ${script_name} 0     k1_hand_stand    0412       0426     01      ${wandb_group_name} &&
bash ${script_name} 0     k1_push_up       0541       0557     01      ${wandb_group_name}


wait