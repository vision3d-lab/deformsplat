#!/bin/bash
SECONDS=0
set -e        # exit when error
# set -o xtrace # print command

object_name=$1
frame_index=$2


# Diva360
data_folder=/data/rvi/dataset
origin_folder=${data_folder}/diva360/${object_name}/frames_1
processed_folder=data/diva360_processed/${object_name}_${frame_index}

# link images
mkdir -p ${processed_folder}/images

cam_list=`ls ${origin_folder}`

for cam_folder in ${cam_list}; do
    filename=${origin_folder}/${cam_folder}/0000${frame_index}.png
    ln -s ${filename} ${processed_folder}/images/${cam_folder}.png
done

if [ ${object_name} == "hour_glass" ]; then
    python gesi/rgb_zero_where_alpha_zero.py --folder_path ${processed_folder}/images/
fi
# create camera meta json
train_json_path=${data_folder}/diva360/${object_name}/transforms_train.json
val_json_path=${data_folder}/diva360/${object_name}/transforms_val.json
merged_json_path=${processed_folder}/cameras.json

jq -s '{
  frames: (.[0].frames + .[1].frames),
  aabb_scale: .[0].aabb_scale
}' ${train_json_path} ${val_json_path} > ${merged_json_path}
sed -i 's|undist/||g; s|/0000[0-9]\{4\}||g' ${merged_json_path}