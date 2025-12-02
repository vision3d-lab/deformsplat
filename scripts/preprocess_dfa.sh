# bash scripts/preprocess_dfa.sh /data/wlsgur4011/DFA /data/wlsgur4011/GESI/DFA_processed/

#!/bin/bash
set -e        # exit when error
# set -o xtrace # print command


if [ "$#" -ne 2 ]; then
    echo "Usage: $0 DATA_DIR OUTPUT_DIR"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"

# 데이터 디렉토리 순회
for datadir in "$DATA_DIR"/*; do
    # if datadir is not panda continue
    if [ "$(basename "$datadir")" != "beagle_dog" ]; then
        continue
    fi

    if [ -d "$datadir" ]; then
        data_name=$(basename "$datadir")
        # img 폴더 아래의 모든 subfolder (예: s1) 순회
        for subfolder in "$datadir"/img/*; do
            if [ "$(basename "$subfolder")" != "s1_24fps" ]; then
                continue
            fi

            if [ -d "$subfolder" ]; then
                subfolder_name=$(basename "$subfolder")
                # subfolder 내의 각 frame 폴더 순회
                for frame_dir in "$subfolder"/*; do
                    frame=$(basename "$frame_dir")
                    # frame 이름이 숫자인지 확인
                    if ! [[ "$frame" =~ ^[0-9]+$ ]]; then
                        continue
                    fi
                    # 5 frame 간격으로 처리 (예: frame % 10 == 0 인 경우만)
                    if (( frame % 5 != 0 )); then
                        continue
                    fi

                    # 출력 경로 생성: OUTPUT_DIR/data_name/subfolder/frame
                    dest_seq_dir="$OUTPUT_DIR/${data_name}\(${subfolder_name}\)/$frame"
                    mkdir -p "$dest_seq_dir/images"

                    # frame 폴더 내의 이미지 파일 처리
                    for img_file in "$frame_dir"/img_*.png; do
                        if [[ "$img_file" == *_alpha.png ]]; then
                            continue
                        fi
                        base=$(basename "$img_file")
                        base_no_ext="${base%.png}"
                        alpha_file="$frame_dir/${base_no_ext}_alpha.png"
                        if [ -f "$alpha_file" ]; then
                            out_file="$dest_seq_dir/images/${base_no_ext}_rgba.png"
                            convert "$img_file" "$alpha_file" -alpha off -compose CopyOpacity -composite "$out_file"
                        else
                            echo "Warning: $alpha_file not found for $img_file"
                        fi
                    done

                    # Intrinsic.inf와 CamPose.inf 복사 (CamPose.inf는 Campose.inf로 이름 변경)
                    intrinsic_src="$datadir/Intrinsic.inf"
                    campose_src="$datadir/CamPose.inf"
                    if [ -f "$intrinsic_src" ]; then
                        cp "$intrinsic_src" "$dest_seq_dir/"
                    else
                        echo "Warning: $intrinsic_src not found."
                    fi
                    if [ -f "$campose_src" ]; then
                        cp "$campose_src" "$dest_seq_dir/Campose.inf"
                    else
                        echo "Warning: $campose_src not found."
                    fi
                done
            fi
        done
    fi
done