import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import argparse
import datetime


def get_bbox(alpha):
    """
    주어진 alpha 채널(2D numpy array)에서 non-zero 영역의 bounding box를 구함.
    반환값: (left, top, right, bottom)
    (right, bottom)은 slicing 시 exclusive index가 되도록 +1 함.
    """
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    return (left, top, right, bottom)


def compute_union_bbox(image_paths):
    """
    image_paths에 있는 모든 이미지의 alpha 채널로부터 union bounding box를 계산.
    """
    union_bbox = None
    for path in image_paths:
        # 알파 채널까지 읽기
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] < 4:
            continue
        alpha = img[..., 3]
        bbox = get_bbox(alpha)
        if bbox is None:
            continue
        if union_bbox is None:
            union_bbox = bbox
        else:
            left, top, right, bottom = union_bbox
            l, t, r, b = bbox
            union_bbox = (min(left, l), min(top, t), max(right, r), max(bottom, b))
    return union_bbox


def create_integrated_video_with_crop(
    image_paths,
    video_paths,
    output_path,
    image_height_ratio=1 / 3,
    timebar_height=10,
    progress_color=(100, 150, 230),
    marker_color=(230, 180, 50),
):
    """
    알파 채널을 이용해 이미지 n장의 union_bbox를 계산한 후,
    해당 bounding box를 이미지와 영상 프레임에 적용하여 crop하고,
    통합 비디오를 만드는 함수.

    파라미터:
    - image_paths: 이미지 파일 경로 리스트 (n+1개)
    - video_paths: 비디오 파일 경로 리스트 (n개)
    - output_path: 최종 통합 비디오 저장 경로
    - image_height_ratio: 전체 높이에서 이미지 영역이 차지하는 비율
    - timebar_height: 시간 바 영역의 높이 (픽셀)

    변경사항:
    1. 시간 바 진행 바가 누적 진행률을 반영하여 첫 번째 이미지 아래 marker에서 시작하도록 수정.
    2. 최종 비디오 프레임에 상하좌우 10픽셀의 검은색 패딩을 추가 (출력 해상도는 원본보다 20픽셀씩 커짐).
    """
    if len(image_paths) != len(video_paths) + 1:
        raise ValueError(
            f"Number of images ({len(image_paths)}) must be one more than number of videos ({len(video_paths)})"
        )

    # 1. 모든 이미지에서 union_bbox 계산
    union_bbox = compute_union_bbox(image_paths)

    # 2. 첫번째 비디오에서 샘플 프레임 추출 (이미지와 비디오의 해상도가 같다고 가정)
    cap_temp = cv2.VideoCapture(video_paths[0])
    ret, sample_frame = cap_temp.read()
    cap_temp.release()
    if not ret:
        raise ValueError("Could not read frame from video.")
    if union_bbox is not None:
        left, top, right, bottom = union_bbox
        sample_frame = sample_frame[top:bottom, left:right]
    frame_height, frame_width = sample_frame.shape[:2]

    # 최종 비디오에 10픽셀 패딩을 추가하므로 최종 해상도
    padded_width = frame_width + 20
    padded_height = frame_height + 20

    # 3. 통합 비디오의 각 영역 크기 계산
    image_section_height = int(frame_height * image_height_ratio)
    video_section_height = frame_height - image_section_height - timebar_height
    num_images = len(image_paths)
    image_slot_width = (
        frame_width // num_images
    )  # 상단 이미지 영역에서 각 이미지가 들어갈 슬롯의 너비

    # 4. 이미지 처리: load -> union_bbox crop -> (알파 채널 제거) -> 리사이즈 -> 캔버스에 중앙 배치
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image {img_path}")
        # union_bbox 적용 (해당 영역만 사용)
        if union_bbox is not None:
            left, top, right, bottom = union_bbox
            img = img[top:bottom, left:right]
        # 알파 채널이 있다면 BGR만 사용
        if img.shape[2] == 4:
            img = img[..., :3]
        img_h, img_w = img.shape[:2]
        img_ratio = img_w / img_h

        # 이미지 슬롯에 맞게 크기 조정 (aspect ratio 유지)
        new_h = image_section_height
        new_w = int(new_h * img_ratio)
        if new_w > image_slot_width:
            new_w = image_slot_width
            new_h = int(new_w / img_ratio)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 검정색 캔버스에 중앙 배치
        img_canvas = np.zeros(
            (image_section_height, image_slot_width, 3), dtype=np.uint8
        )
        y_offset = (image_section_height - new_h) // 2
        x_offset = (image_slot_width - new_w) // 2
        img_canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = (
            img_resized
        )
        images.append(img_canvas)

    # 상단 이미지를 좌우로 연결
    top_row = np.hstack(images)
    # top_row의 너비가 frame_width와 다르면 맞춰줌 (padding 또는 crop)
    if top_row.shape[1] < frame_width:
        pad_width = frame_width - top_row.shape[1]
        top_row = cv2.copyMakeBorder(
            top_row, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    elif top_row.shape[1] > frame_width:
        top_row = top_row[:, :frame_width]

    # 각 이미지 슬롯의 중심 좌표 계산 (time bar 표시용)
    marker_positions = [int((i + 0.5) * image_slot_width) for i in range(num_images)]

    # 5. 비디오 처리: MoviePy로 비디오 클립 로드 및 이어붙이기
    video_clips = [VideoFileClip(video_path) for video_path in video_paths]
    concatenated_clip = concatenate_videoclips(video_clips)
    total_duration = concatenated_clip.duration

    # 6. 출력 비디오 writer 생성 (패딩 적용된 해상도 사용)
    fps = cv2.VideoCapture(video_paths[0]).get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (padded_width, padded_height))

    # 7. 이어붙인 비디오 클립을 임시 파일로 저장
    now = str(datetime.datetime.now())
    temp_video_path = f"{now}.mp4"
    concatenated_clip.write_videofile(temp_video_path, codec="libx264")

    cap = cv2.VideoCapture(temp_video_path)
    frame_count = 0

    # 8. 각 프레임 처리
    while True:
        ret, video_frame = cap.read()
        if not ret:
            break

        # 영상 프레임에 union_bbox 적용
        if union_bbox is not None:
            left, top, right, bottom = union_bbox
            video_frame = video_frame[top:bottom, left:right]

        # 영상 영역 (아래쪽)의 크기를 유지하면서 aspect ratio 유지하여 리사이즈
        video_h, video_w = video_frame.shape[:2]
        video_ratio = video_w / video_h
        target_ratio = frame_width / video_section_height
        if video_ratio > target_ratio:
            new_video_w = frame_width
            new_video_h = int(new_video_w / video_ratio)
        else:
            new_video_h = video_section_height
            new_video_w = int(new_video_h * video_ratio)
        video_frame_resized = cv2.resize(
            video_frame, (new_video_w, new_video_h), interpolation=cv2.INTER_AREA
        )

        # 검정색 캔버스에 영상 프레임 중앙 배치
        video_canvas = np.zeros((video_section_height, frame_width, 3), dtype=np.uint8)
        y_offset = (video_section_height - new_video_h) // 2
        x_offset = (frame_width - new_video_w) // 2
        video_canvas[
            y_offset : y_offset + new_video_h, x_offset : x_offset + new_video_w
        ] = video_frame_resized

        # 시간 바 생성 (전체 영상에 대한 누적 진행률 반영)
        time_bar = np.zeros((timebar_height, frame_width, 3), dtype=np.uint8)
        current_time = frame_count / fps
        progress_fraction = current_time / total_duration
        progress_fraction = min(max(progress_fraction, 0), 1)  # 0~1 범위 클램핑
        # 진행 바가 첫 번째 이미지 marker부터 마지막 이미지 marker까지 확장되도록 계산
        progress_pixel = int(
            marker_positions[0]
            + progress_fraction * (marker_positions[-1] - marker_positions[0])
        )

        # 파란색 진행 바 그리기 (첫 번째 marker부터 진행 중 위치까지)
        cv2.line(
            time_bar,
            (marker_positions[0], timebar_height // 2),
            (progress_pixel, timebar_height // 2),
            progress_color,
            2,
        )
        # 각 이미지의 marker 표시 (노란색 원)
        for pos in marker_positions:
            cv2.circle(
                time_bar,
                (int(pos), timebar_height // 2),
                3,
                marker_color,
                -1,
                lineType=cv2.LINE_AA,
            )

        # 상단 이미지, 시간 바, 영상 영역을 수직으로 결합
        combined_frame = np.vstack([top_row, time_bar, video_canvas])
        # 10픽셀의 검은색 패딩 추가 (상하좌우)
        padded_frame = cv2.copyMakeBorder(
            combined_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        out.write(padded_frame)
        frame_count += 1

    cap.release()
    out.release()
    os.remove(temp_video_path)
    print(f"Integrated video saved to {output_path}")

    return output_path


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', nargs='+', type=str, help='First list of integers')
    parser.add_argument('--video_paths', nargs='+', type=str, help='Second list of integers')
    parser.add_argument('--output_path', type=str, help='Second list of integers')
    
    args = parser.parse_args()
    
    create_integrated_video_with_crop(
        args.image_paths,
        args.video_paths,
        args.output_path,
    )
