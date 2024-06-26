from fractions import Fraction
from pathlib import Path
from typing import Sequence

import ffmpeg
import scipy.io

DATASET_PATH = Path("/mnt/media/Dev/jwo-cv-data/merl-shopping-dataset-original")

LABEL_DIR_PATH = DATASET_PATH / "Labels"
VIDEO_DIR_PATH = DATASET_PATH / "Videos"
VIDEO_FILE_NAME_FORMAT = "{video_name}_crop.mp4"
FRAME_RATE = 30

OUTPUT_DATASET_PATH = Path("/mnt/media/Dev/jwo-cv-data/merl-shopping-dataset")
OUTPUT_CLIP_DIR_PATH = OUTPUT_DATASET_PATH
OUTPUT_CLIP_EXT = ".mp4"


def get_secs_from_frame_nums(
    video_path: Path, start_frame_num: any, end_frame_num: any
) -> float:
    result = ffmpeg.probe(str(video_path))
    fps = int(Fraction(result["streams"][0]["avg_frame_rate"]))
    start_secs = int(start_frame_num) / fps
    end_secs = int(end_frame_num) / fps
    return start_secs, end_secs


def extract_clips_from_video(
    video_name: str,
    start_end_frames: Sequence[Sequence[int]],
    dir_name: str,
):
    (OUTPUT_CLIP_DIR_PATH / dir_name).mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR_PATH / VIDEO_FILE_NAME_FORMAT.format(video_name=video_name)

    for index, start_end_frame in enumerate(start_end_frames):
        print(f"Extracting clip {start_end_frame} of {video_name}.")

        try:
            start_secs, end_secs = get_secs_from_frame_nums(
                video_path, start_end_frame[0], start_end_frame[1]
            )
        except ffmpeg.Error as err:
            print(f"Failed to read {video_path}: {err.stderr}")
            continue

        if start_secs == end_secs:
            print(
                f"Clip {start_end_frame} has the same start and end frame in {video_name}. Skipping."
            )
            continue

        clip_name = f"{video_name}_{index}"
        clip_output_path = OUTPUT_CLIP_DIR_PATH / dir_name / clip_name
        clip_output_path = clip_output_path.with_suffix(OUTPUT_CLIP_EXT)

        try:
            (
                ffmpeg.input(video_path, ss=start_secs, to=end_secs)
                .output(str(clip_output_path))
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as err:
            print(
                f"Failed to extract {start_end_frame} from {video_name}: ", err.stderr
            )

        print(
            f"Saved clip {start_end_frame} as {dir_name}/{clip_name} from {video_name}."
        )


for path in LABEL_DIR_PATH.glob("*.mat"):
    video_name = path.stem.replace("_label", "")
    print(f"Process video {video_name}")

    mat = scipy.io.loadmat(path)
    mat = mat["tlabs"]

    print(f"Extracting 0_pick clips from {video_name}.")
    pick_frames = mat[1][0]
    extract_clips_from_video(video_name, pick_frames, "0_pick")

    print(f"Extracting 1_return clips from {video_name}.")
    return_frames = mat[0][0]
    extract_clips_from_video(video_name, return_frames, "1_return")
