import os
import subprocess
import pandas
from pathlib import Path
from typing import Sequence
import scipy.io

VIDEO_DIR_PATH = Path(
    "/mnt/media/Dev/KLTN/Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset"
)
VIDEO_FILE_NAME_FORMAT = "{video_name}_crop.mp4"
FRAME_RATE = 30
LABEL_DIR_PATH = Path("/mnt/media/Dev/KLTN/Labels_MERL_Shopping_Dataset")

OUTPUT_VIDEO_DIR_PATH = Path("/mnt/media/Dev/KLTN/MERL-Dataset")
OUTPUT_VIDEO_EXT = "mp4"
ONLY_OUTPUT_LABEL = False
OUTPUT_LABEL_PATH = Path("/mnt/media/Dev/KLTN/MERL-Dataset/labels.csv")


def get_time_from_frame_number(frame_number: int) -> float:
    frame_number = int(frame_number)
    return round(frame_number / FRAME_RATE, 2)


def create_clips(
    video_name: str,
    start_end_frames: Sequence[Sequence[int]],
    label: str,
) -> list[str]:
    video_path = VIDEO_DIR_PATH / VIDEO_FILE_NAME_FORMAT.format(video_name=video_name)
    output_video_names: list[str] = []

    for index, start_end_frame in enumerate(start_end_frames):
        print(f"Processing clip {start_end_frame} of {video_name}.")

        start_time = get_time_from_frame_number(start_end_frame[0])
        end_time = get_time_from_frame_number(start_end_frame[1])
        if start_time == end_time:
            print(
                f"Clip {start_end_frame} has the same start and end frame in {video_name}. Skipping."
            )
            continue

        output_video_name = f"{video_name}_{label}_{index}"
        output_video_names.append(output_video_name)

        if ONLY_OUTPUT_LABEL:
            continue

        output_video_path = (
            OUTPUT_VIDEO_DIR_PATH / f"{output_video_name}.{OUTPUT_VIDEO_EXT}"
        )

        with open(os.devnull, "wb") as dev_null:
            try:
                subprocess.check_call(
                    [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        str(start_time),
                        "-to",
                        str(end_time),
                        "-i",
                        video_path,
                        output_video_path,
                    ],
                    stdout=dev_null,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as err:
                print(f"Failed to edit {start_end_frame} from {video_name}: ", err)

        print(f"Saved clip {start_end_frame} as {output_video_name} from {video_name}.")

    return output_video_names


labels = {
    "video_name": [],
    "class": [],
    "label": [],
}

for path in LABEL_DIR_PATH.glob("*.mat"):
    video_name = path.stem.replace("_label", "")
    print(f"Process video {video_name}")

    mat = scipy.io.loadmat(path)
    mat = mat["tlabs"]

    pick_frames = mat[0][0]
    output_video_names = create_clips(video_name, pick_frames, "0_pick")
    for name in output_video_names:
        labels["video_name"].append(name)
        labels["class"].append(0)
        labels["label"].append("pick")

    return_frames = mat[1][0]
    output_video_names = create_clips(video_name, return_frames, "1_return")
    for name in output_video_names:
        labels["video_name"].append(name)
        labels["class"].append(1)
        labels["label"].append("return")

pandas.DataFrame(labels).to_csv(OUTPUT_LABEL_PATH)
print(f"Saved label to {OUTPUT_LABEL_PATH}.")
