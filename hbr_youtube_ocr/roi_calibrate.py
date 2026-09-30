from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one frame from a video, manually select speaker/dialogue ROIs, "
            "and save normalized coordinates to JSON."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Video file path",
    )

    parser.add_argument(
        "--roi-json",
        type=Path,
        default=Path(__file__).with_name("roi.json"),
        help="ROI JSON path",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Frame timestamp in seconds",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Preview width",
    )

    return parser.parse_args()


def resize_for_display(
    frame,
    target_width: int,
):
    height, width = frame.shape[:2]

    if width <= target_width:
        return frame.copy(), 1.0

    scale = target_width / width

    display = cv2.resize(
        frame,
        (
            target_width,
            int(round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    return display, scale


def display_box_to_normalized(
    box,
    display_scale: float,
    source_width: int,
    source_height: int,
):
    x, y, w, h = box

    source_x = x / display_scale
    source_y = y / display_scale
    source_w = w / display_scale
    source_h = h / display_scale

    return {
        "x": source_x / source_width,
        "y": source_y / source_height,
        "w": source_w / source_width,
        "h": source_h / source_height,
    }


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(
            args.video
        )

    cap = cv2.VideoCapture(
        str(args.video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        max(
            args.time,
            0.0,
        ) * 1000.0,
    )

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(
            "Could not read calibration frame"
        )

    display, scale = resize_for_display(
        frame,
        args.width,
    )

    speaker_box = cv2.selectROI(
        "Select speaker ROI",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select speaker ROI"
    )

    dialogue_box = cv2.selectROI(
        "Select dialogue ROI",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(
        "Select dialogue ROI"
    )

    cv2.destroyAllWindows()

    if (
        speaker_box[2] == 0
        or speaker_box[3] == 0
        or dialogue_box[2] == 0
        or dialogue_box[3] == 0
    ):
        raise RuntimeError(
            "ROI selection cancelled"
        )

    source_height, source_width = frame.shape[:2]

    data = {
        "reference_width": source_width,
        "reference_height": source_height,
        "calibration_time_sec": max(
            args.time,
            0.0,
        ),
        "speaker_roi": display_box_to_normalized(
            speaker_box,
            scale,
            source_width,
            source_height,
        ),
        "dialogue_roi": display_box_to_normalized(
            dialogue_box,
            scale,
            source_width,
            source_height,
        ),
    }

    args.roi_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.roi_json.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Saved ROI JSON: {args.roi_json}"
    )


if __name__ == "__main__":
    main()
