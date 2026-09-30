from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2


WINDOW_NAME = "HBR ROI calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate fixed HBR speaker/dialogue ROIs once, save them to JSON, "
            "then OCR a video by timestamp and group dialogue by character."
        )
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Video file path",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "calibrate",
            "extract",
        ],
        default="calibrate",
        help="calibrate: manually select ROIs and save JSON; extract: OCR using saved JSON",
    )

    parser.add_argument(
        "--roi-json",
        type=Path,
        default=Path(__file__).with_name("roi.json"),
        help="ROI JSON path",
    )

    parser.add_argument(
        "--names-file",
        type=Path,
        default=Path(__file__).with_name("character_names.txt"),
        help="Known character names TXT/CSV",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Calibration preview width",
    )

    parser.add_argument(
        "--time",
        type=float,
        default=0.0,
        help="Initial calibration timestamp in seconds",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="OCR sampling interval in seconds",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hbr_ocr_output"),
        help="Extraction output directory",
    )

    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        text or "",
    ).strip()


def load_name_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    names: list[str] = []

    if path.suffix.lower() == ".csv":
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.reader(file)

            for row in reader:
                if not row:
                    continue

                value = row[-1].strip()

                if value and value.lower() != "name":
                    names.append(value)

    else:
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines():
            line = line.strip()

            if line:
                names.append(line)

    return names


def match_known_name(
    raw_name: str,
    candidates: list[str],
) -> tuple[str, float]:
    raw_norm = normalize_text(raw_name)

    if not raw_norm:
        return "", 0.0

    best_name = raw_name
    best_score = 0.0

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)

        if raw_norm == candidate_norm:
            return candidate, 1.0

        score = SequenceMatcher(
            None,
            raw_norm,
            candidate_norm,
        ).ratio()

        if (
            raw_norm in candidate_norm
            or candidate_norm in raw_norm
        ):
            score = max(
                score,
                0.85,
            )

        if score > best_score:
            best_score = score
            best_name = candidate

    return best_name, best_score


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


def normalized_to_box(
    roi: dict,
    width: int,
    height: int,
):
    return (
        int(round(roi["x"] * width)),
        int(round(roi["y"] * height)),
        int(round(roi["w"] * width)),
        int(round(roi["h"] * height)),
    )


def crop_box(
    frame,
    box,
):
    x, y, w, h = box

    return frame[
        y : y + h,
        x : x + w,
    ]


def prepare_ocr_image(
    crop,
):
    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    _, binary = cv2.threshold(
        gray,
        180,
        255,
        cv2.THRESH_BINARY,
    )

    return cv2.resize(
        binary,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )


def calibrate(
    video_path: Path,
    roi_json_path: Path,
    preview_width: int,
    initial_time: float,
):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )
    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )
    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    current_time = min(
        max(
            initial_time,
            0.0,
        ),
        duration,
    )

    paused = True

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        current_time * 1000.0,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            "Could not read calibration frame"
        )

    cv2.destroyAllWindows()
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    while True:
        if not paused:
            ok, next_frame = cap.read()

            if not ok:
                paused = True
            else:
                frame = next_frame

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        current_frame = max(
            current_frame,
            0,
        )

        current_time = (
            current_frame / fps
            if fps > 0
            else 0.0
        )

        display, scale = resize_for_display(
            frame,
            preview_width,
        )

        cv2.putText(
            display,
            (
                f"{current_time:.2f}/{duration:.2f}s  "
                f"{'PAUSE' if paused else 'PLAY'}"
            ),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            display,
            "Space play/pause | A/D frame | J/L 5 sec | R select ROIs | Q quit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(
            1 if not paused else 30
        ) & 0xFF

        if key == 255:
            continue

        if key == ord("q"):
            break

        if key == ord(" "):
            paused = not paused
            continue

        if key == ord("a"):
            paused = True

            target = max(
                current_frame - 1,
                0,
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                target,
            )

            ok, frame = cap.read()
            continue

        if key == ord("d"):
            paused = True

            target = min(
                current_frame + 1,
                max(
                    frame_count - 1,
                    0,
                ),
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                target,
            )

            ok, frame = cap.read()
            continue

        if key == ord("j"):
            paused = True

            target_time = max(
                current_time - 5.0,
                0.0,
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                target_time * 1000.0,
            )

            ok, frame = cap.read()
            continue

        if key == ord("l"):
            paused = True

            target_time = min(
                current_time + 5.0,
                duration,
            )

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                target_time * 1000.0,
            )

            ok, frame = cap.read()
            continue

        if key == ord("r"):
            paused = True

            selection_frame, selection_scale = resize_for_display(
                frame,
                preview_width,
            )

            speaker_box = cv2.selectROI(
                "Select speaker ROI",
                selection_frame,
                showCrosshair=True,
                fromCenter=False,
            )

            cv2.destroyWindow(
                "Select speaker ROI"
            )

            dialogue_box = cv2.selectROI(
                "Select dialogue ROI",
                selection_frame,
                showCrosshair=True,
                fromCenter=False,
            )

            cv2.destroyWindow(
                "Select dialogue ROI"
            )

            if (
                speaker_box[2] == 0
                or speaker_box[3] == 0
                or dialogue_box[2] == 0
                or dialogue_box[3] == 0
            ):
                print(
                    "ROI selection cancelled."
                )
                continue

            source_height, source_width = frame.shape[:2]

            data = {
                "reference_width": source_width,
                "reference_height": source_height,
                "calibration_time_sec": current_time,
                "speaker_roi": display_box_to_normalized(
                    speaker_box,
                    selection_scale,
                    source_width,
                    source_height,
                ),
                "dialogue_roi": display_box_to_normalized(
                    dialogue_box,
                    selection_scale,
                    source_width,
                    source_height,
                ),
            }

            roi_json_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            roi_json_path.write_text(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"Saved ROI JSON: {roi_json_path}"
            )
            print(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            break

    cap.release()
    cv2.destroyAllWindows()


def similar_dialogue(
    first: str,
    second: str,
) -> bool:
    if not first or not second:
        return False

    if (
        first.startswith(second)
        or second.startswith(first)
    ):
        return True

    return (
        SequenceMatcher(
            None,
            first,
            second,
        ).ratio()
        >= 0.82
    )


def merge_samples(
    rows: list[dict],
    interval: float,
) -> list[dict]:
    segments: list[dict] = []
    current = None

    for row in rows:
        speaker = row["speaker"]
        dialogue = row["dialogue"]
        time_sec = row["time_sec"]

        if not speaker or not dialogue:
            if current is not None:
                current["end_sec"] = time_sec
                segments.append(
                    current
                )
                current = None

            continue

        if current is None:
            current = {
                "start_sec": time_sec,
                "end_sec": time_sec + interval,
                "speaker": speaker,
                "dialogue": dialogue,
            }
            continue

        if (
            speaker == current["speaker"]
            and similar_dialogue(
                dialogue,
                current["dialogue"],
            )
        ):
            current["end_sec"] = (
                time_sec + interval
            )

            if len(dialogue) > len(
                current["dialogue"]
            ):
                current["dialogue"] = dialogue

            continue

        segments.append(
            current
        )

        current = {
            "start_sec": time_sec,
            "end_sec": time_sec + interval,
            "speaker": speaker,
            "dialogue": dialogue,
        }

    if current is not None:
        segments.append(
            current
        )

    return segments


def format_timestamp(
    seconds: float,
) -> str:
    total_ms = int(
        round(
            seconds * 1000
        )
    )

    minutes = total_ms // 60000
    remaining_ms = total_ms % 60000
    whole_seconds = remaining_ms // 1000
    milliseconds = remaining_ms % 1000

    return (
        f"{minutes:02d}:"
        f"{whole_seconds:02d}."
        f"{milliseconds:03d}"
    )


def extract(
    video_path: Path,
    roi_json_path: Path,
    names_file: Path,
    interval: float,
    output_dir: Path,
):
    if not roi_json_path.exists():
        raise FileNotFoundError(
            roi_json_path
        )

    roi_data = json.loads(
        roi_json_path.read_text(
            encoding="utf-8"
        )
    )

    known_names = load_name_list(
        names_file
    )

    from manga_ocr import MangaOcr
    from PIL import Image

    mocr = MangaOcr()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )
    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )
    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )
    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    speaker_box = normalized_to_box(
        roi_data["speaker_roi"],
        width,
        height,
    )

    dialogue_box = normalized_to_box(
        roi_data["dialogue_roi"],
        width,
        height,
    )

    total_samples = int(
        duration / interval
    ) + 1

    rows: list[dict] = []

    for sample_index in range(
        total_samples
    ):
        time_sec = (
            sample_index
            * interval
        )

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            time_sec * 1000.0,
        )

        ok, frame = cap.read()

        if not ok:
            break

        speaker_crop = crop_box(
            frame,
            speaker_box,
        )

        dialogue_crop = crop_box(
            frame,
            dialogue_box,
        )

        speaker_image = prepare_ocr_image(
            speaker_crop
        )

        dialogue_image = prepare_ocr_image(
            dialogue_crop
        )

        speaker_raw = normalize_text(
            mocr(
                Image.fromarray(
                    speaker_image
                ).convert("RGB")
            )
        )

        dialogue = normalize_text(
            mocr(
                Image.fromarray(
                    dialogue_image
                ).convert("RGB")
            )
        )

        speaker, score = match_known_name(
            speaker_raw,
            known_names,
        )

        if score < 0.55:
            speaker = ""

        rows.append(
            {
                "time_sec": round(
                    time_sec,
                    3,
                ),
                "speaker_raw": speaker_raw,
                "speaker": speaker,
                "speaker_score": round(
                    score,
                    4,
                ),
                "dialogue": dialogue,
            }
        )

        print(
            f"[{sample_index + 1}/{total_samples}] "
            f"{time_sec:.1f}/{duration:.1f}s "
            f"speaker={speaker or '-'} "
            f"dialogue={dialogue[:30]}"
        )

    cap.release()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_csv_path = (
        output_dir
        / "ocr_samples.csv"
    )

    with raw_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time_sec",
                "speaker_raw",
                "speaker",
                "speaker_score",
                "dialogue",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    segments = merge_samples(
        rows,
        interval,
    )

    segments_csv_path = (
        output_dir
        / "dialogue_segments.csv"
    )

    with segments_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "start_sec",
                "end_sec",
                "speaker",
                "dialogue",
            ],
        )

        writer.writeheader()
        writer.writerows(
            segments
        )

    by_character: dict[
        str,
        list[dict],
    ] = defaultdict(list)

    for segment in segments:
        by_character[
            segment["speaker"]
        ].append(
            segment
        )

    character_dir = (
        output_dir
        / "characters"
    )

    character_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for speaker, speaker_segments in by_character.items():
        text_path = (
            character_dir
            / f"{speaker}.txt"
        )

        lines: list[str] = []

        for segment in speaker_segments:
            start = format_timestamp(
                segment["start_sec"]
            )
            end = format_timestamp(
                segment["end_sec"]
            )

            lines.append(
                f"[{start} - {end}]"
            )
            lines.append(
                segment["dialogue"]
            )
            lines.append(
                ""
            )

        text_path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

    print(
        f"Saved: {raw_csv_path}"
    )
    print(
        f"Saved: {segments_csv_path}"
    )
    print(
        f"Saved character TXT files: {character_dir}"
    )


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(
            args.video
        )

    if args.mode == "calibrate":
        calibrate(
            video_path=args.video,
            roi_json_path=args.roi_json,
            preview_width=args.width,
            initial_time=args.time,
        )

        return

    extract(
        video_path=args.video,
        roi_json_path=args.roi_json,
        names_file=args.names_file,
        interval=args.interval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
