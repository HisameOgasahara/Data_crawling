from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "HBR ROI OCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect the HBR bottom dialogue panel first, then find white speaker/dialogue "
            "text only inside constrained regions."
        )
    )
    parser.add_argument("video", type=Path, help="Video file path")
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Preview width. Aspect ratio is preserved. Default: 1280",
    )
    parser.add_argument(
        "--names-file",
        type=Path,
        default=Path(__file__).with_name("character_names.txt"),
        help="TXT/CSV with known speaker names.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start playback immediately.",
    )
    return parser.parse_args()


def resize_for_display(frame, target_width: int):
    height, width = frame.shape[:2]

    if width <= target_width:
        return frame.copy(), 1.0

    scale = target_width / width
    display_height = int(round(height * scale))

    display = cv2.resize(
        frame,
        (target_width, display_height),
        interpolation=cv2.INTER_AREA,
    )

    return display, scale


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    return re.sub(r"\s+", "", text).strip()


def load_name_list(path: Path | None) -> list[str]:
    if path is None:
        return []

    if not path.exists():
        raise FileNotFoundError(path)

    names: list[str] = []
    suffix = path.suffix.lower()

    if suffix in {".txt", ".list"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if line:
                names.append(line)

    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)

            for row in reader:
                if row and row[-1].strip():
                    value = row[-1].strip()

                    if value.lower() != "name":
                        names.append(value)

    else:
        raise ValueError("names file must be .txt or .csv")

    return names


def match_known_name(
    raw_name: str,
    candidates: list[str],
) -> tuple[str, float]:
    raw_norm = normalize_text(raw_name)

    if not raw_norm or not candidates:
        return raw_name, 0.0

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

        if raw_norm in candidate_norm or candidate_norm in raw_norm:
            score = max(score, 0.85)

        if score > best_score:
            best_score = score
            best_name = candidate

    if best_score >= 0.55:
        return best_name, best_score

    return raw_name, best_score


def crop_from_box(frame, box):
    if box is None:
        return None

    x, y, w, h = box

    return frame[
        y : y + h,
        x : x + w,
    ]


def draw_box(
    display,
    source_box,
    scale,
    label,
    color,
):
    if source_box is None:
        return

    x, y, w, h = source_box

    x1 = int(round(x * scale))
    y1 = int(round(y * scale))
    x2 = int(round((x + w) * scale))
    y2 = int(round((y + h) * scale))

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        display,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def clamp_box(frame, box):
    if box is None:
        return None

    height, width = frame.shape[:2]
    x, y, w, h = box

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def expand_box(
    frame,
    box,
    pad_x,
    pad_y,
):
    if box is None:
        return None

    x, y, w, h = box

    return clamp_box(
        frame,
        (
            x - pad_x,
            y - pad_y,
            w + 2 * pad_x,
            h + 2 * pad_y,
        ),
    )


def union_boxes(boxes):
    if not boxes:
        return None

    x1 = min(x for x, y, w, h in boxes)
    y1 = min(y for x, y, w, h in boxes)
    x2 = max(x + w for x, y, w, h in boxes)
    y2 = max(y + h for x, y, w, h in boxes)

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def detect_dialogue_panel(frame):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        0,
    )

    x1 = int(round(width * 0.02))
    x2 = int(round(width * 0.98))

    y_start = int(round(height * 0.52))
    y_end = int(round(height * 0.82))

    band = gray[
        y_start:y_end,
        x1:x2,
    ].astype(np.float32)

    vertical_radius = max(
        4,
        int(round(height * 0.006)),
    )

    best_y = None
    best_score = -1.0

    for local_y in range(
        vertical_radius,
        band.shape[0] - vertical_radius,
    ):
        above = band[
            local_y - vertical_radius : local_y,
            :
        ].mean(axis=0)

        below = band[
            local_y : local_y + vertical_radius,
            :
        ].mean(axis=0)

        dark_drop = above - below

        broad_drop = np.clip(
            dark_drop,
            0.0,
            None,
        )

        mean_drop = float(
            np.mean(broad_drop)
        )

        coverage = float(
            np.mean(
                dark_drop > 5.0
            )
        )

        score = (
            mean_drop
            * (0.35 + coverage)
        )

        if score > best_score:
            best_score = score
            best_y = y_start + local_y

    if best_y is None:
        return None, 0.0

    if best_score < 1.8:
        return None, best_score

    panel_top = best_y

    panel_box = (
        0,
        panel_top,
        width,
        height - panel_top,
    )

    return panel_box, best_score


def make_white_text_mask(frame):
    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation <= 80)
        & (value >= 185)
    ).astype(np.uint8) * 255

    return mask


def find_line_boxes(
    frame,
    white_mask,
    search_box,
    min_width_ratio,
    max_width_ratio,
    min_height_ratio,
    max_height_ratio,
):
    height, width = frame.shape[:2]

    sx, sy, sw, sh = search_box

    region = white_mask[
        sy : sy + sh,
        sx : sx + sw,
    ]

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(
                5,
                int(round(width * 0.010)),
            ),
            3,
        ),
    )

    connected = cv2.morphologyEx(
        region,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(
            contour
        )

        width_ratio = w / width
        height_ratio = h / height

        if width_ratio < min_width_ratio:
            continue
        if width_ratio > max_width_ratio:
            continue
        if height_ratio < min_height_ratio:
            continue
        if height_ratio > max_height_ratio:
            continue

        boxes.append(
            (
                sx + x,
                sy + y,
                w,
                h,
            )
        )

    boxes.sort(
        key=lambda box: (
            box[1],
            box[0],
        )
    )

    return boxes


def detect_text_rois(
    frame,
    panel_box,
):
    if panel_box is None:
        return (
            None,
            None,
            None,
        )

    height, width = frame.shape[:2]
    panel_x, panel_y, panel_w, panel_h = panel_box

    white_mask = make_white_text_mask(
        frame
    )

    # There is only ONE dialogue region.
    # Once the bottom dialogue panel is found, the dialogue ROI is a
    # deterministic inner rectangle of that panel instead of being rebuilt
    # from individual text-line contours.
    dialogue_roi = clamp_box(
        frame,
        (
            int(round(width * 0.12)),
            panel_y + int(round(panel_h * 0.06)),
            int(round(width * 0.80)),
            int(round(panel_h * 0.82)),
        ),
    )

    # Speaker name is always immediately ABOVE the dialogue panel
    # and restricted to the left side. It can never become larger than
    # this search region.
    name_search = clamp_box(
        frame,
        (
            int(round(width * 0.025)),
            max(
                0,
                panel_y - int(round(height * 0.115)),
            ),
            int(round(width * 0.22)),
            int(round(height * 0.105)),
        ),
    )

    speaker_roi = None

    if name_search is not None:
        name_lines = find_line_boxes(
            frame,
            white_mask,
            name_search,
            min_width_ratio=0.020,
            max_width_ratio=0.18,
            min_height_ratio=0.018,
            max_height_ratio=0.070,
        )

        if name_lines:
            # Pick one name line only; do not union unrelated white regions.
            best_name_box = max(
                name_lines,
                key=lambda box: box[2] * box[3],
            )

            speaker_roi = expand_box(
                frame,
                best_name_box,
                pad_x=int(round(width * 0.012)),
                pad_y=int(round(height * 0.010)),
            )

            # Hard constraint: speaker ROI must stay inside name_search
            # and strictly above dialogue panel.
            nx, ny, nw, nh = name_search
            sx, sy, sw, sh = speaker_roi

            sx1 = max(sx, nx)
            sy1 = max(sy, ny)
            sx2 = min(sx + sw, nx + nw)
            sy2 = min(
                sy + sh,
                panel_y,
            )

            if sx2 > sx1 and sy2 > sy1:
                speaker_roi = (
                    sx1,
                    sy1,
                    sx2 - sx1,
                    sy2 - sy1,
                )
            else:
                speaker_roi = None

    return (
        speaker_roi,
        dialogue_roi,
        white_mask,
    )

def prepare_ocr_image(crop_bgr):
    gray = cv2.cvtColor(
        crop_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
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


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(
            args.video
        )

    known_names = load_name_list(
        args.names_file
    )

    print(
        f"Loaded {len(known_names)} known names from "
        f"{args.names_file}"
    )

    cap = cv2.VideoCapture(
        str(args.video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration_sec = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    paused = not args.autoplay
    manga_ocr = None

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    ok, frame = cap.read()

    if not ok:
        raise RuntimeError(
            "Could not read the first frame."
        )

    while True:
        if not paused:
            ok, next_frame = cap.read()

            if not ok:
                paused = True
            else:
                frame = next_frame

        (
            dialogue_panel,
            panel_score,
        ) = detect_dialogue_panel(
            frame
        )

        (
            speaker_roi,
            dialogue_roi,
            white_mask,
        ) = detect_text_rois(
            frame,
            dialogue_panel,
        )

        display, scale = resize_for_display(
            frame,
            args.width,
        )

        draw_box(
            display,
            dialogue_panel,
            scale,
            "dialogue-panel",
            (0, 255, 255),
        )

        draw_box(
            display,
            speaker_roi,
            scale,
            "speaker-roi",
            (0, 255, 0),
        )

        draw_box(
            display,
            dialogue_roi,
            scale,
            "dialogue-roi",
            (255, 0, 0),
        )

        current_frame = int(
            cap.get(
                cv2.CAP_PROP_POS_FRAMES
            )
        ) - 1

        current_frame = max(
            current_frame,
            0,
        )

        current_sec = (
            current_frame / fps
            if fps > 0
            else 0.0
        )

        status = (
            f"{current_sec:.2f}/{duration_sec:.2f}s  "
            f"frame {current_frame}/{max(frame_count - 1, 0)}  "
            f"{'PAUSE' if paused else 'PLAY'}  "
            f"panel={panel_score:.2f}"
        )

        cv2.putText(
            display,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        delay = (
            1
            if not paused
            else 30
        )

        key = cv2.waitKey(
            delay
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
                max(frame_count - 1, 0),
            )
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                target,
            )
            ok, frame = cap.read()
            continue

        if key == ord("j"):
            paused = True
            target_sec = max(
                current_sec - 5.0,
                0.0,
            )
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                target_sec * 1000.0,
            )
            ok, frame = cap.read()
            continue

        if key == ord("l"):
            paused = True
            target_sec = min(
                current_sec + 5.0,
                duration_sec,
            )
            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                target_sec * 1000.0,
            )
            ok, frame = cap.read()
            continue

        if key == ord("m"):
            cv2.imshow(
                "white-text-mask",
                white_mask,
            )
            continue

        if key == ord("o"):
            paused = True

            if (
                speaker_roi is None
                or dialogue_roi is None
            ):
                print(
                    "speaker/dialogue ROI was not detected "
                    "inside the dialogue panel."
                )
                continue

            if manga_ocr is None:
                print(
                    "Loading MangaOCR on CPU..."
                )

                from manga_ocr import MangaOcr

                manga_ocr = MangaOcr()

                print(
                    "MangaOCR ready."
                )

            from PIL import Image

            speaker_crop = crop_from_box(
                frame,
                speaker_roi,
            )

            dialogue_crop = crop_from_box(
                frame,
                dialogue_roi,
            )

            speaker_img = prepare_ocr_image(
                speaker_crop
            )

            dialogue_img = prepare_ocr_image(
                dialogue_crop
            )

            speaker_text_raw = manga_ocr(
                Image.fromarray(
                    speaker_img
                ).convert("RGB")
            )

            dialogue_text_raw = manga_ocr(
                Image.fromarray(
                    dialogue_img
                ).convert("RGB")
            )

            speaker_text = normalize_text(
                speaker_text_raw
            )

            dialogue_text = normalize_text(
                dialogue_text_raw
            )

            matched_name, score = match_known_name(
                speaker_text,
                known_names,
            )

            print(
                f"[{current_sec:.2f}s]"
            )
            print(
                "dialogue panel:",
                dialogue_panel,
            )
            print(
                "speaker roi   :",
                speaker_roi,
            )
            print(
                "dialogue roi  :",
                dialogue_roi,
            )
            print(
                "speaker raw   :",
                speaker_text,
            )
            print(
                f"speaker match : {matched_name} "
                f"(score={score:.2f})"
            )
            print(
                "dialogue      :",
                dialogue_text,
            )
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
