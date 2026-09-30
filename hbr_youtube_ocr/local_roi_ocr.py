from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "HBR text mask"

# HBR dialogue UI is almost fixed vertically.
# First detect whether the dark bottom dialogue panel exists from the
# grayscale brightness drop across a wide horizontal band.
PANEL_TOP_SEARCH_RATIO = (0.56, 0.80)
NAME_X_RANGE_RATIO = (0.02, 0.28)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect HBR name/dialogue text from grayscale only, "
            "inside fixed UI search zones."
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


def ratio_box(frame, ratio):
    height, width = frame.shape[:2]
    x1r, y1r, x2r, y2r = ratio

    x1 = int(round(width * x1r))
    y1 = int(round(height * y1r))
    x2 = int(round(width * x2r))
    y2 = int(round(height * y2r))

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


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
    return frame[y : y + h, x : x + w]


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


def make_grayscale_text_mask(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    # Keep only very bright glyph-like pixels.
    _, mask = cv2.threshold(
        gray,
        185,
        255,
        cv2.THRESH_BINARY,
    )

    return gray, mask


def detect_text_box_in_zone(
    frame,
    mask,
    search_box,
    min_width_ratio,
    max_width_ratio,
    min_height_ratio,
    max_height_ratio,
    horizontal_close_ratio,
    vertical_gap_ratio,
):
    height, width = frame.shape[:2]

    sx, sy, sw, sh = search_box
    region = mask[sy : sy + sh, sx : sx + sw]

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(
                3,
                int(round(width * horizontal_close_ratio)),
            ),
            3,
        ),
    )

    connected = cv2.morphologyEx(
        region,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    lines = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        wr = w / width
        hr = h / height

        if wr < min_width_ratio or wr > max_width_ratio:
            continue

        if hr < min_height_ratio or hr > max_height_ratio:
            continue

        lines.append(
            (
                sx + x,
                sy + y,
                w,
                h,
            )
        )

    if not lines:
        return None

    lines.sort(
        key=lambda box: box[1]
    )

    groups = []
    current = [lines[0]]

    max_gap = int(round(height * vertical_gap_ratio))

    for box in lines[1:]:
        previous = current[-1]
        previous_bottom = previous[1] + previous[3]
        gap = box[1] - previous_bottom

        if gap <= max_gap:
            current.append(box)
        else:
            groups.append(current)
            current = [box]

    groups.append(current)

    best_group = max(
        groups,
        key=lambda group: sum(
            box[2] * box[3]
            for box in group
        ),
    )

    x1 = min(box[0] for box in best_group)
    y1 = min(box[1] for box in best_group)
    x2 = max(box[0] + box[2] for box in best_group)
    y2 = max(box[1] + box[3] for box in best_group)

    pad_x = int(round(width * 0.012))
    pad_y = int(round(height * 0.010))

    x1 = max(sx, x1 - pad_x)
    y1 = max(sy, y1 - pad_y)
    x2 = min(sx + sw, x2 + pad_x)
    y2 = min(sy + sh, y2 + pad_y)

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def detect_dialogue_panel_top(gray):
    height, width = gray.shape[:2]

    y_start = int(round(height * PANEL_TOP_SEARCH_RATIO[0]))
    y_end = int(round(height * PANEL_TOP_SEARCH_RATIO[1]))

    x1 = int(round(width * 0.08))
    x2 = int(round(width * 0.92))

    band = gray[
        y_start:y_end,
        x1:x2,
    ].astype(np.float32)

    radius = max(
        4,
        int(round(height * 0.006)),
    )

    best_y = None
    best_score = -1.0
    best_coverage = 0.0
    best_drop = 0.0

    for local_y in range(
        radius,
        band.shape[0] - radius,
    ):
        above = band[
            local_y - radius : local_y,
            :
        ].mean(axis=0)

        below = band[
            local_y : local_y + radius,
            :
        ].mean(axis=0)

        drop = above - below

        positive = drop > 0.0
        mean_drop = float(
            drop[positive].mean()
            if np.any(positive)
            else 0.0
        )

        coverage = float(
            np.mean(
                drop >= 8.0
            )
        )

        score = (
            mean_drop
            * coverage
        )

        if score > best_score:
            best_score = score
            best_y = y_start + local_y
            best_coverage = coverage
            best_drop = mean_drop

    # A real dialogue panel causes a broad horizontal darkening.
    if (
        best_y is None
        or best_coverage < 0.45
        or best_drop < 9.0
        or best_score < 4.5
    ):
        return (
            None,
            best_score,
            best_coverage,
            best_drop,
        )

    return (
        best_y,
        best_score,
        best_coverage,
        best_drop,
    )


def detect_name_box(
    frame,
    mask,
    panel_top,
):
    height, width = frame.shape[:2]

    x1 = int(round(width * NAME_X_RANGE_RATIO[0]))
    x2 = int(round(width * NAME_X_RANGE_RATIO[1]))

    name_height = int(round(height * 0.115))
    y1 = max(
        0,
        panel_top - name_height,
    )
    y2 = panel_top

    search_box = (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )

    name_box = detect_text_box_in_zone(
        frame,
        mask,
        search_box,
        min_width_ratio=0.020,
        max_width_ratio=0.20,
        min_height_ratio=0.018,
        max_height_ratio=0.070,
        horizontal_close_ratio=0.008,
        vertical_gap_ratio=0.020,
    )

    if name_box is None:
        return None

    nx, ny, nw, nh = name_box

    # Hard constraints: name must stay completely above the dialogue panel
    # and remain a compact left-side box.
    if ny + nh > panel_top:
        return None

    if nx + nw > x2:
        return None

    if nw > int(round(width * 0.24)):
        return None

    return name_box


def detect_rois(frame):
    gray, mask = make_grayscale_text_mask(
        frame
    )

    (
        panel_top,
        panel_score,
        panel_coverage,
        panel_drop,
    ) = detect_dialogue_panel_top(
        gray
    )

    # No dark bottom panel => no dialogue UI at all.
    if panel_top is None:
        return (
            gray,
            mask,
            None,
            None,
            panel_score,
            panel_coverage,
            panel_drop,
        )

    height, width = frame.shape[:2]

    # Dialogue box is the entire bottom panel, from detected top edge
    # all the way to the bottom of the frame.
    dialogue_box = (
        0,
        panel_top,
        width,
        height - panel_top,
    )

    name_box = detect_name_box(
        frame,
        mask,
        panel_top,
    )

    # In this UI the name box and dialogue box appear together.
    if name_box is None:
        dialogue_box = None

    return (
        gray,
        mask,
        name_box,
        dialogue_box,
        panel_score,
        panel_coverage,
        panel_drop,
    )

def prepare_ocr_image(crop_bgr):
    gray = cv2.cvtColor(
        crop_bgr,
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
    print("Single-window mode: only 'HBR text mask' will be created.")

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

    cv2.destroyAllWindows()

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
            gray,
            mask,
            name_box,
            dialogue_box,
            panel_score,
            panel_coverage,
            panel_drop,
        ) = detect_rois(
            frame
        )

        # Show ONE window only: grayscale/mask composite.
        debug = cv2.cvtColor(
            mask,
            cv2.COLOR_GRAY2BGR,
        )

        display, scale = resize_for_display(
            debug,
            args.width,
        )

        draw_box(
            display,
            name_box,
            scale,
            "name-box",
            (0, 255, 0),
        )

        draw_box(
            display,
            dialogue_box,
            scale,
            "dialogue-box",
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

        cv2.putText(
            display,
            (
                f"{current_sec:.2f}/{duration_sec:.2f}s  "
                f"{'PAUSE' if paused else 'PLAY'}  "
                f"panel={panel_score:.1f} cov={panel_coverage:.2f} drop={panel_drop:.1f}"
            ),
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

        if key == ord("o"):
            paused = True

            if (
                name_box is None
                or dialogue_box is None
            ):
                print(
                    "No valid name/dialogue pair on this grayscale mask."
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
                name_box,
            )

            dialogue_crop = crop_from_box(
                frame,
                dialogue_box,
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
                "name box      :",
                name_box,
            )
            print(
                "dialogue box  :",
                dialogue_box,
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
