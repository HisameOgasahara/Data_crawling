from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "HBR text mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect HBR dialogue UI from grayscale only. "
            "One mask window, one name box, one dialogue box."
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


def make_grayscale_mask(frame):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    _, mask = cv2.threshold(
        gray,
        185,
        255,
        cv2.THRESH_BINARY,
    )

    return gray, mask


def smooth_1d(values, kernel_size):
    kernel_size = max(
        1,
        int(kernel_size),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones(
        kernel_size,
        dtype=np.float32,
    ) / kernel_size

    return np.convolve(
        values,
        kernel,
        mode="same",
    )


def detect_panel_top(gray):
    """
    Detect the dialogue-panel top directly from the grayscale binary mask.

    HBR property used here:
    - when the dialogue panel is visible, from some y downward almost the
      entire lower region becomes black in the thresholded mask
    - above that y, normal background still contains substantial white pixels
    - if almost the whole frame is black, treat it as a scene transition,
      not as dialogue UI
    """
    height, width = gray.shape[:2]

    _, mask = cv2.threshold(
        gray,
        185,
        255,
        cv2.THRESH_BINARY,
    )

    frame_white_ratio = float(
        np.count_nonzero(mask)
        / mask.size
    )

    # Nearly all-black frame: likely fade / scene transition.
    if frame_white_ratio < 0.015:
        return None, 0.0

    x1 = int(round(width * 0.04))
    x2 = int(round(width * 0.96))

    y_start = int(round(height * 0.52))
    y_end = int(round(height * 0.82))

    inner = mask[:, x1:x2]

    row_white_ratio = (
        np.count_nonzero(
            inner,
            axis=1,
        )
        / inner.shape[1]
    )

    # Smooth row occupancy a little so glyphs/noise do not create false edges.
    row_white_ratio = smooth_1d(
        row_white_ratio.astype(np.float32),
        max(
            3,
            int(round(height * 0.008)),
        ),
    )

    # For each y, measure how black the whole suffix [y:bottom] is.
    suffix_mean = np.zeros(
        height,
        dtype=np.float32,
    )

    running_sum = 0.0
    running_count = 0

    for y in range(height - 1, -1, -1):
        running_sum += float(
            row_white_ratio[y]
        )
        running_count += 1
        suffix_mean[y] = (
            running_sum
            / running_count
        )

    # A valid panel top must satisfy:
    # 1) region below is almost entirely black
    # 2) region just above still has visible white structure
    suffix_black_threshold = 0.035
    above_white_threshold = 0.080
    above_window = max(
        6,
        int(round(height * 0.030)),
    )

    best_y = None
    best_score = -1.0

    for y in range(
        y_start,
        y_end,
    ):
        if suffix_mean[y] > suffix_black_threshold:
            continue

        above_start = max(
            0,
            y - above_window,
        )

        above_mean = float(
            row_white_ratio[
                above_start:y
            ].mean()
        )

        if above_mean < above_white_threshold:
            continue

        score = (
            above_mean
            - suffix_mean[y]
        )

        if score > best_score:
            best_score = score
            best_y = y

    if best_y is None:
        return None, 0.0

    return best_y, best_score

def active_row_groups(
    mask,
    search_box,
    min_white_pixels,
    max_gap_rows,
):
    sx, sy, sw, sh = search_box

    region = mask[
        sy : sy + sh,
        sx : sx + sw,
    ]

    row_counts = np.count_nonzero(
        region,
        axis=1,
    )

    active = row_counts >= min_white_pixels

    # Close tiny vertical gaps inside a text line.
    active_u8 = active.astype(
        np.uint8
    ) * 255

    kernel = np.ones(
        (
            max(
                1,
                max_gap_rows,
            ),
            1,
        ),
        dtype=np.uint8,
    )

    active_closed = cv2.morphologyEx(
        active_u8.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        kernel,
    ).ravel() > 0

    groups = []
    start = None

    for index, is_active in enumerate(
        active_closed
    ):
        if is_active and start is None:
            start = index

        if (
            not is_active
            and start is not None
        ):
            groups.append(
                (
                    start,
                    index - 1,
                )
            )
            start = None

    if start is not None:
        groups.append(
            (
                start,
                len(active_closed) - 1,
            )
        )

    return groups, row_counts


def text_box_from_group(
    frame,
    mask,
    search_box,
    row_group,
    pad_x_ratio,
    pad_y_ratio,
):
    height, width = frame.shape[:2]
    sx, sy, sw, sh = search_box
    row_start, row_end = row_group

    y1 = sy + row_start
    y2 = sy + row_end + 1

    strip = mask[
        y1:y2,
        sx : sx + sw,
    ]

    ys, xs = np.where(
        strip > 0
    )

    if len(xs) == 0:
        return None

    x1 = sx + int(xs.min())
    x2 = sx + int(xs.max()) + 1

    pad_x = int(round(
        width * pad_x_ratio
    ))
    pad_y = int(round(
        height * pad_y_ratio
    ))

    x1 = max(
        sx,
        x1 - pad_x,
    )
    x2 = min(
        sx + sw,
        x2 + pad_x,
    )

    y1 = max(
        sy,
        y1 - pad_y,
    )
    y2 = min(
        sy + sh,
        y2 + pad_y,
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1,
    )


def detect_name_text_box(
    frame,
    mask,
    panel_top,
):
    """
    Detect the actual white character-name glyphs only.

    Search is tightly constrained:
    - left side only
    - immediately above dialogue panel
    - never allowed to cross panel_top
    """
    height, width = frame.shape[:2]

    x1 = int(round(width * 0.025))
    x2 = int(round(width * 0.22))

    y2 = max(
        1,
        panel_top,
    )
    y1 = max(
        0,
        y2 - int(round(height * 0.105)),
    )

    region = mask[
        y1:y2,
        x1:x2,
    ]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        region,
        connectivity=8,
    )

    glyph_boxes = []

    min_h = max(
        8,
        int(round(height * 0.018)),
    )
    max_h = int(round(height * 0.070))

    for label_id in range(
        1,
        num_labels,
    ):
        x, y, w, h, area = stats[
            label_id
        ]

        if h < min_h or h > max_h:
            continue

        if w > int(round(width * 0.060)):
            continue

        if area < max(
            12,
            int(round(width * height * 0.00001)),
        ):
            continue

        glyph_boxes.append(
            (
                x1 + x,
                y1 + y,
                w,
                h,
            )
        )

    if len(glyph_boxes) < 2:
        return None

    # Group glyphs by vertical center.
    groups = []

    for box in sorted(
        glyph_boxes,
        key=lambda b: b[0],
    ):
        bx, by, bw, bh = box
        center_y = by + bh / 2

        placed = False

        for group in groups:
            group_center = float(
                np.mean(
                    [
                        gy + gh / 2
                        for gx, gy, gw, gh
                        in group
                    ]
                )
            )

            if abs(
                center_y - group_center
            ) <= int(round(height * 0.020)):
                group.append(box)
                placed = True
                break

        if not placed:
            groups.append([box])

    # Prefer a compact multi-glyph row nearest the dialogue panel.
    candidates = []

    for group in groups:
        if len(group) < 2:
            continue

        gx1 = min(
            b[0]
            for b in group
        )
        gy1 = min(
            b[1]
            for b in group
        )
        gx2 = max(
            b[0] + b[2]
            for b in group
        )
        gy2 = max(
            b[1] + b[3]
            for b in group
        )

        group_width = gx2 - gx1
        group_height = gy2 - gy1

        if group_width < int(round(width * 0.035)):
            continue

        if group_width > int(round(width * 0.18)):
            continue

        if gy2 > panel_top:
            continue

        candidates.append(
            (
                gx1,
                gy1,
                group_width,
                group_height,
            )
        )

    if not candidates:
        return None

    name_text = max(
        candidates,
        key=lambda box: (
            box[1] + box[3],
            box[2],
        ),
    )

    pad_x = int(round(width * 0.012))
    pad_y = int(round(height * 0.010))

    nx, ny, nw, nh = name_text

    nx1 = max(
        x1,
        nx - pad_x,
    )
    ny1 = max(
        y1,
        ny - pad_y,
    )
    nx2 = min(
        x2,
        nx + nw + pad_x,
    )
    ny2 = min(
        panel_top,
        ny + nh + pad_y,
    )

    return (
        nx1,
        ny1,
        nx2 - nx1,
        ny2 - ny1,
    )


def detect_boxes(frame):
    """
    1) Detect dialogue-panel top from grayscale.
    2) dialogue_box is the full bottom panel.
    3) name_box is derived from actual white name glyphs only,
       in a narrow region immediately above panel_top.
    """
    gray, mask = make_grayscale_mask(
        frame
    )

    height, width = frame.shape[:2]

    panel_top, panel_drop = detect_panel_top(
        gray
    )

    if panel_top is None:
        return (
            mask,
            None,
            None,
            panel_drop,
        )

    dialogue_box = (
        0,
        panel_top,
        width,
        height - panel_top,
    )

    name_box = detect_name_text_box(
        frame,
        mask,
        panel_top,
    )

    if name_box is None:
        return (
            mask,
            None,
            None,
            panel_drop,
        )

    return (
        mask,
        name_box,
        dialogue_box,
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
            mask,
            name_box,
            dialogue_box,
            panel_drop,
        ) = detect_boxes(
            frame
        )

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
                f"panel-drop={panel_drop:.1f}"
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
                    "No valid HBR dialogue UI on this frame."
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
