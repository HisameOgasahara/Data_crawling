from __future__ import annotations

import argparse
import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2


WINDOW_NAME = "HBR ROI OCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview HBR-like dialogue UI, auto-detect dark name/dialogue panels, and run MangaOCR on demand."
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
        help="TXT/CSV with known speaker names. Default: character_names.txt beside this script.",
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start playback immediately instead of paused mode.",
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
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    names.append(row[0].strip())
    else:
        raise ValueError("names file must be .txt or .csv")

    return names


def match_known_name(raw_name: str, candidates: list[str]) -> tuple[str, float]:
    raw_norm = normalize_text(raw_name)
    if not raw_norm or not candidates:
        return raw_name, 0.0

    best_name = raw_name
    best_score = 0.0

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)

        if raw_norm == candidate_norm:
            return candidate, 1.0

        score = SequenceMatcher(None, raw_norm, candidate_norm).ratio()
        if raw_norm in candidate_norm or candidate_norm in raw_norm:
            score = max(score, 0.85)

        if score > best_score:
            best_score = score
            best_name = candidate

    if best_score >= 0.55:
        return best_name, best_score

    return raw_name, best_score


def threshold_dark_panel(search_bgr):
    gray = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return mask, gray


def find_panel(
    frame,
    x1_ratio,
    y1_ratio,
    x2_ratio,
    y2_ratio,
    min_w_ratio,
    max_w_ratio,
    min_h_ratio,
    max_h_ratio,
    min_area_ratio,
):
    height, width = frame.shape[:2]

    sx1 = int(width * x1_ratio)
    sy1 = int(height * y1_ratio)
    sx2 = int(width * x2_ratio)
    sy2 = int(height * y2_ratio)

    search = frame[sy1:sy2, sx1:sx2]
    mask, gray = threshold_dark_panel(search)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_box = None
    best_score = -1.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if w < int(width * min_w_ratio) or w > int(width * max_w_ratio):
            continue
        if h < int(height * min_h_ratio) or h > int(height * max_h_ratio):
            continue
        if area < int(width * height * min_area_ratio):
            continue

        mean_darkness = 255.0 - float(
            gray[y : y + h, x : x + w].mean()
        )
        score = area + (y + h) * 10 + mean_darkness * 50

        if score > best_score:
            best_score = score
            best_box = (
                sx1 + x,
                sy1 + y,
                w,
                h,
            )

    return best_box


def detect_dialogue_panel(frame):
    return find_panel(
        frame,
        x1_ratio=0.04,
        y1_ratio=0.68,
        x2_ratio=0.98,
        y2_ratio=0.98,
        min_w_ratio=0.55,
        max_w_ratio=0.98,
        min_h_ratio=0.10,
        max_h_ratio=0.26,
        min_area_ratio=0.05,
    )


def detect_name_panel(frame, dialogue_panel=None):
    if dialogue_panel is not None:
        dx, dy, dw, dh = dialogue_panel
        height, width = frame.shape[:2]

        x1 = max(0.0, dx / width - 0.03)
        y1 = max(0.0, dy / height - 0.08)
        x2 = min(1.0, (dx + dw * 0.35) / width)
        y2 = min(1.0, (dy + dh * 0.20) / height)
    else:
        x1, y1, x2, y2 = 0.02, 0.58, 0.32, 0.84

    return find_panel(
        frame,
        x1_ratio=x1,
        y1_ratio=y1,
        x2_ratio=x2,
        y2_ratio=y2,
        min_w_ratio=0.05,
        max_w_ratio=0.25,
        min_h_ratio=0.03,
        max_h_ratio=0.10,
        min_area_ratio=0.002,
    )


def derive_text_rois(frame, dialogue_panel, name_panel):
    speaker_roi = None

    if name_panel is not None:
        nx, ny, nw, nh = name_panel
        speaker_roi = (
            nx + int(0.08 * nw),
            ny + int(0.10 * nh),
            int(0.84 * nw),
            int(0.78 * nh),
        )
    elif dialogue_panel is not None:
        dx, dy, dw, dh = dialogue_panel
        speaker_roi = (
            dx + int(0.02 * dw),
            max(0, dy - int(0.26 * dh)),
            int(0.18 * dw),
            int(0.22 * dh),
        )

    dialogue_roi = None

    if dialogue_panel is not None:
        dx, dy, dw, dh = dialogue_panel
        dialogue_roi = (
            dx + int(0.04 * dw),
            dy + int(0.10 * dh),
            int(0.92 * dw),
            int(0.76 * dh),
        )

    return speaker_roi, dialogue_roi


def crop_from_box(frame, box):
    if box is None:
        return None

    x, y, w, h = box
    return frame[y : y + h, x : x + w]


def draw_box(display, source_box, scale, label, color):
    if source_box is None:
        return

    x, y, w, h = source_box
    x1 = int(round(x * scale))
    y1 = int(round(y * scale))
    x2 = int(round((x + w) * scale))
    y2 = int(round((y + h) * scale))

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
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


def prepare_ocr_image(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

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
        raise FileNotFoundError(args.video)

    known_names = load_name_list(args.names_file)
    print(
        f"Loaded {len(known_names)} known names from "
        f"{args.names_file}"
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    paused = not args.autoplay
    manga_ocr = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read the first frame.")

    while True:
        if not paused:
            ok, next_frame = cap.read()

            if not ok:
                paused = True
            else:
                frame = next_frame

        dialogue_panel = detect_dialogue_panel(frame)
        name_panel = detect_name_panel(
            frame,
            dialogue_panel,
        )

        speaker_roi, dialogue_roi = derive_text_rois(
            frame,
            dialogue_panel,
            name_panel,
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
            name_panel,
            scale,
            "name-panel",
            (255, 255, 0),
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
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        ) - 1
        current_frame = max(current_frame, 0)
        current_sec = (
            current_frame / fps
            if fps > 0
            else 0.0
        )

        status = (
            f"{current_sec:.2f}/{duration_sec:.2f}s  "
            f"frame {current_frame}/{max(frame_count - 1, 0)}  "
            f"{'PAUSE' if paused else 'PLAY'}"
        )

        cv2.putText(
            display,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display)

        delay = 1 if not paused else 30
        key = cv2.waitKey(delay) & 0xFF

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
                speaker_roi is None
                or dialogue_roi is None
            ):
                print(
                    "ROI detection failed on this frame."
                )
                continue

            if manga_ocr is None:
                print("Loading MangaOCR on CPU...")
                from manga_ocr import MangaOcr

                manga_ocr = MangaOcr()
                print("MangaOCR ready.")

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

            print(f"[{current_sec:.2f}s]")
            print("name panel   :", name_panel)
            print("dialogue pnl :", dialogue_panel)
            print("speaker roi  :", speaker_roi)
            print("dialogue roi :", dialogue_roi)
            print("speaker raw  :", speaker_text)
            print(
                f"speaker match: {matched_name} "
                f"(score={score:.2f})"
            )
            print("dialogue     :", dialogue_text)
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
