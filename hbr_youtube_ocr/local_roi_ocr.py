from __future__ import annotations

import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "HBR ROI OCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview HBR video, auto-detect speaker/dialogue text ROIs, and run MangaOCR on demand."
    )
    parser.add_argument("video", type=Path, help="Video file path")
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Preview window width. Aspect ratio is preserved. Default: 1280",
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
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def detect_text_box(
    frame,
    x1_ratio,
    y1_ratio,
    x2_ratio,
    y2_ratio,
    min_component_area,
    padding_x,
    padding_y,
):
    height, width = frame.shape[:2]

    sx1 = int(width * x1_ratio)
    sy1 = int(height * y1_ratio)
    sx2 = int(width * x2_ratio)
    sy2 = int(height * y2_ratio)

    search = frame[sy1:sy2, sx1:sx2]
    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        185,
        255,
        cv2.THRESH_BINARY,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 3),
    )
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < min_component_area:
            continue

        if h < max(8, int(height * 0.008)):
            continue

        if h > int(height * 0.10):
            continue

        boxes.append((x, y, w, h))

    if not boxes:
        return None

    x_min = min(x for x, y, w, h in boxes)
    y_min = min(y for x, y, w, h in boxes)
    x_max = max(x + w for x, y, w, h in boxes)
    y_max = max(y + h for x, y, w, h in boxes)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(search.shape[1], x_max + padding_x)
    y_max = min(search.shape[0], y_max + padding_y)

    return (
        sx1 + x_min,
        sy1 + y_min,
        x_max - x_min,
        y_max - y_min,
    )


def detect_hbr_rois(frame):
    height, width = frame.shape[:2]

    speaker_box = detect_text_box(
        frame,
        x1_ratio=0.02,
        y1_ratio=0.52,
        x2_ratio=0.34,
        y2_ratio=0.76,
        min_component_area=max(40, int(width * height * 0.000015)),
        padding_x=max(8, int(width * 0.006)),
        padding_y=max(6, int(height * 0.008)),
    )

    dialogue_box = detect_text_box(
        frame,
        x1_ratio=0.08,
        y1_ratio=0.68,
        x2_ratio=0.95,
        y2_ratio=0.93,
        min_component_area=max(60, int(width * height * 0.000020)),
        padding_x=max(12, int(width * 0.008)),
        padding_y=max(8, int(height * 0.010)),
    )

    return speaker_box, dialogue_box


def main():
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    paused = True
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

        speaker_box, dialogue_box = detect_hbr_rois(frame)

        display, scale = resize_for_display(frame, args.width)

        draw_box(display, speaker_box, scale, "speaker", (0, 255, 0))
        draw_box(display, dialogue_box, scale, "dialogue", (255, 0, 0))

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        current_frame = max(current_frame, 0)
        current_sec = current_frame / fps if fps > 0 else 0.0

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
            target = max(current_frame - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            continue

        if key == ord("d"):
            paused = True
            target = min(current_frame + 1, max(frame_count - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            continue

        if key == ord("j"):
            paused = True
            target_sec = max(current_sec - 5.0, 0.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = cap.read()
            continue

        if key == ord("l"):
            paused = True
            target_sec = min(current_sec + 5.0, duration_sec)
            cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
            ok, frame = cap.read()
            continue

        if key == ord("o"):
            paused = True

            if speaker_box is None or dialogue_box is None:
                print("speaker/dialogue ROI detection failed on this frame.")
                continue

            if manga_ocr is None:
                print("Loading MangaOCR on CPU...")
                from manga_ocr import MangaOcr

                manga_ocr = MangaOcr()
                print("MangaOCR ready.")

            from PIL import Image

            speaker_crop = crop_from_box(frame, speaker_box)
            dialogue_crop = crop_from_box(frame, dialogue_box)

            speaker_rgb = cv2.cvtColor(speaker_crop, cv2.COLOR_BGR2RGB)
            dialogue_rgb = cv2.cvtColor(dialogue_crop, cv2.COLOR_BGR2RGB)

            speaker_text = manga_ocr(Image.fromarray(speaker_rgb))
            dialogue_text = manga_ocr(Image.fromarray(dialogue_rgb))

            print(f"[{current_sec:.2f}s]")
            print("speaker ROI :", speaker_box)
            print("dialogue ROI:", dialogue_box)
            print("speaker     :", speaker_text)
            print("dialogue    :", dialogue_text)
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
