from __future__ import annotations

import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "HBR ROI OCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a video, keep fixed ROIs across frames, and run MangaOCR on demand."
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
        scale = 1.0
        return frame.copy(), scale

    scale = target_width / width
    display_height = int(round(height * scale))
    display = cv2.resize(
        frame,
        (target_width, display_height),
        interpolation=cv2.INTER_AREA,
    )
    return display, scale


def display_box_to_source(box, scale: float):
    x, y, w, h = box
    if w == 0 or h == 0:
        return None

    return (
        int(round(x / scale)),
        int(round(y / scale)),
        int(round(w / scale)),
        int(round(h / scale)),
    )


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

    speaker_box = None
    dialogue_box = None
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

        if key == ord("1"):
            paused = True
            selection, selection_scale = resize_for_display(frame, args.width)
            selected = cv2.selectROI(
                "Select speaker ROI",
                selection,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyWindow("Select speaker ROI")
            speaker_box = display_box_to_source(selected, selection_scale)
            print("speaker ROI:", speaker_box)
            continue

        if key == ord("2"):
            paused = True
            selection, selection_scale = resize_for_display(frame, args.width)
            selected = cv2.selectROI(
                "Select dialogue ROI",
                selection,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyWindow("Select dialogue ROI")
            dialogue_box = display_box_to_source(selected, selection_scale)
            print("dialogue ROI:", dialogue_box)
            continue

        if key == ord("o"):
            paused = True

            if speaker_box is None or dialogue_box is None:
                print("Set both ROIs first: 1 = speaker, 2 = dialogue")
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
            print("speaker :", speaker_text)
            print("dialogue:", dialogue_text)
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
