"""
Collect real training frames from video using argus_scene_v1 as pre-annotator.

Workflow:
  1. Run this on any dashcam/traffic video
  2. Frames + YOLO pre-labels saved to data/real_annotations/
  3. Upload to Roboflow (or Label Studio) — verify/correct the boxes
  4. Export back as YOLO format → fine-tune in the notebook

Usage:
  python3 collect_annotations.py video.mp4 [--conf 0.3] [--every 10] [--limit 300]

Output structure:
  data/real_annotations/
    images/   *.jpg  (full frames)
    labels/   *.txt  (YOLO pre-labels — class x_c y_c w h)
    classes.txt
    data.yaml
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

OUT_DIR  = Path("./data/real_annotations")
CLASSES  = ["car_plate", "tuktuk_plate", "motorbike_plate"]


def frame_is_sharp(frame: np.ndarray, thresh: float = 60.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= thresh


def frames_are_similar(a: np.ndarray, b: np.ndarray, thresh: float = 0.97) -> bool:
    """Skip frames that are nearly identical (stopped traffic)."""
    if a is None:
        return False
    a_s = cv2.resize(a, (64, 36))
    b_s = cv2.resize(b, (64, 36))
    diff = np.mean(np.abs(a_s.astype(float) - b_s.astype(float))) / 255.0
    return diff < (1.0 - thresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video",          help="Path to input video")
    parser.add_argument("--conf",   type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--every",  type=int,   default=10,   help="Sample every N frames")
    parser.add_argument("--limit",  type=int,   default=400,  help="Max frames to save")
    parser.add_argument("--no-preann", action="store_true",   help="Save frames without pre-labels (blank labels)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    img_dir = OUT_DIR / "images"
    lbl_dir = OUT_DIR / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Use fast_alpr ONNX detector — reliable on real traffic scenes
    from fast_alpr import ALPR
    alpr = ALPR(
        detector_model="yolo-v9-t-640-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
        detector_conf_thresh=args.conf,
    )
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Video   : {video_path.name}")
    print(f"Frames  : {total}  ({total/fps:.0f}s @ {fps:.0f}fps)")
    print(f"Sampling: every {args.every} frames  |  conf >= {args.conf}")
    print(f"Limit   : {args.limit} saved frames\n")

    saved = 0
    skipped_blur = 0
    skipped_dup  = 0
    skipped_none = 0
    prev_frame   = None
    stem         = video_path.stem[:30].replace(" ", "_")

    for frame_idx in range(0, total, args.every):
        if saved >= args.limit:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Skip blurry frames
        if not frame_is_sharp(frame):
            skipped_blur += 1
            continue

        # Skip near-duplicate frames
        if frames_are_similar(prev_frame, frame):
            skipped_dup += 1
            continue

        # Run detector
        detections = alpr.predict(frame)

        if len(detections) == 0 and not args.no_preann:
            skipped_none += 1
            prev_frame = frame
            continue

        h, w = frame.shape[:2]
        fname = f"{stem}_{frame_idx:07d}"

        # Save image
        img_path = img_dir / f"{fname}.jpg"
        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save YOLO label
        # fast_alpr doesn't classify car/tuktuk/moto — label all as 0 (car_plate).
        # Correct tuk-tuk and motorcycle boxes to class 1/2 in Roboflow.
        lbl_path = lbl_dir / f"{fname}.txt"
        with open(lbl_path, "w") as f:
            for det in detections:
                bb = det.detection.bounding_box
                x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)
                x_c = (x1 + x2) / 2 / w
                y_c = (y1 + y2) / 2 / h
                bw  = (x2 - x1) / w
                bh  = (y2 - y1) / h
                f.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

        prev_frame = frame
        saved += 1

        if saved % 50 == 0:
            print(f"  Saved {saved}/{args.limit}  (frame {frame_idx})")

    cap.release()

    # classes.txt
    (OUT_DIR / "classes.txt").write_text("\n".join(CLASSES) + "\n")

    # data.yaml — for fine-tuning after labeling; user splits train/val themselves
    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.resolve()}\n"
        f"train: images\n"
        f"val:   images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )

    print(f"""
{'='*45}
  Frames saved      : {saved}
  Skipped (blurry)  : {skipped_blur}
  Skipped (similar) : {skipped_dup}
  Skipped (no det.) : {skipped_none}
  Output            : {OUT_DIR.resolve()}
{'='*45}

Next steps:
  1. Upload data/real_annotations/ to Roboflow
     (roboflow.com → New Project → Object Detection)
  2. Verify/correct the pre-labelled boxes
  3. Export as "YOLOv8" format → download ZIP
  4. Extract to data/real_annotations_verified/
  5. Run fine-tuning cell in plates_training/Untitled.ipynb
""")


if __name__ == "__main__":
    main()
