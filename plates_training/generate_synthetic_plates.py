"""
Synthetic Kenyan plate scene generator for fine-tuning argus_scene_v1.
Generates 640x640 scenes with a plate pasted in, YOLO labels, train/val split, data.yaml.

Classes (must match model):
  0 = car_plate
  1 = tuktuk_plate
  2 = motorbike_plate

Usage: python3 generate_synthetic_plates.py
"""

import os, random, shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "/Users/emmanuel/Downloads/License Plate Detection with YoloV8 and EasyOCR Code"
OUT_DIR    = os.path.join(BASE_DIR, "data/square_plates")
FONT_PATH  = "/System/Library/Fonts/SFNSMono.ttf"  # system monospace — bold, readable for plates

COUNT      = 300     # total images
VAL_RATIO  = 0.15    # 15% validation
SCENE_SIZE = 640
PLATE_W    = 260
PLATE_H    = 220

# ── Plate text generators ───────────────────────────────────────────────────────
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

def gen_car_plate():
    """KAB / 123C — class 0"""
    l1 = f"K{random.choice(LETTERS)}{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 0

def gen_tuktuk_plate():
    """KTWA / 123C — class 1"""
    l1 = f"KTW{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 1

def gen_moto_plate():
    """KMCA / 123C — class 2"""
    l1 = f"KMC{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 2

GENERATORS = [gen_car_plate] * 6 + [gen_tuktuk_plate] * 2 + [gen_moto_plate] * 2  # 60/20/20 mix

# ── Augmentation helpers ────────────────────────────────────────────────────────
def random_perspective(img):
    """Mild perspective warp."""
    h, w = img.shape[:2]
    jitter = int(w * 0.07)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def random_blur(img):
    k = random.choice([0, 0, 1, 3])  # mostly sharp
    if k == 0:
        return img
    return cv2.GaussianBlur(img, (2*k+1, 2*k+1), 0)

def add_mud(img, n=25):
    for _ in range(n):
        cv2.circle(img,
                   (random.randint(0, img.shape[1]-1), random.randint(0, img.shape[0]-1)),
                   random.randint(2, 6), (20, 30, 40), -1)
    return img

# ── Setup dirs ──────────────────────────────────────────────────────────────────
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

# ── Verify font ──────────────────────────────────────────────────────────────────
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font not found: {FONT_PATH}")

val_count  = int(COUNT * VAL_RATIO)
train_count = COUNT - val_count
print(f"Generating {train_count} train + {val_count} val images...")

saved = {"train": 0, "val": 0}
errors = 0

for i in range(COUNT):
    split = "val" if i < val_count else "train"
    try:
        # Background
        bg = (random.randint(30, 160),) * 3
        scene = np.full((SCENE_SIZE, SCENE_SIZE, 3), bg, dtype=np.uint8)

        # Plate image
        plate_color = random.choice([(255, 204, 0), (255, 220, 50)])  # slight shade variation
        plate = Image.new("RGB", (PLATE_W, PLATE_H), plate_color)
        draw = ImageDraw.Draw(plate)

        l1, l2, cls_id = random.choice(GENERATORS)()
        f_size = random.randint(62, 80)
        font = ImageFont.truetype(FONT_PATH, f_size)

        w1 = draw.textbbox((0, 0), l1, font=font)[2]
        w2 = draw.textbbox((0, 0), l2, font=font)[2]
        draw.text(((PLATE_W - w1) / 2, 20),            l1, font=font, fill=(0, 0, 0))
        draw.text(((PLATE_W - w2) / 2, 20 + f_size + 5), l2, font=font, fill=(0, 0, 0))

        plate_cv = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)
        plate_cv = add_mud(plate_cv, n=random.randint(10, 30))
        plate_cv = random_perspective(plate_cv)
        plate_cv = random_blur(plate_cv)

        # Paste into scene
        x_min = random.randint(40, SCENE_SIZE - PLATE_W - 40)
        y_min = random.randint(40, SCENE_SIZE - PLATE_H - 40)
        scene[y_min:y_min+PLATE_H, x_min:x_min+PLATE_W] = plate_cv

        # Save image
        fname = f"argus_syn_{i:04d}.jpg"
        img_path = os.path.join(OUT_DIR, split, "images", fname)
        cv2.imwrite(img_path, scene, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # Save label
        x_c = (x_min + PLATE_W / 2) / SCENE_SIZE
        y_c = (y_min + PLATE_H / 2) / SCENE_SIZE
        wn  = PLATE_W / SCENE_SIZE
        hn  = PLATE_H / SCENE_SIZE
        lbl_path = os.path.join(OUT_DIR, split, "labels", fname.replace(".jpg", ".txt"))
        with open(lbl_path, "w") as f:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}\n")

        saved[split] += 1
        if i % 50 == 0:
            print(f"  {i}/{COUNT} — {split}")

    except Exception as e:
        print(f"  ⚠ skipped {i}: {e}")
        errors += 1
        continue  # never stop the loop on one bad sample

# ── data.yaml ──────────────────────────────────────────────────────────────────
yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: {OUT_DIR}
train: train/images
val:   val/images

nc: 3
names:
  0: car_plate
  1: tuktuk_plate
  2: motorbike_plate
""")

# ── classes.txt (for labeling tools) ───────────────────────────────────────────
with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
    f.write("car_plate\ntuktuk_plate\nmotorbike_plate\n")

print(f"""
{'='*40}
  Train images : {saved['train']}
  Val   images : {saved['val']}
  Errors skipped: {errors}
  data.yaml    : {yaml_path}
{'='*40}
""")
