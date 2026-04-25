"""
Synthetic Kenyan plate scene generator for fine-tuning argus_scene_v1.
Generates 640x640 scenes with a plate pasted in, YOLO labels, train/val split, data.yaml.

Classes (must match model):
  0 = car_plate
  1 = tuktuk_plate
  2 = motorbike_plate

Usage: python3 generate_synthetic_plates.py
"""

import os, random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = "/Users/emmanuel/Downloads/License Plate Detection with YoloV8 and EasyOCR Code"
OUT_DIR    = os.path.join(BASE_DIR, "data/square_plates")
FONT_PATH  = "/System/Library/Fonts/SFNSMono.ttf"

COUNT      = 1500    # total images
VAL_RATIO  = 0.15
SCENE_SIZE = 640
PLATE_W    = 260
PLATE_H    = 220

# ── Plate text generators ───────────────────────────────────────────────────────
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

def gen_car_plate():
    l1 = f"K{random.choice(LETTERS)}{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 0

def gen_tuktuk_plate():
    l1 = f"KT{random.choice(LETTERS)}{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 1

def gen_moto_plate():
    l1 = f"KMC{random.choice(LETTERS)}"
    l2 = f"{random.randint(100, 999)}{random.choice(LETTERS)}"
    return l1, l2, 2

GENERATORS = [gen_car_plate] * 6 + [gen_tuktuk_plate] * 2 + [gen_moto_plate] * 2


# ── Background ──────────────────────────────────────────────────────────────────
def apply_complex_background(scene):
    """Gradient + distraction shapes — stops model overfitting to flat colors."""
    h, w = scene.shape[:2]

    # Random vertical gradient
    top    = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    bottom = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
    for y in range(h):
        alpha = y / h
        scene[y] = ((1 - alpha) * top + alpha * bottom).astype(np.uint8)

    # Distraction rectangles (indicators / lights / road markings)
    for _ in range(random.randint(2, 5)):
        color = (
            random.randint(0, 50),
            random.randint(150, 255),
            random.randint(180, 255),
        )  # yellowish in BGR
        pt1 = (random.randint(0, w - 1), random.randint(0, h - 1))
        pt2 = (random.randint(0, w - 1), random.randint(0, h - 1))
        cv2.rectangle(scene, pt1, pt2, color, -1)

    return scene


# ── Augmentation helpers ────────────────────────────────────────────────────────
def random_perspective(img):
    h, w = img.shape[:2]
    jitter = int(w * 0.07)
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def random_blur(img):
    k = random.choice([0, 0, 1, 3])
    if k == 0:
        return img
    return cv2.GaussianBlur(img, (2*k+1, 2*k+1), 0)

def add_mud(img, n=25):
    for _ in range(n):
        cv2.circle(img,
                   (random.randint(0, img.shape[1]-1), random.randint(0, img.shape[0]-1)),
                   random.randint(2, 6), (20, 30, 40), -1)
    return img


# ── Setup ───────────────────────────────────────────────────────────────────────
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, split, "labels"), exist_ok=True)

if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font not found: {FONT_PATH}")

val_count   = int(COUNT * VAL_RATIO)
train_count = COUNT - val_count
print(f"Generating {train_count} train + {val_count} val images...")

saved  = {"train": 0, "val": 0}
errors = 0

for i in range(COUNT):
    split = "val" if i < val_count else "train"
    try:
        # ── Complex background ──────────────────────────────────────────────────
        scene = np.zeros((SCENE_SIZE, SCENE_SIZE, 3), dtype=np.uint8)
        scene = apply_complex_background(scene)

        # ── Plate with color jitter ─────────────────────────────────────────────
        r = random.randint(220, 255)
        g = random.randint(180, 220)
        b = random.randint(0, 40)
        plate_color_rgb = (r, g, b)          # RGB for PIL
        plate_color_bgr = (b, g, r)          # BGR for OpenCV shadow

        plate = Image.new("RGB", (PLATE_W, PLATE_H), plate_color_rgb)
        draw  = ImageDraw.Draw(plate)

        l1, l2, cls_id = random.choice(GENERATORS)()
        f_size = random.randint(62, 80)
        font   = ImageFont.truetype(FONT_PATH, f_size)

        w1 = draw.textbbox((0, 0), l1, font=font)[2]
        w2 = draw.textbbox((0, 0), l2, font=font)[2]
        draw.text(((PLATE_W - w1) / 2, 20),               l1, font=font, fill=(0, 0, 0))
        draw.text(((PLATE_W - w2) / 2, 20 + f_size + 5),  l2, font=font, fill=(0, 0, 0))

        plate_cv = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)
        plate_cv = add_mud(plate_cv, n=random.randint(10, 30))
        plate_cv = random_perspective(plate_cv)
        plate_cv = random_blur(plate_cv)

        # ── Shadow then paste ───────────────────────────────────────────────────
        x_min = random.randint(40, SCENE_SIZE - PLATE_W - 50)
        y_min = random.randint(40, SCENE_SIZE - PLATE_H - 50)

        shadow_off = random.randint(4, 10)
        sx1 = min(x_min + shadow_off, SCENE_SIZE - 1)
        sy1 = min(y_min + shadow_off, SCENE_SIZE - 1)
        sx2 = min(x_min + PLATE_W + shadow_off, SCENE_SIZE - 1)
        sy2 = min(y_min + PLATE_H + shadow_off, SCENE_SIZE - 1)
        cv2.rectangle(scene, (sx1, sy1), (sx2, sy2), (0, 0, 0), -1)

        scene[y_min:y_min+PLATE_H, x_min:x_min+PLATE_W] = plate_cv

        # ── Save image ──────────────────────────────────────────────────────────
        fname    = f"argus_syn_{i:04d}.jpg"
        img_path = os.path.join(OUT_DIR, split, "images", fname)
        cv2.imwrite(img_path, scene, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── YOLO label ──────────────────────────────────────────────────────────
        x_c = (x_min + PLATE_W / 2) / SCENE_SIZE
        y_c = (y_min + PLATE_H / 2) / SCENE_SIZE
        wn  = PLATE_W / SCENE_SIZE
        hn  = PLATE_H / SCENE_SIZE
        lbl_path = os.path.join(OUT_DIR, split, "labels", fname.replace(".jpg", ".txt"))
        with open(lbl_path, "w") as f:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {wn:.6f} {hn:.6f}\n")

        saved[split] += 1
        if i % 100 == 0:
            print(f"  {i}/{COUNT} — {split}")

    except Exception as e:
        print(f"  skipped {i}: {e}")
        errors += 1
        continue

# ── data.yaml ───────────────────────────────────────────────────────────────────
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

with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
    f.write("car_plate\ntuktuk_plate\nmotorbike_plate\n")

print(f"""
{'='*40}
  Train : {saved['train']}
  Val   : {saved['val']}
  Errors: {errors}
  Yaml  : {yaml_path}
{'='*40}
""")
