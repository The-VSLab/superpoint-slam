import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# --- 설정 ---
NUM_IMAGES = 5000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "synthetic_shapes"
IMG_H, IMG_W = 480, 640

def _non_degenerate_triangle():
    while True:
        pt1 = (np.random.randint(20, IMG_W-20), np.random.randint(20, IMG_H-20))
        pt2 = (np.random.randint(20, IMG_W-20), np.random.randint(20, IMG_H-20))
        pt3 = (np.random.randint(20, IMG_W-20), np.random.randint(20, IMG_H-20))
        area2 = abs((pt2[0] - pt1[0]) * (pt3[1] - pt1[1]) - (pt3[0] - pt1[0]) * (pt2[1] - pt1[1]))
        if area2 > 50: return pt1, pt2, pt3

def generate_shapes_with_labels():
    if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"==> Generating {NUM_IMAGES} images & labels (Row, Col format)...")
    
    for idx in tqdm(range(NUM_IMAGES)):
        img = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        points = []

        for _ in range(np.random.randint(5, 11)):
            shape_type = np.random.choice(['rect', 'triangle', 'line', 'circle'])
            color = np.random.randint(100, 255)
            
            if shape_type == 'rect':
                margin = 20
                max_w = min(80, IMG_W - 2 * margin)
                max_h = min(80, IMG_H - 2 * margin)
                if max_w < 20 or max_h < 20:
                    continue
                w = np.random.randint(20, max_w + 1)
                h = np.random.randint(20, max_h + 1)
                x = np.random.randint(margin, IMG_W - w - margin + 1)
                y = np.random.randint(margin, IMG_H - h - margin + 1)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                # [Fix] y, x 순서로 저장 (Row, Col)
                points.append([y, x])
                points.append([y, x+w])
                points.append([y+h, x])
                points.append([y+h, x+w])

            elif shape_type == 'triangle':
                pt1, pt2, pt3 = _non_degenerate_triangle()
                cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, color, -1)
                # [Fix] y, x 순서로 저장
                points.append([pt1[1], pt1[0]])
                points.append([pt2[1], pt2[0]])
                points.append([pt3[1], pt3[0]])
            
            elif shape_type == 'line':
                pt1 = (np.random.randint(0, IMG_W), np.random.randint(0, IMG_H))
                pt2 = (np.random.randint(0, IMG_W), np.random.randint(0, IMG_H))
                cv2.line(img, pt1, pt2, color, thickness=np.random.randint(1, 4))
                # [Fix] y, x 순서로 저장
                points.append([pt1[1], pt1[0]])
                points.append([pt2[1], pt2[0]])

            elif shape_type == 'circle':
                center = (np.random.randint(20, IMG_W-20), np.random.randint(20, IMG_H-20))
                radius = np.random.randint(10, 40)
                cv2.circle(img, center, radius, color, -1)

        noise = np.random.randn(IMG_H, IMG_W) * 5
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(str(OUTPUT_DIR / f"{idx:05d}.jpg"), img)
        if len(points) > 0:
            pts_array = np.array(points, dtype=np.float32)
        else:
            pts_array = np.zeros((0, 2), dtype=np.float32)
        np.savez_compressed(OUTPUT_DIR / f"{idx:05d}.npz", points=pts_array)

if __name__ == "__main__":
    generate_shapes_with_labels()
