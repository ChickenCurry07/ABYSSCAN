"""
ABYSSCAN detection pipeline
Sonar image -> preprocessing -> unsupervised AI anomaly detection -> regions -> confidence -> priority
"""

import numpy as np
import cv2
from sklearn.ensemble import IsolationForest
from scipy import ndimage
import pandas as pd


# ----------------------------------------------------------------------
# 1. Synthetic sonar image (so the app is demo-able with no dataset)
# ----------------------------------------------------------------------
def generate_synthetic_sonar_image(width=640, height=480, seed=None):
    """Generate a synthetic side-scan-sonar-like grayscale image with a
    mottled seabed texture, a central nadir gap, and a handful of
    embedded bright 'objects' that the detector should be able to find.
    Returns (image_uint8, ground_truth_objects)."""
    rng = np.random.default_rng(seed)

    # low-frequency seabed texture
    small = rng.normal(0, 1, (max(4, height // 8), max(4, width // 8))).astype(np.float32)
    base = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=6)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    img = 40 + base * 60  # mid-gray seabed

    # sonar backscatter speckle
    img = img + rng.normal(0, 12, (height, width))

    # nadir gap: darker vertical band down the middle (typical side-scan artifact)
    cx = width // 2
    band = int(width * 0.035)
    img[:, cx - band:cx + band] -= 25

    # embed anomaly objects: bright return + darker acoustic-shadow trail
    n_objects = int(rng.integers(4, 7))
    objects = []
    for _ in range(n_objects):
        ox = int(rng.integers(int(width * 0.12), int(width * 0.88)))
        oy = int(rng.integers(int(height * 0.12), int(height * 0.88)))
        shape = rng.choice(["circle", "rect", "irregular"])
        size = int(rng.integers(10, 24))
        brightness = float(rng.integers(55, 95))

        highlight_mask = np.zeros_like(img, dtype=np.uint8)
        shadow_mask = np.zeros_like(img, dtype=np.uint8)

        if shape == "circle":
            cv2.circle(highlight_mask, (ox, oy), size, 1, -1)
            cv2.circle(shadow_mask, (ox + size + size // 2, oy), int(size * 0.8), 1, -1)
        elif shape == "rect":
            cv2.rectangle(highlight_mask, (ox - size, oy - size // 2), (ox + size, oy + size // 2), 1, -1)
            cv2.rectangle(shadow_mask, (ox + size, oy - size // 2), (ox + int(size * 1.8), oy + size // 2), 1, -1)
        else:
            pts = []
            for a in range(8):
                ang = a / 8 * 2 * np.pi
                r = size * rng.uniform(0.5, 1.2)
                pts.append([int(ox + r * np.cos(ang)), int(oy + r * np.sin(ang))])
            cv2.fillPoly(highlight_mask, [np.array(pts, dtype=np.int32)], 1)
            cv2.circle(shadow_mask, (ox + size, oy), size, 1, -1)

        img = img + highlight_mask.astype(np.float32) * brightness
        img = img - shadow_mask.astype(np.float32) * 22
        objects.append({"x": ox, "y": oy, "size": size, "shape": shape})

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, objects


# ----------------------------------------------------------------------
# 2. Preprocessing: noise reduction + CLAHE
# ----------------------------------------------------------------------
def preprocess(gray_img):
    denoised = cv2.fastNlMeansDenoising(gray_img, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return denoised, enhanced


# ----------------------------------------------------------------------
# 3. Block-level feature extraction
# ----------------------------------------------------------------------
def extract_block_features(img, block_size=16):
    h, w = img.shape
    features, coords = [], []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = img[y:y + block_size, x:x + block_size].astype(np.float32)
            mean_i = block.mean()
            std_i = block.std()
            max_i = block.max()
            lap = cv2.Laplacian(block, cv2.CV_32F)
            edge_energy = lap.var()
            features.append([mean_i, std_i, max_i, edge_energy])
            coords.append((x, y))
    return np.array(features, dtype=np.float32), coords


# ----------------------------------------------------------------------
# 4. Unsupervised anomaly detection (Isolation Forest, fit per-image)
# ----------------------------------------------------------------------
def detect_anomalies(enhanced_img, block_size=16, contamination=0.08, random_state=42,
                      mask_nadir=True, nadir_frac=0.05):
    features, coords = extract_block_features(enhanced_img, block_size)

    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-6
    features_norm = (features - mu) / sigma

    model = IsolationForest(
        n_estimators=200,
        contamination=min(max(contamination, 0.01), 0.4),
        random_state=random_state,
    )
    model.fit(features_norm)
    raw_scores = model.decision_function(features_norm)  # higher = more "normal"
    preds = model.predict(features_norm)  # -1 = anomaly, 1 = normal

    smin, smax = raw_scores.min(), raw_scores.max()
    anomaly_conf = 100 * (smax - raw_scores) / (smax - smin + 1e-9)  # higher = more anomalous

    h, w = enhanced_img.shape
    grid_h, grid_w = h // block_size, w // block_size
    anomaly_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    conf_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    for (x, y), pred, conf in zip(coords, preds, anomaly_conf):
        gx, gy = x // block_size, y // block_size
        if gy < grid_h and gx < grid_w:
            conf_grid[gy, gx] = conf
            if pred == -1:
                anomaly_grid[gy, gx] = 1

    if mask_nadir:
        # side-scan sonar images have a dark, non-informative water-column/nadir
        # gap running down the center of the swath; standard practice is to
        # exclude it before flagging targets, otherwise it dominates as a
        # "rare" (but meaningless) feature pattern.
        band_px = int(w * nadir_frac)
        cx = w // 2
        gx0 = max(0, (cx - band_px) // block_size)
        gx1 = min(grid_w, (cx + band_px) // block_size + 1)
        anomaly_grid[:, gx0:gx1] = 0

    return anomaly_grid, conf_grid


# ----------------------------------------------------------------------
# 5. Merge flagged blocks into regions
# ----------------------------------------------------------------------
def extract_regions(anomaly_grid, conf_grid, block_size, min_blocks=2):
    labeled, num = ndimage.label(anomaly_grid, structure=np.ones((3, 3)))
    regions = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < min_blocks:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        px0, py0 = int(x0 * block_size), int(y0 * block_size)
        px1, py1 = int((x1 + 1) * block_size), int((y1 + 1) * block_size)
        conf = float(conf_grid[ys, xs].mean())
        regions.append({
            "bbox": (px0, py0, px1, py1),
            "confidence": round(conf, 1),
            "area_blocks": int(len(xs)),
        })
    regions.sort(key=lambda r: -r["confidence"])
    for idx, r in enumerate(regions, start=1):
        r["id"] = idx
    return regions


# ----------------------------------------------------------------------
# 6. Priority scoring
# ----------------------------------------------------------------------
def assign_priority(regions):
    for r in regions:
        c, area = r["confidence"], r["area_blocks"]
        if c >= 70 or (c >= 55 and area >= 4):
            r["priority"] = "HIGH"
        elif c >= 45:
            r["priority"] = "MEDIUM"
        else:
            r["priority"] = "LOW"
    return regions


# ----------------------------------------------------------------------
# 7. Drawing + table export
# ----------------------------------------------------------------------
PRIORITY_COLORS_BGR = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 210, 210)}


def draw_regions(gray_img, regions):
    img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        color = PRIORITY_COLORS_BGR[r["priority"]]
        cv2.rectangle(img_color, (x0, y0), (x1, y1), color, 2)
        label = f"R{r['id']} {r['confidence']:.0f}% {r['priority']}"
        y_text = max(12, y0 - 6)
        cv2.putText(img_color, label, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return img_color


def regions_to_dataframe(regions):
    rows = []
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        rows.append({
            "Region": f"R{r['id']}",
            "BBox (x0, y0, x1, y1)": f"({x0}, {y0}, {x1}, {y1})",
            "Anomaly Score": f"{r['confidence']:.1f}%",
            "Area (blocks)": r["area_blocks"],
            "Priority": r["priority"],
        })
    return pd.DataFrame(rows)
