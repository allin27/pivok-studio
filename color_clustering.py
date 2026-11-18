import logging
from typing import Optional
import cv2
import numpy as np

COLOR_DICT_BGR = {
    "SIYAH": (30, 30, 30),
    "BEYAZ": (230, 230, 230),
    "KIRMIZI": (30, 30, 200),
    "SARI": (30, 200, 200),
    "YESIL": (30, 150, 30),
    "MAVI": (200, 100, 30),
    "LACIVERT": (100, 40, 20),
    "GRI": (128, 128, 128),
    "ANTRASIT": (70, 70, 70),
    "KAHVE": (40, 70, 100),
    "BEJ": (180, 210, 220),
    "PUDRA": (190, 180, 230),
    "EKRU": (210, 230, 240),
    "TABA": (50, 100, 180),
    "BORDO": (35, 15, 95),
    "VIZON": (120, 140, 150),
    "PEMBE": (140, 100, 220),
    "MOR": (120, 30, 100),
}


def extract_dominant_lab_from_pose(small_bgr: np.ndarray, pose_data: Optional[dict]) -> Optional[np.ndarray]:
    """
    Extracts dominant garment LAB color from the person's torso/dress region
    using already-computed YOLOv8-pose keypoints/boxes (zero extra neural net calls!).
    """
    try:
        if small_bgr is None or small_bgr.size == 0:
            return None
        h, w = small_bgr.shape[:2]
        roi = None

        if pose_data and pose_data.get("boxes"):
            boxes = pose_data["boxes"]
            best_i = max(
                range(len(boxes)),
                key=lambda i: (boxes[i]["xyxy"][2] - boxes[i]["xyxy"][0])
                * (boxes[i]["xyxy"][3] - boxes[i]["xyxy"][1]),
            )
            x1, y1, x2, y2 = map(int, boxes[best_i]["xyxy"])
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)

            # Use shoulder-to-knee/hip torso band if keypoints available
            cx1, cx2 = x1 + int(bw * 0.24), x2 - int(bw * 0.24)
            cy1, cy2 = y1 + int(bh * 0.24), y2 - int(bh * 0.16)

            if pose_data.get("keypoints") and best_i < len(pose_data["keypoints"]):
                kps = pose_data["keypoints"][best_i]
                if len(kps) >= 13:
                    sh_ys = [kps[idx][1] for idx in (5, 6) if kps[idx][1] > 5]
                    hip_ys = [kps[idx][1] for idx in (11, 12, 13, 14) if kps[idx][1] > 5]
                    sh_xs = [kps[idx][0] for idx in (5, 6) if kps[idx][0] > 5]
                    if sh_ys:
                        cy1 = max(y1, int(max(sh_ys) + bh * 0.03))
                    if hip_ys:
                        cy2 = min(y2, int(max(hip_ys)))
                    if len(sh_xs) == 2:
                        lx, rx = min(sh_xs), max(sh_xs)
                        pad = max(4, int((rx - lx) * 0.10))
                        if rx - lx > 18:
                            cx1, cx2 = int(lx + pad), int(rx - pad)

            cx1, cx2 = max(0, cx1), min(w, cx2)
            cy1, cy2 = max(0, cy1), min(h, cy2)
            if cx2 - cx1 >= 8 and cy2 - cy1 >= 8:
                roi = small_bgr[cy1:cy2, cx1:cx2]

        if roi is None or roi.size == 0:
            roi = small_bgr[int(h * 0.30) : int(h * 0.70), int(w * 0.38) : int(w * 0.62)]

        # Subsample to max ~1600 pixels so K-Means takes < 1 ms!
        if roi.shape[0] > 45 or roi.shape[1] > 45:
            roi = cv2.resize(roi, (min(40, roi.shape[1]), min(40, roi.shape[0])))

        pixels = np.float32(roi.reshape(-1, 3))
        if len(pixels) < 10:
            return None

        n_colors = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
        _, counts = np.unique(labels, return_counts=True)

        best_lab = None
        best_weight = -1.0
        total_pixels = float(sum(counts))
        for i in range(n_colors):
            bgr = palette[i]
            weight = counts[i] / total_pixels
            b, g, r = bgr
            if r > g > b and r > 60 and (r - g) > 15:
                weight *= 0.12  # Skin-tone penalty
            lab_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
            if weight > best_weight:
                best_weight = weight
                best_lab = lab_px
        return best_lab
    except Exception as e:
        logging.error(f"Error in extract_dominant_lab_from_pose: {e}")
        return None


def nearest_color_name(lab: Optional[np.ndarray]) -> str:
    if lab is None:
        return "COLOR"
    best_name, best_dist = "COLOR", float("inf")
    for name, bgr in COLOR_DICT_BGR.items():
        lab2 = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        dist = float(
            np.sqrt(0.5 * (lab[0] - lab2[0]) ** 2 + (lab[1] - lab2[1]) ** 2 + (lab[2] - lab2[2]) ** 2)
        )
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def cluster_analyzed_items(items: list[dict]):
    """
    items: [{'temp_path': ..., 'ext': ..., 'score': float, 'lab': list|np.ndarray|None, 'pose': dict}, ...]
    Returns: (pairs, labels)
    """
    if not items:
        return [], []

    pool = []
    for it in items:
        lab = it.get("lab")
        if lab is not None and not isinstance(lab, np.ndarray):
            lab = np.array(lab, dtype=np.float32)
        pool.append({"item": it, "lab": lab})

    pairs, labels = [], []
    while len(pool) >= 2:
        min_dist, best_pair = float("inf"), (0, 1)
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                lab1, lab2 = pool[i]["lab"], pool[j]["lab"]
                if lab1 is not None and lab2 is not None:
                    base_dist = np.sqrt(
                        0.5 * (lab1[0] - lab2[0]) ** 2
                        + (lab1[1] - lab2[1]) ** 2
                        + (lab1[2] - lab2[2]) ** 2
                    )
                    s1 = pool[i]["item"].get("score", 0.0)
                    s2 = pool[j]["item"].get("score", 0.0)
                    dist = base_dist - 5.0 * abs(s1 - s2)
                else:
                    dist = 9999.0
                if dist < min_dist:
                    min_dist, best_pair = dist, (i, j)

        i, j = best_pair
        p1, p2 = pool[i]["item"], pool[j]["item"]
        dom_lab = pool[i]["lab"]
        pairs.append([p1, p2] if p1.get("score", -999.0) >= p2.get("score", -999.0) else [p2, p1])
        labels.append(nearest_color_name(dom_lab))
        pool.pop(j)
        pool.pop(i)

    if pool:
        pairs.append([pool[0]["item"]])
        labels.append(nearest_color_name(pool[0]["lab"]))

    return pairs, labels
