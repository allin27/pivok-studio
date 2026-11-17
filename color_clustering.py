import logging
import cv2
import numpy as np
import yolo_local

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


def get_pixels_from_image(image_bytes: bytes):
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        max_dim = 640
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]

        pixels = None
        yolo_data = yolo_local.detect_yolo(img, "yolov8n-seg")

        if yolo_data and yolo_data.get("boxes"):
            person_indices = [i for i, b in enumerate(yolo_data["boxes"]) if int(b["cls"]) == 0]
            if person_indices:
                best_i = max(
                    person_indices,
                    key=lambda i: (yolo_data["boxes"][i]["xyxy"][2] - yolo_data["boxes"][i]["xyxy"][0])
                    * (yolo_data["boxes"][i]["xyxy"][3] - yolo_data["boxes"][i]["xyxy"][1]),
                )
                x1, y1, x2, y2 = map(int, yolo_data["boxes"][best_i]["xyxy"])
                if yolo_data.get("masks") and best_i < len(yolo_data["masks"]):
                    poly = np.array(yolo_data["masks"][best_i], np.int32)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly], 1)

                    bw, bh = x2 - x1, y2 - y1
                    crop_x1, crop_x2 = max(0, x1 + int(bw * 0.15)), min(w, x2 - int(bw * 0.15))
                    crop_y1, crop_y2 = max(0, y1 + int(bh * 0.25)), min(h, y2 - int(bh * 0.10))

                    crop_mask = np.zeros((h, w), dtype=bool)
                    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                        crop_mask[crop_y1:crop_y2, crop_x1:crop_x2] = True
                    else:
                        crop_mask[y1:y2, x1:x2] = True

                    valid_pixels = img[(mask > 0) & crop_mask]
                    if len(valid_pixels) > 0:
                        pixels = np.float32(valid_pixels)

        if pixels is None or len(pixels) == 0:
            cropped = img[int(h * 0.3) : int(h * 0.7), int(w * 0.4) : int(w * 0.6)]
            pixels = np.float32(cropped.reshape(-1, 3))
        return pixels
    except Exception as e:
        logging.error(f"Error in get_pixels_from_image: {e}")
        return None


def extract_color_features(image_bytes: bytes):
    pixels = get_pixels_from_image(image_bytes)
    if pixels is None or len(pixels) < 10:
        return None
    try:
        n_colors = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.1)
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 8, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)

        valid_clusters = []
        total_pixels = sum(counts)
        for i in range(n_colors):
            bgr = palette[i]
            weight = counts[i] / total_pixels
            b, g, r = bgr
            if r > g > b and r > 60 and (r - g) > 15:
                weight *= 0.1
            pixel = np.uint8([[bgr]])
            lab_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0][0]
            valid_clusters.append(
                {"lab": np.array([lab_pixel[0], lab_pixel[1], lab_pixel[2]], dtype=np.float32), "weight": weight}
            )
        return valid_clusters
    except Exception as e:
        logging.error(f"Error in extract_color_features: {e}")
        return None


def _nearest_color_name(lab: np.ndarray) -> str:
    if lab is None:
        return "COLOR"
    best_name, best_dist = "COLOR", float("inf")
    for name, bgr in COLOR_DICT_BGR.items():
        lab2 = cv2.cvtColor(np.uint8([[list(bgr)]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        dist = float(np.sqrt(0.5 * (lab[0] - lab2[0]) ** 2 + (lab[1] - lab2[1]) ** 2 + (lab[2] - lab2[2]) ** 2))
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def cluster_photos_by_color(photo_list):
    if not photo_list:
        return [], []

    features = []
    for p in photo_list:
        clusters = extract_color_features(p["bytes"])
        if clusters:
            dominant = max(clusters, key=lambda x: x["weight"])
            features.append({"photo": p, "lab": dominant["lab"]})
        else:
            features.append({"photo": p, "lab": None})

    pairs, labels = [], []
    while len(features) >= 2:
        min_dist, best_pair = float("inf"), (0, 1)
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                lab1, lab2 = features[i]["lab"], features[j]["lab"]
                if lab1 is not None and lab2 is not None:
                    base_dist = np.sqrt(
                        0.5 * (lab1[0] - lab2[0]) ** 2 + (lab1[1] - lab2[1]) ** 2 + (lab1[2] - lab2[2]) ** 2
                    )
                    s1 = features[i]["photo"].get("score", 0.0)
                    s2 = features[j]["photo"].get("score", 0.0)
                    dist = base_dist - 5.0 * abs(s1 - s2)
                else:
                    dist = 9999.0
                if dist < min_dist:
                    min_dist, best_pair = dist, (i, j)

        i, j = best_pair
        p1, p2 = features[i]["photo"], features[j]["photo"]
        dom_lab = features[i]["lab"]
        pairs.append([p1, p2] if p1.get("score", -999.0) >= p2.get("score", -999.0) else [p2, p1])
        labels.append(_nearest_color_name(dom_lab))
        features.pop(j)
        features.pop(i)

    if features:
        pairs.append([features[0]["photo"]])
        labels.append(_nearest_color_name(features[0]["lab"]))

    return pairs, labels
