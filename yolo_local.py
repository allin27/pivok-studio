import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import threading
import cv2
import numpy as np
import onnxruntime as ort

try:
    cv2.setNumThreads(1)
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_pose_session = None
_lock = threading.Lock()


def get_pose_session() -> ort.InferenceSession:
    global _pose_session
    if _pose_session is None:
        with _lock:
            if _pose_session is None:
                onnx_path = os.path.join(MODELS_DIR, "yolov8n-pose.onnx")
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                _pose_session = ort.InferenceSession(
                    onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
    return _pose_session


def preload_models():
    try:
        get_pose_session()
    except Exception as e:
        print(f"[ONNX] Preload warning: {e}")


def _letterbox(img_bgr: np.ndarray, new_size: int = 640):
    h, w = img_bgr.shape[:2]
    scale = min(new_size / float(h), new_size / float(w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_size - nw) / 2.0
    pad_h = (new_size - nh) / 2.0
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    return blob, scale, left, top


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def detect_yolo(img_bgr: np.ndarray, model_name: str = "yolov8n-pose", conf_thresh: float = 0.25) -> dict:
    if img_bgr is None or img_bgr.size == 0:
        return {"boxes": [], "keypoints": [], "shape": [0, 0]}

    orig_h, orig_w = img_bgr.shape[:2]
    blob, scale, pad_x, pad_y = _letterbox(img_bgr, 640)
    sess = get_pose_session()
    inp_name = sess.get_inputs()[0].name
    with _lock:
        outputs = sess.run(None, {inp_name: blob})

    preds = outputs[0][0].T
    out = {"boxes": [], "keypoints": [], "shape": [orig_h, orig_w]}

    scores = preds[:, 4]
    valid = scores >= conf_thresh
    preds = preds[valid]
    scores = scores[valid]
    if len(preds) == 0:
        return out

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes_lb = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = _nms_xyxy(boxes_lb, scores, 0.45)

    for idx in keep:
        bx1 = float(np.clip((boxes_lb[idx, 0] - pad_x) / scale, 0, orig_w))
        by1 = float(np.clip((boxes_lb[idx, 1] - pad_y) / scale, 0, orig_h))
        bx2 = float(np.clip((boxes_lb[idx, 2] - pad_x) / scale, 0, orig_w))
        by2 = float(np.clip((boxes_lb[idx, 3] - pad_y) / scale, 0, orig_h))
        out["boxes"].append({"cls": 0, "conf": float(scores[idx]), "xyxy": [bx1, by1, bx2, by2]})

        kps_raw = preds[idx, 5:].reshape(17, 3)
        kps_list = []
        for kx, ky, kc in kps_raw:
            if kc >= 0.45:
                rx = float(np.clip((kx - pad_x) / scale, 0, orig_w))
                ry = float(np.clip((ky - pad_y) / scale, 0, orig_h))
                kps_list.append([rx, ry])
            else:
                kps_list.append([0.0, 0.0])
        out["keypoints"].append(kps_list)

    return out
