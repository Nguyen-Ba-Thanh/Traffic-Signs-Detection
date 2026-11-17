"""Video inference independent of Streamlit and the desktop UI."""
import csv
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import cv2
import imageio_ffmpeg

VALID_SPEEDS = {str(n) for n in (5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120)}
FIELDS = ["frame", "time_seconds", "class_id", "class_name", "confidence", "speed_kmh", "x1", "y1", "x2", "y2", "speed_source"]


@dataclass(frozen=True)
class Options:
    confidence: float = 0.5
    image_size: int = 640
    max_seconds: int = 60
    max_width: int = 960
    device: str = "cpu"
    ocr_interval: int = 10
    preview_interval: float = 0.5


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


class SpeedCache:
    """One-to-one spatial matching against only the preceding frame, not box order.

    Ambiguous matches trigger fresh OCR. This is a short-lived heuristic, not tracking.
    """
    def __init__(self, interval):
        if interval < 1:
            raise ValueError("OCR interval must be at least 1")
        self.interval = interval
        self.previous = []

    def read_frame(self, reader, frame, boxes, frame_index):
        overlaps = [[box_iou(box, item["bbox"]) for item in self.previous] for box in boxes]
        current = []
        output = []
        used = set()
        calls = 0
        for index, bbox in enumerate(boxes):
            candidates = [i for i, score in enumerate(overlaps[index]) if score >= 0.5]
            match = None
            if len(candidates) == 1:
                candidate = candidates[0]
                contenders = sum(row[candidate] >= 0.5 for row in overlaps)
                if contenders == 1 and candidate not in used:
                    match = self.previous[candidate]
                    used.add(candidate)
            if match is not None and frame_index - match["last_read"] < self.interval:
                value, last_read, source = match["value"], match["last_read"], "cache"
            else:
                x1, y1, x2, y2 = bbox
                value = read_speed(reader, frame[y1:y2, x1:x2])
                last_read, source = frame_index, "ocr"
                calls += 1
            current.append({"bbox": bbox, "value": value, "last_read": last_read})
            output.append((value, source if value else ""))
        self.previous = current
        return output, calls


def read_speed(reader, crop):
    if crop.size == 0:
        return ""
    for text in reader.readtext(crop, detail=0, allowlist="0123456789"):
        value = text.strip().replace(" ", "")
        if value in VALID_SPEEDS:
            return value
    return ""


def analyze_frame(frame, model, options, speed_cache, processed, fps, reader=None):
    height, width = frame.shape[:2]
    rows = []
    prediction_start = time.perf_counter()
    result = model.predict(frame, conf=options.confidence, imgsz=options.image_size,
                           device=options.device, verbose=False)[0]
    # Copy all boxes once: avoids one CUDA synchronization per scalar.
    boxes = result.boxes.cpu() if hasattr(result.boxes, "cpu") else result.boxes
    yolo_seconds = time.perf_counter() - prediction_start
    records = []
    for box in boxes:
        cid, confidence = int(box.cls[0]), float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (max(0, min(width, x1)), max(0, min(height, y1)),
                max(0, min(width, x2)), max(0, min(height, y2)))
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            records.append((cid, confidence, bbox))
    speed_boxes = [bbox for cid, _, bbox in records if model.names[cid] == "Gioi han toc do"]
    ocr_start = time.perf_counter()
    speeds, calls = speed_cache.read_frame(reader, frame, speed_boxes, processed) if reader else ([], 0)
    ocr_seconds = time.perf_counter() - ocr_start
    ocr_calls = calls
    speed_values = iter(speeds)
    annotated = frame.copy()
    for cid, confidence, (x1, y1, x2, y2) in records:
        name = model.names[cid]
        speed, speed_source = next(speed_values) if reader and name == "Gioi han toc do" else ("", "")
        label = f"{name} {confidence:.2f}" + (f" | {speed} km/h" if speed else "")
        color = (60, 180, 255) if speed else (90, 215, 95)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        rows.append(dict(zip(FIELDS, [processed, round(processed / fps, 3), cid, name,
                                      round(confidence, 4), speed, x1, y1, x2, y2, speed_source])))

    return annotated, rows, yolo_seconds, ocr_seconds, ocr_calls


def analyze_video(source, destination, model, options=Options(), reader=None, progress=None):
    """Process sequential frames; output has no audio. Timings include CPU box transfer."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    start = time.perf_counter()
    counts = Counter()
    processed = detections = 0
    yolo_seconds = ocr_seconds = preview_seconds = 0.0
    ocr_calls = 0
    last_preview = 0.0
    speed_cache = SpeedCache(options.ocr_interval)
    raw_path = destination / "annotated.avi"
    csv_path = destination / "detections.csv"
    output_path = destination / "result.mp4"
    try:
        if not cap.isOpened():
            raise ValueError("Không đọc được video. Hãy thử file MP4 mã hóa H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video không có FPS hợp lệ; hãy chuyển sang MP4 trước.")
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        limit = max(1, int(fps * options.max_seconds))
        expected = min(int(total), limit) if math.isfinite(total) and total > 0 else limit
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            log = csv.DictWriter(handle, fieldnames=FIELDS)
            log.writeheader()
            while processed < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                scale = min(1.0, options.max_width / w, 960 / h)
                width, height = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
                frame = cv2.resize(frame, (width, height))
                if writer is None:
                    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Không khởi tạo được bộ ghi video.")
                annotated, rows, yolo_time, ocr_time, calls = analyze_frame(
                    frame, model, options, speed_cache, processed, fps, reader)
                yolo_seconds += yolo_time
                ocr_seconds += ocr_time
                ocr_calls += calls
                log.writerows(rows)
                counts.update(row["class_name"] for row in rows)
                detections += len(rows)
                writer.write(annotated)
                processed += 1
                if progress and (processed == 1 or time.perf_counter() - last_preview >= options.preview_interval):
                    preview_start = time.perf_counter()
                    progress(min(processed / expected, 0.99), annotated, processed)
                    preview_seconds += time.perf_counter() - preview_start
                    last_preview = time.perf_counter()
        if not processed:
            raise ValueError("Video không có frame đọc được.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    encode_start = time.perf_counter()
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw_path),
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
                       check=True, capture_output=True, timeout=300)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("Không mã hóa được MP4 H.264. Hãy thử video ngắn hơn.") from exc
    finally:
        raw_path.unlink(missing_ok=True)
    encode_seconds = time.perf_counter() - encode_start
    elapsed = time.perf_counter() - start
    timings = {"YOLO": yolo_seconds, "OCR": ocr_seconds, "Preview": preview_seconds,
               "MP4 encoding": encode_seconds,
               "Read / draw / write": max(0.0, elapsed - yolo_seconds - ocr_seconds - preview_seconds - encode_seconds)}
    return {"video": output_path, "csv": csv_path, "frames": processed, "detections": detections,
            "counts": dict(counts), "seconds": processed / fps,
            "elapsed": elapsed, "limit_reached": processed >= limit, "timings": timings,
            "processing_fps": processed / elapsed, "ocr_calls": ocr_calls,
            "device": options.device, "source_fps": fps}
