"""Measure the file pipeline with the same video/settings on CPU and optional GPU."""
import argparse
import json
import tempfile
from statistics import median
from pathlib import Path

import cv2
from ultralytics import YOLO

from traffic_web.devices import device_info
from traffic_web.processor import Options, analyze_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    parser.add_argument("--model", type=Path, default=Path("UI/models/best.pt"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.seconds, args.repeats) < 1:
        parser.error("seconds and repeats must be positive")
    info = device_info()
    cap = cv2.VideoCapture(str(args.video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Cannot read video")
    height, width = frame.shape[:2]
    scale = min(1, 960 / width, 960 / height)
    frame = cv2.resize(frame, (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)))
    results = []
    for device in (["cpu", "0"] if info["available"] else ["cpu"]):
        model = YOLO(str(args.model))
        for _ in range(2):
            model.predict(frame, imgsz=args.imgsz, device=device, verbose=False)
        for run in range(args.repeats):
            with tempfile.TemporaryDirectory(prefix="traffic-benchmark-") as work:
                result = analyze_video(args.video, work, model,
                                       Options(image_size=args.imgsz, max_seconds=args.seconds, device=device))
                result = {key: value for key, value in result.items() if key not in ("video", "csv", "counts")}
                result["run"] = run + 1
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {device: {"median_pipeline_fps": median(row["processing_fps"] for row in results if row["device"] == device),
                        "median_yolo_ms_per_frame": median(row["timings"]["YOLO"] / row["frames"] * 1000
                                                           for row in results if row["device"] == device)}
               for device in {row["device"] for row in results}}
    report = {"hardware": info, "image_size": args.imgsz, "ocr": False, "warmup_runs": 2,
              "note": "File pipeline benchmark without UI preview; not realtime camera FPS.", "summary": summary, "results": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
