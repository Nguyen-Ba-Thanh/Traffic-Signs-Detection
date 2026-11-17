import csv
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from traffic_web.processor import Options, SpeedCache, analyze_video, read_speed


def video(path, frames=6):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()


class FakeModel:
    names = {7: "Gioi han toc do"}

    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, frame, **kwargs):
        return [SimpleNamespace(boxes=self.boxes)]


def test_video_limit_ocr_and_csv(tmp_path):
    source = tmp_path / "input.avi"
    video(source)
    box = SimpleNamespace(cls=[7], conf=[0.9], xyxy=[[-2, -2, 35, 35]])
    reader = SimpleNamespace(readtext=lambda *a, **kw: ["60"])
    result = analyze_video(source, tmp_path / "out", FakeModel([box]), Options(max_seconds=1), reader)
    assert result["frames"] == result["detections"] == 5
    assert result["limit_reached"]
    with result["csv"].open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["speed_kmh"] == "60"
    assert rows[-1]["time_seconds"] == "0.8"
    assert rows[0]["x1"] == "0"
    assert result["ocr_calls"] == 1
    assert rows[0]["speed_source"] == "ocr"
    assert rows[-1]["speed_source"] == "cache"
    assert sum(result["timings"].values()) == pytest.approx(result["elapsed"])
    cap = cv2.VideoCapture(str(result["video"]))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
    cap.release()


def test_no_detections_and_eof(tmp_path):
    source = tmp_path / "input.avi"
    video(source, 2)
    result = analyze_video(source, tmp_path / "out", FakeModel([]))
    assert result["frames"] == 2
    assert result["detections"] == 0
    assert not result["limit_reached"]
    assert len(result["csv"].read_text(encoding="utf-8-sig").splitlines()) == 1


def test_invalid_input(tmp_path):
    with pytest.raises(ValueError, match="Không đọc được"):
        analyze_video(tmp_path / "missing.mp4", tmp_path / "out", FakeModel([]))


def test_ocr_rejects_invalid_speed():
    reader = SimpleNamespace(readtext=lambda *a, **kw: ["999", "6 0"])
    assert read_speed(reader, np.zeros((10, 10, 3), np.uint8)) == "60"


def test_cache_matches_positions_not_detection_order():
    values = iter(["30", "80", "60", "100"])
    reader = SimpleNamespace(readtext=lambda *a, **kw: [next(values)])
    cache = SpeedCache(3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    left, right = (0, 0, 20, 20), (60, 0, 80, 20)
    assert cache.read_frame(reader, frame, [left, right], 0)[0] == [("30", "ocr"), ("80", "ocr")]
    assert cache.read_frame(reader, frame, [right, left], 1) == ([("80", "cache"), ("30", "cache")], 0)
    assert cache.read_frame(reader, frame, [left, right], 3)[0] == [("60", "ocr"), ("100", "ocr")]


def test_cache_forgets_disappeared_sign():
    values = iter(["30", "80"])
    reader = SimpleNamespace(readtext=lambda *a, **kw: [next(values)])
    cache = SpeedCache(10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (0, 0, 20, 20)
    cache.read_frame(reader, frame, [box], 0)
    cache.read_frame(reader, frame, [], 1)
    assert cache.read_frame(reader, frame, [box], 2) == ([("80", "ocr")], 1)


def test_cache_does_not_reuse_ambiguous_match():
    reader = SimpleNamespace(readtext=lambda *a, **kw: ["60"])
    cache = SpeedCache(10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    box = (0, 0, 20, 20)
    cache.read_frame(reader, frame, [box], 0)
    assert cache.read_frame(reader, frame, [box, box], 1)[1] == 2
