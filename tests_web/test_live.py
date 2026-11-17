import asyncio
import threading
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np

from traffic_web.live import LiveDetector
from traffic_web.processor import Options


class Model:
    names = {0: "Cam re trai"}

    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        return [SimpleNamespace(boxes=[SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[0, 10, 20, 30]])])]


def make_frame(pts=0):
    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24")
    frame.pts, frame.time_base = pts, Fraction(1, 90000)
    return frame


def test_latest_frame_and_timestamps():
    model = Model()
    worker = LiveDetector(model, Options(), threading.Lock())
    output = asyncio.run(worker.process_queued([make_frame(0), make_frame(3000), make_frame(6000)]))
    assert model.calls == 1
    assert len(output) == 1 and output[0].pts == 6000
    assert output[0].time_base == Fraction(1, 90000)
    state = worker.snapshot()
    assert state["processed"] == 1 and state["dropped"] == 2
    assert state["rows"][0]["class_name"] == "Cam re trai"


def test_busy_does_not_queue_or_infer():
    model, lock = Model(), threading.Lock()
    worker = LiveDetector(model, Options(), lock)
    lock.acquire()
    try:
        assert len(worker.process_frames([make_frame()])) == 1
        assert model.calls == 0 and worker.snapshot()["busy"]
    finally:
        lock.release()
    worker.process_frames([make_frame()])
    assert model.calls == 1 and not worker.snapshot()["busy"]


def test_frame_without_media_timestamp():
    worker = LiveDetector(Model(), Options(), threading.Lock())
    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24")
    assert worker.process_frames([frame])[0].pts is None


def test_error_releases_lock_and_reset_clears_session():
    lock = threading.Lock()
    bad = SimpleNamespace(predict=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("test failure")))
    worker = LiveDetector(bad, Options(), lock)
    assert worker.process_frames([make_frame()])
    assert not lock.locked()
    assert "test failure" in worker.snapshot()["error"]
    worker.reset()
    assert not worker.snapshot()["error"] and not worker.snapshot()["rows"]
