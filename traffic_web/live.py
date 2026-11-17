"""Browser video callbacks. No Streamlit calls from media worker threads."""
import logging
import threading
import time
from collections import deque
from typing import Any

import av
import cv2

from traffic_web.processor import SpeedCache, analyze_frame


class LiveDetector:
    def __init__(self, model, options, inference_lock, reader=None):
        self.model, self.options, self.inference_lock, self.reader = model, options, inference_lock, reader
        self._state_lock = threading.Lock()
        self._times = deque(maxlen=60)
        self._rows = deque(maxlen=100)
        self._stats: dict[str, Any] = {"processed": 0, "dropped": 0, "processing_ms": 0.0, "error": "", "busy": False}
        self._cache = SpeedCache(options.ocr_interval)
        self._index = 0
        self._last_frame = 0.0
        self._start = time.perf_counter()

    def reset(self):
        # Invoked by the media worker after a track ends.
        with self._state_lock:
            self._cache = SpeedCache(self.options.ocr_interval)
            self._times.clear()
            self._rows.clear()
            self._index = 0
            self._last_frame = 0.0
            self._start = time.perf_counter()
            self._stats = {"processed": 0, "dropped": 0, "processing_ms": 0.0, "error": "", "busy": False}

    def snapshot(self):
        with self._state_lock:
            state = dict(self._stats)
            times = list(self._times)
            state["fps"] = ((len(times) - 1) / (times[-1] - times[0])
                            if len(times) > 1 and time.perf_counter() - times[-1] < 2 else 0.0)
            state["rows"] = list(self._rows)
            return state

    async def process_queued(self, frames):
        return self.process_frames(frames)

    def process_frames(self, frames):
        if not frames:
            return []
        source = frames[-1]  # Drop backlog rather than grow camera delay.
        frame = source.to_ndarray(format="bgr24")
        now = time.perf_counter()
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / w, 640 / h)
        frame = cv2.resize(frame, (max(2, int(w * scale)), max(2, int(h * scale))))
        with self._state_lock:
            self._index += len(frames)
            self._stats["dropped"] += len(frames) - 1
        if not self.inference_lock.acquire(blocking=False):
            with self._state_lock:
                self._stats["dropped"] += 1
                self._stats["busy"] = True
            cv2.putText(frame, "Server busy", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)
        else:
            try:
                if now - self._last_frame > 0.75:
                    self._cache = SpeedCache(self.options.ocr_interval)
                self._last_frame = now
                annotated, rows, _, _, _ = analyze_frame(
                    frame, self.model, self.options, self._cache, self._index, 30, self.reader)
                frame = annotated
                duration = time.perf_counter() - now
                for row in rows:
                    row["time_seconds"] = round(now - self._start, 3)
                with self._state_lock:
                    self._stats.update(processed=self._stats["processed"] + 1, processing_ms=duration * 1000,
                                       error="", busy=False)
                    self._times.append(time.perf_counter())
                    self._rows.extend(rows)
                cv2.putText(frame, f"{duration * 1000:.0f} ms/frame", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 230, 100), 2)
            except Exception as exc:
                logging.getLogger(__name__).exception("Camera inference failed")
                with self._state_lock:
                    self._stats["error"] = str(exc)
                cv2.putText(frame, "Inference error - see dashboard", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            finally:
                self.inference_lock.release()
        result = av.VideoFrame.from_ndarray(frame, format="bgr24")
        result.pts = source.pts
        if source.time_base is not None:
            result.time_base = source.time_base
        return [result]
