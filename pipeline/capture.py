# capture.py — Device capture layer.
#
# Wraps camera / video file into a uniform FramePacket producer.
# A dedicated capture thread grabs frames at device FPS, storing
# only the latest frame so consumers never block on stale data.
#
# Supports:
#   - USB webcam (source = int device index)
#   - Video file (source = str file path)
#   - Reconnection on transient failures
#
# Usage::
#
#     src = CaptureSource(source=0, width=640, height=480, fps=30)
#     src.open()
#     frame = src.get_frame()   # returns FramePacket or None
#     src.release()

import cv2
import time
import threading
import numpy as np

from typing import Optional, Union
from pipeline.packets import FramePacket
from pipeline.monitor import tprint


class CaptureSource:
    """Camera or video capture with latest-frame semantics.

    A background thread continuously grabs frames from the device.
    ``get_frame()`` always returns the *most recent* frame, discarding
    any older buffered frames to maintain real-time freshness.
    """

    def __init__(self,
                 source: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 reconnect_attempts: int = 5,
                 reconnect_delay_sec: float = 1.0):
        """
        Args:
            source: Device index (int) or video file path (str).
            width:  Capture width in pixels.
            height: Capture height in pixels.
            fps:    Target capture frame rate.
            reconnect_attempts: Max consecutive reconnect retries.
            reconnect_delay_sec: Delay between reconnect attempts.
        """
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay_sec

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Latest-frame buffer (protected by lock)
        self._lock = threading.Lock()
        self._latest_frame: Optional[FramePacket] = None
        self._frame_counter = 0

    def open(self) -> bool:
        """Open the capture device and start the grab thread.

        Returns:
            True if the device was opened successfully.
        """
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            tprint(f"[capture] FAILED to open source: {self._source}")
            return False

        # Configure device (ignored silently for video files)
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        tprint(f"[capture] opened: {actual_w}x{actual_h} @ {actual_fps:.1f} fps "
               f"(source={self._source})")

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True,
                                        name="CaptureThread")
        self._thread.start()
        return True

    def get_frame(self) -> Optional[FramePacket]:
        """Return the most recent frame, or None if nothing available."""
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None  # consumed
            return frame

    def release(self) -> None:
        """Stop the capture thread and release the device."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        tprint("[capture] released")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # internal methods
    def _grab_loop(self) -> None:
        """Continuously grab frames on a background thread."""
        consecutive_failures = 0

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._try_reconnect():
                    break
                continue

            ret, frame = self._cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > self._reconnect_attempts:
                    tprint("[capture] max failures reached, stopping")
                    break
                # For video file: loop back to beginning
                if isinstance(self._source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            consecutive_failures = 0
            self._frame_counter += 1
            packet = FramePacket(
                frame_id=self._frame_counter,
                timestamp=time.monotonic(),
                image=frame,
                camera_meta={
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2] if frame.ndim == 3 else 1,
                },
            )

            with self._lock:
                self._latest_frame = packet

    def _try_reconnect(self) -> bool:
        """Attempt to reopen the device after a failure."""
        for attempt in range(1, self._reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False
            tprint(f"[capture] reconnect attempt {attempt}/{self._reconnect_attempts}")
            time.sleep(self._reconnect_delay)
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self._source)
            if self._cap.isOpened():
                tprint("[capture] reconnected")
                return True
        tprint("[capture] reconnect failed, giving up")
        return False
