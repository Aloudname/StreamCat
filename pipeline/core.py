# orchestrator.py — Pipeline orchestration and queue management.
#
# Manages the capture -> preprocess -> infer -> postprocess -> display
# dataflow with bounded queues, drop-oldest back-pressure, and
# graceful shutdown via a shared stop event.
#
# Architecture::
#
#     [CaptureThread] ──> capture_q ──> [WorkerThread] ──> result_q ──> [MainThread: display]
#
#     CaptureThread: grabs frames from device, writes latest FramePacket.
#     WorkerThread:  preprocess + batched inference + stitch (sequential per frame).
#     MainThread:    pulls ResultPacket and renders (OpenCV imshow needs main thread).
#
# Queue semantics:
#     ``LatestQueue`` with bounded size — when full the oldest item is
#     silently discarded (real-time: freshness > completeness).
#
# Usage::
#
#     pipeline = StreamPipeline("config/config_stream.yaml")
#     pipeline.run()       # blocking; press 'q' to quit


from munch import Munch
from typing import Optional
import time, numpy as np, threading, collections

from pipeline.monitor import tprint
from pipeline.capture import CaptureSource
from pipeline.display import StreamDisplay
from pipeline.postprocess import TileStitcher
from pipeline.preprocess import StreamPreprocessor
from pipeline.inference import InferClient, batched_infer
from pipeline.monitor import LatencyTracker, StreamMonitor
from pipeline.packets import FramePacket, InferPacket, ResultPacket
# ---------------------------------------------------------------------------
# Drop-oldest bounded queue
# ---------------------------------------------------------------------------

class LatestQueue:
    """Bounded thread-safe queue that discards the oldest item when full.

    Designed for real-time pipelines where freshness matters more
    than processing every single frame.
    """

    def __init__(self, maxsize: int = 2):
        self._deque = collections.deque(maxlen=maxsize)
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)

    def put(self, item) -> None:
        """Enqueue *item*, silently dropping the oldest if at capacity."""
        with self._mutex:
            self._deque.append(item)
            self._not_empty.notify()

    def get(self, timeout: Optional[float] = None):
        """Block until an item is available, then return it (FIFO).

        Returns ``None`` on timeout.
        """
        with self._not_empty:
            while not self._deque:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            return self._deque.popleft()

    def get_latest(self):
        """Non-blocking: return the *newest* item and discard older ones.

        Returns ``None`` if the queue is empty.
        """
        with self._mutex:
            if self._deque:
                item = self._deque[-1]
                self._deque.clear()
                return item
            return None

    def qsize(self) -> int:
        with self._mutex:
            return len(self._deque)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class StreamPipeline:
    """Main orchestrator for the real-time streaming pipeline.

    Instantiation wires up all stages from a single YAML config.
    ``run()`` is blocking and drives the display loop on the main thread.
    """

    def __init__(self, config: Munch):
        """
        Args:
            config: Munch loaded from ``config/config_stream.yaml``.
        """
        self._cfg = config
        self._stop_event = threading.Event()

        # Queues
        qsize = config.pipeline.queue_maxsize
        self._capture_q = LatestQueue(maxsize=qsize)
        self._result_q = LatestQueue(maxsize=qsize)

        # Stages
        self._capture = CaptureSource(
            source=config.capture.source,
            width=config.capture.width,
            height=config.capture.height,
            fps=config.capture.fps,
            reconnect_attempts=config.capture.reconnect_attempts,
            reconnect_delay_sec=config.capture.reconnect_delay_sec,
        )

        self._preprocessor = StreamPreprocessor(
            patch_size=config.preprocess.patch_size,
            stride=config.preprocess.stride,
            normalize_mode=config.preprocess.normalize_mode,
            preprocessor_path=config.preprocess.get("preprocessor_path", ""),
        )

        self._infer_client = InferClient.create(config.inference)

        self._stitcher = TileStitcher(
            patch_size=config.preprocess.patch_size,
            stride=config.preprocess.stride,
            num_classes=config.postprocess.num_classes,
            fusion_mode=config.postprocess.fusion_mode,
            confidence_threshold=config.postprocess.confidence_threshold,
            morphology_kernel=config.postprocess.morphology_kernel,
        )

        colors = [tuple(c) for c in config.display.colormap]
        self._display = StreamDisplay(
            window_name=config.display.window_name,
            class_names=config.postprocess.class_names,
            colors=colors,
            overlay_alpha=config.display.overlay_alpha,
            show_fps=config.display.show_fps,
            show_latency=config.display.show_latency,
            show_class_legend=config.display.show_class_legend,
            max_display_width=config.display.max_display_width,
        )

        # Monitor
        self._tracker = LatencyTracker(
            window_size=config.monitor.latency_window)
        self._monitor = StreamMonitor(
            tracker=self._tracker,
            stop_event=self._stop_event,
            interval_sec=config.monitor.log_interval_sec,
        ) if config.monitor.enable else None

        tprint("[orchestrator] pipeline assembled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all threads and run the display loop (blocking).

        Returns when the user presses 'q' / ESC or the pipeline errors out.
        """
        if not self._capture.open():
            raise RuntimeError("Failed to open capture device")

        # Warmup: discard first N frames to let camera auto-expose
        warmup = self._cfg.pipeline.warmup_frames
        if warmup > 0:
            tprint(f"[orchestrator] warming up ({warmup} frames)...")
            for _ in range(warmup):
                self._capture.get_frame()
                time.sleep(0.05)

        # Start threads
        self._stop_event.clear()

        capture_thread = threading.Thread(target=self._capture_loop,
                                          daemon=True, name="CaptureLoop")
        worker_thread = threading.Thread(target=self._worker_loop,
                                         daemon=True, name="WorkerLoop")
        capture_thread.start()
        worker_thread.start()

        if self._monitor:
            self._monitor.start()

        tprint("[orchestrator] pipeline running — press 'q' to quit")

        try:
            self._display_loop()
        except KeyboardInterrupt:
            tprint("[orchestrator] interrupted")
        finally:
            self._stop_event.set()
            capture_thread.join(timeout=2.0)
            worker_thread.join(timeout=5.0)
            if self._monitor:
                self._monitor.stop()
            self._capture.release()
            self._display.release()
            tprint("[orchestrator] shutdown complete")

    # ------------------------------------------------------------------
    # Thread bodies
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background: continuously grab the latest frame and enqueue."""
        while not self._stop_event.is_set():
            frame = self._capture.get_frame()
            if frame is not None:
                self._capture_q.put(frame)
            else:
                time.sleep(0.001)  # brief yield if no frame yet

    def _worker_loop(self) -> None:
        """Background: preprocess -> infer -> stitch, per frame."""
        batch_size = self._cfg.inference.batch_size

        while not self._stop_event.is_set():
            frame = self._capture_q.get(timeout=0.1)
            if frame is None:
                continue

            try:
                # Preprocess
                t0 = time.monotonic()
                infer_pkt = self._preprocessor.process(frame)
                prep_ms = (time.monotonic() - t0) * 1000.0
                self._tracker.record("preprocess", prep_ms)

                # Inference (batched)
                t1 = time.monotonic()
                infer_pkt.raw_output = batched_infer(
                    self._infer_client, infer_pkt.tiles, batch_size=batch_size)
                infer_ms = (time.monotonic() - t1) * 1000.0
                infer_pkt.infer_ms = infer_ms
                self._tracker.record("inference", infer_ms)

                # Postprocess (stitch)
                t2 = time.monotonic()
                result = self._stitcher.stitch(infer_pkt)
                post_ms = (time.monotonic() - t2) * 1000.0
                self._tracker.record("postprocess", post_ms)

                self._result_q.put(result)

            except Exception as e:
                tprint(f"[worker] error: {e}")

    def _display_loop(self) -> None:
        """Main thread: pull results and render."""
        while not self._stop_event.is_set():
            result = self._result_q.get(timeout=0.05)
            if result is None:
                continue

            self._tracker.tick_frame()
            self._tracker.record("e2e",
                                 (time.monotonic() - result.timestamp) * 1000.0)

            if not self._display.render(result):
                break  # user pressed quit
