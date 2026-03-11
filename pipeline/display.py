# display.py — Real-time visualization and interactive display.
#
# Overlays the segmentation mask on the original camera frame,
# draws a HUD with FPS / latency / class legend, and handles
# keyboard interaction (quit, toggle overlays, etc.).
#
# NOTE: ``cv2.imshow`` must be called from the main thread on macOS.
#       On Linux it is more flexible, but we keep it on main for safety.
#
# Usage::
#
#     disp = StreamDisplay(window_name="LoLA Stream", class_names=["PG","TG"],
#                          colors=[(0,200,0),(0,0,200)])
#     keep_going = disp.render(result_packet)
#     disp.release()

import cv2
import time
import numpy as np

from typing import List, Optional, Tuple
from pipeline.packets import ResultPacket
from pipeline.monitor import tprint


class StreamDisplay:
    """Real-time display with segmentation overlay and metrics HUD.

    Renders:
        - Semi-transparent class-coloured mask on top of the camera frame.
        - FPS counter, per-stage latency, and class legend in a top bar.

    Keyboard controls:
        - ``q`` / ``ESC``: quit.
        - ``o``: toggle overlay on/off.
        - ``l``: toggle latency HUD.
    """

    def __init__(self,
                 window_name: str = "LoLA hsViT Stream",
                 class_names: Optional[List[str]] = None,
                 colors: Optional[List[Tuple[int, int, int]]] = None,
                 overlay_alpha: float = 0.45,
                 show_fps: bool = True,
                 show_latency: bool = True,
                 show_class_legend: bool = True,
                 max_display_width: int = 1280):
        """
        Args:
            window_name:       OpenCV window title.
            class_names:       Human-readable class labels.
            colors:            Per-class BGR colours.
            overlay_alpha:     Transparency of the segmentation overlay.
            show_fps:          Draw FPS counter.
            show_latency:      Draw per-stage latency breakdown.
            show_class_legend: Draw class-colour legend.
            max_display_width: Downscale for display if frame exceeds this width.
        """
        self._window = window_name
        self._class_names = class_names or []
        self._colors = colors or []
        self._alpha = overlay_alpha
        self._show_fps = show_fps
        self._show_latency = show_latency
        self._show_legend = show_class_legend
        self._max_width = max_display_width

        # FPS smoothing
        self._fps_window: List[float] = []
        self._fps_maxlen = 30
        self._last_render_time = time.monotonic()

        self._overlay_on = True  # toggled by 'o' key
        self._latency_on = show_latency

        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

    def render(self, result: ResultPacket) -> bool:
        """Render one frame and return True to continue, False to quit.

        Args:
            result: Completed ResultPacket with mask and latency data.

        Returns:
            ``False`` when the user presses quit (q / ESC).
        """
        now = time.monotonic()

        # -- FPS calculation --
        dt = now - self._last_render_time
        self._last_render_time = now
        if dt > 0:
            self._fps_window.append(1.0 / dt)
            if len(self._fps_window) > self._fps_maxlen:
                self._fps_window.pop(0)
        fps = sum(self._fps_window) / len(self._fps_window) if self._fps_window else 0.0

        # -- Build display canvas --
        canvas = result.original_image.copy()

        if self._overlay_on:
            canvas = self._draw_overlay(canvas, result.segmentation_mask)

        # -- HUD --
        y_offset = 25
        if self._show_fps:
            self._put_text(canvas, f"FPS: {fps:.1f}", (10, y_offset))
            y_offset += 25

        if self._latency_on and result.latency:
            for key, val in result.latency.items():
                self._put_text(canvas, f"{key}: {val:.1f} ms", (10, y_offset))
                y_offset += 22

        if self._show_legend:
            self._draw_legend(canvas, x=canvas.shape[1] - 150, y_start=25)

        # -- Resize if needed --
        if canvas.shape[1] > self._max_width:
            scale = self._max_width / canvas.shape[1]
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_LINEAR)

        cv2.imshow(self._window, canvas)

        # -- Keyboard handling --
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            return False
        if key == ord('o'):
            self._overlay_on = not self._overlay_on
        if key == ord('l'):
            self._latency_on = not self._latency_on
        return True

    def release(self) -> None:
        """Destroy the display window."""
        cv2.destroyWindow(self._window)

    # ---- drawing helpers ----

    def _draw_overlay(self, canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Alpha-blend class colours onto the camera frame."""
        H, W = canvas.shape[:2]
        mH, mW = mask.shape[:2]

        # Resize mask to frame size if needed (nearest to preserve labels)
        if (mH, mW) != (H, W):
            mask = cv2.resize(mask.astype(np.uint8), (W, H),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)

        overlay = np.zeros_like(canvas)
        blend_mask = np.zeros((H, W), dtype=np.uint8)

        for cls_id, color in enumerate(self._colors):
            region = mask == cls_id
            overlay[region] = color
            blend_mask[region] = 1

        # Only blend where there is a valid class prediction
        where = blend_mask.astype(bool)
        canvas[where] = cv2.addWeighted(canvas, 1 - self._alpha,
                                         overlay, self._alpha, 0)[where]
        return canvas

    def _draw_legend(self, canvas: np.ndarray, x: int, y_start: int) -> None:
        """Draw a class-colour legend in the upper-right corner."""
        for i, (name, color) in enumerate(zip(self._class_names, self._colors)):
            y = y_start + i * 25
            cv2.rectangle(canvas, (x, y - 12), (x + 16, y + 4), color, -1)
            self._put_text(canvas, name, (x + 22, y + 2), scale=0.5)

    @staticmethod
    def _put_text(canvas: np.ndarray, text: str, org: Tuple[int, int],
                  scale: float = 0.55, color: Tuple[int, int, int] = (255, 255, 255),
                  thickness: int = 1) -> None:
        """Put text with a dark shadow for readability over any background."""
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
