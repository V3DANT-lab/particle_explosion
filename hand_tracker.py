"""
Hand tracking module using MediaPipe.

Provides the HandTracker class which wraps MediaPipe and produces, per
camera frame, a structured HandState:

  * landmarks      : (21, 2) normalized image-space coords (x, y in [0, 1])
  * palm           : (2,) normalized image-space coords of the palm centroid
  * gesture        : one of "none", "fist", "open"
  * fist_progress  : 0..1 (how long the fist has been held - used to show a
                     charging indicator before the explosion arms)
  * hand_present   : bool

Gesture heuristics
------------------
  * Compute the average fingertip -> palm-center distance for the 5
    fingertips (4, 8, 12, 16, 20).
  * Compute the same distance for the 4 MCP joints (5, 9, 13, 17) as a
    per-hand "scale reference".
  * If the average fingertip distance is LESS than the MCP reference the
    hand is closing - call it a "fist".
  * If the average fingertip distance is MORE than ~1.55x the MCP
    reference the hand is "open".
  * Otherwise the pose is indeterminate.

Because we compare ratios rather than absolute pixel distances, the
heuristic is robust across hands of different sizes and distances.

MediaPipe 1.x note
-----------------
MediaPipe 1.0 removed the legacy ``mp.solutions.*`` API, so this module
uses the modern Tasks API (``HandLandmarker``). That needs the
``hand_landmarker.task`` model file, which ``setup.sh`` downloads into
``./models/``. If the model is missing, ``HandTracker.available`` stays
False and the rest of the app still runs (just without hand tracking)
rather than crashing at start-up.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# 21 MediaPipe hand landmarks (in order):
#   0: wrist
#   1-4: thumb (CMC, MCP, IP, TIP)
#   5-8: index
#   9-12: middle
#   13-16: ring
#   17-20: pinky
FINGERTIPS  = (4, 8, 12, 16, 20)
MCP_JOINTS  = (5, 9, 13, 17)

# Joint indices used to draw the skeleton.
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]

# Where the model lives. Overridable for unusual installs.
MODEL_ENV_VAR = "HAND_LANDMARKER_MODEL"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


def model_path() -> Path:
    """Resolve the hand_landmarker.task model path."""
    env = os.environ.get(MODEL_ENV_VAR)
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    return here / "models" / "hand_landmarker.task"


@dataclass
class HandState:
    landmarks: Optional[np.ndarray] = None     # (21, 2) normalized
    palm: Optional[np.ndarray] = None          # (2,) normalized
    gesture: str = "none"                       # "none" | "fist" | "open"
    fist_progress: float = 0.0                  # 0..1, how charged the fist is
    hand_present: bool = False

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (landmarks_px, palm_px) suitable for drawing."""
        if not self.hand_present or self.landmarks is None:
            return (np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32))
        return self.landmarks, self.palm.reshape(1, 2)


class HandTracker:
    """MediaPipe Tasks hand-landmark wrapper with a small stability filter."""

    # How long the user must hold a fist before the explosion arms.
    FIST_HOLD_TIME = 0.30

    def __init__(self,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 delegate: str = "GPU") -> None:

        self.available = False
        self.error: Optional[str] = None

        path = model_path()
        if not path.exists():
            self.error = (f"hand tracking model not found at {path}. "
                          "Run ./setup.sh to download it.")
            print(f"[warn] {self.error} Hand tracking disabled.", flush=True)
            return

        self._landmarker = None
        last_exc = None
        # GPU delegate is faster but unavailable on some drivers / headless
        # setups; fall back to CPU rather than failing outright.
        for use_delegate in ([delegate, "CPU"] if delegate == "GPU"
                             else [delegate]):
            try:
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                if use_delegate != "CPU":
                    options.delegate = (
                        vision.BaseOptions.Delegate.GPU
                        if use_delegate == "GPU"
                        else vision.BaseOptions.Delegate.CPU)
                self._landmarker = vision.HandLandmarker.create_from_options(
                    options)
                print(f"[tracker] MediaPipe HandLandmarker ready "
                      f"(delegate={use_delegate})", flush=True)
                break
            except Exception as exc:      # pragma: no cover - driver specific
                last_exc = exc
                self._landmarker = None

        if self._landmarker is None:
            self.error = f"could not start MediaPipe HandLandmarker: {last_exc}"
            print(f"[warn] {self.error} Hand tracking disabled.", flush=True)
            return

        self.available = True
        self._last_ts_ms: int = -1
        self._last_gesture: str = "none"
        self._fist_start_time: Optional[float] = None

    # ------------------------------------------------------------------ public
    def process(self, frame_bgr: np.ndarray) -> HandState:
        """Run hand detection on a BGR frame and return a HandState."""
        state = HandState()
        if not self.available:
            return state

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # The VIDEO running mode requires strictly increasing timestamps.
        ts_ms = int(time.monotonic() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        try:
            result = self._landmarker.detect_for_video(image, ts_ms)
        except Exception as exc:      # pragma: no cover - runtime guard
            self.error = f"detection failed: {exc}"
            return state

        if result.hand_landmarks:
            pts = result.hand_landmarks[0]
            arr = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
            # Mirror X so the on-screen hand matches the user's mirror image
            # (the camera is shown mirrored to the user).
            arr[:, 0] = 1.0 - arr[:, 0]

            palm = self._palm_center(arr)
            gesture = self._classify_gesture(arr, palm)

            # A gesture must persist before we accept the change, otherwise
            # we hold the previous one to avoid flicker.
            if gesture in ("fist", "open"):
                self._last_gesture = gesture
            else:
                gesture = self._last_gesture

            # Track fist-hold time to compute the charging progress.
            now = time.monotonic()
            if gesture == "fist":
                if self._fist_start_time is None:
                    self._fist_start_time = now
                held = now - self._fist_start_time
                fist_progress = min(1.0, held / self.FIST_HOLD_TIME)
            else:
                self._fist_start_time = None
                fist_progress = 0.0

            state.landmarks = arr
            state.palm = palm
            state.gesture = gesture
            state.fist_progress = fist_progress
            state.hand_present = True
        else:
            # No hand detected - drop the charge so it cannot stay armed.
            self._fist_start_time = None
            self._last_gesture = "none"
            state.gesture = "none"
            state.fist_progress = 0.0
            state.hand_present = False

        return state

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None
        self.available = False

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _palm_center(landmarks: np.ndarray) -> np.ndarray:
        """Average of wrist + 4 MCP joints gives a stable palm centroid."""
        idxs = [0, 5, 9, 13, 17]
        return landmarks[idxs].mean(axis=0).astype(np.float32)

    @staticmethod
    def _classify_gesture(landmarks: np.ndarray,
                          palm: np.ndarray) -> str:
        """Return 'fist', 'open', or 'indeterminate'."""
        # Distance from each fingertip to palm center
        tip_dists = np.linalg.norm(landmarks[list(FINGERTIPS)] - palm, axis=1)
        # Distance from each MCP joint to palm center (per-hand scale ref)
        mcp_dists = np.linalg.norm(landmarks[list(MCP_JOINTS)] - palm, axis=1)

        ref = float(np.mean(mcp_dists))
        if ref < 1e-4:
            return "indeterminate"

        avg_tip = float(np.mean(tip_dists))
        ratio = avg_tip / ref

        # Also check finger extension: a finger is "extended" if its tip
        # is further from the wrist than its MCP joint.
        wrist = landmarks[0]
        finger_extended = []
        for tip, mcp in zip(FINGERTIPS[1:], MCP_JOINTS):   # skip thumb
            tip_d = np.linalg.norm(landmarks[tip] - wrist)
            mcp_d = np.linalg.norm(landmarks[mcp] - wrist)
            finger_extended.append(tip_d > mcp_d * 1.1)
        extended_count = sum(finger_extended)

        if ratio < 1.1 and extended_count <= 1:
            return "fist"
        if ratio > 1.55 and extended_count >= 4:
            return "open"
        return "indeterminate"
