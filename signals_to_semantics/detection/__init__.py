"""Signal processing and braking-window detection."""

from .braking_events import detect_brakes, ego_kinematics, smooth_series

__all__ = ["detect_brakes", "ego_kinematics", "smooth_series"]
