#!/usr/bin/env python3
"""
Manual referencing and persisted calibration for the turret steppers.

There are no endstops, so every axis is referenced by hand: jog it to a visual
reference (e.g. barrel level), declare that position zero, and keep later moves
inside soft limits saved relative to it.

  * Calibration — a small JSON file (default config/calibration.json) holding
    per-axis values between runs. Positions are raw Stepper.position values
    in µsteps, relative to that axis's zero.

        {
          "tilt": {"min_steps": -400, "max_steps": 800,
                   "last_position": 120, "last_position_saved_at": "…"}
        }

  * clamp_steps() shortens a relative move so it ends inside the saved limits.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'calibration.json'
)


class Calibration:
    """Per-axis calibration dicts, loaded from and saved to a JSON file."""

    def __init__(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        self.path = path
        self.data: dict = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def axis(self, name: str) -> dict:
        """The (mutable) calibration dict for one axis, created if missing."""
        return self.data.setdefault(name, {})

    def save(self) -> None:
        """Write atomically so a crash mid-save can't leave a corrupt file."""
        d = os.path.dirname(self.path) or '.'
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix='.calibration-', suffix='.json')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
                f.write('\n')
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise


def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')


def limits(cal: dict) -> tuple[Optional[int], Optional[int]]:
    return cal.get('min_steps'), cal.get('max_steps')


def clamp_steps(position: int, steps: int, cal: dict) -> int:
    """Shorten a relative move so it ends inside the axis's soft limits."""
    lo, hi = limits(cal)
    target = position + steps
    # Only the limit in the direction of travel applies. If already beyond it,
    # the move is refused (min/max with position), never reversed.
    if steps > 0 and hi is not None:
        target = min(target, max(hi, position))
    if steps < 0 and lo is not None:
        target = max(target, min(lo, position))
    return target - position
