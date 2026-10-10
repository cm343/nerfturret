"""Host health for the UI: CPU temperature, throttling flags, load."""

import os
import subprocess
import time

_THROTTLE_BITS = {
    0: 'under-voltage now', 1: 'frequency capped now', 2: 'throttled now',
    3: 'soft temp limit now', 16: 'under-voltage occurred', 17: 'frequency cap occurred',
    18: 'throttling occurred', 19: 'soft temp limit occurred',
}
_cache: dict = {'t': 0.0, 'throttled': None}


def cpu_temp() -> float | None:
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return round(int(f.read().strip()) / 1000, 1)
    except (OSError, ValueError):
        return None


def throttled() -> list[str] | None:
    """Decoded `vcgencmd get_throttled` flags (cached for 5 s). None off-Pi."""
    now = time.monotonic()
    if now - _cache['t'] < 5:
        return _cache['throttled']
    _cache['t'] = now
    try:
        out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                             text=True, timeout=1).stdout
        value = int(out.strip().split('=')[1], 16)
        _cache['throttled'] = [msg for bit, msg in _THROTTLE_BITS.items() if value >> bit & 1]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        _cache['throttled'] = None
    return _cache['throttled']


def snapshot() -> dict:
    try:
        load = round(os.getloadavg()[0], 2)
    except OSError:
        load = None
    return {'cpu_temp': cpu_temp(), 'throttled': throttled(), 'load': load}
