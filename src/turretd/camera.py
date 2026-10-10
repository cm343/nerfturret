"""
Camera service for turretd: keeps the latest JPEG frame for MJPEG streaming.

Backends:
  * "picam" — Picamera2 with the hardware MJPEG encoder: every encoder buffer
    is one complete JPEG, so the Pi's CPU never touches pixels.
  * "sim"   — synthetic scene that shifts with the pan/tilt angles, so the UI
    behaves realistically without a Pi (needs numpy; cv2 or simplejpeg).

If no real frame arrives for a while (camera missing, ribbon cable loose),
a grey placeholder frame keeps the stream alive and `error` says why.

Besides the JPEGs, both backends publish a small grayscale frame (≤ 640 px
wide, same field of view) with the time its exposure started, for tracking:
gray_after(t) returns the first one exposed after t, so callers can wait
until the turret has stopped and settled before they measure anything.
"""

import collections
import io
import math
import threading
import time
from typing import Callable, Optional

PLACEHOLDER_AFTER_S = 2.0
PLACEHOLDER_FPS = 2.0
GRAY_MAX_WIDTH = 640
GRAY_RING = 24                    # ~0.8 s of tracking frames kept for look-back

_QUALITY_JPEG = {'low': 50, 'medium': 70, 'high': 85, 'very_high': 95}


def encode_jpeg(bgr, quality: int = 85) -> Optional[bytes]:
    try:
        import simplejpeg
        return simplejpeg.encode_jpeg(bgr, quality=quality, colorspace='BGR')
    except ImportError:
        pass
    try:
        import cv2
        ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None
    except ImportError:
        return None


def parse_resolution(s: str) -> tuple[int, int]:
    w, h = s.lower().split('x')
    return int(w), int(h)


def gray_size(size: tuple[int, int]) -> tuple[int, int]:
    """Tracking frame size: same aspect as `size`, at most GRAY_MAX_WIDTH wide, even."""
    w, h = size
    if w <= GRAY_MAX_WIDTH:
        return w, h
    return GRAY_MAX_WIDTH, int(round(h * GRAY_MAX_WIDTH / w / 2)) * 2


def _sensor_clock_offset(sensor_s: float) -> float:
    """sensor timestamp + offset = time.monotonic(). libcamera stamps with
    CLOCK_BOOTTIME on some kernels and CLOCK_MONOTONIC on others: take the one
    the timestamp is just behind."""
    mono = time.monotonic()
    for clk in (getattr(time, 'CLOCK_BOOTTIME', None), time.CLOCK_MONOTONIC):
        if clk is None:
            continue
        now = time.clock_gettime(clk)
        if 0 <= now - sensor_s < 2.0:
            return mono - now
    return mono - sensor_s - 0.05            # unknown clock: assume 50 ms old


class CameraService:
    def __init__(self, backend: str = 'picam',
                 view: Optional[Callable[[], tuple[float, float]]] = None) -> None:
        self.backend = backend
        self.view = view or (lambda: (0.0, 0.0))     # (pan°, tilt°) for the sim
        self.params: dict = {}
        self.frame: Optional[bytes] = None
        self.seq = 0
        self.real_seq = 0
        self.last_real = 0.0
        self.fps = 0.0
        self.error: Optional[str] = None
        self.size = (0, 0)
        self.gray = None                     # latest tracking frame (uint8, h×w)
        self.gray_ts = 0.0                   # its exposure start, time.monotonic()
        self.gray_exp = 0.0                  # its exposure time (s)
        self.gray_ring = collections.deque(maxlen=GRAY_RING)   # (ts, exp, gray)
        self.sim_t0 = None                   # sim: time origin of the drifting target
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._picam = None
        self._fps_count, self._fps_t0 = 0, time.monotonic()
        self._lock = threading.Lock()

    # ── frames ───────────────────────────────────────────────────────────────
    def _publish(self, jpeg: bytes, real: bool = True) -> None:
        with self._cond:
            self.frame = jpeg
            self.seq += 1
            if real:
                self.real_seq += 1
                self.last_real = time.monotonic()
                self.error = None
                self._fps_count += 1
                now = time.monotonic()
                if now - self._fps_t0 >= 1.0:
                    self.fps = self._fps_count / (now - self._fps_t0)
                    self._fps_count, self._fps_t0 = 0, now
            self._cond.notify_all()

    def _publish_gray(self, gray, ts: float, exposure: float = 0.0) -> None:
        with self._cond:
            self.gray, self.gray_ts, self.gray_exp = gray, ts, exposure
            self.gray_ring.append((ts, exposure, gray))
            self._cond.notify_all()

    def gray_since(self, t_min: float) -> list:
        """Buffered tracking frames exposed at or after t_min, oldest first:
        [(ts, exposure_s, gray), …] (at most the last ~0.8 s)."""
        with self._cond:
            return [f for f in self.gray_ring if f[0] >= t_min]

    def gray_next(self, after_ts: float, timeout: float = 1.0):
        """The next frame exposed after after_ts: (ts, exposure_s, gray) or None.
        Blocking: call from a worker thread."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: (self.gray is not None and self.gray_ts > after_ts) or self._stop.is_set(),
                timeout)
            return (self.gray_ts, self.gray_exp, self.gray) if ok and self.gray is not None else None

    def gray_after(self, t_min: float, timeout: float = 1.0):
        """First tracking frame exposed at or after t_min (monotonic): (ts, gray),
        or None on timeout. Blocking: call from a worker thread."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: (self.gray is not None and self.gray_ts >= t_min) or self._stop.is_set(),
                timeout)
            return (self.gray_ts, self.gray) if ok and self.gray is not None else None

    def wait_frame(self, last_seq: int, timeout: float = 1.0) -> tuple[int, Optional[bytes]]:
        """Block until a frame newer than last_seq exists (or timeout)."""
        with self._cond:
            self._cond.wait_for(lambda: self.seq != last_seq or self._stop.is_set(), timeout)
            return self.seq, self.frame

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self, params: dict) -> None:
        with self._lock:
            self.params = dict(params)
            self._stop.clear()
            self.size = parse_resolution(params.get('camera.resolution', '1280x720'))
            if self.backend == 'picam':
                try:
                    self._start_picam()
                except Exception as e:  # noqa: BLE001 — any libcamera failure
                    self.error = f'camera failed to start: {e}'
                    self._close_picam()
            elif self.backend == 'sim':
                self._spawn(self._sim_loop)
            self._spawn(self._placeholder_loop)

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            with self._cond:
                self._cond.notify_all()
            for t in self._threads:
                t.join(timeout=2)
            self._threads.clear()
            self._close_picam()

    def restart(self, params: dict) -> None:
        self.stop()
        self.start(params)

    def _spawn(self, fn) -> None:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        self._threads.append(t)

    def status(self) -> dict:
        stale = time.monotonic() - self.last_real > PLACEHOLDER_AFTER_S
        err = self.error
        if err is None and stale and self.backend != 'none':
            err = ('no frames from the camera — check the ribbon cable at both ends'
                   if self.real_seq == 0 or self.backend == 'picam' else 'camera stalled')
        return {'backend': self.backend, 'fps': round(self.fps, 1) if not stale else 0.0,
                'size': list(self.size), 'error': err, 'frames': self.real_seq,
                'tracking': self.gray is not None and time.monotonic() - self.gray_ts < 2.0,
                # exposure start → now, for the latest tracking frame (sanity check
                # of the sensor clock mapping: expect ~1–3 frame times)
                'gray_age_ms': round((time.monotonic() - self.gray_ts) * 1000)
                if self.gray is not None else None,
                'exposure_ms': round(self.gray_exp * 1000, 1)}

    # ── picamera2 ────────────────────────────────────────────────────────────
    def _start_picam(self) -> None:
        from libcamera import Transform
        from picamera2 import Picamera2
        from picamera2.encoders import MJPEGEncoder, Quality
        from picamera2.outputs import FileOutput

        p = self.params
        fps = int(p.get('camera.fps', 30))
        frame_us = int(1_000_000 / fps)
        cam = Picamera2()
        gw, gh = gray_size(self.size)
        controls = {'FrameDurationLimits': (frame_us, frame_us)}
        exp_ms = float(p.get('camera.exposure_ms', 0) or 0)
        if exp_ms > 0:
            # fixed shutter; auto exposure keeps adjusting the gain
            controls['ExposureTime'] = int(min(exp_ms * 1000, frame_us))
        cfg = cam.create_video_configuration(
            main={'size': self.size, 'format': 'YUV420'},
            lores={'size': (gw, gh), 'format': 'YUV420'},
            controls=controls,
            transform=Transform(hflip=int(bool(p.get('camera.hflip'))),
                                vflip=int(bool(p.get('camera.vflip')))),
        )
        cam.configure(cfg)
        quality = {'low': Quality.LOW, 'medium': Quality.MEDIUM, 'high': Quality.HIGH,
                   'very_high': Quality.VERY_HIGH}[p.get('camera.quality', 'high')]

        service = self

        class _Sink(io.BufferedIOBase):
            # picamera2's FileOutput wants a BufferedIOBase; each MJPEG
            # encoder buffer written here is one complete JPEG frame.
            def writable(self) -> bool:
                return True

            def write(self, buf) -> int:
                service._publish(bytes(buf))
                return len(buf)

        clock = {}

        def on_request(request) -> None:
            # The Y plane of the YUV420 lores stream is the grayscale image.
            try:
                md = request.get_metadata()
                ts = md.get('SensorTimestamp')
                exp = md.get('ExposureTime', 0) / 1e6
                y = request.make_array('lores')[:gh, :gw].copy()
            except Exception:  # noqa: BLE001 — never break the camera thread
                return
            if ts is None:
                service._publish_gray(y, time.monotonic() - 0.05, exp)
                return
            if 'offset' not in clock:
                clock['offset'] = _sensor_clock_offset(ts / 1e9)
            service._publish_gray(y, ts / 1e9 + clock['offset'], exp)

        cam.post_callback = on_request
        self._picam = cam
        cam.start_recording(MJPEGEncoder(), FileOutput(_Sink()), quality=quality)

    def _close_picam(self) -> None:
        cam, self._picam = self._picam, None
        if cam is None:
            return
        try:
            cam.stop_recording()
        except Exception:  # noqa: BLE001
            pass
        try:
            cam.close()
        except Exception:  # noqa: BLE001
            pass

    # ── placeholder / sim ────────────────────────────────────────────────────
    def _placeholder_loop(self) -> None:
        import numpy as np
        w, h = self.size
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        img[::32, :, :] = 60
        img[:, ::32, :] = 60
        jpeg = encode_jpeg(img, 60)
        while not self._stop.wait(1.0 / PLACEHOLDER_FPS):
            if jpeg and time.monotonic() - self.last_real > PLACEHOLDER_AFTER_S:
                self._publish(jpeg, real=False)

    def _sim_loop(self) -> None:
        import numpy as np
        try:
            import cv2
        except ImportError:
            cv2 = None
        w, h = self.size
        fps = int(self.params.get('camera.fps', 30))
        quality = _QUALITY_JPEG.get(self.params.get('camera.quality', 'high'), 85)
        scene = SimScene(self.params, (w, h))
        gw, gh = gray_size((w, h))
        t0 = self.sim_t0 = time.monotonic()
        while not self._stop.wait(1.0 / fps):
            ts = time.monotonic()
            pan, tilt = self.view()
            frame = scene.render(pan, tilt, ts - t0)
            if cv2 is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if (gw, gh) != (w, h):
                    gray = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
                self._publish_gray(gray, ts)
            jpeg = encode_jpeg(frame, quality)
            if jpeg:
                self._publish(jpeg)


class SimScene:
    """A textured synthetic world seen through a pinhole camera: gridlines every
    5° (labelled every 15°), random speckle for the tracker to lock on, and a
    red square drifting about so motion is visible while the turret is still."""

    PX_PER_DEG = 24
    SPAN = (400, 160)                        # world size in degrees (pan, tilt)

    def __init__(self, params: dict, size: tuple[int, int]) -> None:
        import numpy as np
        try:
            import cv2
        except ImportError:
            cv2 = None
        self.cv2, self.np = cv2, np
        self.size = size
        self.hfov = float(params.get('camera.hfov', 53.5))
        self.vfov = float(params.get('camera.vfov', 30.1))
        d = self.PX_PER_DEG
        ww, wh = self.SPAN[0] * d, self.SPAN[1] * d
        rng = np.random.default_rng(7)
        noise = rng.random((wh // 4, ww // 4)).astype(np.float32)
        if cv2 is not None:
            noise = np.clip(cv2.resize(noise, (ww, wh), interpolation=cv2.INTER_CUBIC), 0, 1)
        else:
            noise = np.kron(noise, np.ones((4, 4), np.float32))
        base = (30 + 50 * noise).astype(np.uint8)
        world = np.stack([base + 8, base + 4, base], axis=2)
        for deg in range(-self.SPAN[0] // 2, self.SPAN[0] // 2 + 1, 5):
            x = int(ww / 2 + deg * d)
            if 0 <= x < ww:
                world[:, x:x + (2 if deg % 15 == 0 else 1)] = (90, 90, 90)
        for deg in range(-self.SPAN[1] // 2, self.SPAN[1] // 2 + 1, 5):
            y = int(wh / 2 - deg * d)
            if 0 <= y < wh:
                world[y:y + (2 if deg % 15 == 0 else 1), :] = (90, 90, 90)
        if cv2 is not None:
            for dx in range(-195, 196, 15):
                for dy in range(-75, 76, 15):
                    cv2.putText(world, f'{dx},{dy}', (int(ww / 2 + dx * d) + 4,
                                int(wh / 2 - dy * d) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (150, 150, 150), 1, cv2.LINE_AA)
        self.world = world
        w, h = size
        u = (np.arange(w) + 0.5) / w * 2 - 1
        v = (np.arange(h) + 0.5) / h * 2 - 1
        self.ax = np.degrees(np.arctan(u * np.tan(np.radians(self.hfov) / 2))).astype(np.float32)
        self.ay = np.degrees(np.arctan(v * np.tan(np.radians(self.vfov) / 2))).astype(np.float32)

    def render(self, pan: float, tilt: float, t: float = 0.0):
        np, cv2, d = self.np, self.cv2, self.PX_PER_DEG
        wh, ww = self.world.shape[:2]
        w, h = self.size
        mx = ww / 2 + (pan + self.ax) * d               # world column per image column
        my = wh / 2 - (tilt - self.ay) * d              # world row per image row (+v down)
        if cv2 is not None:
            frame = cv2.remap(self.world, np.broadcast_to(mx, (h, w)).astype(np.float32),
                              np.broadcast_to(my[:, None], (h, w)).astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            xi = np.clip(mx.astype(int), 0, ww - 1)
            yi = np.clip(my.astype(int), 0, wh - 1)
            frame = self.world[yi[:, None], xi[None, :]]
        # the drifting target (a red/white checker), drawn in image space
        az, el = self.target_at(t)
        tx_deg, ty_deg = az - pan, el - tilt
        if abs(tx_deg) < self.hfov / 2 and abs(ty_deg) < self.vfov / 2:
            tx = int((math.tan(math.radians(tx_deg)) / math.tan(math.radians(self.hfov) / 2) + 1) / 2 * w)
            ty = int((1 - math.tan(math.radians(ty_deg)) / math.tan(math.radians(self.vfov) / 2)) / 2 * h)
            r = max(10, w // 40)
            if r <= tx < w - r and r <= ty < h - r:
                frame[ty - r:ty + r, tx - r:tx + r] = (40, 60, 220)
                frame[ty - r:ty, tx - r:tx] = (230, 230, 230)
                frame[ty:ty + r, tx:tx + r] = (230, 230, 230)
        return frame

    @staticmethod
    def target_at(t: float) -> tuple[float, float]:
        """True (azimuth, elevation) in ° of the drifting target at sim time t."""
        return 20 * math.sin(t * 0.5), 8 * math.sin(t * 0.8)
