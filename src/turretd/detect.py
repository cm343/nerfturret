"""
Object detection for turretd: the Pi sends its small, timestamped colour
frames to a YOLO server (turret_ui.py on the laptop starts one and tells the
Pi its URL) and keeps the boxes that come back.

Each result carries the exposure time of the frame it was computed on, so a
box converts to a direction in the room with the turret position of that
moment — that is what lets follow mode combine slow detections (a few per
second, ~100–250 ms late) with its own 30 fps template tracking.

  * detect.enabled        send frames at most detect.max_rate per second
  * detect.<class>        which classes count as targets
  * detect.auto_track     start following the best target by itself (never
                          for detect.manual_pause s after a manual move)
"""

import asyncio
import collections
import itertools
import time
from typing import Optional

TARGET_CLASSES = ('airplane', 'drone', 'helicopter')
RESULT_KEEP_S = 2.0


def class_key(label: str) -> str:
    """Model label → config key: 'AirPlane' → 'airplane', 'c-helicopter' → 'helicopter'."""
    lab = label.lower().replace('-', '')
    for k in TARGET_CLASSES:
        if k in lab:
            return k
    return lab


class DetectionService:
    def __init__(self, ctrl) -> None:
        self.c = ctrl
        self.url: Optional[str] = None       # set by the UI relay (or --detect-url)
        self.results: collections.deque = collections.deque()
        self.ids = itertools.count(1)
        self.error: Optional[str] = None
        self.fps = 0.0
        self.infer_ms = None
        self.last_manual = 0.0
        self._task: Optional[asyncio.Task] = None
        self._session = None

    @property
    def cfg(self):
        return self.c.cfg

    def is_target(self, label: str) -> bool:
        return bool(self.cfg.get(f'detect.{class_key(label)}', False))

    def manual_input(self) -> None:
        """A manual move happened: keep auto-track from grabbing the turret."""
        self.last_manual = time.monotonic()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session is not None:
            await self._session.close()

    # ── results ──────────────────────────────────────────────────────────────
    def latest(self, max_age: float = 0.6) -> Optional[dict]:
        if not self.results:
            return None
        r = self.results[-1]
        return r if time.monotonic() - r['received'] <= max_age else None

    def since(self, rid: int) -> list:
        return [r for r in self.results if r['id'] > rid]

    def get(self, rid: int) -> Optional[dict]:
        return next((r for r in self.results if r['id'] == rid), None)

    def best_target(self) -> Optional[tuple[dict, dict]]:
        r = self.latest()
        if r is None:
            return None
        cands = [d for d in r['dets'] if d['target']]
        return (r, max(cands, key=lambda d: d['conf'])) if cands else None

    def telemetry(self) -> dict:
        r = self.latest()
        return {
            'enabled': bool(self.cfg['detect.enabled']), 'url': self.url, 'error': self.error,
            'fps': round(self.fps, 1), 'infer_ms': self.infer_ms,
            'result': None if r is None else {
                'id': r['id'], 'age_ms': round((time.monotonic() - r['ts']) * 1000),
                'boxes': [{'label': d['label'], 'conf': round(d['conf'], 2), 'box': d['box'],
                           'target': d['target']} for d in r['dets']]},
        }

    # ── loop ─────────────────────────────────────────────────────────────────
    async def _loop(self) -> None:
        import aiohttp
        from .aim import Aimer
        last_ts, n, t0 = 0.0, 0, time.monotonic()
        while True:
            cam = self.c.camera
            if (not self.cfg['detect.enabled'] or not self.url or cam is None
                    or not hasattr(cam, 'color_bgr')):
                self.fps = 0.0
                await asyncio.sleep(0.2)
                continue
            got = cam.color_bgr()
            if got is None or got[0] <= last_ts:
                await asyncio.sleep(0.01)
                continue
            t_start = time.monotonic()
            ts, exp, bgr = got
            last_ts = ts
            try:
                jpeg = await Aimer.cpu(_encode, bgr)
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                async with self._session.post(
                        self.url, data=jpeg, params={'conf': self.cfg['detect.conf']},
                        headers={'Content-Type': 'image/jpeg'},
                        timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    payload = await resp.json()
                self.error = None
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError) as e:
                self.error = f'detector not reachable ({type(e).__name__})'
                await asyncio.sleep(1.0)
                continue
            h, w = bgr.shape[:2]
            dets = []
            for d in payload.get('detections', []):
                x1, y1, x2, y2 = d['bbox']
                dets.append({'label': d['label'], 'conf': float(d['conf']),
                             'box': [round(x1 / w * 2 - 1, 4), round(y1 / h * 2 - 1, 4),
                                     round(x2 / w * 2 - 1, 4), round(y2 / h * 2 - 1, 4)],
                             'target': self.is_target(d['label'])})
            now = time.monotonic()
            self.infer_ms = payload.get('infer_ms')
            self.results.append({'id': next(self.ids), 'ts': ts, 'exp': exp, 'size': (w, h),
                                 'dets': dets, 'received': now})
            while self.results and now - self.results[0]['received'] > RESULT_KEEP_S:
                self.results.popleft()
            n += 1
            if now - t0 >= 1.0:
                self.fps, n, t0 = n / (now - t0), 0, now
            self._maybe_auto_track()
            wait = 1.0 / self.cfg['detect.max_rate'] - (time.monotonic() - t_start)
            if wait > 0:
                await asyncio.sleep(wait)

    def _maybe_auto_track(self) -> None:
        cfg, aim = self.cfg, self.c.aim
        if not cfg['detect.auto_track'] or aim.following or self.c.estop_reason is not None:
            return
        if time.monotonic() - self.last_manual < cfg['detect.manual_pause']:
            return
        best = self.best_target()
        if best is not None:
            r, d = best
            self.c.emit('info', f'auto-track: following {d["label"]} ({d["conf"]:.0%})')
            aim.follow_detection(r, d)


def _encode(bgr) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return buf.tobytes()
