"""
Closed-loop aiming on top of the open-loop moves: click refinement and the
automatic field-of-view calibration. Both use tracker.py on the camera's
grayscale frames and only ever measure on frames whose exposure started
after pan and tilt stopped and aim.settle passed, so motion blur and frame
latency never enter a measurement.

Click to aim (aim.mode):
  open  turn by the angle computed from the field of view, done.
  auto  the same move first; if the clicked spot is trackable (enough texture,
        turret was standing still when you clicked), find it again after the
        turret settles and correct the residual, up to aim.max_iter times.
        Anything not good enough → stay with the open-loop result and say why.

FOV calibration: lock on a feature, step the axis in aim.cal_step increments
to both sides until the feature nears the frame edge, then fit the pinhole
field of view to all (angle, position) pairs.
"""

import asyncio
import os
import time
from typing import Optional

try:
    from . import tracker                  # needs OpenCV: sudo apt install python3-opencv
except ImportError:                        # without it, click-to-aim stays open-loop
    tracker = None

EDGE = 0.85                    # stop a calibration sweep when |u| exceeds this
DEBUG_DIR = os.environ.get('TURRETD_TRACK_DEBUG', '/tmp/turret-track')
DEBUG_KEEP = 30
POLL_S = 0.01
MOVE_TIMEOUT_S = 10.0


class AimAbort(Exception):
    pass


class Aimer:
    def __init__(self, ctrl) -> None:
        self.c = ctrl
        self.task: Optional[asyncio.Task] = None
        self.still_since: Optional[float] = time.monotonic()
        # Shown in telemetry: what the tracker did last and where the feature is.
        self.status: dict = {'busy': None, 'state': 'idle', 'msg': '', 'point': None}

    # ── plumbing ─────────────────────────────────────────────────────────────
    @property
    def cfg(self):
        return self.c.cfg

    @property
    def available(self) -> bool:
        return (tracker is not None and self.c.camera is not None
                and hasattr(self.c.camera, 'gray_after'))

    def tick(self) -> None:
        """Called from the controller tick: track when pan/tilt came to rest."""
        if self._moving():
            self.still_since = None
        elif self.still_since is None:
            self.still_since = time.monotonic()

    def _moving(self) -> bool:
        return any(self.c.motion.axis(self.c.axis_index(a)).moving for a in ('pan', 'tilt'))

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            self._set('idle', 'cancelled')
        self.task = None

    def _set(self, state: str, msg: str = '', busy=None, point=None) -> None:
        self.status = {'busy': busy, 'state': state, 'msg': msg, 'point': point,
                       't': time.time()}

    def _report(self, level: str, state: str, msg: str, point=None) -> None:
        """Final outcome of a click: status for the video overlay + a log line."""
        self._set(state, msg, point=point)
        self.c.emit(level, f'aim: {msg}')

    def _start(self, coro) -> None:
        self.cancel()
        self.task = asyncio.create_task(coro)

    async def _wait_still(self) -> float:
        """Wait until pan and tilt are stopped; returns when they stopped."""
        end = time.monotonic() + MOVE_TIMEOUT_S
        await asyncio.sleep(2 * POLL_S)          # let a just-sent move start
        while self._moving() or self.still_since is None:
            if self.c.estop_reason is not None:
                raise AimAbort('E-STOP')
            if time.monotonic() > end:
                raise AimAbort('axis did not stop')
            await asyncio.sleep(POLL_S)
            self.tick()
        return self.still_since

    async def _settled_frame(self, since: float):
        """First tracking frame exposed ≥ aim.settle after `since`."""
        t_min = since + self.cfg['aim.settle'] / 1000
        got = await asyncio.get_running_loop().run_in_executor(
            None, self.c.camera.gray_after, t_min, 2.0)
        if got is None:
            raise AimAbort('no camera frames')
        return got[1]

    async def _locate(self, patch, ex, ey, radius, since: float):
        """Find `patch` in the first frame exposed ≥ aim.settle after `since`.
        While the only problem is blur (turret still shaking), keep taking newer
        frames up to aim.settle_max. Returns (gray, match, settle_s)."""
        loop = asyncio.get_running_loop()
        t_min = since + self.cfg['aim.settle'] / 1000
        deadline = since + max(self.cfg['aim.settle_max'], self.cfg['aim.settle']) / 1000
        while True:
            got = await loop.run_in_executor(None, self.c.camera.gray_after, t_min, 2.0)
            if got is None:
                raise AimAbort('no camera frames')
            ts, gray = got
            m = self._find(gray, patch, ex, ey, radius)
            blurred = m is not None and not m.ok and m.reason.startswith('image blurred')
            if not blurred or ts >= deadline:
                return gray, m, ts - since
            t_min = ts + 1e-3

    def _dump(self, kind: str, gray, patch, ex, ey, radius, m) -> Optional[str]:
        """Save what the tracker saw on a failure: template | best match, and the
        frame with the search window (red) and the match (green)."""
        try:
            import cv2
            import numpy as np
            os.makedirs(DEBUG_DIR, exist_ok=True)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            r = int(radius)
            cv2.rectangle(img, (int(ex) - r, int(ey) - r), (int(ex) + r, int(ey) + r), (0, 0, 255), 1)
            ph, pw = patch.img.shape
            if m is not None:
                cv2.rectangle(img, (int(m.x) - pw // 2, int(m.y) - ph // 2),
                              (int(m.x) + pw // 2, int(m.y) + ph // 2), (0, 255, 0), 1)
            tile = cv2.resize(patch.img, (pw * 3, ph * 3), interpolation=cv2.INTER_NEAREST)
            row = [cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)]
            if m is not None:
                y0, x0 = int(round(m.y)) - ph // 2, int(round(m.x)) - pw // 2
                found = gray[max(0, y0):y0 + ph, max(0, x0):x0 + pw]
                if found.shape == patch.img.shape:
                    row.append(cv2.cvtColor(cv2.resize(found, (pw * 3, ph * 3),
                                            interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR))
            strip = np.hstack(row)
            canvas = np.zeros((img.shape[0] + strip.shape[0], img.shape[1], 3), np.uint8)
            canvas[:img.shape[0]] = img
            canvas[img.shape[0]:, :min(strip.shape[1], img.shape[1])] = strip[:, :img.shape[1]]
            path = os.path.join(DEBUG_DIR, f'{time.strftime("%H%M%S")}-{kind}.png')
            cv2.imwrite(path, canvas)
            for old in sorted(os.listdir(DEBUG_DIR))[:-DEBUG_KEEP]:
                os.remove(os.path.join(DEBUG_DIR, old))
            return path
        except Exception:  # noqa: BLE001 — diagnostics must never break aiming
            return None

    def _find(self, gray, patch, ex, ey, radius):
        return tracker.find(gray, patch, ex, ey, radius,
                            min_score=self.cfg['aim.min_score'],
                            min_margin=self.cfg['aim.min_margin'],
                            min_sharp=self.cfg['aim.min_sharp'])

    def _patch(self, gray, u, v):
        h, w = gray.shape
        half = int(self.cfg['aim.patch']) // 2
        p = tracker.make_patch(gray, tracker.to_px(u, w), tracker.to_px(v, h), half)
        if p is None:
            return None, 'too close to the frame edge'
        p.frame_shape = gray.shape
        if p.texture < self.cfg['aim.min_texture']:
            return None, f'too little texture there ({p.texture:.1f} < {self.cfg["aim.min_texture"]:.1f})'
        return p, ''

    def _norm(self, gray, x, y) -> tuple[float, float]:
        h, w = gray.shape
        return tracker.to_norm(x, w), tracker.to_norm(y, h)

    # ── click to aim ─────────────────────────────────────────────────────────
    async def click(self, x: float, y: float) -> str:
        """x, y relative to the crosshair (UI convention). Moves open-loop right
        away; in auto mode refines in the background. Returns the mode used."""
        self.cancel()
        cx, cy = self.cfg['camera.crosshair_x'], self.cfg['camera.crosshair_y']
        u, v = x + cx, y + cy
        patch, why = None, ''
        if self.cfg['aim.mode'] == 'auto':
            if not self.available:
                why = 'no camera' if tracker is not None else 'OpenCV not installed on the Pi'
            elif self._moving() or self.still_since is None:
                why = 'turret was moving when you clicked'
            else:
                got = self.c.camera.gray
                # The frame you clicked on must show the turret at rest.
                if got is None or self.c.camera.gray_ts < self.still_since + self.cfg['aim.settle'] / 1000:
                    why = 'turret had not settled when you clicked'
                else:
                    patch, why = self._patch(got, u, v)
        self.c.move_open_loop(u, v, cx, cy)
        if patch is None:
            if why:
                self._report('warn', 'open', f'tracking not used ({why}) — open-loop move only')
            else:
                self._set('open', 'open loop')
            return 'open'
        self._set('tracking', 'refining…', busy='aim', point=[u, v])
        self._start(self._refine(patch, u, v))
        return 'auto'

    async def _refine(self, patch, pu: float, pv: float) -> None:
        """patch was cut at image point (pu, pv); find it near the crosshair."""
        cx, cy = self.cfg['camera.crosshair_x'], self.cfg['camera.crosshair_y']
        tol = self.cfg['aim.tolerance']
        hfov, vfov = self.cfg['camera.hfov'], self.cfg['camera.vfov']
        radius = self.cfg['aim.search'] / 100
        first_err = None
        try:
            for it in range(int(self.cfg['aim.max_iter']) + 1):
                since = await self._wait_still()
                h, w = patch.frame_shape
                ex, ey = tracker.to_px(cx, w), tracker.to_px(cy, h)
                # Perspective: the spot looks smaller near the centre than at the edge.
                p = tracker.rescaled(patch, tracker.lens_scale(pu, cx, hfov),
                                     tracker.lens_scale(pv, cy, vfov))
                gray, m, settle_s = await self._locate(p, ex, ey, radius * w, since)
                if m is None or not m.ok:
                    why = m.reason if m else 'spot out of view'
                    if m is not None and m.reason.startswith('image blurred'):
                        why += f' {settle_s * 1000:.0f} ms after stopping — turret still shaking? ' \
                               'raise aim.settle_max'
                    self._dump('aim', gray, p, ex, ey, radius * w, m)
                    self._report('warn', 'lost', f'tracking failed after the move ({why}) — '
                                 'kept the open-loop aim' if it == 0 else
                                 f'tracking lost after {it} correction(s) ({why}) — stopped there')
                    return
                u, v = self._norm(gray, m.x, m.y)
                # Re-cut the template here: later corrections are small and
                # compare like with like.
                fresh = tracker.make_patch(gray, m.x, m.y, patch.img.shape[0] // 2)
                if fresh is not None:
                    fresh.frame_shape = gray.shape
                if fresh is not None and fresh.texture >= self.cfg['aim.min_texture']:
                    patch, pu, pv = fresh, u, v
                radius = max(0.05, 0.5 * radius)
                d_pan, d_tilt = self.c.view_delta(u, v, cx, cy)
                err = max(abs(d_pan), abs(d_tilt))
                if first_err is None:
                    first_err = err
                if err <= tol:
                    self._report('info', 'locked', f'tracking locked: {err:.2f}° off after {it} '
                                 f'correction{"" if it == 1 else "s"} (open-loop move was off by '
                                 f'{first_err:.2f}°)', point=[u, v])
                    return
                if it == int(self.cfg['aim.max_iter']):
                    break
                self._set('tracking', f'correcting {err:.2f}°', busy='aim', point=[u, v])
                self.c.move_by_deg(d_pan, d_tilt)
            self._report('warn', 'locked', f'tracking gave up after {it} corrections, {err:.2f}° '
                         'left (raise aim.tolerance, or check for skipped steps)', point=[u, v])
        except AimAbort as e:
            self._report('warn', 'lost', f'tracking stopped: {e}')
        except asyncio.CancelledError:
            raise

    def _first_step_hint(self, gray, patch, x, y, est, axis) -> str:
        """After the first calibration step lost the spot: search the whole frame
        to tell 'moved much more than expected' from 'unrecognisable'."""
        h, w = gray.shape
        m = tracker.find(gray, patch, w / 2, h / 2, max(w, h),
                         min_score=self.cfg['aim.min_score'], min_margin=self.cfg['aim.min_margin'],
                         min_sharp=0.0)
        if m is None or not m.ok:
            return ''
        moved = abs(m.x - x) if axis == 'pan' else abs(m.y - y)
        if moved > 2.5 * est + 10:
            return (f' — but it was found {moved:.0f} px away instead of ~{est:.0f} px: '
                    f'the axis turns ~{moved / est:.1f}× more than commanded '
                    '(gear ratio / microsteps), or the field of view setting is far off')
        return ''

    # ── automatic FOV calibration ────────────────────────────────────────────
    def calibrate(self, axis: str, u: float, v: float) -> None:
        if not self.available:
            raise AimAbort('no camera' if tracker is not None else 'OpenCV not installed on the Pi')
        self._set('calibrating', 'locking on…', busy='fov', point=[u, v])
        self._start(self._calibrate(axis, u, v))

    async def _calibrate(self, axis: str, u: float, v: float) -> None:
        fov_key = 'camera.hfov' if axis == 'pan' else 'camera.vfov'
        step = self.cfg['aim.cal_step']
        start = self.c.position_deg(axis)
        try:
            gray = await self._settled_frame(await self._wait_still())
            patch, why = self._patch(gray, u, v)
            if patch is None:
                raise AimAbort(f'pick another spot: {why}')
            h, w = gray.shape
            size = w if axis == 'pan' else h
            est = step / self.cfg[fov_key] * size      # expected px per step
            thetas, coords = [0.0], [u if axis == 'pan' else v]
            stops = []                                  # why each sweep ended
            for direction in (1, -1):
                side = '+' if direction > 0 else '−'
                if direction == -1:
                    self.c.goto_deg(axis, start)
                    await self._wait_still()
                x, y, delta = patch.x, patch.y, None
                last_theta = 0.0
                for _ in range(200):
                    self.c.move_by_deg(**{f'd_{axis}': direction * step})
                    since = await self._wait_still()
                    theta = self.c.position_deg(axis) - start
                    if abs(theta - last_theta) < step / 4:
                        stops.append(f'{side}: soft limit')
                        break
                    if delta is None:                   # unknown direction on the image
                        radius = 2.5 * est + 10
                        ex, ey = x, y
                    else:
                        radius = 0.5 * abs(delta) + 6
                        ex, ey = (x + delta, y) if axis == 'pan' else (x, y + delta)
                    # Rescale for perspective at the last known position (one
                    # template for the whole sweep: no drift from re-cutting).
                    pu, pv = tracker.to_norm(ex, w), tracker.to_norm(ey, h)
                    p = tracker.rescaled(patch, tracker.lens_scale(u, pu, self.cfg['camera.hfov']),
                                         tracker.lens_scale(v, pv, self.cfg['camera.vfov']))
                    gray, m, settle_s = await self._locate(p, ex, ey, radius, since)
                    if m is None or not m.ok:
                        why = m.reason if m else 'search window left the frame'
                        if delta is None:
                            why += self._first_step_hint(gray, p, x, y, est, axis)
                        elif m is not None and m.reason.startswith('image blurred'):
                            why += f' {settle_s * 1000:.0f} ms after stopping'
                        self._dump(f'fov{side}', gray, p, ex, ey, radius, m)
                        stops.append(f'{side} after {len(thetas) - 1} steps: {why}')
                        break
                    delta = (m.x - x) if axis == 'pan' else (m.y - y)
                    x, y, last_theta = m.x, m.y, theta
                    mu, mv = self._norm(gray, m.x, m.y)
                    thetas.append(theta)
                    coords.append(mu if axis == 'pan' else mv)
                    self._set('calibrating', f'{len(thetas)} samples', busy='fov', point=[mu, mv])
                    if abs(coords[-1]) > EDGE:
                        stops.append(f'{side}: frame edge')
                        break
                    if abs(theta) > 0.9 * self.cfg[fov_key]:
                        stops.append(f'{side}: max angle')
                        break
            self.c.goto_deg(axis, start)
            if len(thetas) < 5:
                raise AimAbort(f'only {len(thetas)} samples. ' + '; '.join(stops))
            span = max(coords) - min(coords)
            if span < 0.5:
                raise AimAbort(f'feature crossed only {span / 2:.0%} of the frame')
            fov, _sign, rms = tracker.fit_fov(thetas, coords)
            if rms > self.cfg['aim.cal_max_rms']:
                raise AimAbort(f'fit too poor ({rms:.2f}° rms) — check gear ratio, microsteps '
                               'and for skipped steps')
            self.c.set_config({fov_key: round(fov, 2)})
            self._set('done', f'{"horizontal" if axis == "pan" else "vertical"} field of view = '
                      f'{fov:.2f}° ({len(thetas)} samples, {rms:.3f}° rms, saved)')
            if rms > 0.05:
                self.status['msg'] += ' — rms is high-ish: wobble or skipped steps?'
            self.c.emit('info', self.status['msg'])
        except AimAbort as e:
            self.c.goto_deg(axis, start)
            self._set('failed', f'FOV calibration: {e}')
            self.c.emit('warn', self.status['msg'])
