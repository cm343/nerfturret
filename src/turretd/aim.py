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
import time
from typing import Optional

try:
    from . import tracker                  # needs OpenCV: sudo apt install python3-opencv
except ImportError:                        # without it, click-to-aim stays open-loop
    tracker = None

EDGE = 0.85                    # stop a calibration sweep when |u| exceeds this
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
            self._set('open', f'open loop only: {why}' if why else 'open loop', point=None)
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
        try:
            for it in range(int(self.cfg['aim.max_iter']) + 1):
                gray = await self._settled_frame(await self._wait_still())
                h, w = gray.shape
                ex, ey = tracker.to_px(cx, w), tracker.to_px(cy, h)
                # Perspective: the spot looks smaller near the centre than at the edge.
                p = tracker.rescaled(patch, tracker.lens_scale(pu, cx, hfov),
                                     tracker.lens_scale(pv, cy, vfov))
                m = self._find(gray, p, ex, ey, radius * w)
                if m is None or not m.ok:
                    self._set('lost', f'kept open-loop aim: {m.reason if m else "feature out of view"}')
                    return
                u, v = self._norm(gray, m.x, m.y)
                # Re-cut the template here: later corrections are small and
                # compare like with like.
                fresh = tracker.make_patch(gray, m.x, m.y, patch.img.shape[0] // 2)
                if fresh is not None and fresh.texture >= self.cfg['aim.min_texture']:
                    patch, pu, pv = fresh, u, v
                radius = max(0.05, 0.5 * radius)
                d_pan, d_tilt = self.c.view_delta(u, v, cx, cy)
                err = max(abs(d_pan), abs(d_tilt))
                if err <= tol:
                    self._set('locked', f'on target: {err:.2f}° after {it} correction'
                              + ('' if it == 1 else 's'), point=[u, v])
                    return
                if it == int(self.cfg['aim.max_iter']):
                    break
                self._set('tracking', f'correcting {err:.2f}°', busy='aim', point=[u, v])
                self.c.move_by_deg(d_pan, d_tilt)
            self._set('locked', f'stopped after {it} corrections, {err:.2f}° left '
                      '(raise aim.tolerance, or check for skipped steps)', point=[u, v])
        except AimAbort as e:
            self._set('lost', f'refine stopped: {e}')
        except asyncio.CancelledError:
            raise

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
            for direction in (1, -1):
                if direction == -1:
                    self.c.goto_deg(axis, start)
                    gray = await self._settled_frame(await self._wait_still())
                x, y, delta = patch.x, patch.y, None
                last_theta = 0.0
                for _ in range(200):
                    self.c.move_by_deg(**{f'd_{axis}': direction * step})
                    gray = await self._settled_frame(await self._wait_still())
                    theta = self.c.position_deg(axis) - start
                    if abs(theta - last_theta) < step / 4:
                        break                           # soft limit reached
                    if delta is None:                   # unknown direction on the image
                        radius = 2.5 * est + 10
                        ex, ey = x, y
                    else:
                        radius = 0.5 * est + 6
                        ex, ey = (x + delta, y) if axis == 'pan' else (x, y + delta)
                    # Rescale for perspective at the last known position (one
                    # template for the whole sweep: no drift from re-cutting).
                    pu, pv = self._norm(gray, ex, ey)
                    p = tracker.rescaled(patch, tracker.lens_scale(u, pu, self.cfg['camera.hfov']),
                                         tracker.lens_scale(v, pv, self.cfg['camera.vfov']))
                    m = self._find(gray, p, ex, ey, radius)
                    if m is None or not m.ok:
                        break
                    delta = (m.x - x) if axis == 'pan' else (m.y - y)
                    x, y, last_theta = m.x, m.y, theta
                    mu, mv = self._norm(gray, m.x, m.y)
                    thetas.append(theta)
                    coords.append(mu if axis == 'pan' else mv)
                    self._set('calibrating', f'{len(thetas)} samples', busy='fov', point=[mu, mv])
                    if abs(coords[-1]) > EDGE or abs(theta) > 0.9 * self.cfg[fov_key]:
                        break
            self.c.goto_deg(axis, start)
            if len(thetas) < 5:
                raise AimAbort(f'only {len(thetas)} samples — pick a feature nearer the centre '
                               'or a smaller aim.cal_step')
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
            self.c.emit('info', self.status['msg'])
        except AimAbort as e:
            self.c.goto_deg(axis, start)
            self._set('failed', f'FOV calibration: {e}')
            self.c.emit('warn', self.status['msg'])
