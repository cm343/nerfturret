"""
Closed-loop aiming on top of the open-loop moves: click refinement and the
automatic field-of-view calibration. Both use tracker.py on the camera's
grayscale frames and only ever measure on frames whose exposure started
after pan and tilt stopped and aim.settle passed, so motion blur and frame
latency never enter a measurement.

Click to aim (aim.mode):
  open  turn by the angle computed from the field of view, done.
  auto  the same move first; then measure how far the camera really turned
        from many scene features (tracker.scene_motion), plus the clicked spot
        itself when it has texture, and correct the residual, up to
        aim.max_iter times. Not measurable → keep the open-loop result, say why.

FOV calibration: step the axis aim.cal_steps × aim.cal_step each way; every
scene feature tracked across a step gives (before, after, angle); fit the
pinhole field of view to all of them. No feature needs to be picked.
"""

import asyncio
import functools
import os
import time
from typing import Optional

try:
    from . import tracker                  # needs OpenCV: sudo apt install python3-opencv
except ImportError:                        # without it, click-to-aim stays open-loop
    tracker = None
from .follow import Follower

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
        self.follower = Follower(ctrl, self)

    # ── plumbing ─────────────────────────────────────────────────────────────
    @property
    def cfg(self):
        return self.c.cfg

    @property
    def available(self) -> bool:
        return (tracker is not None and self.c.camera is not None
                and hasattr(self.c.camera, 'gray_after'))

    def tick(self) -> None:
        """Called from the controller tick: track when pan/tilt came to rest,
        and keep the turret position history for follow mode."""
        self.follower.sample_pose()
        if self._moving():
            self.still_since = None
        elif self.still_since is None:
            self.still_since = time.monotonic()

    def _moving(self) -> bool:
        return any(self.c.motion.axis(self.c.axis_index(a)).moving for a in ('pan', 'tilt'))

    @property
    def following(self) -> bool:
        return self.status.get('busy') == 'follow' and self.task is not None and not self.task.done()

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            was = self.status.get('busy')
            self.task.cancel()
            self._set('idle', 'follow stopped' if was == 'follow' else 'cancelled')
        self.task = None

    def follow_detection(self, result: dict, d: dict) -> None:
        """Start following a detected object (clicked box or auto-track)."""
        if not self.available:
            raise AimAbort('no camera')
        self.cancel()
        u1, v1, u2, v2 = d['box']
        u, v = (u1 + u2) / 2, (v1 + v2) / 2
        self._set('following', f'locking on {d["label"]}…', busy='follow', point=[u, v])
        self._start(self.follower.run(u, v, seed=(result, d)))

    def stop_follow(self) -> None:
        """Stop button: end following and bring pan/tilt to a ramped stop."""
        if self.following:
            self.cancel()
            for a in ('pan', 'tilt'):
                self.c.motion.send('stop', self.c.axis_index(a))
            self.c.emit('info', 'aim: follow stopped')

    def _set(self, state: str, msg: str = '', busy=None, point=None) -> None:
        self.status = {'busy': busy, 'state': state, 'msg': msg, 'point': point,
                       't': time.time()}

    def _report(self, level: str, state: str, msg: str, point=None) -> None:
        """Final outcome of a click: status for the video overlay + a log line."""
        self._set(state, msg, point=point)
        self.c.emit(level, msg if msg.startswith('FOV') or state == 'done' else f'aim: {msg}')

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
        p.u, p.v = u, v
        if p.texture < self.cfg['aim.min_texture']:
            return None, f'too little texture there ({p.texture:.1f} < {self.cfg["aim.min_texture"]:.1f})'
        return p, ''

    def _norm(self, gray, x, y) -> tuple[float, float]:
        h, w = gray.shape
        return tracker.to_norm(x, w), tracker.to_norm(y, h)

    async def _sharp_frame(self, ref, since: float):
        """First frame exposed ≥ aim.settle after `since`; while it is clearly
        blurrier than `ref` (turret still shaking), take newer ones up to
        aim.settle_max. Returns (gray, seconds after stopping)."""
        loop = asyncio.get_running_loop()
        t_min = since + self.cfg['aim.settle'] / 1000
        deadline = since + max(self.cfg['aim.settle_max'], self.cfg['aim.settle']) / 1000
        ref_sharp = await self.cpu(tracker.sharpness, ref) if ref is not None else 0.0
        while True:
            got = await loop.run_in_executor(None, self.c.camera.gray_after, t_min, 2.0)
            if got is None:
                raise AimAbort('no camera frames')
            ts, gray = got
            if (ref_sharp <= 0 or ts >= deadline
                    or await self.cpu(tracker.sharpness, gray) >= self.cfg['aim.min_sharp'] * ref_sharp):
                return gray, ts - since
            t_min = ts + 1e-3

    @staticmethod
    async def cpu(fn, *args, **kw):
        """Run image processing in a worker thread: tens of ms on the Pi must not
        stall the event loop (heartbeats, watchdog, commands, telemetry)."""
        return await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(fn, *args, **kw))

    async def _scene(self, ref, cur, expect: tuple[float, float]):
        """Scene motion, rejected unless it roughly agrees with the commanded turn."""
        sm = await self.cpu(tracker.scene_motion, ref, cur, self.cfg['camera.hfov'],
                            self.cfg['camera.vfov'], min_points=int(self.cfg['aim.min_points']),
                            expect=expect)
        if sm.ok and not (tracker.plausible(sm.d_pan, expect[0])
                          and tracker.plausible(sm.d_tilt, expect[1])):
            sm.ok = False
            sm.reason = (f'scene says the turret turned {sm.d_pan:+.1f}°/{sm.d_tilt:+.1f}°, '
                         f'commanded {expect[0]:+.1f}°/{expect[1]:+.1f}° — field of view far off, '
                         'or skipped steps')
        return sm

    def _rest_frame(self):
        """The latest frame if it shows the turret at rest, else (None, why)."""
        if self._moving() or self.still_since is None:
            return None, 'turret was moving when you clicked'
        gray = self.c.camera.gray
        if gray is None or self.c.camera.gray_ts < self.still_since + self.cfg['aim.settle'] / 1000:
            return None, 'turret had not settled when you clicked'
        return gray, ''

    # ── click to aim ─────────────────────────────────────────────────────────
    async def click(self, x: float, y: float) -> str:
        """x, y relative to the crosshair (UI convention). Moves open-loop right
        away; in auto mode refines in the background. Returns the mode used."""
        self.cancel()
        cx, cy = self.cfg['camera.crosshair_x'], self.cfg['camera.crosshair_y']
        u, v = x + cx, y + cy
        if self.cfg['aim.mode'] == 'follow':
            if not self.available:
                raise AimAbort('no camera' if tracker is not None else 'OpenCV not installed on the Pi')
            self._set('following', 'locking on…', busy='follow', point=[u, v])
            self._start(self.follower.run(u, v))
            return 'follow'
        ref, why = None, ''
        if self.cfg['aim.mode'] == 'auto':
            if not self.available:
                why = 'no camera' if tracker is not None else 'OpenCV not installed on the Pi'
            else:
                ref, why = self._rest_frame()
        self.c.move_open_loop(u, v, cx, cy)
        if ref is None:
            if why:
                self._report('warn', 'open', f'tracking not used ({why}) — open-loop move only')
            else:
                self._set('open', 'open loop')
            return 'open'
        patch, _ = self._patch(ref, u, v)         # optional: the spot itself, if trackable
        self._set('tracking', 'refining…', busy='aim', point=[u, v])
        self._start(self._refine(ref, u, v, patch))
        return 'auto'

    async def _refine(self, ref, u: float, v: float, patch) -> None:
        """Bring the clicked point (u, v of frame `ref`) onto the crosshair.

        The point is followed as two angles off the camera axis. After every
        move, the turn the camera actually made is measured from many scene
        features (works on a plain clicked spot), and, if the spot itself is
        trackable, its template is matched near where the scene says it is (a
        direct measurement of the target; wins when both are available)."""
        cx, cy = self.cfg['camera.crosshair_x'], self.cfg['camera.crosshair_y']
        tol = self.cfg['aim.tolerance']
        hfov, vfov = self.cfg['camera.hfov'], self.cfg['camera.vfov']
        h, w = ref.shape
        alpha, beta = tracker.frame_angle(u, hfov), tracker.frame_angle(v, vfov)
        a_cross, b_cross = tracker.frame_angle(cx, hfov), tracker.frame_angle(cy, vfov)
        first_err, how = None, ''
        # the open-loop move click() just sent
        commanded = (alpha - a_cross, -(beta - b_cross))
        try:
            for it in range(int(self.cfg['aim.max_iter']) + 1):
                since = await self._wait_still()
                cur, settle_s = await self._sharp_frame(ref, since)
                sm = await self._scene(ref, cur, commanded)
                notes = []
                if sm.ok:
                    d_p, d_t = sm.at(tracker.from_angle(alpha, hfov), tracker.from_angle(beta, vfov))
                    alpha, beta = alpha - d_p, beta + d_t
                    how = f'scene, {sm.n} features'
                pu, pv = tracker.from_angle(alpha, hfov), tracker.from_angle(beta, vfov)
                if patch is not None:
                    ex, ey = tracker.to_px(pu, w), tracker.to_px(pv, h)
                    radius = (0.03 if sm.ok else self.cfg['aim.search'] / 100) * w
                    pp = tracker.rescaled(patch, tracker.lens_scale(patch.u, pu, hfov),
                                          tracker.lens_scale(patch.v, pv, vfov))
                    m = await self.cpu(self._find, cur, pp, ex, ey, radius)
                    if m is not None and m.ok:
                        mu, mv = self._norm(cur, m.x, m.y)
                        if sm.ok:
                            off = max(abs(tracker.frame_angle(mu, hfov) - alpha),
                                      abs(tracker.frame_angle(mv, vfov) - beta))
                            if off > 3 * tol:
                                notes.append(f'spot is {off:.2f}° off the scene estimate — target moved?')
                        alpha, beta = tracker.frame_angle(mu, hfov), tracker.frame_angle(mv, vfov)
                        how = 'spot' + (f' + scene, {sm.n} features' if sm.ok else '')
                        fresh = tracker.make_patch(cur, m.x, m.y, patch.img.shape[0] // 2)
                        if fresh is not None and fresh.texture >= self.cfg['aim.min_texture']:
                            fresh.u, fresh.v = mu, mv
                            patch = fresh
                    elif not sm.ok:
                        self._dump('aim', cur, pp, ex, ey, radius, m)
                        sm.reason += f'; spot: {m.reason if m else "out of view"}'
                if not sm.ok and not how.startswith('spot'):
                    why = sm.reason
                    if settle_s * 1000 >= self.cfg['aim.settle_max'] - 50:
                        why += f' (frames still blurred {settle_s * 1000:.0f} ms after stopping?)'
                    self._report('warn', 'lost', f'tracking failed after the move ({why}) — '
                                 'kept the open-loop aim' if it == 0 else
                                 f'tracking lost after {it} correction(s) ({why}) — stopped there')
                    return
                point = [tracker.from_angle(alpha, hfov), tracker.from_angle(beta, vfov)]
                d_pan, d_tilt = alpha - a_cross, -(beta - b_cross)
                err = max(abs(d_pan), abs(d_tilt))
                if first_err is None:
                    first_err = err
                for n in notes:
                    self.c.emit('warn', f'aim: {n}')
                if err <= tol:
                    self._report('info', 'locked', f'tracking locked ({how}): {err:.2f}° off after '
                                 f'{it} correction{"" if it == 1 else "s"} (open-loop move was off '
                                 f'by {first_err:.2f}°)', point=point)
                    return
                if it == int(self.cfg['aim.max_iter']):
                    break
                self._set('tracking', f'correcting {err:.2f}° ({how})', busy='aim', point=point)
                self.c.move_by_deg(d_pan, d_tilt)
                commanded = (d_pan, d_tilt)
                ref = cur                          # next measurement: from this frame on
            self._report('warn', 'locked', f'tracking gave up after {it} corrections, {err:.2f}° '
                         'left (raise aim.tolerance, or check for skipped steps)', point=point)
        except AimAbort as e:
            self._report('warn', 'lost', f'tracking stopped: {e}')

    # ── automatic FOV calibration ────────────────────────────────────────────
    def calibrate(self, axis: str) -> None:
        if not self.available:
            raise AimAbort('no camera' if tracker is not None else 'OpenCV not installed on the Pi')
        self._set('calibrating', 'starting…', busy='fov')
        self._start(self._calibrate(axis))

    async def _calibrate(self, axis: str) -> None:
        """Step the axis aim.cal_steps times each way by aim.cal_step; between
        consecutive frames, every scene feature gives a (before, after, angle)
        triple; fit the field of view to all of them."""
        fov_key = 'camera.hfov' if axis == 'pan' else 'camera.vfov'
        step, n_steps = self.cfg['aim.cal_step'], int(self.cfg['aim.cal_steps'])
        start = self.c.position_deg(axis)
        thetas, a0, a1, stops, n_ok = [], [], [], [], 0
        try:
            await self._wait_still()
            for direction in (1, -1):
                side = '+' if direction > 0 else '−'
                if direction == -1:
                    self.c.goto_deg(axis, start)
                ref, _ = await self._sharp_frame(None, await self._wait_still())
                prev = self.c.position_deg(axis)
                for k in range(n_steps):
                    self.c.move_by_deg(**{f'd_{axis}': direction * step})
                    cur, _ = await self._sharp_frame(ref, await self._wait_still())
                    pos = self.c.position_deg(axis)
                    if abs(pos - prev) < step / 4:
                        stops.append(f'{side}: soft limit after {k} steps')
                        break
                    expect = (pos - prev, 0.0) if axis == 'pan' else (0.0, pos - prev)
                    sm = await self._scene(ref, cur, expect)
                    if sm.p0 is None or len(sm.p0) < self.cfg['aim.min_points']:
                        stops.append(f'{side} step {k + 1}: {sm.reason}')
                        break
                    h, w = cur.shape
                    i, size = (0, w) if axis == 'pan' else (1, h)
                    thetas += [pos - prev] * len(sm.p0)
                    a0 += list(tracker.to_norm(sm.p0[:, i], size))
                    a1 += list(tracker.to_norm(sm.p1[:, i], size))
                    n_ok += 1
                    self._set('calibrating', f'{n_ok} steps, {len(thetas)} feature pairs', busy='fov')
                    ref, prev = cur, pos
            self.c.goto_deg(axis, start)
            if n_ok < 3:
                raise AimAbort(f'only {n_ok} usable steps. ' + '; '.join(stops))
            # Pan points move along x for a pan step (tilt: y); the fit ignores sign.
            fov, _sign, rms, n = await self.cpu(tracker.fit_fov_pairs, thetas, a0, a1)
            if rms > self.cfg['aim.cal_max_rms']:
                raise AimAbort(f'fit too poor ({rms:.3f}° rms over {n} pairs) — wobble, skipped '
                               'steps, or a moving scene')
            self.c.set_config({fov_key: round(fov, 2)})
            self._report('info', 'done', f'{"horizontal" if axis == "pan" else "vertical"} field '
                         f'of view = {fov:.2f}° ({n_ok} steps, {n} feature pairs, {rms:.3f}° rms, saved)')
        except AimAbort as e:
            self.c.goto_deg(axis, start)
            self._report('warn', 'failed', f'FOV calibration: {e}')
