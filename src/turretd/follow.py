"""
Click-to-follow: lock onto the clicked object and keep the turret on it.

Per camera frame (~30 Hz):
  1. Turret direction at the frame: pan/tilt step counts interpolated to the
     middle of the frame's exposure (PoseHistory; the camera timestamps every
     frame with its exposure start and length).
  2. Object in the frame: template match around where the filter predicts it
     (template from the click, slowly refreshed while the match is strong).
  3. Object direction in the room = turret direction + the object's angle off
     the optical axis (pinhole model). Unlike the pixel offset alone, this is
     independent of how the turret itself moved and doesn't go stale.
  4. Filter: per axis an alpha-beta (constant-velocity) filter — position and
     velocity, corrected by each measurement; between frames the trajectory is
     extrapolated linearly, x(t) = x + v·(t − t_frame).
  5. Aim: every frame, hand the motion worker the predicted direction at
         now + lead,  lead = distance / dart speed + aim.follow_latency
     (minus the crosshair offset) together with the object's angular speed.
     The worker's track mode moves that aim point on at that speed between
     frames and runs the axis at the same speed plus a correction for the
     remaining gap, so the turret glides instead of stopping at each update.
  6. On target when the turret direction at the frame is within
     aim.follow_tolerance of where it should have pointed, for
     aim.follow_frames frames in a row. Firing stays manual.

The object is clicked on a video frame that is already ~aim.click_latency
old in the browser: the template is cut from the buffered frame of that time
and tracked forward through the newer buffered frames to catch up.
"""

import asyncio
import bisect
import collections
import time
from typing import Optional

try:
    from . import tracker
except ImportError:
    tracker = None

ALPHA, BETA = 0.5, 0.15          # alpha-beta filter gains (per measurement)
TEMPLATE_BLEND = 0.15            # template refresh weight on strong matches
REFRESH_SCORE = 0.85


class PoseHistory:
    """(time, pan steps, tilt steps) samples; pose at any recent time by linear
    interpolation. Fed from the controller tick and the follow loop."""

    def __init__(self, maxlen: int = 400) -> None:
        self.t: collections.deque = collections.deque(maxlen=maxlen)
        self.p: collections.deque = collections.deque(maxlen=maxlen)

    def add(self, t: float, pan: int, tilt: int) -> None:
        if self.t and t <= self.t[-1]:
            return
        self.t.append(t)
        self.p.append((pan, tilt))

    def at(self, t: float) -> Optional[tuple[float, float]]:
        if not self.t:
            return None
        ts = list(self.t)
        if t <= ts[0]:
            return self.p[0]
        if t >= ts[-1]:
            return self.p[-1]
        i = bisect.bisect_left(ts, t)
        t0, t1 = ts[i - 1], ts[i]
        (a0, b0), (a1, b1) = self.p[i - 1], self.p[i]
        f = (t - t0) / (t1 - t0)
        return a0 + f * (a1 - a0), b0 + f * (b1 - b0)


class AlphaBeta:
    """Constant-velocity tracker for one angle."""

    def __init__(self, x: float, t: float) -> None:
        self.x, self.v, self.t = x, 0.0, t

    def predict(self, t: float) -> float:
        return self.x + self.v * (t - self.t)

    def update(self, meas: float, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        xp = self.predict(t)
        r = meas - xp
        self.x = xp + ALPHA * r
        self.v += BETA * r / dt
        self.t = t


class Follower:
    def __init__(self, ctrl, aimer) -> None:
        self.c = ctrl
        self.a = aimer
        self.pose = PoseHistory()

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def cfg(self):
        return self.c.cfg

    def sample_pose(self) -> None:
        m = self.c.motion
        self.pose.add(time.monotonic(), m.axis(self.c.axis_index('pan')).pos,
                      m.axis(self.c.axis_index('tilt')).pos)

    def turret_at(self, t: float) -> Optional[tuple[float, float]]:
        """Turret (pan°, tilt°) at time t."""
        p = self.pose.at(t)
        if p is None:
            return None
        return self.c.to_deg('pan', p[0]), self.c.to_deg('tilt', p[1])

    def lead_s(self) -> float:
        c = self.cfg
        return c['aim.target_distance'] / max(1.0, c['aim.dart_speed']) + c['aim.follow_latency']

    def _status(self, state, msg, point=None, err=None, on_target=False, target=None):
        self.a._set(state, msg, busy='follow' if state == 'following' else None, point=point)
        self.a.status.update({'on_target': on_target, 'err': err, 'target': target,
                              'following': state == 'following'})

    # ── main loop ────────────────────────────────────────────────────────────
    async def run(self, u: float, v: float) -> None:
        cam, cfg = self.c.camera, self.cfg
        hfov, vfov = cfg['camera.hfov'], cfg['camera.vfov']
        cx, cy = cfg['camera.crosshair_x'], cfg['camera.crosshair_y']
        a_cross, b_cross = tracker.frame_angle(cx, hfov), tracker.frame_angle(cy, vfov)
        loop = asyncio.get_running_loop()
        try:
            # 1. The frame the user clicked on, and the frames since.
            t_click = time.monotonic() - cfg['aim.click_latency'] / 1000
            frames = cam.gray_since(0.0)
            # the newest frame exposed at or before the click time, and all after it
            before = [i for i, f in enumerate(frames) if f[0] <= t_click]
            frames = frames[before[-1]:] if before else frames
            if not frames:
                got = await loop.run_in_executor(None, cam.gray_next, 0.0, 1.0)
                if got is None:
                    raise _Lost('no camera frames')
                frames = [got]
            ts, exp, gray = frames[0]
            h, w = gray.shape
            patch = tracker.make_patch(gray, tracker.to_px(u, w), tracker.to_px(v, h),
                                       int(cfg['aim.follow_patch']) // 2)
            if patch is None:
                raise _Lost('too close to the frame edge')
            if patch.texture < cfg['aim.min_texture']:
                raise _Lost(f'too little texture to follow ({patch.texture:.1f})')
            tmpl = patch.img.astype('float32')
            pose = self.turret_at(ts + exp / 2)
            if pose is None:
                raise _Lost('no turret position history')
            az0 = pose[0] + tracker.frame_angle(u, hfov)
            el0 = pose[1] - tracker.frame_angle(v, vfov)
            faz, fel = AlphaBeta(az0, ts + exp / 2), AlphaBeta(el0, ts + exp / 2)
            last_ts, misses, good_run = ts, 0, 0
            pending = frames[1:]
            speed = {ax: self.c.speed_steps(ax, cfg[f'{ax}.speed'] * cfg[f'{ax}.fast_factor'])
                     for ax in ('pan', 'tilt')}
            self._status('following', 'locked on', point=[u, v])
            self.c.emit('info', 'aim: following target')
            while True:
                # 2. Next frame (buffered ones first, to catch up after the click).
                if pending:
                    ts, exp, gray = pending.pop(0)
                else:
                    got = await loop.run_in_executor(None, cam.gray_next, last_ts, 1.0)
                    if got is None:
                        raise _Lost('camera stopped')
                    ts, exp, gray = got
                last_ts = ts
                self.sample_pose()
                t_mid = ts + exp / 2
                pose = self.turret_at(t_mid)
                pan_t, tilt_t = pose
                # 3. Where the filter expects the object in this frame.
                az_p, el_p = faz.predict(t_mid), fel.predict(t_mid)
                pu = tracker.from_angle(az_p - pan_t, hfov)
                pv = tracker.from_angle(tilt_t - el_p, vfov)
                radius = (0.05 + 0.04 * min(misses, 5)) * w \
                    + 2 * abs(faz.v) * (1 / 30) * w / hfov
                p = tracker.Patch(tmpl.astype('uint8'), 0, 0, patch.texture,
                                  tracker.sharpness(tmpl.astype('uint8')))
                m = None
                if abs(pu) < 1.2 and abs(pv) < 1.2:
                    m = await self.a.cpu(tracker.find, gray, p, tracker.to_px(pu, w),
                                         tracker.to_px(pv, h), radius,
                                         min_score=cfg['aim.follow_min_score'],
                                         min_margin=cfg['aim.min_margin'], min_sharp=0.0)
                if m is not None and m.ok:
                    misses = 0
                    mu, mv = tracker.to_norm(m.x, w), tracker.to_norm(m.y, h)
                    faz.update(pan_t + tracker.frame_angle(mu, hfov), t_mid)
                    fel.update(tilt_t - tracker.frame_angle(mv, vfov), t_mid)
                    if m.score >= REFRESH_SCORE:
                        cut = tracker.make_patch(gray, m.x, m.y, tmpl.shape[0] // 2)
                        if cut is not None:
                            tmpl = (1 - TEMPLATE_BLEND) * tmpl + TEMPLATE_BLEND * cut.img
                    point = [mu, mv]
                else:
                    misses += 1
                    point = None
                    if (misses / 30) > cfg['aim.follow_lost']:
                        why = m.reason if m is not None else 'out of view'
                        raise _Lost(f'target lost ({why})')
                # 4. Aim error at this frame: where the turret pointed vs. where it should have.
                lead = self.lead_s()
                want_pan = faz.predict(t_mid + lead) - a_cross
                want_tilt = fel.predict(t_mid + lead) + b_cross
                err = max(abs(want_pan - pan_t), abs(want_tilt - tilt_t))
                good_run = good_run + 1 if (point and err <= cfg['aim.follow_tolerance']) else 0
                on_target = good_run >= int(cfg['aim.follow_frames'])
                # 5. Command: the predicted direction at now + lead.
                now = time.monotonic()
                if not pending:
                    for ax, f, off in (('pan', faz, -a_cross), ('tilt', fel, b_cross)):
                        spd = self.c.steps_per_deg(ax)
                        self.c.motion.send('track', self.c.axis_index(ax),
                                           (f.predict(now + lead) + off) * spd, f.v * spd, speed[ax])
                self._status('following',
                             f'following: {err:.2f}° off' + (' — ON TARGET' if on_target else '')
                             + ('' if point else f' (coasting, {misses} frames)'),
                             point=point, err=round(err, 3), on_target=on_target,
                             target=[round(faz.x, 2), round(fel.x, 2),
                                     round(faz.v, 1), round(fel.v, 1)])
        except _Lost as e:
            self.c.motion.send('stop', self.c.axis_index('pan'))
            self.c.motion.send('stop', self.c.axis_index('tilt'))
            self._status('lost', f'follow stopped: {e}')
            self.c.emit('warn', f'aim: follow stopped: {e}')
        # Cancelled (Stop button, new click, manual move, E-STOP): whoever
        # cancelled decides what the axes do next, so no stop is sent here.


class _Lost(Exception):
    pass
