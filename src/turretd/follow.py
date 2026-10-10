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

Detections (detect.py, YOLO on the laptop) can start a follow (a clicked
box, or auto-track) and keep correcting it: every new result is converted to
room directions with the turret position at *its* frame time and matched to
the track. A matching box re-anchors the template when the template has lost
the object or drifted off it (more than half a box away), and keeps the
follow alive while the template alone could not.
"""

import asyncio
import bisect
import collections
import csv
import os
import time
from typing import Optional

try:
    from . import tracker
except ImportError:
    tracker = None

ALPHA, BETA = 0.35, 0.05         # alpha-beta filter gains (per measurement): calm
REFRESH_MAX_TURRET_SPEED = 10.0  # °/s: only refresh the template from sharp frames
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
    async def run(self, u: float, v: float, seed=None) -> None:
        """Follow the object at (u, v) of the frame clicked ~aim.click_latency
        ago, or — seed = (detection result, box) — the detected object."""
        cam, cfg = self.c.camera, self.cfg
        det = getattr(self.c, 'detect', None)
        hfov, vfov = cfg['camera.hfov'], cfg['camera.vfov']
        cx, cy = cfg['camera.crosshair_x'], cfg['camera.crosshair_y']
        a_cross, b_cross = tracker.frame_angle(cx, hfov), tracker.frame_angle(cy, vfov)
        loop = asyncio.get_running_loop()
        try:
            # 1. The frame the user clicked on, and the frames since.
            t_click = time.monotonic() - cfg['aim.click_latency'] / 1000
            if seed is not None:                       # the detection's own frame
                t_click = seed[0]['ts']
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
            half = int(cfg['aim.follow_patch']) // 2
            if seed is not None:
                half = _box_half(seed[1]['box'], w, h)
            patch = tracker.make_patch(gray, tracker.to_px(u, w), tracker.to_px(v, h), half)
            if patch is None:
                raise _Lost('too close to the frame edge')
            if patch.texture < (0.5 if seed is not None else cfg['aim.min_texture']):
                raise _Lost(f'too little texture to follow ({patch.texture:.1f})')
            anchor = patch.img                       # never changes: guards against drift
            tmpl = patch.img.astype('float32')
            t_off = cfg['aim.frame_time_offset'] / 1000
            pose = self.turret_at(ts + exp / 2 + t_off)
            if pose is None:
                raise _Lost('no turret position history')
            az0 = pose[0] + tracker.frame_angle(u, hfov)
            el0 = pose[1] - tracker.frame_angle(v, vfov)
            faz, fel = AlphaBeta(az0, ts + exp / 2 + t_off), AlphaBeta(el0, ts + exp / 2 + t_off)
            last_ts, misses, good_run = ts, 0, 0
            pending = frames[1:]
            speed = {ax: self.c.speed_steps(ax, min(cfg['aim.follow_max_speed'],
                                                    cfg[f'{ax}.speed'] * cfg[f'{ax}.fast_factor']))
                     for ax in ('pan', 'tilt')}
            still = cfg['aim.follow_still_speed']
            self.log = []
            det_rid = seed[0]['id'] if seed is not None else (
                det.results[-1]['id'] if det is not None and det.results else 0)
            last_det_match = time.monotonic() if seed is not None else 0.0
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
                t_mid = ts + exp / 2 + t_off
                pan_t, tilt_t = self.turret_at(t_mid)
                pan_b, tilt_b = self.turret_at(t_mid - 0.05)
                turret_speed = max(abs(pan_t - pan_b), abs(tilt_t - tilt_b)) / 0.05
                # 3. Where the filter expects the object in this frame.
                az_p, el_p = faz.predict(t_mid), fel.predict(t_mid)
                pu = tracker.from_angle(az_p - pan_t, hfov)
                pv = tracker.from_angle(tilt_t - el_p, vfov)
                radius = (0.05 + 0.04 * min(misses, 5)) * w \
                    + 2 * abs(faz.v) * (1 / 30) * w / hfov
                m = None
                if abs(pu) < 1.2 and abs(pv) < 1.2:
                    # current (refreshed) template first, the original as fallback
                    for img in (tmpl.astype('uint8'), anchor):
                        p = tracker.Patch(img, 0, 0, patch.texture, tracker.sharpness(img))
                        cand = await self.a.cpu(tracker.find, gray, p, tracker.to_px(pu, w),
                                                tracker.to_px(pv, h), radius,
                                                min_score=cfg['aim.follow_min_score'],
                                                min_margin=cfg['aim.follow_min_margin'],
                                                min_sharp=0.0)
                        if m is None or (cand is not None and (cand.ok, cand.score) > (m.ok, m.score)):
                            m = cand
                        if m is not None and m.ok:
                            break
                if m is not None and m.ok:
                    misses = 0
                    mu, mv = tracker.to_norm(m.x, w), tracker.to_norm(m.y, h)
                    faz.update(pan_t + tracker.frame_angle(mu, hfov), t_mid)
                    fel.update(tilt_t - tracker.frame_angle(mv, vfov), t_mid)
                    if m.score >= REFRESH_SCORE and turret_speed < REFRESH_MAX_TURRET_SPEED:
                        cut = tracker.make_patch(gray, m.x, m.y, tmpl.shape[0] // 2)
                        if cut is not None:
                            tmpl = (1 - TEMPLATE_BLEND) * tmpl + TEMPLATE_BLEND * cut.img
                    point = [mu, mv]
                else:
                    misses += 1
                    point = None
                # Detections: re-anchor / keep alive (see module docstring).
                if det is not None and cfg['detect.enabled']:
                    for r in det.since(det_rid):
                        det_rid = r['id']
                        got_det = self._use_detection(r, faz, fel, hfov, vfov, misses > 0)
                        if got_det is not None:
                            last_det_match = time.monotonic()
                            new_patch, frame_ts = got_det
                            if new_patch is not None:
                                patch, anchor = new_patch, new_patch.img
                                tmpl = new_patch.img.astype('float32')
                                misses = 0
                if misses and (misses / 30) > cfg['aim.follow_lost'] and \
                        time.monotonic() - last_det_match > cfg['aim.follow_lost']:
                    why = m.reason if m is not None else 'out of view'
                    raise _Lost(f'target lost ({why})')
                # 4. Aim error at this frame: where the turret pointed vs. where it should have.
                # A target slower than `still` counts as standing: no lead from noise.
                v_az = faz.v if abs(faz.v) >= still else 0.0
                v_el = fel.v if abs(fel.v) >= still else 0.0
                lead = self.lead_s()
                want_pan = faz.predict(t_mid) + v_az * lead - a_cross
                want_tilt = fel.predict(t_mid) + v_el * lead + b_cross
                err = max(abs(want_pan - pan_t), abs(want_tilt - tilt_t))
                good_run = good_run + 1 if (point and err <= cfg['aim.follow_tolerance']) else 0
                on_target = good_run >= int(cfg['aim.follow_frames'])
                # 5. Command: the predicted direction at now + lead.
                now = time.monotonic()
                if not pending:
                    for ax, f, vel, off in (('pan', faz, v_az, -a_cross), ('tilt', fel, v_el, b_cross)):
                        spd = self.c.steps_per_deg(ax)
                        aim = f.x + vel * (now - f.t + lead) + off
                        self.c.motion.send('track', self.c.axis_index(ax), aim * spd, vel * spd,
                                           speed[ax], cfg['aim.follow_gain'])
                self.log.append((round(t_mid, 4), round(exp * 1000, 1), round(pan_t, 3),
                                 round(tilt_t, 3), round(turret_speed, 1),
                                 int(bool(point)), *(point or ('', '')),
                                 round(m.score, 3) if m else '', round(m.margin, 3) if m else '',
                                 round(pan_t + tracker.frame_angle(point[0], hfov), 3) if point else '',
                                 round(tilt_t - tracker.frame_angle(point[1], vfov), 3) if point else '',
                                 round(faz.x, 3), round(faz.v, 2), round(fel.x, 3), round(fel.v, 2),
                                 round(err, 3)))
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
        finally:
            self._save_log()

    def _use_detection(self, r, faz, fel, hfov, vfov, template_lost):
        """Match a detection result to the track. Returns None (no match), or
        (new_patch or None, frame ts): a new template when the current one lost
        the object or drifted more than half a box off it; the filter is
        shifted onto the detection in that case."""
        t = r['ts'] + r['exp'] / 2 + self.cfg['aim.frame_time_offset'] / 1000
        pose = self.turret_at(t)
        if pose is None:
            return None
        pan_t, tilt_t = pose
        az_p, el_p = faz.predict(t), fel.predict(t)
        best = None
        for d in r['dets']:
            if not d['target']:
                continue
            u1, v1, u2, v2 = d['box']
            cu, cv = (u1 + u2) / 2, (v1 + v2) / 2
            az = pan_t + tracker.frame_angle(cu, hfov)
            el = tilt_t - tracker.frame_angle(cv, vfov)
            size = max(tracker.frame_angle(u2, hfov) - tracker.frame_angle(u1, hfov),
                       tracker.frame_angle(v2, vfov) - tracker.frame_angle(v1, vfov))
            dist = max(abs(az - az_p), abs(el - el_p))
            if dist <= max(2.0, 1.5 * size) and (best is None or dist < best[0]):
                best = (dist, d, az, el, size, cu, cv)
        if best is None:
            return None
        dist, d, az, el, size, cu, cv = best
        if not template_lost and dist <= size / 2:
            return None, r['ts']                    # template still on the object
        frame = self.c.camera.gray_at(r['ts'])
        if frame is None:
            return None, r['ts']
        fts, _exp, gray = frame
        h, w = gray.shape
        p = tracker.make_patch(gray, tracker.to_px(cu, w), tracker.to_px(cv, h),
                               _box_half(d['box'], w, h))
        if p is None:
            return None, r['ts']
        faz.x += az - az_p                          # shift the track onto the box
        fel.x += el - el_p
        return p, fts

    def _save_log(self) -> None:
        """Per-frame record of the last follow run, for diagnosis."""
        rows = getattr(self, 'log', None)
        if not rows:
            return
        try:
            from .aim import DEBUG_DIR
            os.makedirs(DEBUG_DIR, exist_ok=True)
            path = os.path.join(DEBUG_DIR, f'follow-{time.strftime("%H%M%S")}.csv')
            with open(path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(['t', 'exp_ms', 'pan', 'tilt', 'turret_speed', 'found', 'u', 'v',
                             'score', 'margin', 'az_meas', 'el_meas', 'az', 'az_v', 'el', 'el_v',
                             'err'])
                wr.writerows(rows)
        except OSError:
            pass
        self.log = []


class _Lost(Exception):
    pass


def _box_half(box, w: int, h: int) -> int:
    """Template half-size for a detection box: ~80 % of its geometric mean size."""
    u1, v1, u2, v2 = box
    bw, bh = (u2 - u1) / 2 * w, (v2 - v1) / 2 * h
    return int(max(5, min(50, 0.4 * (max(bw, 1) * max(bh, 1)) ** 0.5)))
