"""
Feature tracking on grayscale frames: pure functions, no I/O (easy to test).

  * make_patch()  cuts a template around a clicked point and scores how
    trackable it is (texture = smallest eigenvalue of the structure tensor:
    a corner or blob scores high, a plain wall or a straight edge scores low).
  * find()        template-matches the patch in a search window, refines the
    peak to sub-pixel precision and judges the match:
      score   normalized correlation of the best match (1 = identical)
      margin  best minus second-best peak (low = repetitive pattern, ambiguous)
      sharp   sharpness of the match relative to the template (motion blur,
              vibration and defocus lower it)
  * fit_fov()     fits the field of view to (turret angle, image position)
    pairs with the pinhole model, see frame_angle in controller.py.

Coordinates: pixels (x right, y down, pixel centres at i + 0.5 in the
continuous frame) or normalized u, v in [-1, 1] across the frame like the UI.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def frame_angle(frac: float, fov_deg: float) -> float:
    """Angle (°) off the optical axis of a point at `frac` of the half-frame
    (pinhole camera): tan(angle) = frac · tan(fov/2)."""
    return math.degrees(math.atan(frac * math.tan(math.radians(fov_deg) / 2)))


def from_angle(angle_deg: float, fov_deg: float) -> float:
    """Inverse of frame_angle: normalized frame coordinate of an angle."""
    return math.tan(math.radians(angle_deg)) / math.tan(math.radians(fov_deg) / 2)


def to_px(u: float, size: int) -> float:
    return (u + 1) / 2 * size - 0.5


def to_norm(px: float, size: int) -> float:
    return (px + 0.5) / size * 2 - 1


def texture(img: np.ndarray) -> float:
    """√(smallest structure-tensor eigenvalue), in grey levels per pixel."""
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8
    a, b, c = float((gx * gx).mean()), float((gx * gy).mean()), float((gy * gy).mean())
    lam = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b * b)
    return math.sqrt(max(lam, 0.0))


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: drops when the image is blurred."""
    return float(cv2.Laplacian(img, cv2.CV_32F).var())


@dataclass
class Patch:
    img: np.ndarray
    x: float                 # centre in the frame it was cut from (px)
    y: float
    texture: float
    sharp: float
    frame_shape: tuple = (0, 0)        # (h, w) of the frame it was cut from
    u: float = 0.0                     # where it was cut, normalized
    v: float = 0.0


def make_patch(gray: np.ndarray, x: float, y: float, half: int) -> Optional[Patch]:
    """Template of (2·half+1)² px centred on (x, y); None if it doesn't fit."""
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    if xi - half < 0 or yi - half < 0 or xi + half >= w or yi + half >= h:
        return None
    img = gray[yi - half:yi + half + 1, xi - half:xi + half + 1].copy()
    return Patch(img, float(xi), float(yi), texture(img), sharpness(img))


def rescaled(patch: Patch, sx: float, sy: float) -> Patch:
    """The patch as it will look after its image scale changes by (sx, sy)."""
    if abs(sx - 1) < 0.01 and abs(sy - 1) < 0.01:
        return patch
    h, w = patch.img.shape
    nw, nh = max(5, int(round(w * sx)) | 1), max(5, int(round(h * sy)) | 1)
    img = cv2.resize(patch.img, (nw, nh), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    return Patch(img, patch.x, patch.y, texture(img), sharpness(img), patch.frame_shape,
                 patch.u, patch.v)


def lens_scale(frac_from: float, frac_to: float, fov_deg: float) -> float:
    """Pinhole camera: how much bigger a feature looks at frac_to than at
    frac_from (along one axis). Magnification grows as 1 + (u·tan(fov/2))²."""
    k = math.tan(math.radians(fov_deg) / 2)
    return (1 + (frac_to * k) ** 2) / (1 + (frac_from * k) ** 2)


@dataclass
class Match:
    x: float
    y: float
    score: float
    margin: float
    sharp: float             # relative to the template
    ok: bool = False
    reason: str = ''


def _subpixel(r: np.ndarray, i: int, j: int) -> tuple[float, float]:
    """Parabolic peak refinement around r[i, j]; returns (dy, dx) in ±0.5."""
    def para(a, b, c):
        d = a - 2 * b + c
        return 0.0 if d >= 0 else max(-0.5, min(0.5, 0.5 * (a - c) / d))
    dy = para(r[i - 1, j], r[i, j], r[i + 1, j]) if 0 < i < r.shape[0] - 1 else 0.0
    dx = para(r[i, j - 1], r[i, j], r[i, j + 1]) if 0 < j < r.shape[1] - 1 else 0.0
    return dy, dx


def find(gray: np.ndarray, patch: Patch, ex: float, ey: float, radius: float, *,
         min_score: float = 0.8, min_margin: float = 0.1,
         min_sharp: float = 0.5) -> Optional[Match]:
    """Search for `patch` within `radius` px of (ex, ey). None if the window
    doesn't overlap the frame enough to hold the template."""
    ph, pw = patch.img.shape
    hh, hw = ph // 2, pw // 2
    h, w = gray.shape
    r = int(math.ceil(radius))
    x0, y0 = max(0, int(ex) - r - hw), max(0, int(ey) - r - hh)
    x1, y1 = min(w, int(ex) + r + hw + 2), min(h, int(ey) + r + hh + 2)
    if x1 - x0 < pw + 2 or y1 - y0 < ph + 2:
        return None
    res = cv2.matchTemplate(gray[y0:y1, x0:x1], patch.img, cv2.TM_CCOEFF_NORMED)
    _, score, _, (j, i) = cv2.minMaxLoc(res)
    dy, dx = _subpixel(res, i, j)
    # second-best peak: mask out the main one (± template size)
    masked = res.copy()
    masked[max(0, i - hh):i + hh + 1, max(0, j - hw):j + hw + 1] = -1
    second = float(masked.max()) if masked.size else -1.0
    found = gray[y0 + i:y0 + i + ph, x0 + j:x0 + j + pw]
    m = Match(x=float(x0 + j + hw + dx), y=float(y0 + i + hh + dy), score=float(score),
              margin=float(score) - second,
              sharp=sharpness(found) / patch.sharp if patch.sharp > 0 else 0.0)
    if m.score < min_score:
        m.reason = f'match too weak ({m.score:.2f})'
    elif m.margin < min_margin:
        m.reason = 'ambiguous (repeating pattern)'
    elif m.sharp < min_sharp:
        m.reason = f'image blurred ({m.sharp:.0%} of reference sharpness)'
    else:
        m.ok = True
    return m


def fit_fov(thetas, us, lo: float = 10.0, hi: float = 170.0,
            step: float = 0.01) -> tuple[float, float, float]:
    """Field of view (°) that best explains feature positions `us` (normalized,
    along the axis) at turret angles `thetas` (°, relative). A feature at u sits
    atan(u·tan(fov/2)) off-axis, and that angle must change 1:1 with the turret
    (sign either way). Returns (fov, sign, rms residual in °)."""
    th = np.asarray(thetas, dtype=np.float64)
    u = np.asarray(us, dtype=np.float64)
    fovs = np.arange(lo, hi + step / 2, step)
    k = np.tan(np.radians(fovs) / 2)[:, None]
    phi = np.degrees(np.arctan(u[None, :] * k))            # (n_fov, n_samples)
    best = (math.inf, lo, 1.0)
    for s in (1.0, -1.0):
        r = phi - s * th[None, :]
        r -= r.mean(axis=1, keepdims=True)                 # unknown starting angle
        rms = np.sqrt((r * r).mean(axis=1))
        i = int(rms.argmin())
        if rms[i] < best[0]:
            best = (float(rms[i]), float(fovs[i]), s)
    return best[1], best[2], best[0]


# ── scene motion: how far the camera turned, from many features ──────────────
@dataclass
class SceneMotion:
    d_pan: float = 0.0           # ° the camera turned (same sign as Controller.view_delta)
    d_tilt: float = 0.0
    n: int = 0                   # inlier features
    found: int = 0               # features tracked reliably (before outlier rejection)
    detected: int = 0            # corners detected in the reference frame
    rms: float = 0.0             # ° spread of the inliers
    p0: Optional[np.ndarray] = None   # reliable pairs (px), N×2, for FOV fitting
    p1: Optional[np.ndarray] = None
    ok: bool = False
    reason: str = ''
    _fit: tuple = ()                  # per-axis quadratic of the turn vs. position

    def at(self, u: float, v: float) -> tuple[float, float]:
        """(d_pan, d_tilt) as seen by a point at (u, v) of the reference frame.
        With an exact field of view every feature reports the same turn; with an
        inexact one it varies smoothly across the frame (pan with u, tilt with v),
        so read it off a fit at the point instead of using the mean."""
        if not self._fit:
            return self.d_pan, self.d_tilt
        cp, ct = self._fit
        return float(np.polyval(cp, u)), float(np.polyval(ct, v))


def _angles(px: np.ndarray, size: int, fov: float) -> np.ndarray:
    k = math.tan(math.radians(fov) / 2)
    return np.degrees(np.arctan(((px + 0.5) / size * 2 - 1) * k))


def scene_motion(ref: np.ndarray, cur: np.ndarray, hfov: float, vfov: float, *,
                 max_points: int = 200, min_points: int = 20,
                 expect: Optional[tuple[float, float]] = None) -> SceneMotion:
    """Camera rotation between two frames of a static scene, from up to
    max_points corners tracked with pyramidal Lucas–Kanade.

    Pure rotation keeps each point's horizontal angle atan(u·tan(hfov/2))
    changing by exactly the pan angle (and likewise vertically for tilt), so
    every feature votes for (d_pan, d_tilt); the median and a MAD cut throw out
    things that moved on their own (people, the target) and bad tracks.

    expect = the commanded (d_pan, d_tilt): features start where that turn
    would put them. Tracking is also tried from a whole-frame phase-correlation
    guess; the attempt with more consistent features wins."""
    h, w = ref.shape
    margin = 12
    mask = np.zeros_like(ref)
    mask[margin:h - margin, margin:w - margin] = 255
    pts = cv2.goodFeaturesToTrack(ref, max_points, 0.01, max(8, w // 50), mask=mask, blockSize=7)
    detected = 0 if pts is None else len(pts)
    if pts is None or detected < min_points:
        return SceneMotion(detected=detected,
                           reason=f'too little texture in the scene ({detected} features)')
    p0 = pts.reshape(-1, 2)
    guesses = []
    if expect is not None:
        a = _angles(p0[:, 0], w, hfov) - expect[0]
        b = _angles(p0[:, 1], h, vfov) + expect[1]
        ok = (np.abs(a) < 89) & (np.abs(b) < 89)
        gx = (np.tan(np.radians(a)) / math.tan(math.radians(hfov) / 2) + 1) / 2 * w - 0.5
        gy = (np.tan(np.radians(b)) / math.tan(math.radians(vfov) / 2) + 1) / 2 * h - 0.5
        if ok.all():
            guesses.append(np.stack([gx, gy], axis=1))
    (sx, sy), _ = cv2.phaseCorrelate(ref.astype(np.float32), cur.astype(np.float32))
    guesses.append(p0 + np.float32([sx, sy]))
    best = None
    for g in guesses:
        r = _track(ref, cur, pts, g.astype(np.float32).reshape(-1, 1, 2), hfov, vfov, min_points)
        r.detected = detected
        if best is None or r.n > best.n:
            best = r
        if r.ok and r.n >= 0.6 * detected:
            break                                  # good enough, skip the other guess
    if best.ok and best.n < max(min_points, detected // 4):
        best.ok = False
        best.reason = f'only {best.n} of {detected} features consistent (turned too far, or blur)'
    return best


def _track(ref, cur, pts, init, hfov, vfov, min_points) -> SceneMotion:
    h, w = ref.shape
    res = SceneMotion()
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, cur, pts, init, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, ref, nxt, pts.copy(),
                                            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
    p0, p1 = pts.reshape(-1, 2), nxt.reshape(-1, 2)
    fb = np.linalg.norm(back.reshape(-1, 2) - p0, axis=1)
    inside = ((p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h))
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5) & inside
    p0, p1 = p0[good], p1[good]
    res.found, res.p0, res.p1 = len(p0), p0, p1
    if len(p0) < min_points:
        res.reason = f'only {len(p0)} of {len(pts)} features tracked (blur, or turned too far)'
        return res
    dp = _angles(p0[:, 0], w, hfov) - _angles(p1[:, 0], w, hfov)
    dt = _angles(p1[:, 1], h, vfov) - _angles(p0[:, 1], h, vfov)
    keep = np.ones(len(dp), bool)
    for d in (dp, dt):
        med = np.median(d)
        mad = 1.4826 * np.median(np.abs(d - med))
        keep &= np.abs(d - med) <= max(0.05, 4 * mad)
    res.n = int(keep.sum())
    if res.n < min_points:
        res.reason = f'features disagree ({res.n} of {len(p0)} consistent): moving scene?'
        return res
    res.d_pan, res.d_tilt = float(dp[keep].mean()), float(dt[keep].mean())
    res.rms = float(math.sqrt(((dp[keep] - res.d_pan) ** 2).mean()
                              + ((dt[keep] - res.d_tilt) ** 2).mean()))
    if res.n >= 12:
        u0 = (p0[keep, 0] + 0.5) / w * 2 - 1
        v0 = (p0[keep, 1] + 0.5) / h * 2 - 1
        res._fit = (np.polyfit(u0, dp[keep], 2), np.polyfit(v0, dt[keep], 2))
    res.ok = True
    return res


def plausible(measured: float, commanded: float) -> bool:
    """Measured turn vs. the commanded one: same direction and within 3× for
    real moves (the field of view setting may be off, so not tighter)."""
    if abs(commanded) < 1.0:
        return abs(measured - commanded) < 1.5
    r = measured / commanded
    return 1 / 3 <= r <= 3


def fit_fov_pairs(thetas, a0, a1, lo: float = 10.0, hi: float = 170.0) -> tuple[float, float, float, int]:
    """FOV from feature pairs: each pair (a0 → a1, normalized coordinate along
    the axis) was seen before/after the turret turned by theta (°). For the right
    fov, atan(a0·k) − atan(a1·k) = ±theta for every pair. Coarse-to-fine grid
    search, then one round of outlier rejection. Returns (fov, sign, rms°, n)."""
    th = np.asarray(thetas, np.float64)
    a0 = np.asarray(a0, np.float64)
    a1 = np.asarray(a1, np.float64)
    keep = np.ones(len(th), bool)

    def search(lo_, hi_, step, sel):
        fovs = np.arange(lo_, hi_ + step / 2, step)
        k = np.tan(np.radians(fovs) / 2)[:, None]
        d = np.degrees(np.arctan(a0[sel][None] * k) - np.arctan(a1[sel][None] * k))
        best = (math.inf, lo_, 1.0)
        for s in (1.0, -1.0):
            rms = np.sqrt(((d - s * th[sel][None]) ** 2).mean(axis=1))
            i = int(rms.argmin())
            if rms[i] < best[0]:
                best = (float(rms[i]), float(fovs[i]), s)
        return best

    for _ in range(2):
        _, fov, _ = search(lo, hi, 0.5, keep)
        rms, fov, s = search(max(lo, fov - 1), min(hi, fov + 1), 0.01, keep)
        k = math.tan(math.radians(fov) / 2)
        r = np.degrees(np.arctan(a0 * k) - np.arctan(a1 * k)) - s * th
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r - np.median(r[keep])) <= max(0.02, 4 * mad)
    rms, fov, s = search(max(lo, fov - 1), min(hi, fov + 1), 0.01, keep)
    return fov, s, rms, int(keep.sum())
