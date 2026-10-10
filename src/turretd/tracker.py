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
    return Patch(img, patch.x, patch.y, texture(img), sharpness(img))


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
