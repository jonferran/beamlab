"""Beam analysis: pure numpy, no camera, no UI.

Far-field geometry: the LBA-C-100 is a camera behind a lens focused at infinity,
so a spot's POSITION on the sensor is the beam's pointing angle and its SIZE is
the beam's divergence:   angle [rad] = distance on sensor [mm] / focal length [mm].

Widths by threshold crossing on the profile:
  FWHM  = width where the profile crosses 50 % of its peak
  1/e^2 = width where it crosses 13.5 % of its peak
plus the ISO 11146 second-moment width D4-sigma, which needs no threshold.
"""
import math

import numpy as np

LEVEL_FWHM = 0.5
LEVEL_E2 = 0.135          # 1/e^2 = 0.1353


# ----------------------------------------------------------------- colour

def split_bayer_gr8(raw):
    """BayerGR8 (row 0: G R G R..., row 1: B G B G...) -> R, G, B planes at half
    resolution (one value per 2x2 cell; G is the mean of its two pixels).
    Returns float32 arrays of shape (H//2, W//2)."""
    h, w = raw.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    r = raw[0:h2:2, 1:w2:2].astype(np.float32)
    g = (raw[0:h2:2, 0:w2:2].astype(np.float32) + raw[1:h2:2, 1:w2:2]) * 0.5
    b = raw[1:h2:2, 0:w2:2].astype(np.float32)
    return r, g, b


# Where each Bayer colour sits inside its 2x2 cell (analysis px from the cell
# centre); BayerGR8: G (0,0)+(1,1), R (1,0), B (0,1) in sensor px.
BAYER_OFFSET = np.array([[+0.25, -0.25],      # R
                         [0.0, 0.0],          # G
                         [-0.25, +0.25]])     # B


def bayer_channel_max(raw):
    """Peak raw value of each colour, read on the Bayer pixels themselves (the
    exposure is judged before any colour correction or unmixing)."""
    h, w = raw.shape
    return (int(raw[0::2, 1::2].max()),
            int(max(raw[0::2, 0::2].max(), raw[1::2, 1::2].max())),
            int(raw[1::2, 0::2].max()))


# ----------------------------------------------------------------- geometry

def _box_blur(img, r=2):
    """(2r+1)^2 mean filter via cumulative sums (no scipy)."""
    k = 2 * r + 1
    p = np.pad(img.astype(np.float64), r, mode='edge')
    c = p.cumsum(axis=0).cumsum(axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _block_mean(img, f):
    h, w = img.shape
    h2, w2 = h // f * f, w // f * f
    # two float32 sums are ~5x faster than .mean(axis=(1, 3)) (float64 + overhead)
    a = img[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    if img.dtype == np.uint8:                 # raw frames: integer sums, no float copy
        s = a.sum(axis=3, dtype=np.uint16).sum(axis=1, dtype=np.uint32)
        return s.astype(np.float32) * np.float32(1.0 / (f * f))
    if img.dtype == np.uint16:                # 12-bit raw frames
        s = a.sum(axis=3, dtype=np.uint32).sum(axis=1, dtype=np.uint32)
        return s.astype(np.float32) * np.float32(1.0 / (f * f))
    return a.sum(axis=3, dtype=np.float32).sum(axis=1, dtype=np.float32) * np.float32(1.0 / (f * f))


# a beam counts as present only if its smoothed peak stands this many noise
# sigmas above the dark level (otherwise noise alone produced 70-120 mrad "widths")
MIN_SNR = 8.0


def _grow_blob(mask, seed):
    """The connected part of `mask` (8-neighbours) that contains `seed` (y, x).
    Grown by repeated dilation inside the mask: a few dozen cheap passes on the
    small block-averaged image (no scipy needed)."""
    # work in a small window around the seed, doubled whenever the spot reaches its
    # edge: a spot is ~30 block px across, the image ~480 x 270 (50 ms -> a few ms)
    H, W = mask.shape
    sy, sx = seed
    r = 16
    comp = None
    while True:
        y0, y1 = max(0, sy - r), min(H, sy + r + 1)
        x0, x1 = max(0, sx - r), min(W, sx + r + 1)
        m = mask[y0:y1, x0:x1]
        c = np.zeros_like(m)
        if comp is None:
            c[sy - y0, sx - x0] = True
        else:
            c[cy0 - y0:cy0 - y0 + comp.shape[0], cx0 - x0:cx0 - x0 + comp.shape[1]] = comp
        n = int(c.sum())
        while True:
            g = c.copy()
            g[1:] |= c[:-1]
            g[:-1] |= c[1:]
            g[:, 1:] |= g[:, :-1].copy()
            g[:, :-1] |= g[:, 1:].copy()
            g &= m
            k = int(g.sum())
            if k == n:
                break
            c, n = g, k
        touches = ((y0 > 0 and c[0].any()) or (y1 < H and c[-1].any())
                   or (x0 > 0 and c[:, 0].any()) or (x1 < W and c[:, -1].any()))
        if not touches:
            out = np.zeros_like(mask)
            out[y0:y1, x0:x1] = c
            return out
        comp, cy0, cx0 = c, y0, x0
        r *= 2


def spot_mask_info(img, level=0.05, f=4):
    """(block image, dark, noise, peak, mask of pixels above `level` of the peak)."""
    sm = _block_mean(img.astype(np.float32, copy=False), f) if f > 1 else img.astype(np.float32)
    flat = sm.ravel()
    sub = flat[::7] if flat.size > 20000 else flat       # dark level / noise from a sample
    dark = float(np.median(sub))
    noise = float(np.median(np.abs(sub - dark))) * 1.4826
    noise = max(noise, 0.5 / f)                          # 8-bit quantisation floor
    peak = float(sm.max()) - dark
    return sm, dark, noise, peak, (sm - dark >= peak * level)


SPOT_CHOICES = ('right', 'middle', 'left', 'brightest')


def _find_spots(sm, dark, peak, mask, min_rel=0.15, max_spots=9):
    """Separate spots on the block image: list of dicts {mask, peak, seed, cx, box}
    (box = block-px bbox), each with a peak of at least `min_rel` of the brightest
    (specks of noise or speckle are not counted), brightest first."""
    rest = mask.copy()
    out = []
    while len(out) < max_spots and rest.any():
        vals = np.where(rest, sm - dark, -np.inf)
        i = np.unravel_index(int(np.argmax(vals)), sm.shape)
        if vals[i] < min_rel * peak:
            break
        blob = _grow_blob(rest, i)
        rest &= ~blob
        ys, xs = np.nonzero(blob)
        out.append({'mask': blob, 'peak': float(vals[i]), 'seed': i, 'cx': float(xs.mean()),
                    'box': (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)})
    return out


def _pick(spots, spot):
    """The spot to measure: 'right' / 'left' / 'middle' (by position, left to right)
    or 'brightest'."""
    if not spots:
        return None
    if spot in ('right', 'left', 'middle') and len(spots) > 1:
        by_x = sorted(spots, key=lambda s: s['cx'])
        return by_x[-1] if spot == 'right' else by_x[0] if spot == 'left' else by_x[len(by_x) // 2]
    return spots[0]


def count_spots(img, level=0.05, min_rel=0.15, max_spots=9):
    """How many separate spots the picture has (each with a peak of at least
    `min_rel` of the brightest; specks of noise or speckle are not counted)."""
    f = 4 if min(img.shape) >= 64 else 1
    sm, dark, noise, peak, mask = spot_mask_info(img, level, f)
    if peak <= MIN_SNR * noise:
        return 0
    return len(_find_spots(sm, dark, peak, mask, min_rel, max_spots))


def auto_roi(img, level=0.05, margin=2.6, min_size=16, blob=True, spot='brightest'):
    """ROI around ONE spot: the connected patch of pixels above `level` of that
    spot's own peak (ABOVE THE DARK LEVEL), bounding box grown by `margin` (x the box
    size) so the wings are inside, but never into another spot. Which spot: `spot`
    = 'brightest', or by position 'right' / 'middle' / 'left' (the LBA always shows
    3 spots side by side and Jon measures the right one). Before this, every pixel
    above 5 % went into one box, so the 1/e2 and D4s 'widths' spanned all three
    spots. blob=False: every pixel above the level (no spot picking).
    2.6 x the 5 % box reaches ~6 sigma from the centre, the room the ISO
    3 x D4-sigma aperture needs (1.5 x reached only 3.7 sigma: D4s read ~4 % small).
    Found on a 4x4 block-averaged copy (16x fewer pixels: fast, and the averaging
    stops noise from winning). Returns (x0, y0, x1, y1) in img coordinates, or
    None if there is no beam (peak < MIN_SNR noise sigmas)."""
    f = 4 if min(img.shape) >= 64 else 1
    sm, dark, noise, peak, mask = spot_mask_info(img, level, f)
    if peak <= MIN_SNR * noise:
        return None
    others = []
    if blob:
        if spot == 'brightest':
            mask = _grow_blob(mask, np.unravel_index(int(np.argmax(sm)), sm.shape))
            spots = None
        else:
            spots = _find_spots(sm, dark, peak, mask)
            chosen = _pick(spots, spot)
            if chosen is None:
                return None
            # re-cut at `level` of THIS spot's own peak (a dimmer side spot would
            # otherwise get a wider, lower cut than the brightest one)
            own = (sm - dark >= chosen['peak'] * level) & ~np.logical_or.reduce(
                [s['mask'] for s in spots if s is not chosen] or [np.zeros_like(mask)])
            mask = _grow_blob(own, chosen['seed'])
            others = [s['box'] for s in spots if s is not chosen]
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    xs0, xs1 = xs.min() * f, (xs.max() + 1) * f          # back to full-res pixels
    ys0, ys1 = ys.min() * f, (ys.max() + 1) * f
    cx, cy = 0.5 * (xs0 + xs1), 0.5 * (ys0 + ys1)
    hw = max(min_size / 2, (xs1 - xs0) * margin / 2)
    hh = max(min_size / 2, (ys1 - ys0) * margin / 2)
    h, w = img.shape
    x0, y0 = max(0, int(cx - hw)), max(0, int(cy - hh))
    x1, y1 = min(w, int(math.ceil(cx + hw)) + 1), min(h, int(math.ceil(cy + hh)) + 1)
    # keep the margin out of the neighbouring spots: stop halfway to them (only
    # across a real gap; a neighbour whose faint edge overlaps this spot's own box
    # is left alone - clipping there turned the box inside out and stopped the
    # camera loop with an empty power ROI)
    full = (x0, y0, x1, y1)
    for bx0, by0, bx1, by1 in others:
        ox0, oy0, ox1, oy1 = bx0 * f, by0 * f, bx1 * f, by1 * f
        if ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1:
            continue                                     # does not reach into the box
        gx = max(ox0 - xs1, xs0 - ox1)                   # gap between the spots, x / y
        gy = max(oy0 - ys1, ys0 - oy1)
        if gx <= 0 and gy <= 0:
            continue
        if gx >= gy:
            if ox0 >= xs1:
                x1 = min(x1, int((xs1 + ox0) / 2))
            else:
                x0 = max(x0, int(math.ceil((ox1 + xs0) / 2)))
        else:
            if oy0 >= ys1:
                y1 = min(y1, int((ys1 + oy0) / 2))
            else:
                y0 = max(y0, int(math.ceil((oy1 + ys0) / 2)))
    if x1 - x0 < max(4, xs1 - xs0) or y1 - y0 < max(4, ys1 - ys0):
        return full                                      # never smaller than the spot itself
    return (x0, y0, x1, y1)


def subtract_background(img, border=4, noise_k=3.0):
    """Remove the dark level (median of the ROI's outer `border` pixels) and the
    noise floor: pixels within noise_k x the edge noise are set to 0, as ISO 11146
    asks, so noise in the wings does not inflate second-moment widths."""
    if img.shape[0] <= 2 * border or img.shape[1] <= 2 * border:
        return np.clip(img - float(np.min(img)), 0, None)
    edge = np.concatenate([img[:border].ravel(), img[-border:].ravel(),
                           img[:, :border].ravel(), img[:, -border:].ravel()])
    med = float(np.median(edge))
    sd = float(np.median(np.abs(edge - med))) * 1.4826
    # the dark level is the MEAN of the edge (median is biased on 8-bit integer
    # data: dark 2.5 +-1 reads as 2 or 3, and over a big ROI that bias adds up to
    # more than a dim laser's whole signal); outliers beyond 4 sd are dropped
    keep = edge[np.abs(edge - med) <= max(4 * sd, 1.0)]
    dark = float(keep.mean()) if keep.size else med
    out = img - dark
    if noise_k > 0:
        out[out < noise_k * max(sd, 0.5)] = 0
    return out


def width_at_level(profile, level):
    """Full width where `profile` crosses `level` x its peak, between the
    OUTERMOST crossings, with linear interpolation between samples.
    Returns (width_px, left_px, right_px) or (nan, nan, nan)."""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 3:
        return math.nan, math.nan, math.nan
    peak = p.max()
    if peak <= 0:
        return math.nan, math.nan, math.nan
    thr = peak * level
    above = np.nonzero(p >= thr)[0]
    i0, i1 = above[0], above[-1]
    if i0 == 0 or i1 == p.size - 1:          # beam runs off the ROI edge
        return math.nan, math.nan, math.nan
    left = (i0 - 1) + (thr - p[i0 - 1]) / (p[i0] - p[i0 - 1])
    right = i1 + (p[i1] - thr) / (p[i1] - p[i1 + 1])
    return right - left, left, right


def moments(img):
    """Centroid and ISO 11146 D4-sigma widths (px) of a background-free image."""
    total = float(img.sum())
    if total <= 0:
        return None
    h, w = img.shape
    x = np.arange(w, dtype=np.float64)
    y = np.arange(h, dtype=np.float64)
    px, py = img.sum(axis=0, dtype=np.float64), img.sum(axis=1, dtype=np.float64)
    cx, cy = (px * x).sum() / total, (py * y).sum() / total
    vx = (px * (x - cx) ** 2).sum() / total
    vy = (py * (y - cy) ** 2).sum() / total
    return cx, cy, 4 * math.sqrt(vx), 4 * math.sqrt(vy)


def quick_centroid(raw, spot='brightest'):
    """Beam centroid of one raw frame (analysis px), or None. Used to tell whether
    the beam moved between frames before averaging them."""
    # 4x4 sensor-pixel blocks (= 2x2 analysis px, every Bayer colour inside): no
    # colour split needed, 16x fewer pixels, and averaging keeps the centroid exact
    sm = _block_mean(raw, 4)
    roi = auto_roi(sm, spot=spot)
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    m = moments(subtract_background(sm[y0:y1, x0:x1]))
    if m is None:
        return None
    # block centre (k + 1.5 sensor px) -> analysis px (cell k covers 2k..2k+1, centre 2k+0.5)
    return ((m[0] + x0) * 2.0 + 0.5, (m[1] + y0) * 2.0 + 0.5)


class MotionAwareStack(object):
    """Frame averaging that never smears a moving beam.

    Averaging N frames of a beam that moves blurs it along its path, so its widths
    grow with its speed (and look as if they depended on position). Here a frame is
    only averaged with earlier ones if the beam centre is within `tol_px` of theirs;
    otherwise the average restarts from this frame. Still beam: full noise
    reduction. Moving beam: single frames, no smear."""

    def __init__(self, tol_px=0.5):
        self.tol = tol_px
        self.frames = []
        self.cents = []
        self.restarts = 0
        self.spot = 'brightest'      # which spot's centre decides "the beam moved"

    def clear(self):
        self.frames, self.cents = [], []

    def add(self, raw, n):
        c = quick_centroid(raw, self.spot) if n > 1 else None
        if n > 1 and c is not None and self.cents:
            ref = self.cents[-1]
            if ref is None or abs(c[0] - ref[0]) > self.tol or abs(c[1] - ref[1]) > self.tol:
                self.clear()
                self.restarts += 1
        self.frames.append(raw)
        self.cents.append(c)
        if len(self.frames) > n:
            del self.frames[:len(self.frames) - n]
            del self.cents[:len(self.cents) - n]
        if len(self.frames) == 1:
            return self.frames[0]
        acc = self.frames[0].astype(np.float32)
        for fr in self.frames[1:]:
            acc += fr
        acc *= np.float32(1.0 / len(self.frames))
        return acc

    @property
    def count(self):
        return len(self.frames)


def moments_iso(img, iterations=6):
    """ISO 11146-3 second moments: background-subtracted data is NOT clipped (noise
    stays zero-mean, so no bias), and the moments are taken inside a rectangle of
    3 x the D4-sigma width around the centroid, iterated until it settles. A hard
    noise cut-off instead removes the faint wings and reads ~3-6 % narrow.
    Returns (cx, cy, d4s_x, d4s_y) in img px, or None."""
    h, w = img.shape
    x0, y0, x1, y1 = 0, 0, w, h
    res = None
    for _ in range(iterations):
        a = img[y0:y1, x0:x1]
        px = a.sum(axis=0, dtype=np.float64)
        py = a.sum(axis=1, dtype=np.float64)
        tot = px.sum()
        if tot <= 0:
            return res
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        cx, cy = (px * xs).sum() / tot, (py * ys).sum() / tot
        vx = max((px * (xs - cx) ** 2).sum() / tot, 1e-6)
        vy = max((py * (ys - cy) ** 2).sum() / tot, 1e-6)
        dx, dy = 4 * math.sqrt(vx), 4 * math.sqrt(vy)
        new = (max(0, int(round(cx - 1.5 * dx))), max(0, int(round(cy - 1.5 * dy))),
               min(w, int(round(cx + 1.5 * dx)) + 1), min(h, int(round(cy + 1.5 * dy)) + 1))
        res = (cx, cy, dx, dy)
        if new == (x0, y0, x1, y1) or new[2] - new[0] < 3 or new[3] - new[1] < 3:
            break
        x0, y0, x1, y1 = new
    return res


# ----------------------------------------------------------------- power

def channel_rates(r, g, b, exposure_us):
    """Summed signal of each colour plane per millisecond of exposure."""
    ms = max(exposure_us, 1.0) / 1000.0
    return np.array([r.sum(), g.sum(), b.sum()], dtype=np.float64) / ms


def solve_power(rates, response):
    """rates (3,) = response (3 x k) @ powers (k,) -> powers in W.
    Column j of `response` = channel rates per watt of laser j (calibrated by
    shooting a known power of that laser alone). Least squares, so 1-3 lasers
    work; negative results (noise) are clipped to 0."""
    if response is None or response.shape[1] == 0:
        return None
    p, *_ = np.linalg.lstsq(response, rates, rcond=None)
    return np.clip(p, 0, None)


# ----------------------------------------------------------------- one frame

class Optics(object):
    """Sensor-to-angle scale. mrad per raw sensor pixel = pitch / focal * 1000."""

    def __init__(self, pixel_pitch_um=3.45, focal_mm=50.0):
        self.pixel_pitch_um = pixel_pitch_um
        self.focal_mm = focal_mm

    def mrad_per_px(self, binning=2):
        # analysis runs on 2x2 Bayer cells -> one analysis pixel = 2 sensor pixels
        return binning * self.pixel_pitch_um * 1e-3 / self.focal_mm * 1000.0


def analyse(raw, exposure_us, optics, roi=None, auto=True, response=None,
            zero_px=None, dark=None, planes=None, spot='brightest'):
    """Full measurement of one (possibly stacked) BayerGR8 frame.
    roi is in analysis (half-res) pixels. dark = mean frame taken with the laser
    off (same exposure), subtracted first. planes = (r, g, b) if the caller already
    split this (dark-subtracted) frame. Returns a dict (all angles in mrad)."""
    if planes is not None:
        r, g, b = planes
    else:
        if dark is not None and dark.shape == raw.shape:
            raw = raw.astype(np.float32) - dark
        r, g, b = split_bayer_gr8(raw)
    lum = r + g + b
    no_beam = False
    if auto or roi is None:
        found = auto_roi(lum, spot=spot)
        no_beam = found is None
        roi = found or (0, 0, lum.shape[1], lum.shape[0])
    else:
        x0, y0, x1, y1 = roi
        no_beam = auto_roi(lum[y0:y1, x0:x1]) is None        # same SNR test inside a fixed ROI
    x0, y0, x1, y1 = roi
    # background removed but NOT clipped: zero-mean noise averages out of the
    # profiles and moments instead of biasing them (spatial data first)
    sub = subtract_background(lum[y0:y1, x0:x1], noise_k=0)
    out = {'roi': roi, 'shape': lum.shape, 'exposure_us': exposure_us,
           'peak_raw': bayer_channel_max(raw)}
    k = optics.mrad_per_px()
    out['mrad_per_px'] = k
    # no beam (e.g. laser off during the zero): say so instead of measuring noise,
    # which gave 70-120 mrad of "divergence" from pure noise
    m = None if no_beam else moments_iso(sub)
    if m is None:
        out['ok'] = False
        out['no_beam'] = True
        return out
    cx, cy, d4x, d4y = m
    px, py = sub.sum(axis=0), sub.sum(axis=1)
    fx = width_at_level(px, LEVEL_FWHM)
    fy = width_at_level(py, LEVEL_FWHM)
    ex = width_at_level(px, LEVEL_E2)
    ey = width_at_level(py, LEVEL_E2)
    # whole-beam centre: correct the Bayer sub-pixel offsets, weighted by how much
    # light each colour carries (a pure red beam would otherwise read 0.25 px off)
    sums = np.array([float(np.clip(c[y0:y1, x0:x1], 0, None).sum()) for c in (r, g, b)])
    off = (sums @ BAYER_OFFSET) / sums.sum() if sums.sum() > 0 else np.zeros(2)
    cxa, cya = cx + x0 + off[0], cy + y0 + off[1]
    zx, zy = zero_px if zero_px else (lum.shape[1] / 2.0, lum.shape[0] / 2.0)
    out.update({
        'ok': True,
        # separate spots in the picture; only the brightest is measured
        'n_spots': count_spots(lum),
        'cx_px': cxa, 'cy_px': cya,
        'profile_x': px, 'profile_y': py,
        'fwhm_x': fx[0] * k, 'fwhm_y': fy[0] * k,
        'e2_x': ex[0] * k, 'e2_y': ey[0] * k,
        'd4s_x': d4x * k, 'd4s_y': d4y * k,
        'cross_x': (fx[1], fx[2], ex[1], ex[2]),
        'cross_y': (fy[1], fy[2], ey[1], ey[2]),
        # alignment: pointing angle relative to the zero point (image centre by default)
        'align_x': (cxa - zx) * k, 'align_y': -(cya - zy) * k,   # +y = up
    })
    out['fwhm_geo'] = math.sqrt(out['fwhm_x'] * out['fwhm_y']) if fx[0] == fx[0] and fy[0] == fy[0] else math.nan
    out['e2_geo'] = math.sqrt(out['e2_x'] * out['e2_y']) if ex[0] == ex[0] and ey[0] == ey[0] else math.nan
    # power needs ALL the light: a generous ROI that does not shrink for a dim laser
    # (a beam-sized ROI would cut a different share of the wings at each brightness)
    proi = auto_roi(lum, level=0.01, margin=3.0, spot=spot) or roi
    if not auto and roi is not None:
        proi = roi
    if proi[2] - proi[0] < 2 or proi[3] - proi[1] < 2:
        proi = roi                                       # never measure an empty box
    px0, py0, px1, py1 = proi
    rr, gg, bb = (c[py0:py1, px0:px1] for c in (r, g, b))
    # power: plain dark-level subtraction, NO noise clipping (clipping throws away
    # the wings of dim beams and biases watts low by an amount that varies with brightness)
    rates = channel_rates(subtract_background(rr, noise_k=0), subtract_background(gg, noise_k=0),
                          subtract_background(bb, noise_k=0), exposure_us)
    out['rates'] = rates
    pw = solve_power(rates, response) if response is not None else None
    out['powers'] = pw
    if pw is not None:
        tot = float(pw.sum())
        out['power_total'] = tot
        # power density: watts per mrad of beam (1/e^2 width), per axis and per mrad^2
        if out['e2_x'] > 0 and out['e2_y'] > 0:
            out['dens_x'] = tot / out['e2_x']
            out['dens_y'] = tot / out['e2_y']
            out['dens_point'] = tot / (math.pi / 4 * out['e2_x'] * out['e2_y'])
    return out


def robust_peak(raw, top_n=12):
    """Brightest level of the beam that is not a single hot or noisy pixel: the
    mean of the `top_n` brightest Bayer pixels of the brightest colour."""
    best = 0.0
    for plane in (raw[0::2, 1::2], raw[0::2, 0::2], raw[1::2, 1::2], raw[1::2, 0::2]):
        flat = plane.ravel()
        n = min(top_n, flat.size)
        v = float(np.partition(flat, flat.size - n)[flat.size - n:].mean())
        best = max(best, v)
    return best


class AutoExposure(object):
    """Auto exposure that settles in a few frames and never hunts.

    - measures a ROBUST peak (mean of the brightest few Bayer pixels, so one hot
      pixel does not drive it) and the saturated fraction of the beam
    - works in log space: next = current * target/peak, damped, with a dead band
      (hysteresis) around the target so it stops adjusting once it is close
    - saturated: steps down by how saturated it is (x0.5 up to x0.1), because a
      clipped peak says nothing about how far over it is
    - dark: searches UP geometrically (x4 per step) instead of jumping to 5 ms
    - waits for a frame actually exposed at the new value before judging again
      (cameras deliver 1-2 frames from the old setting after a change)
    - reports when it is pinned at the shortest or longest exposure
    """

    def __init__(self, target=0.85, band=0.08, full=255.0):
        self.target = target          # fraction of full scale for the robust peak
        self.band = band              # +-band around target counts as settled
        self.full = full
        self.pending = None           # exposure we asked for, until a frame confirms it
        self.wait = 0
        self.state = 'idle'

    def update(self, raw, exposure_us, lo_us, hi_us):
        """-> new exposure in us (or the current one). Sets self.state to one of
        settled / adjusting / saturated / dark / at_min / at_max / waiting."""
        if self.pending is not None:
            if abs(exposure_us - self.pending) > max(2.0, 0.02 * self.pending) and self.wait < 6:
                self.wait += 1                 # still a frame from before the change
                self.state = 'waiting'
                return exposure_us
            self.pending, self.wait = None, 0
            self.state = 'waiting'
            return exposure_us                 # skip one more: this frame may straddle the change
        peak = robust_peak(raw)
        sat = float(np.count_nonzero(raw >= self.full - 1))
        lit = float(np.count_nonzero(raw >= 0.5 * max(peak, 1.0))) or 1.0
        frac = peak / self.full
        if sat >= 3:                                   # clipped: step down by how bad it is
            ratio = sat / lit
            factor = 0.5 if ratio < 0.1 else (0.25 if ratio < 0.4 else 0.1)
            new, self.state = exposure_us * factor, 'saturated'
        elif peak < 6 * self.full / 255.0:             # nothing visible: search upwards
            new, self.state = exposure_us * 4.0, 'dark'
        elif abs(frac - self.target) <= self.band:
            self.state = 'settled'
            return exposure_us
        else:
            ratio = self.target / max(frac, 1e-3)
            ratio = min(max(ratio, 0.1), 10.0)
            new = exposure_us * ratio ** 0.85          # damped step in log space
            self.state = 'adjusting'
        clamped = float(min(max(new, lo_us), hi_us))
        if clamped != new:
            self.state = 'at_min' if new < lo_us else 'at_max'
        if abs(clamped - exposure_us) < 0.5:
            return exposure_us
        self.pending = clamped
        return clamped


def next_exposure(exposure_us, peak_raw, lo_us, hi_us, target=0.85, full=255):
    """Simple one-shot rule (kept for reference / tests; the app uses AutoExposure)."""
    """Auto exposure: scale so the brightest Bayer pixel sits at `target` of full
    Saturated -> halve. Dark -> back to 5 ms."""
    peak = max(peak_raw)
    if peak < 8 * full / 255.0:
        new = 5000.0
    elif peak >= full - 1:
        new = exposure_us * 0.5
    else:
        new = exposure_us * (target * full) / peak
        if abs(new - exposure_us) < 0.05 * exposure_us:     # dead band
            return exposure_us
        new = exposure_us + 0.7 * (new - exposure_us)        # damp
    return float(min(max(new, lo_us), hi_us))
