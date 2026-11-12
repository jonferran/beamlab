"""Camera + analysis process. The UI never touches the camera or does numpy work:
it sends commands on `cmd_q` and shows whatever arrives on `out_q`.

Messages out (dicts with 'type'):
  'opened'  {name, model, serial, exposure_range}
  'error'   {text}
  'frame'   {preview (HxWx3 uint8), result (analyse() dict, profiles as lists), fps, temp}
  'calibrated' {laser, rates_per_w}
"""
import queue
import time
import traceback

import numpy as np

from . import analysis
from .cameras import open_camera

PREVIEW_MAX_W = 960

# The display picture is worked on in horizontal bands on several cores, like a
# pixel shader: numpy releases the GIL inside its loops, so threads really run in
# parallel. Created on first use, inside the camera process.
_POOL = None
_BANDS = 1


def _parallel(fn, n_rows, min_rows=64):
    """Run fn(r0, r1) over row bands of 0..n_rows on the thread pool -> list."""
    global _POOL, _BANDS
    if _POOL is None:
        import os
        from concurrent.futures import ThreadPoolExecutor
        _BANDS = max(1, min(6, (os.cpu_count() or 2) - 1))
        _POOL = ThreadPoolExecutor(_BANDS) if _BANDS > 1 else False
    n = min(_BANDS, max(1, n_rows // min_rows))
    if n <= 1 or not _POOL:
        return [fn(0, n_rows)]
    edges = [n_rows * i // n for i in range(n + 1)]
    futs = [_POOL.submit(fn, edges[i], edges[i + 1]) for i in range(n)]
    return [f.result() for f in futs]


def unmix_planes(r, g, b, response):
    """Per-pixel laser intensities (k planes) from the R,G,B Bayer planes, using
    the calibrated response so a green laser that leaks into the red pixels is
    shown as green only. Without a calibration, the raw planes are the lasers."""
    stack = np.stack([r, g, b], axis=-1)                 # H,W,3
    if response is None:
        return stack
    return np.clip(stack @ unmix_matrix(response).T, 0, None)   # H,W,k


def unmix_matrix(response):
    """k x 3 matrix that turns camera R,G,B into laser intensities (each laser's
    response column scaled so its main colour = 1)."""
    response = np.asarray(response, np.float64)
    norm = response / np.maximum(response.max(axis=0, keepdims=True), 1e-12)
    return np.linalg.pinv(norm).astype(np.float32)       # k x 3


def colour_matrix(sigs):
    """3x3 colour matrix from the learned laser colours (display only): column i =
    how laser i shows in the camera's R, G, B pixels. A laser not learned yet keeps
    its own camera colour (column = unit vector). None when nothing is learned."""
    if not sigs or not any(sigs):
        return None
    m = np.eye(3)
    for i, s in enumerate(list(sigs)[:3]):
        if s and max(s) > 0:
            m[:, i] = np.asarray(s, float) / max(s)
    if abs(np.linalg.det(m)) < 1e-3:         # two lasers learned the same: unusable
        return None
    return m


# Where each Bayer colour really sits inside its 2x2 cell, in analysis px relative
# to the cell centre (BayerGR8: G at (0,0) and (1,1), R at (1,0), B at (0,1) sensor px).
# Without this, red reads 0.25 px right/up and blue 0.25 px left/down of green: a
# false colour-to-colour misalignment of ~0.035 mrad with the default optics.
BAYER_OFFSET = analysis.BAYER_OFFSET


def plane_offsets(response, k):
    """Sub-pixel offset of each laser plane: the raw colours' own offsets, or for
    unmixed lasers the offsets weighted by how much each colour sees that laser."""
    if response is None:
        return BAYER_OFFSET[:k]
    w = np.abs(np.asarray(response, float))                 # 3 x k
    w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    return (w.T @ BAYER_OFFSET)                             # k x 2


def laser_spots(planes, rel_level=0.05, offsets=None, mrad_per_px=None, divs=None, spot='brightest'):
    """For each laser plane: (centroid, spot box) in analysis px, so auto zoom can
    keep every spot in view and the picture can mark each colour. The box is the
    spot's extent down to ~2 % of its peak (wings included). (None, None) for a
    laser that is not on (peak below rel_level of the brightest).
    If `divs` is a list, each laser's own divergence is appended to it (dict of
    fwhm/e2/d4s x/y in mrad, or None) - same method as the whole-beam numbers."""
    # one contiguous copy per laser: max() over the interleaved H,W,k array took 10 ms
    pl = [np.ascontiguousarray(planes[..., j]) for j in range(planes.shape[-1])]
    peaks = [float(p.max()) for p in pl]
    top = max(peaks) if peaks else 0.0
    cxy, boxes = [], []
    for j, p in enumerate(pl):
        div = None
        if top <= 0 or peaks[j] < rel_level * top or peaks[j] < 4:
            cxy.append(None)
            boxes.append(None)
            if divs is not None:
                divs.append(None)
            continue
        box = analysis.auto_roi(p, level=0.02, margin=1.0, spot=spot)   # the measured spot, wings included
        roi = analysis.auto_roi(p, spot=spot)       # measuring ROI: same rule as the whole beam
        if box is None or roi is None:
            cxy.append(None)
            boxes.append(None)
            if divs is not None:
                divs.append(None)
            continue
        x0, y0, x1, y1 = roi
        sub = analysis.subtract_background(p[y0:y1, x0:x1], noise_k=0)
        m = analysis.moments_iso(sub)
        if m is None:
            cxy.append(None)
        else:
            ox, oy = offsets[j] if offsets is not None and j < len(offsets) else (0.0, 0.0)
            cxy.append((m[0] + x0 + ox, m[1] + y0 + oy))
            if mrad_per_px:
                k = mrad_per_px
                px, py = sub.sum(axis=0), sub.sum(axis=1)
                div = {'fwhm_x': analysis.width_at_level(px, analysis.LEVEL_FWHM)[0] * k,
                       'fwhm_y': analysis.width_at_level(py, analysis.LEVEL_FWHM)[0] * k,
                       'e2_x': analysis.width_at_level(px, analysis.LEVEL_E2)[0] * k,
                       'e2_y': analysis.width_at_level(py, analysis.LEVEL_E2)[0] * k,
                       'd4s_x': m[2] * k, 'd4s_y': m[3] * k}
        boxes.append(box)
        if divs is not None:
            divs.append(div)
    return cxy, boxes


def laser_centroids(planes, rel_level=0.05):
    """Centroid of each laser's own spot (analysis px), or None if that laser is off."""
    return laser_spots(planes, rel_level)[0]


def render(planes, display, roi, detail_box=None):
    """Display image with every laser painted in the colour the user chose for it
    (so a laser blocked by your goggles can be shown in a colour they pass).
    display = {'colors': [(r,g,b) 0-255 per laser], 'auto_bright': bool, 'gain': float,
    'atten' / 'boost' / 'lcontrast': per laser, 'contrast': all, 'raw_planes': bool,
    'unmix': k x 3 matrix applied to `planes` (camera R,G,B) first, or None}
    -> (preview of the whole sensor, its step).
    With detail_box = (x0, y0, x1, y1) analysis px (the zoomed-in window): also a
    picture of just that window at up to full analysis resolution, painted with the
    same brightness scale -> (preview, step, {'img', 'box', 'step'} or None).

    Like a pixel shader it only works on the pixels it sends (the <= 960 px wide
    preview, the zoom window) plus the ROI for the brightness peak - never on the
    whole 4K frame - and the per-laser curves are lookup tables, not powers."""
    U = display.get('unmix')
    U = None if U is None else np.asarray(U, np.float32)              # k x 3
    k = planes.shape[-1] if U is None else U.shape[0]
    cols = np.array(display['colors'][:k], dtype=np.float32) / 255.0   # k,3
    if cols.shape[0] < k:
        cols = np.vstack([cols, np.ones((k - cols.shape[0], 3), np.float32)])

    def per_laser_list(key, dflt):
        v = list(display.get(key) or [])[:k]
        return np.array(v + [dflt] * (k - len(v)), np.float32)
    # per-laser brightness (0..1) x boost; auto brightness is measured BEFORE them,
    # so turning a laser down really dims it
    f = np.clip(per_laser_list('atten', 1.0), 0, 1) * np.clip(per_laser_list('boost', 1.0), 0.05, 50.0)
    gam = (2.0 ** (np.clip(per_laser_list('lcontrast', 0.0), -100, 100) / 50.0)).astype(np.float32)
    balance = bool(display.get('balance'))
    raw = bool(display.get('raw_planes')) and U is None
    gain = float(display.get('gain', 1.0))
    cmax = np.maximum(cols.max(axis=1), 1e-6)            # brightest channel of each colour
    H, W = planes.shape[:2]
    step = max(1, int(np.ceil(W / PREVIEW_MAX_W)))

    def mm(a, M):
        """(h,w,n) @ (n,m). einsum, NOT `@`: `@` calls the bundled BLAS, which is not
        safe to call from several threads at once (it crashed the process with an
        access violation when the bands ran in parallel); einsum uses numpy's own
        loops and is as fast here."""
        M = np.asarray(M, np.float32)
        return np.einsum('...n,nm->...m', a, M)

    def lasers(pl):
        pl = np.ascontiguousarray(pl, dtype=np.float32)   # strided slices are slow to work on
        if U is not None:
            pl = mm(pl, U.T)
            np.maximum(pl, 0, out=pl)
        return pl
    # the regions that are sent (+ the ROI, at full resolution, for the peak)
    regions = [(planes[::step, ::step], 0, 0, step)]
    dbox = None
    if detail_box is not None:
        bx0, by0, bx1, by1 = (int(round(v)) for v in detail_box)
        bx0, by0 = max(0, bx0), max(0, by0)
        bx1, by1 = min(W, bx1), min(H, by1)
        if bx1 - bx0 >= 8 and by1 - by0 >= 8:
            dstep = max(1, int(np.ceil((bx1 - bx0) / PREVIEW_MAX_W)))
            if dstep < step:                     # only worth sending when it is finer
                dbox = (bx0, by0, bx1, by1, dstep)
                regions.append((planes[by0:by1:dstep, bx0:bx1:dstep], bx0, by0, dstep))
    # stage 1, in parallel bands: separate the lasers, plain colours, band peak
    L, base, peaks, lpeaks = [], [], [], []
    for pl, _, _, _ in regions:
        def stage1(r0, r1, pl=pl):
            l = lasers(pl[r0:r1])
            b = mm(l, cols)
            lp = l.reshape(-1, l.shape[-1]).max(axis=0) if (balance and l.size) else None
            return l, b, float(b.max()) if b.size else 0.0, lp
        parts = _parallel(stage1, pl.shape[0])
        L.append([p[0] for p in parts])                  # kept as bands for stage 2
        base.append([p[1] for p in parts])
        peaks += [p[2] for p in parts]
        lpeaks += [p[3] for p in parts if p[3] is not None]
    rx0, ry0 = max(0, int(roi[0])), max(0, int(roi[1]))
    rx1, ry1 = min(W, int(roi[2])), min(H, int(roi[3]))
    roi_l = None
    if rx1 > rx0 and ry1 > ry0 and (rx1 - rx0) * (ry1 - ry0) < 0.5 * H * W:
        roi_l = lasers(planes[ry0:ry1, rx0:rx1])         # the beam at full resolution
        if balance:
            lpeaks.append(roi_l.reshape(-1, k).max(axis=0))
    if display.get('auto_bright', True):
        peak = max(peaks)
        if roi_l is not None:
            peak = max(peak, float(mm(roi_l, cols).max()))
        scale = 255.0 / peak if peak > 1 else 1.0
    else:
        scale = 255.0 / float(display.get('full', 255.0))   # full scale -> white
    norm = scale / 255.0                                 # -> 0..1 at the auto-brightness peak
    if balance and lpeaks:
        # Balance lasers: each laser's own brightest point to full brightness. The
        # camera is far less sensitive to 638 nm red than the eye-balanced lasers
        # look (IR-cut filter, sensor response), so red came out ~5x dimmer than
        # green / blue. A laser that is off (below 3 % of the brightest) is left as
        # it is, so its noise is not blown up; at most 30x.
        lp = np.max(np.stack(lpeaks), axis=0) * cmax     # each laser's peak as shown
        on = lp >= 0.03 * max(float(lp.max()), 1e-12)
        bf = np.where(on, 1.0 / np.maximum(lp * norm, 1e-12), 1.0)
        f = f * np.clip(bf, 1.0 / 30, 30.0).astype(np.float32)
    per_laser = bool((np.abs(f - 1) > 1e-3).any() or (np.abs(gam - 1) > 1e-3).any())
    NQ = 4096                                    # lookup steps (finer than the 256 output levels)
    curves = table = None
    if per_laser:
        x = np.arange(NQ + 1, dtype=np.float32) / NQ
        curves = np.stack([x ** g for g in gam]).astype(np.float32)   # k x (NQ+1)

    def index(v, coef):
        """v * coef -> lookup index 0..NQ (in place on a temporary)."""
        t = v * np.float32(coef * NQ)
        np.clip(t, 0, NQ, out=t)
        t += 0.5
        return t.astype(np.int32)

    def shade(l, b):
        """lasers (h,w,k) + their plain colours b (h,w,3) -> 0..255 float."""
        if not per_laser:
            return b * np.float32(scale * gain)
        if raw:
            # not separated: the planes are the camera's raw R/G/B pixels and one
            # laser shows in all three (a red laser leaks into G and B). Changing only
            # its main plane would shift its hue, so each pixel is changed as a whole,
            # by the settings of the planes it is made of (keeps the hue).
            # (measured: max over the last axis and (N,3) table reads are slow in numpy;
            # whole-plane ops, 1-D lookups and flattened matrix products are fast)
            ls = [l[..., i] for i in range(k)]
            tot = ls[0] + ls[1]
            for i in range(2, k):
                tot += ls[i]
            np.maximum(tot, 1e-6, out=tot)
            m = np.maximum(np.maximum(b[..., 0], b[..., 1]), b[..., 2])   # pixel brightness
            qi = index(m * mm(l, f[:, None])[..., 0] / tot, norm)        # x its lasers' boost
            m2 = ls[0] * curves[0].take(qi)
            for i in range(1, k):
                m2 += ls[i] * curves[i].take(qi)
            m2 /= tot                                    # its lasers' curves, by share
            m2 *= np.float32(255.0 * gain)
            m2 /= np.maximum(m, 1e-9)
            return b * m2[..., None]
        # separated (calibrated or learned): each plane is one laser, own curve
        u2 = np.empty(l.shape, np.float32)
        for i in range(k):
            u2[..., i] = curves[i].take(index(l[..., i], norm * cmax[i] * f[i]))
        return mm(u2, cols / cmax[:, None] * np.float32(255.0 * gain))
    # master contrast (-100..100): a gamma curve, 2^(c/50) = 0.25..4. Below 0 lifts the
    # faint wings, above 0 keeps only the bright core. Display only.
    c = float(display.get('contrast', 0) or 0)
    lut = None
    if abs(c) > 0.5:
        g = 2.0 ** (max(-100.0, min(100.0, c)) / 50.0)
        lut = np.round(255.0 * (np.arange(256) / 255.0) ** g).astype(np.uint8)
    rc = np.array(display.get('roi_color', (0, 255, 0)), np.uint8)

    def paint(lb, bb, ox, oy, st):
        """bands of lasers / plain colours -> uint8 picture with contrast and the ROI
        drawn (ox, oy = where this part starts on the sensor, analysis px; st = its
        subsampling). Each band is shaded on its own core."""
        rows = [x.shape[0] for x in bb]
        out = np.empty((sum(rows), bb[0].shape[1], 3), np.uint8)
        starts = np.concatenate([[0], np.cumsum(rows)]).astype(int)

        def stage2(i):
            v = shade(lb[i], bb[i])
            np.clip(v, 0, 255, out=v)
            o = v.astype(np.uint8)
            out[starts[i]:starts[i + 1]] = lut[o] if lut is not None else o
        if len(bb) > 1 and _POOL:
            for f in [_POOL.submit(stage2, i) for i in range(len(bb))]:
                f.result()
        else:
            for i in range(len(bb)):
                stage2(i)
        if roi[2] - 1 > roi[0] and roi[3] - 1 > roi[1]:
            h, w = out.shape[:2]
            x0 = int(round((roi[0] - ox) / st))
            y0 = int(round((roi[1] - oy) / st))
            x1 = int(round((min(roi[2], W) - 1 - ox) / st))
            y1 = int(round((min(roi[3], H) - 1 - oy) / st))
            xs0, xs1 = max(x0, 0), min(x1, w)
            ys0, ys1 = max(y0, 0), min(y1, h)
            for yy in (y0, y1):
                if 0 <= yy < h and xs1 > xs0:
                    out[yy, xs0:xs1] = rc
            for xx in (x0, x1):
                if 0 <= xx < w and ys1 > ys0:
                    out[ys0:ys1, xx] = rc
        return out
    preview = paint(L[0], base[0], 0, 0, step)
    if detail_box is None:
        return preview, step
    detail = None
    if dbox is not None:
        bx0, by0, bx1, by1, dstep = dbox
        detail = {'img': paint(L[1], base[1], bx0, by0, dstep), 'box': (bx0, by0, bx1, by1), 'step': dstep}
    return preview, step, detail


def _put_latest(q, msg):
    """Never block the camera loop on a slow UI: drop the oldest frame instead."""
    try:
        q.put_nowait(msg)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


def run(cmd_q, out_q, cfg):
    """Process entry. cfg: backend, serial, exposure_us, auto_exposure, stack,
    roi, auto_roi, optics {pixel_pitch_um, focal_mm}, response (3xk list or None),
    display {...}, zero_px (the crosshair, analysis px; alignment is measured from it)."""
    import faulthandler
    from .settings import ERROR_LOG, log_error
    # a hard crash (e.g. inside a camera DLL) or a stuck loop still leaves a stack trace
    try:
        _crash_fh = open(ERROR_LOG, 'a')
        faulthandler.enable(file=_crash_fh)
    except OSError:
        _crash_fh = None
    log_error('Camera process started')
    cam = None
    try:
        cam = open_camera(cfg.get('backend', 'auto'), cfg.get('serial'))
        cam.set_exposure(cfg.get('exposure_us', 5000.0))
        _put_latest(out_q, {'type': 'opened', 'name': cam.name, 'model': cam.model,
                            'serial': cam.serial, 'exposure_range': cam.exposure_range,
                            'pixel_format': getattr(cam, 'bayer', 'BayerGR8')})
    except Exception as e:  # noqa: BLE001
        log_error('Camera open failed: %s' % e)
        out_q.put({'type': 'error', 'text': str(e)})
        return
    log_error('Camera opened: %s %s %s, exposure range %s, pixel format %s' % (
        cam.name, cam.model, cam.serial, cam.exposure_range, getattr(cam, 'bayer', 'BayerGR8')))
    exp = float(cfg.get('exposure_us', 5000.0))
    full = float(getattr(cam, 'full', 255.0))     # 4095 for a 12-bit camera, 255 for 8-bit
    ae = analysis.AutoExposure(full=full)
    # Zero on connect: keep the laser off until the app is running, so the first
    # frames are a dark frame (sensor black level + fixed-pattern offset).
    zero = [int(cfg.get('zero_frames', 10)), None, 0] if cfg.get('zero_on_connect', True) else None
    dark, dark_exp = None, None
    stack = analysis.MotionAwareStack()
    calib = None            # [laser, power_w, n_left, rates_sum, n]
    t_last, fps = time.time(), 0.0
    n_frames = 0
    try:
        while True:
            # stall detector: if this pass takes > limit, dump where it is stuck to errors.log
            if _crash_fh is not None:
                limit = max(20.0, 6 * exp / 1e6 * max(1, int(cfg.get('stack', 1))))
                faulthandler.dump_traceback_later(limit, repeat=True, file=_crash_fh)
            # ------------------------------------------------ commands
            while True:
                try:
                    c = cmd_q.get_nowait()
                except queue.Empty:
                    break
                op = c[0]
                if op == 'quit':
                    return
                if not (op == 'set' and isinstance(c[1], dict) and set(c[1]) == {'detail_view'}):
                    # (the zoom window is sent often while zooming: not logged)
                    log_error('Command at frame %d: %s %s' % (n_frames, op, sorted(c[1].keys())
                                                              if op == 'set' and isinstance(c[1], dict) else ''))
                if op == 'set':
                    cfg.update(c[1])
                    if 'exposure_us' in c[1]:
                        exp = float(c[1]['exposure_us'])
                        cam.set_exposure(exp)
                        stack.clear()
                    if 'stack' in c[1]:
                        stack.clear()
                elif op == 'calibrate':        # ('calibrate', laser_index, power_w, frames)
                    calib = [c[1], float(c[2]), int(c[3]), np.zeros(3), 0]
                elif op == 'zero':             # ('zero', frames): laser OFF -> dark frame
                    zero = [int(c[1]), None, 0]
                elif op == 'clear_zero':
                    dark = None
                    dark_exp = None
            # ------------------------------------------------ grab + stack
            raw, exp_used, temp = cam.grab()
            n_frames += 1
            n = max(1, int(cfg.get('stack', 1)))
            stack.spot = cfg.get('spot', 'right')
            frame = stack.add(raw, n)        # restarts itself when the beam moves
            if zero is not None and zero[2] == 0 and zero[1] is None and max(analysis.bayer_channel_max(raw)) > 60 * full / 255.0:
                out_q.put({'type': 'error', 'text': 'Zero skipped: the sensor sees light. Turn the '
                           'laser off and press "Zero (laser off)".'})
                zero = None
            if zero is not None:
                rf = raw.astype(np.float64)
                zero[1] = rf if zero[1] is None else zero[1] + rf
                zero[2] += 1
                if zero[2] >= zero[0]:
                    dark = (zero[1] / zero[2]).astype(np.float32)
                    dark_exp = exp_used
                    out_q.put({'type': 'zeroed', 'exposure_us': exp_used,
                               'level': float(dark.mean())})
                    zero = None
            # the dark frame is the sensor's black level / fixed pattern, which does not
            # depend on exposure at these times (ms), so it is used at every exposure
            use_dark = dark
            # ------------------------------------------------ analyse
            resp = cfg.get('response')
            resp = np.array(resp, dtype=float) if resp else None
            optics = analysis.Optics(**cfg.get('optics', {}))
            work = frame.astype(np.float32) - use_dark if use_dark is not None else frame
            rgb = analysis.split_bayer_gr8(work)          # split ONCE, shared by analysis + display
            res = analysis.analyse(work, exp_used, optics, roi=cfg.get('roi'),
                                   auto=cfg.get('auto_roi', True), response=resp,
                                   zero_px=cfg.get('zero_px'), planes=rgb,
                                   spot=cfg.get('spot', 'right'))
            res['dark'] = 'on' if use_dark is not None else ('recording' if zero is not None else 'none')
            if zero is not None:
                res['powers'] = None          # no watts until the zero is in
            res['peak_raw'] = analysis.bayer_channel_max(raw)     # judge the raw frame
            res['full'] = full                                    # its saturation level
            res['avg'] = (stack.count, n)                         # frames averaged / asked for
            # ------------------------------------------------ auto exposure
            ae.target = float(cfg.get('ae_target', 0.85))
            if cfg.get('auto_exposure') and (calib is not None or zero is not None):
                # calibrating / zeroing: the exposure is held still (a power
                # calibration needs one fixed exposure; a dark frame with the laser off
                # would otherwise drive auto exposure to its maximum); afterwards auto
                # exposure carries on by itself - no need to switch it off and on
                ae.pending = None
                ae.state = 'held while calibrating' if calib is not None else 'held while zeroing'
                res['ae_state'] = ae.state
            elif cfg.get('auto_exposure'):
                lo, hi = cam.exposure_range
                new = ae.update(raw, exp_used, lo, hi)
                if new != exp_used:
                    exp = new
                    cam.set_exposure(exp)
                    stack.clear()
                res['ae_state'] = ae.state
            else:
                ae.pending, ae.state = None, 'off'
            # ------------------------------------------------ calibration
            if calib is not None and res.get('ok'):
                if max(res['peak_raw']) >= full * 254.0 / 255.0:
                    out_q.put({'type': 'error', 'text': 'Calibration stopped: the sensor is '
                               'saturated. Lower the exposure (or the laser power) and try again.'})
                    calib = None
                else:
                    calib[3] += res['rates']
                    calib[4] += 1
                    if calib[4] >= calib[2]:
                        rates = calib[3] / calib[4]
                        out_q.put({'type': 'calibrated', 'laser': calib[0],
                                   'rates_per_w': (rates / calib[1]).tolist()})
                        calib = None
            # ------------------------------------------------ preview
            r, g, b = (np.clip(p, 0, None) for p in rgb)
            planes = unmix_planes(r, g, b, resp)
            # plane j is laser idx[j]: all three raw colours, or only the calibrated lasers
            idx = list(cfg.get('laser_idx') or [0, 1, 2]) if resp is not None else [0, 1, 2]
            res['laser_idx'] = idx
            divs = []
            res['laser_cxy'], res['laser_box'] = laser_spots(
                planes, offsets=plane_offsets(resp, planes.shape[-1]),
                mrad_per_px=res['mrad_per_px'], divs=divs, spot=cfg.get('spot', 'right'))
            res['laser_div'] = divs                 # per colour divergence (mrad), by plane
            disp = dict(cfg['display'])
            sig_m = colour_matrix(cfg.get('colour_sig')) if resp is None else None
            if sig_m is not None:
                # learned colours (display only): overlapping beams separate on screen,
                # so each laser's sliders act on that laser alone (render applies the
                # matrix only to the pixels it draws)
                disp['unmix'] = unmix_matrix(sig_m)
            disp['colors'] = [cfg['display']['colors'][i] for i in idx]
            att = cfg['display'].get('atten') or [1.0, 1.0, 1.0]
            disp['atten'] = [att[i] for i in idx]
            for key, dflt in (('boost', 1.0), ('lcontrast', 0.0)):
                v = list(cfg['display'].get(key) or [])
                v += [dflt] * (3 - len(v))
                disp[key] = [v[i] for i in idx]
            disp['raw_planes'] = resp is None and sig_m is None
            disp['full'] = full
            out = render(planes, disp, res['roi'], cfg.get('detail_view'))
            preview, step = out[0], out[1]
            detail = out[2] if len(out) > 2 else None
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - t_last, 1e-6)
            t_last = now
            for key in ('profile_x', 'profile_y', 'rates', 'powers'):
                if res.get(key) is not None:
                    res[key] = np.asarray(res[key]).tolist()
            _put_latest(out_q, {'type': 'frame', 'preview': preview, 'step': step, 'detail': detail,
                                'result': res, 'fps': fps, 'temp': temp,
                                'calibrating': calib is not None})
    except Exception:  # noqa: BLE001
        log_error('Camera loop stopped at frame %d:\n%s' % (n_frames, traceback.format_exc()))
        try:            # never block forever on a full queue (the window may be busy)
            out_q.put({'type': 'error', 'text': 'Camera loop stopped:\n' + traceback.format_exc(limit=3)},
                      timeout=5)
        except queue.Full:
            pass
    finally:
        if _crash_fh is not None:
            faulthandler.cancel_dump_traceback_later()
        if cam is not None:
            cam.close()
        log_error('Camera process ending after %d frames' % n_frames)
