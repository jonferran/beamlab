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


def unmix_planes(r, g, b, response):
    """Per-pixel laser intensities (k planes) from the R,G,B Bayer planes, using
    the calibrated response so a green laser that leaks into the red pixels is
    shown as green only. Without a calibration, the raw planes are the lasers."""
    stack = np.stack([r, g, b], axis=-1)                 # H,W,3
    if response is None:
        return stack
    norm = response / np.maximum(response.max(axis=0, keepdims=True), 1e-12)
    pinv = np.linalg.pinv(norm)                          # k x 3
    return np.clip(stack @ pinv.T, 0, None)              # H,W,k


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


def laser_spots(planes, rel_level=0.05, offsets=None, mrad_per_px=None, divs=None):
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
        box = analysis.auto_roi(p, level=0.02, margin=1.0)
        roi = analysis.auto_roi(p)                  # measuring ROI: same rule as the whole beam
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


def render(planes, display, roi):
    """Display image with every laser painted in the colour the user chose for it
    (so a laser blocked by your goggles can be shown in a colour they pass).
    display = {'colors': [(r,g,b) 0-255 per laser], 'auto_bright': bool, 'gain': float}"""
    k = planes.shape[-1]
    cols = np.array(display['colors'][:k], dtype=np.float32) / 255.0   # k,3
    if cols.shape[0] < k:
        cols = np.vstack([cols, np.ones((k - cols.shape[0], 3), np.float32)])
    # per-laser display attenuation (0..1): dim a strong laser so the others show;
    # applied BEFORE auto brightness, so the attenuated laser really ends up dimmer
    att = list(display.get('atten') or [])[:k]
    att += [1.0] * (k - len(att))
    cols = cols * np.clip(np.array(att, np.float32), 0, 1)[:, None]
    img = planes @ cols                                  # H,W,3
    if display.get('auto_bright', True):
        peak = float(img.max())
        scale = 255.0 / peak if peak > 1 else 1.0
    else:
        scale = 1.0
    img = np.clip(img * scale * float(display.get('gain', 1.0)), 0, 255).astype(np.uint8)
    # draw the ROI
    x0, y0, x1, y1 = roi
    x1, y1 = min(x1, img.shape[1]) - 1, min(y1, img.shape[0]) - 1
    if x1 > x0 and y1 > y0:
        rc = np.array(display.get('roi_color', (0, 255, 0)), np.uint8)
        img[y0, x0:x1] = rc
        img[y1, x0:x1] = rc
        img[y0:y1, x0] = rc
        img[y0:y1, x1] = rc
    step = max(1, int(np.ceil(img.shape[1] / PREVIEW_MAX_W)))
    return img[::step, ::step].copy(), step


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
    cam = None
    try:
        cam = open_camera(cfg.get('backend', 'auto'), cfg.get('serial'))
        cam.set_exposure(cfg.get('exposure_us', 5000.0))
        _put_latest(out_q, {'type': 'opened', 'name': cam.name, 'model': cam.model,
                            'serial': cam.serial, 'exposure_range': cam.exposure_range})
    except Exception as e:  # noqa: BLE001
        out_q.put({'type': 'error', 'text': str(e)})
        return
    exp = float(cfg.get('exposure_us', 5000.0))
    ae = analysis.AutoExposure()
    # Zero on connect: keep the laser off until the app is running, so the first
    # frames are a dark frame (sensor black level + fixed-pattern offset).
    zero = [int(cfg.get('zero_frames', 10)), None, 0] if cfg.get('zero_on_connect', True) else None
    dark, dark_exp = None, None
    stack = analysis.MotionAwareStack()
    calib = None            # [laser, power_w, n_left, rates_sum, n]
    t_last, fps = time.time(), 0.0
    try:
        while True:
            # ------------------------------------------------ commands
            while True:
                try:
                    c = cmd_q.get_nowait()
                except queue.Empty:
                    break
                op = c[0]
                if op == 'quit':
                    return
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
            n = max(1, int(cfg.get('stack', 1)))
            frame = stack.add(raw, n)        # restarts itself when the beam moves
            if zero is not None and zero[2] == 0 and zero[1] is None and max(analysis.bayer_channel_max(raw)) > 60:
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
                                   zero_px=cfg.get('zero_px'), planes=rgb)
            res['dark'] = 'on' if use_dark is not None else ('recording' if zero is not None else 'none')
            if zero is not None:
                res['powers'] = None          # no watts until the zero is in
            res['peak_raw'] = analysis.bayer_channel_max(raw)     # judge the raw frame
            res['avg'] = (stack.count, n)                         # frames averaged / asked for
            # ------------------------------------------------ auto exposure
            ae.target = float(cfg.get('ae_target', 0.85))
            if cfg.get('auto_exposure') and calib is None:
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
                if max(res['peak_raw']) >= 254:
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
                mrad_per_px=res['mrad_per_px'], divs=divs)
            res['laser_div'] = divs                 # per colour divergence (mrad), by plane
            disp = dict(cfg['display'])
            disp['colors'] = [cfg['display']['colors'][i] for i in idx]
            att = cfg['display'].get('atten') or [1.0, 1.0, 1.0]
            disp['atten'] = [att[i] for i in idx]
            preview, step = render(planes, disp, res['roi'])
            now = time.time()
            fps = 0.8 * fps + 0.2 / max(now - t_last, 1e-6)
            t_last = now
            for key in ('profile_x', 'profile_y', 'rates', 'powers'):
                if res.get(key) is not None:
                    res[key] = np.asarray(res[key]).tolist()
            _put_latest(out_q, {'type': 'frame', 'preview': preview, 'step': step,
                                'result': res, 'fps': fps, 'temp': temp,
                                'calibrating': calib is not None})
    except Exception:  # noqa: BLE001
        out_q.put({'type': 'error', 'text': 'Camera loop stopped:\n' + traceback.format_exc(limit=3)})
    finally:
        if cam is not None:
            cam.close()
