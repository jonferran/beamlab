"""Tests against the simulated camera (known truth). Run: python -m pytest -q tests
or plain:  python tests/test_beamlab.py"""
import math
import multiprocessing as mp
import os
import queue
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402
os.environ.setdefault('BEAMLAB_DIR', tempfile.mkdtemp(prefix='beamlab_test_'))  # never touch the real settings/logs
from beamlab import analysis, worker  # noqa: E402
from beamlab.cameras import SimCamera  # noqa: E402

OPT = analysis.Optics(pixel_pitch_um=3.45, focal_mm=50.0)
K = OPT.mrad_per_px()


def sim(**kw):
    kw.setdefault('wander', False)
    kw.setdefault('fps', 1000)
    c = SimCamera(width=640, height=480, **kw)
    return c


def test_widths_match_gaussian_truth():
    c = sim(sigma_px=(18.0, 12.0), powers_w=(0.5, 0, 0))
    c.set_exposure(400)
    raw, exp, _ = c.grab()
    assert max(analysis.bayer_channel_max(raw)) < 250, 'test frame must not saturate'
    r = analysis.analyse(raw, exp, OPT)
    sx, sy = 18.0 / 2, 12.0 / 2          # analysis px are 2x2 Bayer cells
    for key, exp_w in (('fwhm_x', 2.3548 * sx), ('fwhm_y', 2.3548 * sy),
                       ('e2_x', 4 * sx), ('e2_y', 4 * sy), ('d4s_x', 4 * sx), ('d4s_y', 4 * sy)):
        got = r[key] / K
        assert abs(got - exp_w) / exp_w < 0.06, (key, got, exp_w)


def test_width_at_level_interpolates():
    x = np.arange(101, dtype=float)
    p = np.exp(-(x - 50.3) ** 2 / (2 * 7.0 ** 2))
    w, l, rr = analysis.width_at_level(p, 0.5)
    assert abs(w - 2.3548 * 7.0) < 0.05
    assert abs(0.5 * (l + rr) - 50.3) < 0.02
    # beam running off the edge -> nan, not a wrong number
    assert math.isnan(analysis.width_at_level(p[:52], 0.5)[0])


def test_alignment_sign_and_scale():
    c = sim(powers_w=(0.3, 0, 0))
    c.set_exposure(2000)
    raw, exp, _ = c.grab()
    r = analysis.analyse(raw, exp, OPT)
    # sim spot is +60 sensor px right and 30 px UP of centre
    assert abs(r['align_x'] - 30 * K) < 0.02 * 30 * K + 0.01
    assert abs(r['align_y'] - 15 * K) < 0.02 * 15 * K + 0.01


def _dark(exp, seed=99):
    """'Zero (laser off)': mean of 10 frames with every laser off."""
    c = sim(powers_w=(0, 0, 0), seed=seed)
    c.set_exposure(exp)
    return np.mean([c.grab()[0].astype(np.float32) for _ in range(10)], axis=0)


def _rates(cam, exp, dark=None):
    cam.set_exposure(exp)
    acc = np.zeros(3)
    for _ in range(15):                      # the app calibrates over 15 frames too
        raw, e, _ = cam.grab()
        assert max(analysis.bayer_channel_max(raw)) < 254, 'power test frame saturated'
        acc += analysis.analyse(raw, e, OPT, dark=dark)['rates']
    return acc / 15


def test_power_calibration_unmixes_three_lasers():
    cols = []
    for j, pw in enumerate((0.5, 0.4, 0.3)):
        p = [0.0, 0.0, 0.0]
        p[j] = pw
        cols.append(_rates(sim(powers_w=p, seed=j + 5), 300, _dark(1000)) / pw)
    resp = np.array(cols).T
    truth = np.array([0.35, 0.22, 0.41])
    cam = sim(powers_w=truth, seed=11)
    got = analysis.solve_power(_rates(cam, 450, _dark(1000)), resp)    # different exposure on purpose
    err = np.abs(got - truth) / truth
    # measured on the simulator: 2-4 % (green up to 8 %: the blue laser leaks ~25 %
    # into the green pixels). Power is secondary here (spatial data first), and
    # the fast simulator's coarse noise limits this, so the bar is 6 % / 10 %.
    assert err[0] < 0.06 and err[2] < 0.06 and err[1] < 0.10, (got, err)


def test_auto_exposure_settles_from_dark_and_from_saturated():
    for start in (30.0, 150000.0):
        cam = sim(powers_w=(0.4, 0.3, 0.2))
        cam.set_exposure(start)
        ae = analysis.AutoExposure(target=0.85)
        exp = start
        states = []
        for _ in range(40):
            raw, used, _ = cam.grab()
            new = ae.update(raw, used, *cam.exposure_range)
            states.append(ae.state)
            if new != used:
                cam.set_exposure(new)
        raw, used, _ = cam.grab()
        frac = analysis.robust_peak(raw) / 255.0
        assert abs(frac - 0.85) <= 0.1, (start, frac, states)
        assert states[-5:].count('settled') + states[-5:].count('waiting') == 5, states
        # no hunting once settled
        changes = 0
        for _ in range(15):
            raw, used, _ = cam.grab()
            if ae.update(raw, used, *cam.exposure_range) != used:
                changes += 1
        assert changes == 0


def test_auto_exposure_ignores_a_hot_pixel():
    cam = sim(powers_w=(0.05, 0, 0))
    cam.set_exposure(4000)
    raw, used, _ = cam.grab()
    base = analysis.robust_peak(raw)
    raw2 = raw.copy()
    raw2[3, 3] = 255
    assert abs(analysis.robust_peak(raw2) - base) < 0.15 * 255 / 12 + 1


def test_colour_remap_paints_green_laser_magenta():
    cam = sim(powers_w=(0, 0.4, 0))
    cam.set_exposure(2000)
    raw, _, _ = cam.grab()
    r, g, b = analysis.split_bayer_gr8(raw.astype(np.float32))
    # calibrated response: the green laser alone is column 1
    resp = np.array(SimCamera.CROSSTALK)
    planes = worker.unmix_planes(r, g, b, resp)
    disp = {'colors': [[255, 0, 0], [255, 0, 255], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}
    img, _ = worker.render(planes, disp, (0, 0, 1, 1))
    cy, cx = np.unravel_index(np.argmax(img.sum(axis=2)), img.shape[:2])
    px = img[cy, cx].astype(int)
    assert px[0] > 200 and px[2] > 200 and px[1] < 20, px


def test_snapshot_overlay_keeps_both_beams():
    """A snapped red beam (laser now off) stays visible beside the live green one."""
    from PIL import Image
    from beamlab.app import overlay_snapshot
    live = np.zeros((60, 80, 3), np.uint8)
    live[30, 50] = (0, 255, 0)                     # live green spot
    snap = np.zeros((120, 160, 3), np.uint8)       # snapshot at a finer preview step
    snap[60:62, 40:42] = (255, 0, 0)               # snapped red spot at the same sensor place as (30, 20)
    out = overlay_snapshot(live, Image.fromarray(snap), 1.0)
    assert tuple(out[30, 50]) == (0, 255, 0)
    assert out[30, 20, 0] > 200 and out[30, 20, 1] == 0
    half = overlay_snapshot(live, Image.fromarray(snap), 0.5)
    assert 100 < half[30, 20, 0] < 140 and tuple(half[30, 50]) == (0, 255, 0)


def test_contrast_curve_display_only():
    planes = np.zeros((40, 40, 3), np.float32)
    planes[20, 20, 0] = 100.0                      # peak
    planes[20, 10, 0] = 25.0                       # faint wing, 25 % of the peak
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 0, 255]], 'auto_bright': True, 'gain': 1.0}
    wing = {}
    for c in (-100, 0, 100):
        img, _ = worker.render(planes, dict(base, contrast=c), (0, 0, 1, 1))
        assert img[20, 20, 0] == 255, c            # the peak stays full brightness
        wing[c] = int(img[20, 10, 0])
    assert wing[-100] > wing[0] > wing[100], wing  # left lifts the wing, right darkens it
    # 25 % wing: gamma 1 -> 64, gamma 0.25 (-100) -> 0.25^0.25 = 71 % -> 180, gamma 4 -> 0
    assert abs(wing[0] - 64) <= 1 and abs(wing[-100] - 180) <= 1 and wing[100] <= 1, wing


def test_zoom_window_rendered_at_full_resolution():
    """Zoomed in, the camera process sends the window at full analysis resolution
    (the whole-sensor preview has only every 2nd pixel on a 4K sensor)."""
    H, W = 1080, 1919
    yy, xx = np.mgrid[0:H, 0:W]
    planes = np.zeros((H, W, 3), np.float32)
    planes[..., 1] = 100 * np.exp(-((xx - 900.3) ** 2 + (yy - 500.7) ** 2) / (2 * 6.0 ** 2))
    disp = {'colors': [[255, 0, 0], [0, 255, 0], [0, 0, 255]], 'auto_bright': True, 'gain': 1.0}
    prev, step = worker.render(planes, disp, (850, 450, 951, 551))
    prev2, step2, det = worker.render(planes, disp, (850, 450, 951, 551), (845, 445, 955, 555))
    assert step == step2 == 2 and np.array_equal(prev, prev2)          # whole preview unchanged
    assert det['step'] == 1 and det['box'] == (845, 445, 955, 555) and det['img'].shape == (110, 110, 3)
    assert np.array_equal(det['img'][1::2, 1::2], prev[223:278, 423:478])  # same pixels, same brightness
    assert (det['img'][5, 5:105] == (0, 255, 0)).all()                 # ROI drawn in the window too
    assert worker.render(planes, disp, (0, 0, 1, 1), (0, 0, W, H))[2] is None   # no gain zoomed out


def test_three_spots_measure_the_chosen_one():
    """The LBA always shows 3 spots side by side (Jon uses the right one). The
    widths must be the chosen spot's own, not the span of all three."""
    H, W = 540, 960                                   # analysis px
    yy, xx = np.mgrid[0:H, 0:W]
    sig = 10.0

    def spot(cx, cy, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    lum = spot(200, 300, 60) + spot(480, 300, 120) + spot(760, 290, 60)   # middle brightest
    # Bayer frame with that light in the green pixels only (R=B=0), + dark level 2
    raw = np.full((2 * H, 2 * W), 2.0, np.float32)
    raw[0::2, 0::2] += lum / 2
    raw[1::2, 1::2] += lum / 2
    o = analysis.Optics()
    k = o.mrad_per_px()
    d4_true = 4 * sig * k
    for which, cx_true in (('right', 760), ('left', 200), ('middle', 480), ('brightest', 480)):
        res = analysis.analyse(raw, 1000, o, spot=which)
        assert res['ok'] and res['n_spots'] == 3, which
        assert abs(res['cx_px'] - cx_true) < 1.0, (which, res['cx_px'])
        assert abs(res['d4s_x'] - d4_true) < 0.03 * d4_true, (which, res['d4s_x'], d4_true)
        assert abs(res['e2_x'] - d4_true) < 0.05 * d4_true, (which, res['e2_x'], d4_true)
        x0, y0, x1, y1 = res['roi']
        assert not (x0 <= 480 <= x1) or which in ('middle', 'brightest'), (which, res['roi'])
    # per colour: the chosen spot too
    r, g, b = analysis.split_bayer_gr8(raw)
    divs = []
    cxy, _ = worker.laser_spots(worker.unmix_planes(r, g, b, None), mrad_per_px=k, divs=divs, spot='right')
    assert abs(cxy[1][0] - 760) < 1.0 and abs(divs[1]['d4s_x'] - d4_true) < 0.03 * d4_true
    # one spot only: every choice measures it
    one = np.full((2 * H, 2 * W), 2.0, np.float32)
    one[0::2, 0::2] += spot(300, 200, 100) / 2
    one[1::2, 1::2] += spot(300, 200, 100) / 2
    for which in analysis.SPOT_CHOICES:
        res = analysis.analyse(one, 1000, o, spot=which)
        assert res['n_spots'] == 1 and abs(res['cx_px'] - 300) < 1.0, which


def test_12bit_camera_pipeline():
    """A 12-bit camera (0..4095): auto exposure settles at its own full scale,
    widths are the same as with 8-bit, and a dim beam has ~16x more levels."""
    c12 = SimCamera(bits=12, wander=False, fps=1000, powers_w=(0, 0.4, 0))
    c8 = SimCamera(bits=8, wander=False, fps=1000, powers_w=(0, 0.4, 0))
    assert c12.full == 4095.0 and c8.full == 255.0
    # auto exposure, 12-bit: lands near 85 % of 4095 (it used to aim at 255)
    ae = analysis.AutoExposure(full=c12.full)
    exp = 50.0
    c12.set_exposure(exp)
    for _ in range(40):
        raw, e, _ = c12.grab()
        new = ae.update(raw, e, 20, 400000)
        if new != e:
            c12.set_exposure(new)
    assert raw.dtype == np.uint16 and ae.state == 'settled', ae.state
    assert 0.75 * 4095 < analysis.robust_peak(raw) < 0.95 * 4095, analysis.robust_peak(raw)
    # same light, both depths: same widths (truth sigma 18 x 12 sensor px = 9 x 6 analysis px)
    o = analysis.Optics()
    k = o.mrad_per_px()
    w = {}
    for cam in (c12, c8):
        cam.set_exposure(c12.exp)
        got = []
        for _ in range(4):
            raw, e, _ = cam.grab()
            res = analysis.analyse(raw, e, o)
            assert res['ok'], cam.full
            got.append((res['d4s_x'] / k, res['d4s_y'] / k))
        w[cam.full] = np.mean(got, axis=0)
        assert abs(w[cam.full][0] - 36.0) < 0.04 * 36 and abs(w[cam.full][1] - 24.0) < 0.04 * 24, (cam.full, w[cam.full])
    assert np.all(np.abs(w[4095.0] - w[255.0]) < 0.02 * w[255.0]), w
    # a DIM beam (peak ~20 of 255): the 12-bit picture has many more distinct levels
    dim = {}
    for cam in (c12, c8):
        cam.set_exposure(c12.exp * 20.0 / 217.0)
        raw, e, _ = cam.grab()
        r, g, b = analysis.split_bayer_gr8(raw)
        dim[cam.full] = len(np.unique(raw[raw > raw.mean() + 3 * cam.full / 255.0]))
    assert dim[4095.0] > 8 * dim[255.0], dim


def test_every_bayer_order_converts_to_gr8():
    """The LBA's Basler daA3840-45uc only offers BayerRG8: its red pixels must land
    in the R plane, and the sensor centre must not move."""
    from beamlab.cameras import to_gr8, BAYER_ORDERS
    H, W = 8, 12
    tile = {'R': 10, 'G': 20, 'B': 30}
    layouts = {'BayerGR8': ('GR', 'BG'), 'BayerRG8': ('RG', 'GB'),
               'BayerGB8': ('GB', 'RG'), 'BayerBG8': ('BG', 'GR')}
    assert 'BayerRG12' in BAYER_ORDERS and BAYER_ORDERS.index('BayerRG12') < BAYER_ORDERS.index('BayerRG8')
    for order in BAYER_ORDERS:                     # 12-bit names too (same layouts)
        rows = layouts[order[:7] + '8']
        dt = np.uint16 if order.endswith('12') else np.uint8
        raw = np.array([[tile[rows[y % 2][x % 2]] for x in range(W)] for y in range(H)], dt)
        gr = to_gr8(raw, order)
        r, g, b = analysis.split_bayer_gr8(gr.astype(np.float32))
        assert (r == 10).all() and (g == 20).all() and (b == 30).all(), order
        # same centre: equal crop on both sides
        assert (H - gr.shape[0]) % 2 == 0 and (W - gr.shape[1]) % 2 == 0, order


def test_widths_do_not_depend_on_position_with_averaging():
    """Reported issue: with 'Average frames' at 3, divergence changed with the beam's
    position on the moving sim beam (smear grows with speed). Widths must stay put."""
    cam = SimCamera(width=1440, height=1080, fps=10000, powers_w=(0.05, 0.04, 0.03))
    cam.set_exposure(1500)
    stack = analysis.MotionAwareStack()
    fw, e2 = [], []
    for i in range(160):
        cam.t0 = time.time() - i * 0.25          # walk through the whole wander path
        raw, exp, _ = cam.grab()
        frame = stack.add(raw, 3)
        r = analysis.analyse(frame, exp, OPT)
        if i >= 3:
            fw.append(r['fwhm_x'] / K)
            e2.append(r['e2_x'] / K)
    fw, e2 = np.array(fw), np.array(e2)
    truth_fw, truth_e2 = 2.3548 * 9, 36.0
    assert abs(fw.mean() - truth_fw) / truth_fw < 0.03, fw.mean()
    assert (fw.max() - fw.min()) / fw.mean() < 0.06, (fw.min(), fw.max())
    assert (e2.max() - e2.min()) / e2.mean() < 0.05, (e2.min(), e2.max())


def test_exposure_capped_at_400ms():
    from beamlab.cameras import open_camera, MAX_EXPOSURE_US
    cam = open_camera('sim')
    assert MAX_EXPOSURE_US == 400000.0 and cam.exposure_range[1] == 400000.0, cam.exposure_range
    # auto exposure can reach the cap on a very dim beam, and not beyond
    dim = SimCamera(width=640, height=480, wander=False, fps=10000, powers_w=(0.0002, 0, 0))
    dim.exposure_range = cam.exposure_range
    dim.set_exposure(1000)
    ae = analysis.AutoExposure()
    for _ in range(60):
        raw, used, _ = dim.grab()
        new = ae.update(raw, used, *dim.exposure_range)
        if new != used:
            dim.set_exposure(new)
    assert dim.exp <= 400000.0 and dim.exp > 100000.0, dim.exp


def test_still_beam_is_still_averaged():
    cam = sim(powers_w=(0.05, 0.04, 0.03))
    cam.set_exposure(1500)
    stack = analysis.MotionAwareStack()
    for _ in range(8):
        stack.add(cam.grab()[0], 5)
    assert stack.count == 5 and stack.restarts == 0


def test_colours_on_one_spot_read_the_same_position():
    """Three lasers on exactly the same spot must read the SAME centre (Bayer
    sub-pixel offsets corrected), or the app reports a false misalignment."""
    cam = SimCamera(width=640, height=480, wander=False, fps=10000, powers_w=(0.3, 0.3, 0.3))
    cam.CROSSTALK = np.eye(3)                     # each colour sees only its own laser
    cam.set_exposure(400)
    acc = np.zeros((3, 2))
    n = 12
    for _ in range(n):
        raw, _, _ = cam.grab()
        r, g, b = analysis.split_bayer_gr8(raw.astype(np.float32))
        planes = worker.unmix_planes(r, g, b, None)
        cxy, _ = worker.laser_spots(planes, offsets=worker.plane_offsets(None, 3))
        acc += np.array(cxy)
    acc /= n
    # truth: sensor (cx, cy) -> analysis px = (c - 0.5) / 2
    tx, ty = (cam.truth['cx_sensor'] - 0.5) / 2, (cam.truth['cy_sensor'] - 0.5) / 2
    for j, name in enumerate('RGB'):
        assert abs(acc[j, 0] - tx) < 0.05 and abs(acc[j, 1] - ty) < 0.05, (name, acc[j], (tx, ty))
    # and without the correction the error is the 0.25 px it fixes
    raw, _, _ = cam.grab()
    r, g, b = analysis.split_bayer_gr8(raw.astype(np.float32))
    plain, _ = worker.laser_spots(worker.unmix_planes(r, g, b, None))
    # (red samples sit +0.25 px right, so its uncorrected centre reads 0.25 px LEFT; blue the opposite)
    assert abs((plain[0][0] - plain[2][0]) + 0.5) < 0.1, plain


def test_per_colour_divergence_matches_truth():
    """Each colour's own divergence (separate spots, different sizes) must match
    that colour's true beam - not the mix."""
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sig = [(6.0, 4.0), (9.0, 7.0), (4.0, 10.0)]      # analysis px, per colour
    cen = [(80, 90), (200, 150), (320, 200)]
    rng = np.random.default_rng(3)
    planes = np.zeros((h, w, 3), np.float32)
    for j in range(3):
        (sx, sy), (cx, cy) = sig[j], cen[j]
        planes[..., j] = 180 * np.exp(-((xx - cx) ** 2 / (2 * sx * sx) + (yy - cy) ** 2 / (2 * sy * sy)))
    planes += rng.normal(0, 1.0, planes.shape).astype(np.float32)
    divs = []
    worker.laser_spots(planes, mrad_per_px=1.0, divs=divs)    # 1 mrad/px -> widths in px
    for j, (sx, sy) in enumerate(sig):
        d = divs[j]
        for key, truth in (('fwhm_x', 2.3548 * sx), ('fwhm_y', 2.3548 * sy), ('e2_x', 4 * sx),
                           ('e2_y', 4 * sy), ('d4s_x', 4 * sx), ('d4s_y', 4 * sy)):
            assert abs(d[key] - truth) / truth < 0.04, ('RGB'[j], key, d[key], truth)


def test_display_attenuation_per_laser():
    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    planes = np.zeros((h, w, 3), np.float32)
    planes[..., 0] = 50 * np.exp(-((xx - 40) ** 2 + (yy - 60) ** 2) / 50.0)      # weak red
    planes[..., 1] = 200 * np.exp(-((xx - 120) ** 2 + (yy - 60) ** 2) / 50.0)    # strong green
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}
    img, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    red_full = int(img[60, 40, 0])
    att = dict(base, atten=[1.0, 0.2, 1.0])
    img2, _ = worker.render(planes, att, (0, 0, 1, 1))
    # green turned down to 20 % really is 20 % as bright; red is unchanged
    assert abs(int(img2[60, 120, 1]) - 0.2 * int(img[60, 120, 1])) <= 2, (img[60, 120], img2[60, 120])
    assert abs(int(img2[60, 40, 0]) - red_full) <= 1, (red_full, img2[60, 40, 0])
    # 0 % hides that laser completely; the measurement planes are untouched
    img3, _ = worker.render(planes, dict(base, atten=[1.0, 0.0, 1.0]), (0, 0, 1, 1))
    assert img3[..., 1].max() == 0 and abs(int(img3[60, 40, 0]) - red_full) <= 1
    assert planes[60, 120, 1] > 199
    # the only laser on: the slider still dims it (auto brightness no longer undoes it)
    solo = planes.copy()
    solo[..., 1] = 0
    full, _ = worker.render(solo, dict(base), (0, 0, 1, 1))
    half, _ = worker.render(solo, dict(base, atten=[0.5, 1.0, 1.0]), (0, 0, 1, 1))
    assert full[60, 40, 0] == 255 and abs(int(half[60, 40, 0]) - 128) <= 1, (full[60, 40], half[60, 40])


def test_per_laser_boost_and_contrast():
    """Each laser's own Boost and Contrast change only that laser (calibrated planes),
    and the defaults draw exactly what was drawn before."""
    h, w = 60, 120
    yy, xx = np.mgrid[0:h, 0:w]
    planes = np.zeros((h, w, 3), np.float32)
    planes[..., 0] = 100 * np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 60.0)    # red
    planes[..., 1] = 100 * np.exp(-((xx - 90) ** 2 + (yy - 30) ** 2) / 60.0)    # green
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}
    ref, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    same, _ = worker.render(planes, dict(base, boost=[1, 1, 1], lcontrast=[0, 0, 0]), (0, 0, 1, 1))
    assert np.array_equal(ref, same)
    # red at 0.5x: its peak halves, green untouched
    b, _ = worker.render(planes, dict(base, boost=[0.5, 1, 1]), (0, 0, 1, 1))
    assert abs(int(b[30, 30, 0]) - 128) <= 1 and np.abs(b[:, 60:].astype(int) - ref[:, 60:]).max() <= 1
    # green boosted 4x: a faint wing (5 % of the peak) shows 4x brighter, the peak clips at 255
    wx = 90 + int(round(math.sqrt(-60.0 * math.log(0.05))))
    b4, _ = worker.render(planes, dict(base, boost=[1, 4, 1]), (0, 0, 1, 1))
    assert b4[30, 90, 1] == 255 and abs(int(b4[30, wx, 1]) - 4 * int(ref[30, wx, 1])) <= 4
    # red contrast -100: its wing comes up, green's does not move
    wr = 30 + int(round(math.sqrt(-60.0 * math.log(0.25))))
    c, _ = worker.render(planes, dict(base, lcontrast=[-100, 0, 0]), (0, 0, 1, 1))
    assert c[30, wr, 0] > ref[30, wr, 0] + 60 and c[30, 30, 0] == 255
    assert np.abs(c[..., 1].astype(int) - ref[..., 1]).max() <= 1        # green's own channel unchanged


def test_learned_colours_separate_overlapping_beams():
    """Red and green beams on the SAME spot, each leaking into the other camera
    colours. With their learned colours, boosting red changes only red."""
    h, w = 60, 60
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 40.0)
    red_sig, green_sig = np.array([1.0, 0.18, 0.06]), np.array([0.08, 1.0, 0.3])
    raw = (100 * spot)[..., None] * red_sig + (80 * spot)[..., None] * green_sig     # H,W,3 camera R,G,B
    r, g, b = raw[..., 0], raw[..., 1], raw[..., 2]
    m = worker.colour_matrix([red_sig.tolist(), green_sig.tolist(), None])
    planes = worker.unmix_planes(r, g, b, m)
    assert abs(planes[30, 30, 0] - 100) < 0.5 and abs(planes[30, 30, 1] - 80) < 0.5 and planes[30, 30, 2] < 0.5
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}
    ref, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    up, _ = worker.render(planes, dict(base, boost=[0.5, 1, 1]), (0, 0, 1, 1))
    assert up[30, 30, 0] < ref[30, 30, 0] * 0.6 and np.abs(up[..., 1].astype(int) - ref[..., 1]).max() <= 1, (ref[30, 30], up[30, 30])
    # nothing learned, or two lasers learned the same colour -> no separation
    assert worker.colour_matrix(None) is None and worker.colour_matrix([None, None, None]) is None
    assert worker.colour_matrix([red_sig.tolist(), red_sig.tolist(), None]) is None


def test_balance_lasers():
    """The camera sees the red laser ~5x weaker than green (Jon: had to boost red
    5x). Balance lasers: each laser's own peak at full brightness; a laser that is
    off is not blown up from its noise."""
    h, w = 60, 180
    yy, xx = np.mgrid[0:h, 0:w]

    def spot(cx):
        return np.exp(-((xx - cx) ** 2 + (yy - 30) ** 2) / 40.0)
    planes = np.zeros((h, w, 3), np.float32)
    planes[..., 0] = 40 * spot(30)                                  # red, weak in the camera
    planes[..., 1] = 200 * spot(90)                                 # green
    planes[..., 2] = 0.5 * np.abs(np.sin(xx * 1.7 + yy))            # blue laser OFF: noise only
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}
    off, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    on, _ = worker.render(planes, dict(base, balance=True), (0, 0, 1, 1))
    assert off[30, 30, 0] < 60 and off[30, 90, 1] == 255, (off[30, 30], off[30, 90])
    assert on[30, 30, 0] >= 253 and on[30, 90, 1] >= 253, (on[30, 30], on[30, 90])
    assert on[..., 2].max() <= off[..., 2].max() + 1, (on[..., 2].max(), off[..., 2].max())
    # the user's own boost still works on top (red at 0.5x of balanced)
    half, _ = worker.render(planes, dict(base, balance=True, boost=[0.5, 1, 1]), (0, 0, 1, 1))
    assert abs(int(half[30, 30, 0]) - 128) <= 2, half[30, 30]
    # and without calibration (raw camera colours), the same
    raw_on, _ = worker.render(planes, dict(base, balance=True, raw_planes=True), (0, 0, 1, 1))
    assert raw_on[30, 30, 0] >= 250 and raw_on[30, 90, 1] >= 250, (raw_on[30, 30], raw_on[30, 90])


def test_per_laser_settings_keep_the_hue_without_calibration():
    h, w = 60, 60
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 40.0)
    planes = np.stack([200 * spot, 40 * spot, 15 * spot], axis=-1).astype(np.float32)   # red + leaks
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0,
            'raw_planes': True}
    full, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    for extra in ({'boost': [0.4, 1, 1]}, {'lcontrast': [-80, 0, 0]}, {'lcontrast': [80, 0, 0]}):
        img, _ = worker.render(planes, dict(base, **extra), (0, 0, 1, 1))
        for (y, x) in ((30, 30), (30, 34)):
            a, b = full[y, x].astype(float), img[y, x].astype(float)
            if a.sum() < 30 or b.sum() < 30:
                continue
            assert np.abs(a / a.sum() - b / b.sum()).max() < 0.04, (extra, a, b)
        assert not np.array_equal(img, full), extra                       # it did change something


def test_brightness_slider_keeps_the_hue_without_calibration():
    """Uncalibrated, a red laser shows in the raw R plane AND leaks into G and B.
    Turning the Red slider down must dim the spot, not change its colour."""
    h, w = 60, 60
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-((xx - 30) ** 2 + (yy - 30) ** 2) / 40.0)
    planes = np.stack([200 * spot, 40 * spot, 15 * spot], axis=-1).astype(np.float32)   # R, G leak, B leak
    base = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0,
            'raw_planes': True}
    full, _ = worker.render(planes, dict(base), (0, 0, 1, 1))
    dim, _ = worker.render(planes, dict(base, atten=[0.3, 1.0, 1.0]), (0, 0, 1, 1))
    a, b = full[30, 30].astype(float), dim[30, 30].astype(float)
    assert b.sum() < 0.6 * a.sum(), (a, b)                              # clearly dimmer
    hue_a, hue_b = a / a.sum(), b / b.sum()
    assert np.abs(hue_a - hue_b).max() < 0.03, (hue_a, hue_b)            # same colour


def test_laser_centroids_separate_colours():
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    planes = np.zeros((h, w, 3), np.float32)
    for j, (cx, cy) in enumerate(((100, 80), (160, 120), (220, 150))):
        planes[..., j] = 200 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 6.0 ** 2))
    got = worker.laser_centroids(planes)
    for (gx, gy), (cx, cy) in zip(got, ((100, 80), (160, 120), (220, 150))):
        assert abs(gx - cx) < 0.2 and abs(gy - cy) < 0.2
    planes[..., 2] = 0
    assert worker.laser_centroids(planes)[2] is None


def test_worker_process_end_to_end():
    cmd_q, out_q = mp.Queue(), mp.Queue(maxsize=3)
    cfg = {'backend': 'sim', 'exposure_us': 100.0, 'auto_exposure': True, 'stack': 1, 'auto_roi': True,
           'optics': {'pixel_pitch_um': 3.45, 'focal_mm': 50.0}, 'zero_px': None, 'response': None,
           'laser_idx': [], 'display': {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]],
                                        'auto_bright': True, 'gain': 1.0}}
    p = mp.Process(target=worker.run, args=(cmd_q, out_q, cfg), daemon=True)
    p.start()
    try:
        got_open, frames, last = False, 0, None
        zero_exps = set()
        t_end = time.time() + 25
        while time.time() < t_end and frames < 60:
            try:
                m = out_q.get(timeout=5)
            except queue.Empty:
                continue
            if m['type'] == 'error':
                raise AssertionError(m['text'])
            if m['type'] == 'opened':
                got_open = True
            if m['type'] == 'frame':
                frames += 1
                last = m
                if m['result'].get('ae_state') == 'held while zeroing':
                    zero_exps.add(m['result']['exposure_us'])
        assert got_open and frames >= 30, (got_open, frames)
        # the zero on connect (laser off) held the exposure still instead of letting
        # auto exposure run it up to the maximum
        assert len(zero_exps) == 1, zero_exps
        assert last['preview'].ndim == 3 and last['preview'].dtype == np.uint8
        assert last['result']['ok'] and last['result']['ae_state'] in ('settled', 'waiting'), last['result'].get('ae_state')
        # calibrate the red channel through the worker WITH auto exposure left on:
        # the exposure is held while calibrating, auto exposure carries on afterwards
        cmd_q.put(('calibrate', 0, 0.6, 5))
        t_end = time.time() + 15
        cal, held_exps, after = None, set(), None
        while time.time() < t_end and (cal is None or after is None):
            try:
                m = out_q.get(timeout=5)
            except queue.Empty:
                continue
            if m['type'] == 'calibrated':
                cal = m
            elif m['type'] == 'error':
                raise AssertionError(m['text'])
            elif m['type'] == 'frame':
                st = m['result'].get('ae_state')
                if st == 'held while calibrating':
                    held_exps.add(m['result']['exposure_us'])
                elif cal is not None:
                    after = st
        assert cal and cal['laser'] == 0 and len(cal['rates_per_w']) == 3
        assert len(held_exps) == 1, held_exps
        assert after in ('settled', 'waiting', 'adjusting'), after
    finally:
        cmd_q.put(('quit',))
        p.join(5)
        if p.is_alive():
            p.terminate()


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            t = time.time()
            try:
                fn()
                print('PASS %-50s %.1fs' % (name, time.time() - t))
            except Exception as e:  # noqa: BLE001
                fails += 1
                print('FAIL %-50s %r' % (name, e))
    sys.exit(1 if fails else 0)
