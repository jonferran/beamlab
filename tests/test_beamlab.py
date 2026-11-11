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
    # green turned down to 20 % -> red is no longer drowned by auto brightness
    assert img2[60, 120, 1] < img[60, 120, 1] * 0.9 or img2[60, 40, 0] > red_full * 2, (img[60, 120], img2[60, 120])
    assert img2[60, 40, 0] > red_full * 2, (red_full, img2[60, 40, 0])
    # 0 % hides that laser completely; the measurement planes are untouched
    img3, _ = worker.render(planes, dict(base, atten=[1.0, 0.0, 1.0]), (0, 0, 1, 1))
    assert img3[..., 1].max() == 0 and img3[60, 40, 0] > 200
    assert planes[60, 120, 1] > 199


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
        assert got_open and frames >= 30, (got_open, frames)
        assert last['preview'].ndim == 3 and last['preview'].dtype == np.uint8
        assert last['result']['ok'] and last['result']['ae_state'] in ('settled', 'waiting'), last['result'].get('ae_state')
        # calibrate the red channel through the worker
        cmd_q.put(('set', {'auto_exposure': False}))
        cmd_q.put(('calibrate', 0, 0.6, 5))
        t_end = time.time() + 15
        cal = None
        while time.time() < t_end and cal is None:
            try:
                m = out_q.get(timeout=5)
            except queue.Empty:
                continue
            if m['type'] == 'calibrated':
                cal = m
            elif m['type'] == 'error':
                raise AssertionError(m['text'])
        assert cal and cal['laser'] == 0 and len(cal['rates_per_w']) == 3
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
