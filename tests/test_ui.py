"""UI checks with the real window and simulated frames (no worker process).
Run: python tests/test_ui.py      (opens and closes a window briefly)"""
import os
import queue
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402
os.environ.setdefault('BEAMLAB_DIR', tempfile.mkdtemp(prefix='beamlab_test_'))  # never touch the real settings/logs
from beamlab import analysis, settings, worker  # noqa: E402
from beamlab.cameras import SimCamera  # noqa: E402

# keep the user's real settings and snapshots out of the test
_tmp = tempfile.mkdtemp(prefix='beamlab_ui_')
settings.DIR = _tmp
settings.FILE = os.path.join(_tmp, 'settings.json')
settings.SNAP_DIR = os.path.join(_tmp, 'snapshots')
settings.ERROR_LOG = os.path.join(_tmp, 'errors.log')

import tkinter as tk  # noqa: E402
from beamlab import app as A  # noqa: E402

CAM = SimCamera(width=1440, height=1080, wander=False, fps=1000, powers_w=(0.05, 0.04, 0.03))
CAM.set_exposure(2000)
DISPLAY = {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]], 'auto_bright': True, 'gain': 1.0}


def frame_msg():
    raw, exp, temp = CAM.grab()
    res = analysis.analyse(raw, exp, analysis.Optics())
    r, g, b = analysis.split_bayer_gr8(raw)
    planes = worker.unmix_planes(r, g, b, None)
    res['laser_idx'] = [0, 1, 2]
    divs = []
    res['laser_cxy'], res['laser_box'] = worker.laser_spots(planes, mrad_per_px=res['mrad_per_px'], divs=divs)
    res['laser_div'] = divs
    prev, step = worker.render(planes, DISPLAY, res['roi'])
    for k in ('profile_x', 'profile_y', 'rates'):
        res[k] = np.asarray(res[k]).tolist()
    return {'type': 'frame', 'preview': prev, 'step': step, 'result': res, 'fps': 30.0, 'temp': 35.0}


def pump(root, t=0.1):
    end = time.time() + t
    while time.time() < end:
        root.update()


def main():
    root = tk.Tk()
    app = A.App(root)
    root.title('BeamLab UI TEST (not the app - closes by itself)')   # never mistaken for a running BeamLab
    pump(root, 0.3)
    app.serial = 'SIM0001'
    msg = frame_msg()
    app.last = msg
    app.draw(msg)
    pump(root)
    res = msg['result']
    H, W = res['shape']
    fails = []
    app.zoom_follow.set('Beam')
    app.draw(msg)

    def check(cond, what):
        print(('PASS ' if cond else 'FAIL ') + what)
        if not cond:
            fails.append(what)

    # ---- readout: divergence per colour, no position block
    txt = app.readout.get('1.0', 'end')
    check('POSITION' not in txt and 'dx' not in txt, 'readout has no position (dx/dy) lines')
    rows = [l for l in txt.splitlines() if l.strip().startswith(('Red', 'Green', 'Blue', 'Whole beam'))]
    check(len(rows) == 4 and all(len(l.split()) >= 7 for l in rows), 'readout shows divergence for the beam and each colour')
    width = int(app.readout.cget('width'))
    longest = max(len(l) for l in txt.splitlines())
    check(longest <= width, 'every readout line fits the box (%d <= %d chars), nothing wraps' % (longest, width))

    def is_num(s):
        try:
            float(s)
            return True
        except ValueError:
            return s == '--'
    for l in rows:
        parts = l.split()
        name_len = 2 if parts[0] == 'Whole' else 1
        vals = parts[name_len:]
        check(len(vals) == 6 and all(is_num(v) for v in vals),
              '%s row = name + exactly 6 values, no stray digits' % parts[0])
    check(app.readout.cget('wrap') == 'none', 'readout never wraps lines')
    # ---- profile plots are cropped to the beam: the 1/e2 edges sit well inside the plot
    pw = app.prof_x.winfo_width()
    edges = sorted(app.prof_x.coords(i)[0] for i in app.prof_x.find_all()
                   if app.prof_x.type(i) == 'line' and app.prof_x.itemcget(i, 'fill') == '#ef9a9a'
                   and app.prof_x.coords(i)[0] == app.prof_x.coords(i)[2])
    check(len(edges) == 2 and edges[1] - edges[0] > 0.4 * pw and edges[0] > 0.1 * pw and edges[1] < 0.9 * pw,
          'X profile is cropped to the beam (1/e2 edges at %s of %d px)' % ([int(e) for e in edges], pw))
    # ---- narrow window: the readout keeps its full width, the graphs shrink
    root.geometry('1100x800')
    pump(root, 0.4)
    app.draw(msg)
    pump(root, 0.2)
    need = app.readout.winfo_reqwidth()
    have = app.readout.winfo_width()
    right_edge = app.readout.winfo_rootx() + have
    bottom_edge = app.bottom.winfo_rootx() + app.bottom.winfo_width()
    check(have >= need - 2 and right_edge <= bottom_edge + 1,
          'narrow window: readout not cut off (%d of %d px, inside the frame)' % (have, need))
    # what matters on any screen (CI screens are small, so the "wide" window may
    # already be narrow): graphs + readout together fit inside the window
    used = app.prof_x.winfo_width() + app.prof_y.winfo_width() + app.readout.winfo_width()
    check(used <= app.bottom.winfo_width() and app.prof_x.winfo_width() <= pw,
          'narrow window: graphs shrink to fit beside the readout (%d + %d + %d <= %d px)'
          % (app.prof_x.winfo_width(), app.prof_y.winfo_width(), app.readout.winfo_width(),
             app.bottom.winfo_width()))
    # hide the graphs
    app.show_profiles.set(False)
    app.on_show_profiles()
    pump(root, 0.2)
    check(not app.prof_x.winfo_ismapped() and not app.prof_y.winfo_ismapped() and app.readout.winfo_ismapped(),
          'Show X/Y profiles off hides both graphs, readout stays')
    app.show_profiles.set(True)
    app.on_show_profiles()
    pump(root, 0.2)
    check(app.prof_x.winfo_ismapped() and app.prof_y.winfo_ismapped(), 'and back on shows them again')
    root.geometry('1500x900')
    pump(root, 0.4)
    # ---- zoom 1x shows the whole image
    x0, y0, vw, vh = app._view
    check(abs(vw - W) < 1e-6 and abs(vh - H) < 1e-6 and x0 == 0 and y0 == 0, 'zoom 1x = whole sensor')
    # ---- zoom 4x, follow beam: window is 1/4 size and centred on the beam
    app.set_zoom(4.0)
    for _ in range(30):                       # smoothing converges
        app.draw(msg)
    x0, y0, vw, vh = app._view
    cx, cy = x0 + vw / 2, y0 + vh / 2
    check(abs(vw - W / 4) < 1e-6, 'zoom 4x window is a quarter of the width')
    check(abs(cx - res['cx_px']) < 1.0 and abs(cy - res['cy_px']) < 1.0, 'zoom follows the beam centre')
    # ---- crosshair readout: (0, 0) at the sensor centre, no "sensor centre" text
    app.set_crosshair(None)
    txt = app.cross_info.cget('text')
    check('+0.000' in txt and 'centre' not in txt.lower(), 'crosshair at the centre reads (0, 0): %r' % txt)
    # ---- a plain left click (press + release, no drag) puts the crosshair at the cursor
    ox, oy, sc, gx0, gy0 = app._geom['live']
    tx, ty = res['cx_px'] + 5.0, res['cy_px'] - 3.0
    ev = type('E', (), {'x': ox + (tx - gx0) * sc, 'y': oy + (ty - gy0) * sc})()
    app.on_live_press(ev)
    app.on_live_release(ev)
    c = app.s['crosshair_px']
    check(c is not None and abs(c[0] - tx) < 0.5 and abs(c[1] - ty) < 0.5,
          'left click places the crosshair at the cursor (also when zoomed)')
    x, y, xm, ym = app.cross_coords()
    check(abs(x - (tx - W / 2.0)) < 0.5 and abs(y - (H / 2.0 - ty)) < 0.5 and (xm > 0) == (x > 0),
          'crosshair coordinates are relative to the sensor centre, +y up (%.1f, %.1f px)' % (x, y))
    # ---- snapshot overlay: the snapped beam stays on the live view after the laser is off
    app.hide_snapshot()
    check(app.chk_overlay.instate(['disabled']) and app.overlay_reason.cget('text'),
          'Overlay on live is greyed out (with a reason) until there is a snapshot')
    app.save_snapshot()
    pump(root, 0.1)
    check(not app.chk_overlay.instate(['disabled']), 'Overlay on live is available once a snapshot is shown')
    seen = {}
    orig_render = app._render

    def spy(canvas, key, img, step, view, marks, **kw):
        if key == 'live':
            seen['img'], seen['marks'] = np.asarray(img), list(marks)
        return orig_render(canvas, key, img, step, view, marks, **kw)
    app._render = spy
    dark = dict(msg, preview=np.zeros_like(msg['preview']))          # all lasers now off
    app.draw_views(dark)
    check(seen['img'].max() == 0, 'overlay off: the live view is dark with the lasers off')
    app.overlay_on.set(True)
    app.on_overlay()
    app.draw_views(dark)
    check(seen['img'].max() > 200 and abs(int(seen['img'].sum()) - int(msg['preview'].sum())) < 0.02 * msg['preview'].sum(),
          'overlay on: the snapped beams show on the live view')
    check(sum(1 for m in seen['marks'] if len(m) > 2 and m[2] == 'ring') == len(app.snapshot['meta']['marks']) > 0,
          'overlay on: each snapped laser centre is drawn as a ring')
    app.scale_overlay.set(50)
    app.on_overlay()
    app.draw_views(dark)
    check(100 < seen['img'].max() < 140, 'overlay strength 50 %% dims the snapshot (peak %d)' % seen['img'].max())
    app.draw_views(msg)                                              # live beams on again
    check(seen['img'].max() == 255, 'live beams stay at full brightness under the overlay')
    app.hide_snapshot()
    app.draw_views(dark)
    check(not app.overlay_on.get() and seen['img'].max() == 0, 'Hide snapshot also ends the overlay')
    app._render = orig_render
    app.scale_overlay.set(100)
    app.on_overlay()
    # ---- contrast slider reaches the display settings sent to the camera process
    sent = []
    orig_send = app.send_set
    app.send_set = lambda d: sent.append(d)
    app.contrast.set(-40)
    app.on_contrast()
    pump(root, 0.15)
    check(sent and sent[-1]['display'].get('contrast') == -40, 'Contrast slider is sent with the display settings')
    app.contrast.set(0)
    app.on_contrast()
    pump(root, 0.15)
    # ---- per-laser boost and contrast sliders
    check(len(app.boost_vars) == 3 and len(app.lcon_vars) == 3, 'each laser has its own Boost and Contrast slider')
    app.boost_vars[1].set(2.5)
    app.on_laser_tone(app.boost_vars[1], app.boost_labels[1], '%.1fx')
    app.lcon_vars[2].set(-30)
    app.on_laser_tone(app.lcon_vars[2], app.lcon_labels[2], '%+d')
    pump(root, 0.15)
    d = sent[-1]['display'] if sent else {}
    check(d.get('boost') == [1.0, 2.5, 1.0] and d.get('lcontrast') == [0, 0, -30]
          and app.boost_labels[1].cget('text') == '2.5x',
          'per-laser Boost / Contrast are sent with the display settings (%s, %s)' % (d.get('boost'), d.get('lcontrast')))
    app.reset_laser_tone()
    d = sent[-1]['display']
    check(d['boost'] == [1.0, 1.0, 1.0] and d['lcontrast'] == [0, 0, 0] and d['atten'] == [1.0, 1.0, 1.0],
          'Reset puts every laser back to 100 %, 1.0x, +0')
    # ---- buttons that need a beam come alive when the beam appears AFTER connecting
    #      (Jon: Learn and crosshair Snap stayed grey with the laser on)
    class _Alive(object):
        def is_alive(self):
            return True
        exitcode = None
    saved = (app.proc, app.out_q, app.last, app.send_set)
    app.proc, app.out_q, app.last = _Alive(), queue.Queue(), None
    app.send_set = lambda d: None
    real = dict(msg, calibrating=False)                          # as the camera process sends them
    app.out_q.put(dict(real, result=dict(res, ok=False)))        # first frame: laser still off
    app._poll_body()
    grey = app.learn_btns[0].instate(['disabled']) and app.btn_snapcross.instate(['disabled'])
    app.out_q.put(real)                                          # laser turned on
    app._poll_body()
    check(grey and not app.learn_btns[0].instate(['disabled']) and not app.btn_snapcross.instate(['disabled']),
          'Learn and crosshair Snap come alive when a beam appears after connecting')
    app.out_q.put(dict(real, result=dict(res, ok=False)))        # laser off again
    app._poll_body()
    check(app.learn_btns[0].instate(['disabled']) and app.btn_snapcross.instate(['disabled']),
          'and grey out again (with a reason) when the beam goes away')
    app.proc, app.out_q, app.last, app.send_set = saved
    app._beam_seen = None
    app._refresh_blocked()
    # ---- Calibrate works with Auto exposure ON (the camera process holds the exposure)
    saved_proc = app.proc
    app.proc = type('P', (), {'is_alive': lambda self: True, 'exitcode': None})()
    app.auto_exp.set(True)
    app.pw_entries[0].delete(0, 'end')
    app.pw_entries[0].insert(0, '0.5')
    app._refresh_blocked()
    check(not app.pw_buttons[0].instate(['disabled']) and 'Auto exposure' not in app.pw_reason.cget('text'),
          'Calibrate is available with Auto exposure on (no "turn it off first")')
    app.pw_entries[0].delete(0, 'end')
    app.proc = saved_proc
    app._refresh_blocked()
    # ---- control panels in Jon's order, top to bottom
    root.update_idletasks()
    boxes = sorted(app._boxes.items(), key=lambda kv: kv[1].winfo_y())
    order = [t.split(' (')[0].replace('Display colours', 'Colours') for t, _ in boxes]
    check(order == ['Camera', 'Crosshair', 'View', 'Exposure', 'Colours', 'Snapshot', 'Optics',
                    'Power calibration', 'Log measurements'],
          'control panels top to bottom: %s' % ', '.join(order))
    # ---- which spot is measured (the LBA shows 3)
    sent_spot = []
    orig_send2 = app.send_set
    app.send_set = lambda d: sent_spot.append(d)
    check(app.spot.get() == 'Right spot', 'Measure defaults to the right spot')
    app.spot.set('Left spot')
    app.on_spot()
    check(sent_spot and sent_spot[-1] == {'spot': 'left'} and app._collect()['spot'] == 'left',
          'choosing Left spot is sent to the camera process and saved')
    app.draw_readout(dict(msg, result=dict(res, n_spots=3)))
    check('3 spots seen: measuring the left spot' in app.readout.get('1.0', 'end'),
          'the readout says how many spots there are and which one is measured')
    app.spot.set('Right spot')
    app.on_spot()
    app.send_set = orig_send2
    # ---- right-click a slider = its default
    app.boost_vars[0].set(3.0)
    app.gain.set(4.0)
    sc_boost = next(w for w, fn in app._reset_fns if isinstance(w, tk.Scale) and str(w.cget('variable')) == str(app.boost_vars[0]))
    sc_gain = next(w for w, fn in app._reset_fns if isinstance(w, tk.Scale) and str(w.cget('variable')) == str(app.gain))
    for w_ in (sc_boost, sc_gain):
        w_.event_generate('<Button-3>', x=5, y=5)
    pump(root, 0.15)
    check(abs(app.boost_vars[0].get() - 1.0) < 1e-9 and abs(app.gain.get() - 1.0) < 1e-9,
          'right-click puts a slider back to its default (boost %.1f, master boost %.1f)'
          % (app.boost_vars[0].get(), app.gain.get()))
    check(len(app._reset_fns) >= 15, 'every slider has a right-click reset (%d)' % len(app._reset_fns))
    # ---- learn a laser's colour (one laser on), saved per camera and sent to the camera process
    sent.clear()
    app.learn_colour(1)
    for _ in range(app.LEARN_FRAMES):
        app._learn_frame(dict(res, ok=True, peak_raw=[60, 200, 90], rates=[8.0, 100.0, 30.0]))
    sig = app._colour_sigs()[1]
    check(sig == [0.08, 1.0, 0.3] and app.learn_labels[1].cget('text') == 'learned',
          'Learn stores the laser colour as seen by the camera (%s)' % (sig,))
    cs = [d['colour_sig'] for d in sent if 'colour_sig' in d]
    check(cs and cs[-1][1] == [0.08, 1.0, 0.3], 'learned colour is sent to the camera process')
    app.learn_colour(0)
    app._learn_frame(dict(res, ok=True, peak_raw=[255, 40, 20], rates=[100.0, 18.0, 6.0]))
    check(app._learning is None and app._colour_sigs()[0] is None and 'saturated' in app.learn_reason.cget('text'),
          'Learn refuses a saturated picture and says why')
    app.learn_colour(2)
    app._learn_frame(dict(res, ok=False))
    check(app._colour_sigs()[2] is None and 'No beam' in app.learn_reason.cget('text'),
          'Learn with no beam says so')
    app.forget_colours()
    check(app._colour_sigs() == [None, None, None] and app.btn_forget.instate(['disabled']),
          'Forget learned colours clears them')
    # ---- zoomed in: the window asks for the zoomed area at full resolution and draws it
    frame_msg_saved = msg
    sent.clear()
    app.zoom_follow.set('Beam')
    app.set_zoom(12.0)
    for _ in range(30):
        app.draw(msg)
    reqs = [d['detail_view'] for d in sent if 'detail_view' in d]
    x0, y0, vw, vh = app._view
    want = reqs[-1] if reqs else None
    check(want is not None and want[0] <= x0 and want[1] <= y0 and x0 + vw <= want[2] and y0 + vh <= want[3],
          'zoomed in: the camera process is asked for the zoomed window (covers the view)')
    check(len(reqs) <= 4, 'the zoom window is not re-sent every frame while the view holds still (%d)' % len(reqs))
    # (this test camera is small, so its whole picture is already full resolution;
    # stand in a window cut from it, as the 4K camera process would send)
    bx0, by0, bx1, by1 = (int(round(v)) for v in want)
    det = {'img': msg['preview'][by0:by1, bx0:bx1].copy(), 'box': (bx0, by0, bx1, by1), 'step': 1}
    msg = dict(msg, step=2, preview=msg['preview'][::2, ::2].copy())    # coarse whole picture, like 4K
    seen = {}

    def spy2(canvas, key, img, step, view, marks, **kw):
        if key == 'live':
            seen['step'], seen['origin'] = step, kw.get('origin')
        return orig_render(canvas, key, img, step, view, marks, **kw)
    app._render = spy2
    app.draw_views(dict(msg, detail=det))
    check(det is not None and seen.get('step') == det['step'] < msg['step'] and seen.get('origin') == det['box'][:2],
          'zoomed in: the live view draws the full-resolution window (step %s, was %d)' % (seen.get('step'), msg['step']))
    vw_now = app._view[2]
    check(want[2] - want[0] >= 2.5 * vw_now and want[2] - want[0] <= worker.PREVIEW_MAX_W,
          'zoom window has a wide margin (%.0f px for a %.0f px view) but stays full resolution'
          % (want[2] - want[0], vw_now))
    # a snapshot taken zoomed in keeps the full-resolution window (and reopens with it)
    snaps = {}

    def spy3(canvas, key, img, step, view, marks, **kw):
        snaps[key] = (step, kw.get('origin'))
        return orig_render(canvas, key, img, step, view, marks, **kw)
    app._render = spy3
    app.last = dict(msg, detail=det)
    app.save_snapshot()
    app.draw_snapshot()
    check(snaps.get('snap') == (1, det['box'][:2]), 'zoomed-in snapshot is drawn from its full-resolution window %s'
          % (snaps.get('snap'),))
    p = app.snapshot['path']
    app.hide_snapshot()
    orig_ask = A.filedialog.askopenfilename
    A.filedialog.askopenfilename = lambda **k: p
    app.open_snapshot()
    A.filedialog.askopenfilename = orig_ask
    app.draw_snapshot()
    check(app.snapshot.get('detail') is not None and snaps.get('snap') == (1, det['box'][:2]),
          'reopened snapshot is still full resolution')
    app.hide_snapshot()
    app.last = frame_msg_saved
    app._render = spy2
    app.set_zoom(1.0)
    for _ in range(30):
        app.draw(msg)
    app.draw_views(dict(msg, detail=det))
    check(seen.get('step') == msg['step'] and sent[-1].get('detail_view', 1) is None,
          'zoomed out: back to the whole-sensor picture, zoom window cancelled')
    app._render = orig_render
    app.send_set = orig_send
    msg = frame_msg_saved
    app.set_zoom(4.0)                         # as the checks below expect
    for _ in range(30):
        app.draw(msg)
    check(not hasattr(app, 'btn_place'), 'no "Place by click" button any more')
    # ---- auto exposure stays ON while it moves the slider itself
    app.auto_exp.set(True)
    for us in (900.0, 1500.0, 2500.0, 4000.0):
        app._set_exp_widgets(us)
        pump(root, 0.05)                     # Tk delivers the slider command when idle
    check(app.auto_exp.get(), 'auto exposure stays on while it moves the slider')
    # ... and a real drag by the user still takes over
    app._slider_user(True)
    app.exp_slider.set(app.exp_slider.get() - 40)
    pump(root, 0.05)
    app._slider_user(False)
    pump(root, 0.3)
    check(not app.auto_exp.get(), 'dragging the slider yourself switches auto exposure off')
    check(not app._dragging, 'slider is "not dragging" again after release')
    # ---- exposure range ends at 400 ms
    app.on_opened({'serial': 'SIM0001', 'name': 'Simulator', 'model': 'x', 'exposure_range': (20.0, 10e6)})
    check(app.exposure_range[1] == 400000.0 and '400 ms' in app.exp_range_label.cget('text'),
          'exposure slider ends at 400 ms (%s)' % app.exp_range_label.cget('text'))
    app._set_exp_widgets(400000.0)
    check(app.exp_slider.get() == 1000, 'slider at its end = 400 ms')
    # ---- crosshair mode centres on the crosshair
    app.zoom_follow.set('Crosshair')
    app.draw(msg)
    x0, y0, vw, vh = app._view
    check(abs(x0 + vw / 2 - tx) < 0.5, 'zoom can centre on the crosshair')
    # ---- wheel zooms
    z = app.zoom.get()
    app.on_live_wheel(type('E', (), {'delta': 120})())
    check(app.zoom.get() > z, 'mouse wheel zooms in')
    # ---- the zoom never leaves the image
    app.zoom_follow.set('Manual')
    app._pan = [-500.0, -500.0]
    app.draw(msg)
    x0, y0, vw, vh = app._view
    check(x0 >= 0 and y0 >= 0, 'manual pan is clamped to the image')
    # ---- snapshot uses the same window, side by side
    app.zoom_follow.set('Beam')
    app.save_snapshot()
    pump(root, 0.2)
    app.draw(msg)
    pump(root)
    check(app.snapc.winfo_ismapped(), 'snapshot shown next to the live view')
    check(app._geom['snap'][3:] == app._geom['live'][3:], 'snapshot starts at the view it was taken with')
    # the snapshot must NOT move when the live beam moves or the live zoom changes
    import copy
    snap_before = tuple(app._snap_view)
    moved = copy.deepcopy(msg)
    moved['result']['cx_px'] += 120
    moved['result']['cy_px'] += 80
    for _ in range(20):
        app.draw(moved)
    app.set_zoom(app.zoom.get() * 1.5)
    app.draw(moved)
    check(tuple(app._snap_view) == snap_before and app._geom['snap'][3:] == snap_before[:2],
          'snapshot stays still while the live beam moves and the live zoom changes')
    check(app._geom['live'][3:] != app._geom['snap'][3:], '... while the live view did move')
    z_snap = 640.0 / app._snap_view[2] if False else app._snap_view[2]
    app.on_snap_wheel(type('E', (), {'delta': 120})())
    check(app._snap_view[2] < z_snap, 'mouse wheel on the snapshot zooms the snapshot only')
    app.draw(msg)
    snaps = sorted(os.listdir(settings.SNAP_DIR))
    check(any(s.endswith('_raw.png') for s in snaps) and any(s.endswith('.json') for s in snaps),
          'snapshot saved as PNG + raw PNG + JSON')
    lines = app._snap_lines(app.snapshot['meta'])
    check(sum(1 for l in lines if l.strip().startswith(('Red', 'Green', 'Blue', 'Beam'))) == 4,
          'snapshot box shows divergence for the beam and each colour')
    # ---- Auto ROI off freezes the box; a left-drag draws a new one
    sent = []
    app.send_set = lambda d: sent.append(d)
    app.auto_roi.set(False)
    app.on_auto_roi()
    check(sent and sent[-1].get('auto_roi') is False and sent[-1].get('roi') == [int(v) for v in res['roi']],
          'unticking Auto ROI freezes the current ROI and sends it')
    app.zoom_follow.set('Beam')
    app.draw(msg)
    ox, oy, sc, gx0, gy0 = app._geom['live']
    a0, a1 = (res['cx_px'] - 30, res['cy_px'] - 20), (res['cx_px'] + 40, res['cy_px'] + 25)
    e = lambda a: type('E', (), {'x': ox + (a[0] - gx0) * sc, 'y': oy + (a[1] - gy0) * sc})()
    app.on_live_press(e(a0))
    app.on_live_drag(e(a1))
    app.on_live_release(e(a1))
    roi = sent[-1].get('roi')
    check(roi and abs(roi[0] - a0[0]) <= 1.5 and abs(roi[3] - a1[1]) <= 1.5 and sent[-1]['auto_roi'] is False,
          'left-drag on the picture draws the ROI (%s)' % roi)
    raw, exp, _ = CAM.grab()
    fixed = analysis.analyse(raw, exp, analysis.Optics(), roi=tuple(roi), auto=False)
    check(tuple(fixed['roi']) == tuple(roi), 'the analysis really uses the fixed ROI')
    app.auto_roi.set(True)
    app.on_auto_roi()
    check(sent[-1] == {'auto_roi': True}, 'ticking Auto ROI goes back to following the spot')
    del app.send_set
    # ---- idle full screen
    app.fs_enabled.set(True)
    app.fs_timeout.set(5)
    app._last_act = time.time() - 6
    app._ptr = root.winfo_pointerxy()
    app.idle_tick()
    pump(root, 0.3)
    check(app._fs and not app.left_wrap.winfo_ismapped(), 'idle timeout -> full screen, controls hidden')
    check(bool(root.attributes('-fullscreen')), 'window is full screen')
    app.draw(msg)
    check(len(app.live.find_withtag('hud')) > 0, 'full screen shows the compact readout')
    # a key press during the first 0.8 s is ignored (layout change), then wakes it
    app.on_activity(type('E', (), {'type': 'KeyPress'})())
    check(app._fs, 'events right after entering are ignored')
    app._ignore_until = 0
    app.on_activity(type('E', (), {'type': 'KeyPress'})())
    pump(root, 0.3)
    check(not app._fs and app.left_wrap.winfo_ismapped() and app.bottom.winfo_ismapped(),
          'key press restores the normal GUI')
    # mouse movement wakes it too
    app.enter_fullscreen()
    pump(root, 0.2)
    app._ignore_until = 0
    px, py = root.winfo_pointerxy()
    app._ptr = (px + 50, py + 50)            # pretend the pointer was elsewhere 250 ms ago
    app.idle_tick()
    pump(root, 0.3)
    check(not app._fs, 'mouse movement restores the normal GUI')
    # ---- an error while drawing a frame must not stop the picture (it used to, silently)
    import queue as _q
    shown_boxes = []
    A.messagebox.showerror = lambda *a, **k: shown_boxes.append(a)
    real_draw = app.draw
    calls = {'n': 0}

    def flaky_draw(m):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ValueError('test error in draw')
        real_draw(m)
    app.draw = flaky_draw
    app.out_q = _q.Queue()
    after_before = len(root.tk.splitlist(root.tk.call('after', 'info')))
    app.out_q.put(msg)
    app.poll()
    check(len(root.tk.splitlist(root.tk.call('after', 'info'))) == after_before + 1,
          'picture loop keeps running after an error while drawing')
    log_txt = open(settings.ERROR_LOG, encoding='utf-8').read() if os.path.exists(settings.ERROR_LOG) else ''
    check('test error in draw' in log_txt and len(shown_boxes) == 1, 'the error is written to errors.log and shown once')
    n0 = app._frame_no
    app.out_q.put(msg)
    app.poll()
    check(app._frame_no == n0 + 1 and calls['n'] == 2, 'the next frame draws normally')
    app.out_q.put(msg)
    app.draw = lambda m: (_ for _ in ()).throw(ValueError('test error in draw'))
    app.poll()
    check(len(shown_boxes) == 1, 'the same error again is logged but not shown again')
    app.draw = real_draw
    app.out_q = None
    # the idle timer must survive an error inside it (a Combobox popdown used to kill it)
    real_body = app._idle_body

    def boom():
        raise KeyError('popdown')
    app._idle_body = boom
    before = len(root.tk.splitlist(root.tk.call('after', 'info')))
    app.idle_tick()
    after = len(root.tk.splitlist(root.tk.call('after', 'info')))
    app._idle_body = real_body
    check(after == before + 1, 'idle timer keeps running after an error inside it')
    # a real grab (dialog open) is detected without crashing
    top = tk.Toplevel(root)
    top.grab_set()
    app._last_act = time.time() - 100
    app.last = msg
    app._idle_body()
    check(not app._fs, 'no full screen while a dialog holds the mouse')
    top.grab_release()
    top.destroy()
    # no full screen while a dialog holds the grab or with no camera picture
    app.last = None
    app._last_act = time.time() - 100
    app.idle_tick()
    check(not app._fs, 'no full screen without a live picture')
    app.last = msg
    # ================= auto zoom
    import copy
    app.show_cross.set(True)
    app.set_crosshair(None)
    app.set_zoom(1.0)
    app.zoom_follow.set(app.AUTO)
    app.auto_cross.set(False)
    zs = []
    for _ in range(80):
        app.draw(msg)
        zs.append(app.zoom.get())
    check(zs[-1] > 2.0, 'auto zoom zooms in on the spots (%.2fx)' % zs[-1])
    check(all(b >= a - 1e-9 for a, b in zip(zs, zs[1:])), 'auto zoom-in is smooth (never steps back out)')
    check(abs(zs[-1] - zs[-10]) / zs[-1] < 0.01, 'auto zoom settles, no pumping')

    def inside(view, box):
        x0, y0, vw, vh = view
        return box[0] >= x0 and box[1] >= y0 and box[2] <= x0 + vw and box[3] <= y0 + vh
    check(all(inside(app._view, b) for b in res['laser_box'] if b), 'all spots inside the auto view')
    # a beam adjusted too far: the blue spot jumps 300 px away -> zoom out on the NEXT frame
    stray = copy.deepcopy(msg)
    r2 = stray['result']
    bx = list(r2['laser_box'][2])
    dx, dy = 250, 180
    r2['laser_box'][2] = (bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy)
    r2['laser_cxy'][2] = (r2['laser_cxy'][2][0] + dx, r2['laser_cxy'][2][1] + dy)
    z_before = app.zoom.get()
    app.draw(stray)
    check(app.zoom.get() < z_before, 'stray beam -> zooms out at once (%.2fx -> %.2fx)' % (z_before, app.zoom.get()))
    check(all(inside(app._view, b) for b in r2['laser_box'] if b), 'stray beam is inside the view on that same frame')
    # beam comes back -> zooms in again, slowly
    z_out = app.zoom.get()
    app.draw(msg)
    z1 = app.zoom.get()
    check(z_out <= z1 < z_out * 1.5, 'after the beam returns it eases back in (no jump): %.2fx -> %.2fx' % (z_out, z1))
    for _ in range(80):
        app.draw(msg)
    check(app.zoom.get() > z_out * 1.5, 'and ends up zoomed in again')
    # keep crosshair in view
    app.auto_cross.set(True)
    app.set_crosshair([20.0, 20.0])
    app.zoom_follow.set(app.AUTO)
    app.draw(msg)
    x0, y0, vw, vh = app._view
    check(x0 <= 20 <= x0 + vw and y0 <= 20 <= y0 + vh, 'auto keeps the crosshair in view')
    app.set_crosshair(None)
    # nothing visible -> whole sensor
    dark = copy.deepcopy(msg)
    dark['result']['laser_box'] = [None, None, None]
    dark['result']['ok'] = False
    app.draw(dark)
    check(app.zoom.get() == 1.0, 'no beam -> auto shows the whole sensor')
    # a hand zoom takes over from auto
    app.draw(msg)
    app.on_live_wheel(type('E', (), {'delta': 120})())
    check(app.zoom_follow.get() == 'Beam', 'mouse wheel switches auto off (to Beam)')
    root.destroy()
    print('%d failed' % len(fails))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
