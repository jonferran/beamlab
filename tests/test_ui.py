"""UI checks with the real window and simulated frames (no worker process).
Run: python tests/test_ui.py      (opens and closes a window briefly)"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamlab import analysis, settings, worker  # noqa: E402
from beamlab.cameras import SimCamera  # noqa: E402

# keep the user's real settings and snapshots out of the test
_tmp = tempfile.mkdtemp(prefix='beamlab_ui_')
settings.DIR = _tmp
settings.FILE = os.path.join(_tmp, 'settings.json')
settings.SNAP_DIR = os.path.join(_tmp, 'snapshots')

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
    check(app.prof_x.winfo_width() < pw, 'narrow window: profile graphs got narrower (%d -> %d px)'
          % (pw, app.prof_x.winfo_width()))
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
    # ---- a click maps back to analysis px through the zoom
    app.click_mode = 'crosshair'
    ox, oy, sc, gx0, gy0 = app._geom['live']
    tx, ty = res['cx_px'] + 5.0, res['cy_px'] - 3.0
    ev = type('E', (), {'x': ox + (tx - gx0) * sc, 'y': oy + (ty - gy0) * sc})()
    app.on_live_click(ev)
    c = app.s['crosshair_px']
    check(abs(c[0] - tx) < 0.5 and abs(c[1] - ty) < 0.5, 'click places the crosshair at the right spot when zoomed')
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
