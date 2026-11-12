"""BeamLab: a desktop beam analyzer for the LBA-C-100 far-field analyzer (and any
Basler / Imaging Source BayerGR8 camera). Tkinter UI; the camera and all image
work run in a separate process (beamlab.worker), so the window never stalls.
"""
import csv
import datetime
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import settings as S
from . import worker
from . import __version__
from .cameras import MAX_EXPOSURE_US
from .settings import log_error

BG = '#202124'
PANEL = '#2b2c2f'
FG = '#e8e8e8'
DIM = '#9aa0a6'
WARN = '#ffb74d'
BAD = '#ef5350'
GOOD = '#81c784'

COLOR_PRESETS = {
    'True colours': [[255, 0, 0], [0, 255, 0], [0, 80, 255]],
    'All white': [[255, 255, 255]] * 3,
    'All red': [[255, 0, 0]] * 3,
    'All green': [[0, 255, 0]] * 3,
    'All blue': [[0, 90, 255]] * 3,
    'All yellow': [[255, 230, 0]] * 3,
    'All magenta': [[255, 0, 255]] * 3,
}


def fmt(v, nd=3, unit=''):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '--'
    return ('%.' + str(nd) + 'f%s') % (v, unit)


def pick_detail(view, img, step, detail):
    """The picture to draw `view` (x0, y0, w, h analysis px) from: the zoomed-in
    full-resolution window when it covers the view, else the whole-sensor picture.
    img / detail['img'] may be numpy arrays or PIL images -> (img, step, origin)."""
    if detail:
        x0, y0, vw, vh = view
        bx0, by0, bx1, by1 = detail['box']
        if bx0 <= x0 + 1e-6 and by0 <= y0 + 1e-6 and x0 + vw <= bx1 + 1e-6 and y0 + vh <= by1 + 1e-6:
            return detail['img'], detail['step'], (bx0, by0)
    return img, step, (0, 0)


def overlay_snapshot(live, snap, strength=1.0, region=None):
    """Lay a snapshot over the live preview: each pixel keeps the brighter of the
    two (the snapshot dimmed to `strength`), so a beam you snapped and then turned
    off stays on screen next to the live one. live: HxWx3 uint8; snap: PIL image
    of the whole sensor (any preview step); region = the part of `snap` (its px)
    that `live` shows, default all of it -> HxWx3 uint8."""
    h, w = live.shape[:2]
    if region is not None:
        snap = snap.resize((w, h), Image.BICUBIC, box=tuple(region))
    elif snap.size != (w, h):   # area average when shrinking keeps a tight spot sharp
        snap = snap.resize((w, h), Image.BOX if snap.size[0] > w else Image.BILINEAR)
    s = np.asarray(snap.convert('RGB'), dtype=np.uint8)
    if strength < 0.999:
        s = (s.astype(np.float32) * max(0.0, float(strength))).astype(np.uint8)
    return np.maximum(live, s)


def hexcol(c):
    return '#%02x%02x%02x' % tuple(int(x) for x in c)


class App(object):
    def __init__(self, root):
        self.root = root
        self.s = S.load()
        self.proc = None
        self.cmd_q = None
        self.out_q = None
        self.serial = None
        self.cam_name = ''
        self.exposure_range = (20.0, MAX_EXPOSURE_US)
        self.last = None             # last 'frame' message
        self.snapshot = None         # {'img': PIL.Image, 'meta': dict, 'path': str}
        self.csv_file = None
        self.csv_writer = None
        self.calibrating = None      # laser index while a calibration runs
        self._exp_send_job = None
        self._geom = {}              # canvas -> (ox, oy, display px per analysis px, x0, y0)
        self._zc = None              # smoothed zoom centre (analysis px)
        self._pan = None             # manual zoom centre (analysis px)
        self._view = None            # (x0, y0, w, h) visible window, analysis px
        self._fs = False             # idle full screen active
        self._last_act = time.time()
        self._ptr = None
        self._ignore_until = 0.0
        self._frame_no = 0           # frames shown since connecting
        self._last_frame_t = None
        self._stall_reported = False
        self._errors_shown = set()
        root.report_callback_exception = self._on_tk_error   # no silent errors in callbacks
        root.title('BeamLab - laser beam analyzer')
        root.configure(bg=BG)
        root.geometry('1500x900')
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self._style()
        self._build()
        self._refresh_blocked()
        root.after(15, self.poll)

    # ================================================================ UI build
    def _style(self):
        st = ttk.Style(self.root)
        try:
            st.theme_use('clam')
        except tk.TclError:
            pass
        for w in ('TFrame', 'TLabelframe', 'TLabel', 'TCheckbutton', 'TRadiobutton'):
            st.configure(w, background=PANEL, foreground=FG)
        st.configure('TLabelframe.Label', background=PANEL, foreground=FG, font=('Segoe UI', 9, 'bold'))
        st.configure('Dim.TLabel', foreground=DIM)
        st.configure('Warn.TLabel', foreground=WARN)
        st.configure('Big.TLabel', font=('Consolas', 11))
        st.configure('TButton', padding=(6, 2))

    def _build(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill='both', expand=True)
        # ---- left control column (scrollable)
        left_wrap = ttk.Frame(outer)
        left_wrap.pack(side='left', fill='y')
        self.left_wrap = left_wrap
        cv = tk.Canvas(left_wrap, width=330, bg=PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(left_wrap, orient='vertical', command=cv.yview)
        self.left = ttk.Frame(cv)
        self.left.bind('<Configure>', lambda e: cv.configure(scrollregion=cv.bbox('all')))
        cv.create_window((0, 0), window=self.left, anchor='nw')
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side='left', fill='y')
        sb.pack(side='left', fill='y')
        cv.bind_all('<MouseWheel>', lambda e: cv.yview_scroll(int(-e.delta / 120), 'units')
                    if str(e.widget).startswith(str(cv)) else None)
        self._build_camera(self.left)
        self._build_exposure(self.left)
        self._build_view(self.left)
        self._build_colours(self.left)
        self._build_crosshair(self.left)
        self._build_snapshot(self.left)
        self._build_power(self.left)
        self._build_optics(self.left)
        self._build_log(self.left)
        # Jon's order, top to bottom (built in the order above: some boxes use
        # widgets of others, so they are only re-packed here)
        for title in self.PANEL_ORDER:
            f = self._boxes.get(title)
            if f is not None:
                f.pack_forget()
                f.pack(fill='x', padx=6, pady=4)
        # ---- right: views + readouts
        right = ttk.Frame(outer)
        right.pack(side='left', fill='both', expand=True)
        self.right = right
        views = ttk.Frame(right)
        views.pack(side='top', fill='both', expand=True)
        self.live = tk.Canvas(views, bg='black', highlightthickness=0, cursor='crosshair')
        self.live.pack(side='left', fill='both', expand=True, padx=(4, 2), pady=4)
        self.live.bind('<ButtonPress-1>', self.on_live_press)
        self.live.bind('<B1-Motion>', self.on_live_drag)
        self.live.bind('<ButtonRelease-1>', self.on_live_release)
        self.live.bind('<MouseWheel>', self.on_live_wheel)
        self.live.bind('<ButtonPress-3>', self.on_pan_start)
        self.live.bind('<B3-Motion>', self.on_pan_drag)
        self.snapc = tk.Canvas(views, bg='#111', highlightthickness=0)
        self.snapc.bind('<MouseWheel>', self.on_snap_wheel)
        self._snap_view = None       # the snapshot's own FIXED window (analysis px)
        self._roi_drag = None
        self._live_item = None
        self._snap_item = None
        bottom = ttk.Frame(right)
        bottom.pack(side='top', fill='x')
        self.bottom = bottom
        # wrap='none': a line that is too long must be cut off, never wrapped - a wrapped
        # last digit showed up as a "random integer" at the start of the next row
        self.readout = tk.Text(bottom, width=68, height=15, bg='#111', fg=FG, bd=0, wrap='none',
                               font=('Consolas', 10))
        # packed FIRST (right side) so it always gets its full width; the profile graphs
        # only get what is left over and shrink (small requested width) - before, they
        # kept their size and the readout's right edge was cut off
        self.readout.pack(side='right', fill='y', padx=(2, 4), pady=(0, 4))
        self.prof_x = tk.Canvas(bottom, width=60, height=150, bg='#111', highlightthickness=0)
        self.prof_y = tk.Canvas(bottom, width=60, height=150, bg='#111', highlightthickness=0)
        self._pack_profiles(self.s.get('show_profiles', True))
        for tag, col in (('dim', DIM), ('warn', WARN), ('bad', BAD), ('good', GOOD), ('h', '#ffffff'),
                         ('c0', '#ff8a80'), ('c1', '#b9f6ca'), ('c2', '#82b1ff')):
            self.readout.tag_configure(tag, foreground=col)
        self.status = ttk.Label(self.root, text='Not connected.', style='Dim.TLabel', anchor='w')
        self.status.pack(side='bottom', fill='x')
        # activity tracking for the idle full screen (pointer movement is polled)
        for seq in ('<Key>', '<ButtonPress>', '<MouseWheel>'):
            self.root.bind_all(seq, self.on_activity, add='+')
        self.root.after(250, self.idle_tick)

    def _pack_profiles(self, show):
        if show:
            if not self.prof_x.winfo_ismapped():
                self.prof_x.pack(side='left', fill='both', expand=True, padx=(4, 2), pady=(0, 4))
                self.prof_y.pack(side='left', fill='both', expand=True, padx=(2, 2), pady=(0, 4))
        else:
            self.prof_x.pack_forget()
            self.prof_y.pack_forget()

    def on_show_profiles(self):
        self.s['show_profiles'] = self.show_profiles.get()
        self._pack_profiles(self.show_profiles.get())

    PANEL_ORDER = ('Camera', 'Crosshair (alignment reference)', 'View', 'Exposure',
                   'Display colours (for goggles)', 'Snapshot', 'Optics (pixels -> mrad)',
                   'Power calibration', 'Log measurements')

    def _box(self, parent, title):
        f = ttk.LabelFrame(parent, text=title, padding=6)
        f.pack(fill='x', padx=6, pady=4)
        self._boxes = getattr(self, '_boxes', {})
        self._boxes[title] = f
        return f

    def _reason(self, parent):
        lab = ttk.Label(parent, text='', style='Warn.TLabel', wraplength=290)
        return lab

    # ---------------------------------------------------------------- camera
    def _build_camera(self, p):
        f = self._box(p, 'Camera')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Source').pack(side='left')
        self.backend = tk.StringVar(value=self.s['backend'])
        cb = ttk.Combobox(r, textvariable=self.backend, width=22, state='readonly',
                          values=['auto', 'pylon', 'ic4', 'sim'])
        cb.pack(side='left', padx=4)
        ttk.Label(f, text='auto = Basler, then Imaging Source;  sim = built-in test beam',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(4, 0))
        self.btn_connect = ttk.Button(r, text='Connect', command=self.connect)
        self.btn_connect.pack(side='left')
        self.btn_disconnect = ttk.Button(r, text='Disconnect', command=self.disconnect)
        self.btn_disconnect.pack(side='left', padx=4)
        self.cam_info = ttk.Label(f, text='', style='Dim.TLabel', wraplength=290)
        self.cam_info.pack(anchor='w')

    # ---------------------------------------------------------------- exposure
    def _build_exposure(self, p):
        f = self._box(p, 'Exposure')
        r = ttk.Frame(f)
        r.pack(fill='x')
        self.exp_label = ttk.Label(r, text='', style='Big.TLabel', width=12)
        self.exp_label.pack(side='left')
        self.auto_exp = tk.BooleanVar(value=self.s['auto_exposure'])
        ttk.Checkbutton(r, text='Auto', variable=self.auto_exp,
                        command=self.on_auto_exp).pack(side='left', padx=6)
        ttk.Label(r, text='target').pack(side='left')
        self.ae_target = tk.IntVar(value=int(round(100 * self.s.get('ae_target', 0.85))))
        sp = ttk.Spinbox(r, from_=40, to=95, increment=5, width=4, textvariable=self.ae_target,
                         command=self.on_ae_target)
        sp.pack(side='left')
        sp.bind('<Return>', lambda e: self.on_ae_target())
        ttk.Label(r, text='%').pack(side='left')
        # logarithmic slider: 0..1000 -> log10(min)..log10(max)
        self.exp_slider = tk.Scale(f, from_=0, to=1000, orient='horizontal', showvalue=False,
                                   length=290, bg=PANEL, fg=FG, troughcolor='#444',
                                   highlightthickness=0, command=self.on_exp_slider)
        self.exp_slider.pack(fill='x')
        self._on_right_reset(self.exp_slider, self.reset_exposure)
        self._dragging = False
        self._slider_release_job = None
        self.exp_slider.bind('<ButtonPress-1>', lambda e: self._slider_user(True))
        self.exp_slider.bind('<ButtonRelease-1>', lambda e: self._slider_user(False))
        self.exp_slider.bind('<KeyPress>', lambda e: self._slider_user(True))
        self.exp_slider.bind('<KeyRelease>', lambda e: self._slider_user(False))
        self.exp_range_label = ttk.Label(f, text='', style='Dim.TLabel')
        self.exp_range_label.pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Exact (us)').pack(side='left')
        self.exp_entry = ttk.Entry(r, width=10)
        self.exp_entry.pack(side='left', padx=4)
        self.exp_entry.bind('<Return>', lambda e: self.on_exp_entry())
        ttk.Button(r, text='Set', command=self.on_exp_entry).pack(side='left')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(4, 0))
        ttk.Label(r, text='Average frames').pack(side='left')
        self.stack = tk.IntVar(value=self.s['stack'])
        sc = tk.Scale(r, from_=1, to=30, orient='horizontal', variable=self.stack, length=150,
                      bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0,
                      command=lambda v: self.send_set({'stack': int(float(v))}))
        sc.pack(side='left')
        self._on_right_reset(sc, lambda: (self.stack.set(int(S.DEFAULTS['stack'])),
                                          self.send_set({'stack': int(S.DEFAULTS['stack'])})))
        self.auto_roi = tk.BooleanVar(value=self.s['auto_roi'])
        ttk.Checkbutton(f, text='Auto ROI (follow the spot)', variable=self.auto_roi,
                        command=self.on_auto_roi).pack(anchor='w')
        # the LBA shows 3 spots side by side: measure the one you use
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Measure').pack(side='left')
        self.spot = tk.StringVar(value=self.SPOT_NAMES.get(self.s.get('spot', 'right'), 'Right spot'))
        cb = ttk.Combobox(r, textvariable=self.spot, values=list(self.SPOT_NAMES.values()), width=14,
                          state='readonly')
        cb.pack(side='left', padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self.on_spot())
        self.roi_info = ttk.Label(f, text='', style='Dim.TLabel', wraplength=290)
        self.roi_info.pack(anchor='w')
        self._roi_text()
        self._exp_range_text()
        self._set_exp_widgets(self.s['exposure_us'])

    def _roi_text(self):
        if self.auto_roi.get():
            t = 'ROI follows the spot. Left-drag on the picture to draw your own.'
        else:
            r = self.s.get('roi')
            t = ('Fixed ROI x %d-%d, y %d-%d (analysis px). Left-drag to redraw; tick Auto to '
                 'follow the spot again.' % (r[0], r[2], r[1], r[3])) if r else 'Fixed ROI'
        self.roi_info.configure(text=t)

    def on_auto_roi(self):
        """Unticking freezes the ROI where it is now (before, nothing changed because
        there was no manual ROI to fall back to)."""
        if self.auto_roi.get():
            self.send_set({'auto_roi': True})
        else:
            roi = None
            if self.last and self.last['result'].get('roi'):
                roi = [int(v) for v in self.last['result']['roi']]
            roi = roi or self.s.get('roi')
            if roi is None and self.last:
                h, w = self.last['result']['shape']
                roi = [0, 0, int(w), int(h)]
            self.s['roi'] = roi
            self.send_set({'auto_roi': False, 'roi': roi})
        self._roi_text()

    def set_roi(self, roi):
        self.s['roi'] = roi
        self.auto_roi.set(False)
        self.send_set({'auto_roi': False, 'roi': roi})
        self._roi_text()

    def _exp_range_text(self):
        lo, hi = self.exposure_range
        f = lambda us: ('%.0f ms' % (us / 1000)) if us >= 1000 else ('%.0f us' % us)
        self.exp_range_label.configure(text='slider: %s ... %s (logarithmic)' % (f(lo), f(hi)))

    def _exp_to_slider(self, us):
        lo, hi = self.exposure_range
        return int(round(1000 * (math.log10(us) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))))

    def _slider_to_exp(self, v):
        lo, hi = self.exposure_range
        return 10 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * float(v) / 1000.0)

    def _set_exp_widgets(self, us):
        self.exp_label.configure(text=('%.1f ms' % (us / 1000)) if us >= 1000 else ('%.0f us' % us))
        # Tk calls the slider's command LATER (when idle), after any "set by code" flag
        # would already be cleared - so the app's own moves looked like a user drag and
        # switched auto exposure off. on_exp_slider now only acts on real user input.
        self.exp_slider.set(self._exp_to_slider(min(max(us, self.exposure_range[0]), self.exposure_range[1])))
        if self.root.focus_get() is not self.exp_entry:
            self.exp_entry.delete(0, 'end')
            self.exp_entry.insert(0, '%.0f' % us)

    def _slider_user(self, active):
        """Mouse/keyboard on the exposure slider. Release is delayed a little so the
        deferred command from the last movement still counts as the user's."""
        if active:
            if self._slider_release_job:
                self.root.after_cancel(self._slider_release_job)
                self._slider_release_job = None
            self._dragging = True
        else:
            self._slider_release_job = self.root.after(150, self._slider_released)

    def _slider_released(self):
        self._slider_release_job = None
        self._dragging = False

    def on_exp_slider(self, v):
        if not self._dragging:
            return                        # moved by the app (auto exposure / typed value): ignore
        us = self._slider_to_exp(v)
        if self.auto_exp.get():           # dragging takes over from auto
            self.auto_exp.set(False)
            self.send_set({'auto_exposure': False})
        self.s['exposure_us'] = us
        self.exp_label.configure(text=('%.1f ms' % (us / 1000)) if us >= 1000 else ('%.0f us' % us))
        if self._exp_send_job:             # send at most every 60 ms while dragging
            self.root.after_cancel(self._exp_send_job)
        self._exp_send_job = self.root.after(60, self._send_exp_now)

    def _send_exp_now(self):
        self._exp_send_job = None
        self.send_set({'exposure_us': self.s['exposure_us']})

    SPOT_NAMES = {'right': 'Right spot', 'middle': 'Middle spot', 'left': 'Left spot',
                  'brightest': 'Brightest spot'}

    def spot_key(self):
        name = self.spot.get()
        return next((k for k, v in self.SPOT_NAMES.items() if v == name), 'right')

    def on_spot(self):
        self.s['spot'] = self.spot_key()
        self.send_set({'spot': self.s['spot']})

    def _on_right_reset(self, widget, fn):
        """Right-click a slider = back to its default value."""
        self._reset_fns = getattr(self, '_reset_fns', [])
        self._reset_fns.append((widget, fn))
        widget.bind('<Button-3>', lambda e: (fn(), 'break')[1])

    def reset_exposure(self):
        """Right-click on the exposure slider: the default exposure (auto exposure off)."""
        self.exp_entry.delete(0, 'end')
        self.exp_entry.insert(0, '%g' % S.DEFAULTS['exposure_us'])
        self.on_exp_entry()

    def on_exp_entry(self):
        try:
            us = float(self.exp_entry.get())
        except ValueError:
            messagebox.showerror('Exposure', 'Type the exposure in microseconds, e.g. 5000 for 5 ms.')
            return
        lo, hi = self.exposure_range
        us = min(max(us, lo), hi)
        self.auto_exp.set(False)
        self.s['exposure_us'] = us
        self._set_exp_widgets(us)
        self.send_set({'auto_exposure': False, 'exposure_us': us})

    def on_ae_target(self):
        try:
            v = min(max(int(self.ae_target.get()), 40), 95) / 100.0
        except (ValueError, tk.TclError):
            return
        self.s['ae_target'] = v
        self.send_set({'ae_target': v})

    def on_auto_exp(self):
        self.s['auto_exposure'] = self.auto_exp.get()
        self.send_set({'auto_exposure': self.auto_exp.get()})

    # ---------------------------------------------------------------- view: zoom + full screen
    def _build_view(self, p):
        f = self._box(p, 'View')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Zoom').pack(side='left')
        self.zoom = tk.DoubleVar(value=float(self.s.get('zoom', 1.0)))
        sc = tk.Scale(r, from_=1, to=16, resolution=0.25, orient='horizontal', variable=self.zoom,
                      length=170, bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0,
                      command=lambda v: self.on_zoom_slider())
        sc.pack(side='left')
        self._on_right_reset(sc, lambda: self.set_zoom(1.0))
        ttk.Button(r, text='1x', width=3, command=lambda: self.set_zoom(1.0)).pack(side='left', padx=2)
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Centre on').pack(side='left')
        self.zoom_follow = tk.StringVar(value=self.s.get('zoom_follow', self.AUTO))
        cb = ttk.Combobox(r, textvariable=self.zoom_follow, values=[self.AUTO, 'Beam', 'Crosshair', 'Manual'],
                          width=18, state='readonly')
        cb.pack(side='left', padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self._redraw_now())
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Auto zoom: at most').pack(side='left')
        self.auto_zmax = tk.DoubleVar(value=float(self.s.get('auto_zoom_max', 8.0)))
        ttk.Spinbox(r, from_=1, to=16, increment=1, width=4, textvariable=self.auto_zmax).pack(side='left', padx=2)
        ttk.Label(r, text='x').pack(side='left')
        self.auto_cross = tk.BooleanVar(value=self.s.get('auto_zoom_crosshair', True))
        ttk.Checkbutton(r, text='keep crosshair in view', variable=self.auto_cross).pack(side='left', padx=4)
        ttk.Label(f, text='Auto zooms in on all spots and zooms straight out when a beam strays. '
                  'Mouse wheel zooms; right-drag pans (switches to Manual).',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')
        self.show_profiles = tk.BooleanVar(value=self.s.get('show_profiles', True))
        ttk.Checkbutton(f, text='Show X/Y profile graphs', variable=self.show_profiles,
                        command=self.on_show_profiles).pack(anchor='w', pady=(4, 0))
        # idle full screen ("screensaver")
        self.fs_enabled = tk.BooleanVar(value=self.s.get('fs_enabled', False))
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(6, 0))
        ttk.Checkbutton(r, text='Full screen after', variable=self.fs_enabled,
                        command=self._touch).pack(side='left')
        self.fs_timeout = tk.IntVar(value=int(self.s.get('fs_timeout_s', 60)))
        ttk.Spinbox(r, from_=5, to=3600, increment=5, width=5,
                    textvariable=self.fs_timeout).pack(side='left', padx=2)
        ttk.Label(r, text='s idle').pack(side='left')
        r = ttk.Frame(f)
        r.pack(fill='x')
        self.btn_fs = ttk.Button(r, text='Full screen now', command=self.enter_fullscreen)
        self.btn_fs.pack(side='left')
        self.fs_info = ttk.Label(f, text='Live (and snapshot) fill the screen; move the mouse or '
                                 'press a key to come back.', style='Dim.TLabel', wraplength=290)
        self.fs_info.pack(anchor='w')

    AUTO = 'Auto (fit all beams)'

    def set_zoom(self, z):
        if self.zoom_follow.get() == self.AUTO:     # a zoom by hand takes over from auto
            self.zoom_follow.set('Beam')
        self._auto_setting = True
        self.zoom.set(min(max(z, 1.0), 16.0))
        self._auto_setting = False
        self._redraw_now()

    def on_zoom_slider(self):
        if getattr(self, '_auto_setting', False):
            return
        if self.zoom_follow.get() == self.AUTO:
            self.zoom_follow.set('Beam')
        self._redraw_now()

    def _auto_fit(self, shape, res):
        """Auto zoom: the window that holds every laser's spot (and the crosshair),
        with a margin. Zooms OUT at once when a beam strays toward the edge (so a
        beam adjusted too far stays on screen), zooms IN slowly and only when there
        is clearly room (no pumping). Returns (zoom, centre) or None."""
        h, w = shape
        boxes = [b for b in (res.get('laser_box') or []) if b] if res else []
        if not boxes and res and res.get('ok') and res.get('roi'):
            boxes = [res['roi']]
        if not boxes:
            return 1.0, (w / 2.0, h / 2.0)          # nothing visible: show the whole sensor
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        if self.auto_cross.get() and self.show_cross.get():
            c = self.s.get('crosshair_px') or (w / 2.0, h / 2.0)
            x0, y0, x1, y1 = min(x0, c[0]), min(y0, c[1]), max(x1, c[0]), max(y1, c[1])
        bw, bh = x1 - x0, y1 - y0
        pad = max(0.3 * max(bw, bh), 12.0)
        need_w, need_h = bw + 2 * pad, bh + 2 * pad
        try:
            zmax = min(max(float(self.auto_zmax.get()), 1.0), 16.0)
        except (ValueError, tk.TclError):
            zmax = 8.0
        z_fit = max(1.0, min(w / need_w, h / need_h, zmax))
        z = float(self.zoom.get())
        cur = self._view
        box_c = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        # is any part of the spots (unpadded) outside, or within 5 % of the edge?
        edge_hit = cur is None or (x0 < cur[0] + 0.05 * cur[2] or x1 > cur[0] + 0.95 * cur[2] or
                                   y0 < cur[1] + 0.05 * cur[3] or y1 > cur[1] + 0.95 * cur[3])
        if z_fit < z * 0.98 or (edge_hit and z_fit <= z):
            return z_fit, box_c                      # snap out (and re-centre) right away
        if edge_hit:
            return z, box_c                          # same zoom, jump the centre to bring it back
        if z_fit > z * 1.2:
            # ease in, in log space and at most 8 % per frame: ~1-2 s from 1x to 8x
            return z * min((z_fit / z) ** 0.08, 1.08), None
        return z, None

    def _set_zoom_quiet(self, z):
        self._auto_setting = True
        self.zoom.set(round(z * 4) / 4.0 if abs(z - round(z * 4) / 4.0) < 1e-9 else z)
        self._auto_setting = False

    def on_live_wheel(self, ev):
        self.set_zoom(self.zoom.get() * (1.25 if ev.delta > 0 else 0.8))
        return 'break'                    # do not also scroll the control panel

    def on_pan_start(self, ev):
        self._pan_from = (ev.x, ev.y, self._current_centre())

    def on_pan_drag(self, ev):
        g = self._geom.get('live')
        if not g or not getattr(self, '_pan_from', None) or not self._pan_from[2]:
            return
        x, y, (cx, cy) = self._pan_from
        sc = g[2]
        self._pan = [cx - (ev.x - x) / sc, cy - (ev.y - y) / sc]
        self.zoom_follow.set('Manual')
        self._redraw_now()

    def _current_centre(self):
        v = self._view
        return (v[0] + v[2] / 2.0, v[1] + v[3] / 2.0) if v else None

    def _compute_view(self, shape, res):
        """Visible window (x0, y0, w, h) in analysis px for the zoom and centre mode.
        The beam-follow centre is smoothed so the zoomed picture does not shake."""
        h, w = shape
        mode = self.zoom_follow.get()
        target = None
        snap = False
        if mode == self.AUTO:
            fit = self._auto_fit(shape, res)
            z_new, centre = fit
            self._set_zoom_quiet(z_new)
            if centre is not None:
                self._zc, snap = centre, True
            else:                                     # ease the centre toward the spots
                boxes = [b for b in (res.get('laser_box') or []) if b] if res else []
                if boxes:
                    target = ((min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2.0,
                              (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2.0)
        z = max(1.0, float(self.zoom.get()))
        vw, vh = w / z, h / z
        if snap:
            target = self._zc
        elif mode == 'Beam' and res and res.get('ok'):
            target = (res['cx_px'], res['cy_px'])
        elif mode == 'Crosshair':
            c = self.s.get('crosshair_px')
            target = tuple(c) if c else (w / 2.0, h / 2.0)
        elif mode == 'Manual' and self._pan:
            target = tuple(self._pan)
        if target is None:
            target = self._zc or (w / 2.0, h / 2.0)
        if self._zc is None or snap or mode not in ('Beam', self.AUTO):
            self._zc = target
        else:
            a = 0.25
            self._zc = (self._zc[0] + a * (target[0] - self._zc[0]),
                        self._zc[1] + a * (target[1] - self._zc[1]))
        cx = min(max(self._zc[0], vw / 2.0), w - vw / 2.0)
        cy = min(max(self._zc[1], vh / 2.0), h - vh / 2.0)
        if mode == 'Manual' and self._pan:
            self._pan = [cx, cy]          # do not let the pan run off the image
        return (cx - vw / 2.0, cy - vh / 2.0, vw, vh)

    def _redraw_now(self):
        if self.last:
            self.draw_views(self.last)
        elif self.snapshot:
            self.draw_snapshot()

    def _touch(self):
        self._last_act = time.time()

    def on_activity(self, ev=None):
        if time.time() < self._ignore_until:
            return
        self._last_act = time.time()
        if self._fs:
            self.exit_fullscreen()
            return 'break' if ev is not None and str(ev.type) == 'KeyPress' else None

    def idle_tick(self):
        """Every 250 ms: wake/enter full screen and autosave. Always reschedules,
        even if something in it fails (one exception used to stop it for good)."""
        try:
            self._idle_body()
        except Exception as e:  # noqa: BLE001
            print('idle_tick error (timer keeps running): %r' % (e,))
        finally:
            try:
                self.root.after(250, self.idle_tick)
            except tk.TclError:
                pass                       # window already destroyed

    def _idle_body(self):
        try:
            ptr = self.root.winfo_pointerxy()
        except tk.TclError:
            ptr = None
        now = time.time()
        if ptr and self._ptr and now >= self._ignore_until and \
                (abs(ptr[0] - self._ptr[0]) > 3 or abs(ptr[1] - self._ptr[1]) > 3):
            self._last_act = now
            if self._fs:
                self.exit_fullscreen()
        self._ptr = ptr
        if not self._fs and self.fs_enabled.get() and self.last is not None:
            try:
                timeout = max(5, int(self.fs_timeout.get()))
            except (ValueError, tk.TclError):
                timeout = 60
            # raw Tcl call: grab_current() raises KeyError('popdown') while a
            # Combobox list is open (Tk's internal widget), which killed this timer
            try:
                dialog_open = bool(self.root.tk.call('grab', 'current'))
            except tk.TclError:
                dialog_open = True
            if not dialog_open and now - self._last_act >= timeout:
                self.enter_fullscreen()
        # autosave every 5 s, so settings survive a crash or a killed process
        if now - getattr(self, '_last_save', 0) > 5.0:
            self._last_save = now
            try:
                S.save(self._collect())
            except (OSError, tk.TclError, ValueError):
                pass

    def enter_fullscreen(self):
        if self._fs:
            return
        self._fs = True
        self._ignore_until = time.time() + 0.8      # layout change makes fake motion/keys
        self.left_wrap.pack_forget()
        self.bottom.pack_forget()
        self.status.pack_forget()
        self.root.attributes('-fullscreen', True)
        self.root.focus_set()                       # the waking key must not type into a box
        for c in (self.live, self.snapc):
            try:
                c.configure(cursor='none')
            except tk.TclError:
                pass
        self.root.after(100, self._redraw_now)

    def exit_fullscreen(self):
        if not self._fs:
            return
        self._fs = False
        self._ignore_until = time.time() + 0.5
        self._last_act = time.time()
        self.root.attributes('-fullscreen', False)
        self.left_wrap.pack(side='left', fill='y', before=self.right)
        self.bottom.pack(side='top', fill='x')
        self.status.pack(side='bottom', fill='x')
        self.live.configure(cursor='crosshair')
        self.snapc.configure(cursor='')
        self.live.delete('hud')
        self.root.after(100, self._redraw_now)

    # ---------------------------------------------------------------- colours
    def _build_colours(self, p):
        f = self._box(p, 'Display colours (for goggles)')
        ttk.Label(f, text='Paint each laser in any colour on screen, e.g. a colour your '
                  'goggles let through. Measurements are not affected.',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')
        self.swatches = []
        self.atten_vars, self.atten_labels = [], []
        self.boost_vars, self.boost_labels, self.lcon_vars, self.lcon_labels = [], [], [], []
        self.learn_btns, self.learn_labels = [], []
        self._learning = None        # [laser, rates list, frames wanted] while learning
        for i, name in enumerate(S.LASERS):
            r = ttk.Frame(f)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text='%s laser shows as' % name, width=18).pack(side='left')
            b = tk.Button(r, width=6, relief='flat', bg=hexcol(self.s['display']['colors'][i]),
                          activebackground=hexcol(self.s['display']['colors'][i]),
                          command=lambda i=i: self.pick_colour(i))
            b.pack(side='left')
            self.swatches.append(b)
            # learn how this laser shows in the camera's R/G/B pixels (display only),
            # so overlapping beams separate on screen without a power calibration
            lb = ttk.Button(r, text='Learn', width=6, command=lambda i=i: self.learn_colour(i))
            lb.pack(side='left', padx=(6, 2))
            ll = ttk.Label(r, text='', style='Dim.TLabel', width=9)
            ll.pack(side='left')
            self.learn_btns.append(lb)
            self.learn_labels.append(ll)
            # on-screen attenuation of this laser (display only)
            r = ttk.Frame(f)
            r.pack(fill='x')
            ttk.Label(r, text='   brightness', width=12, style='Dim.TLabel').pack(side='left')
            v = tk.IntVar(value=int(round(100 * self.s['display'].get('atten', [1, 1, 1])[i])))
            sc = tk.Scale(r, from_=0, to=100, orient='horizontal', variable=v, length=150,
                          bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0, showvalue=False,
                          command=lambda val, i=i: self.on_atten(i))
            sc.pack(side='left')
            self._on_right_reset(sc, lambda v=v, i=i: (v.set(100), self.on_atten(i)))
            lab = ttk.Label(r, text='%d %%' % v.get(), width=6)
            lab.pack(side='left')
            self.atten_vars.append(v)
            self.atten_labels.append(lab)
            # this laser's own boost and contrast (display only)
            for kind, lo, hi, res, fmt_ in (('boost', 0.2, 10.0, 0.1, '%.1fx'),
                                            ('contrast', -100, 100, 5, '%+d')):
                r = ttk.Frame(f)
                r.pack(fill='x')
                ttk.Label(r, text='   ' + kind, width=12, style='Dim.TLabel').pack(side='left')
                if kind == 'boost':
                    v = tk.DoubleVar(value=float(self.s['display'].get('boost', [1, 1, 1])[i]))
                else:
                    v = tk.IntVar(value=int(self.s['display'].get('lcontrast', [0, 0, 0])[i]))
                lab = ttk.Label(r, text=fmt_ % v.get(), width=6)
                sc = tk.Scale(r, from_=lo, to=hi, resolution=res, orient='horizontal', variable=v,
                              length=150, bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0,
                              showvalue=False,
                              command=lambda val, v=v, lab=lab, fmt_=fmt_: self.on_laser_tone(v, lab, fmt_))
                sc.pack(side='left')
                dflt = 1.0 if kind == 'boost' else 0
                self._on_right_reset(sc, lambda v=v, lab=lab, fmt_=fmt_, dflt=dflt:
                                     (v.set(dflt), self.on_laser_tone(v, lab, fmt_)))
                lab.pack(side='left')
                (self.boost_vars if kind == 'boost' else self.lcon_vars).append(v)
                (self.boost_labels if kind == 'boost' else self.lcon_labels).append(lab)
        self.learn_reason = self._reason(f)
        self.learn_reason.pack(anchor='w')
        self.learn_info = ttk.Label(f, text='Learn: turn ONE laser on, press its Learn. Then its sliders '
                                    'act on that laser only, even where beams overlap.',
                                    style='Dim.TLabel', wraplength=290)
        self.learn_info.pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(2, 0))
        ttk.Button(r, text='Reset sliders', command=self.reset_laser_tone).pack(side='left')
        self.btn_forget = ttk.Button(r, text='Forget learned colours', command=self.forget_colours)
        self.btn_forget.pack(side='left', padx=4)
        ttk.Label(f, text='Right-click any slider to put it back to its default.',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(4, 0))
        ttk.Label(r, text='Preset').pack(side='left')
        self.preset = tk.StringVar(value='')
        cb = ttk.Combobox(r, textvariable=self.preset, values=list(COLOR_PRESETS), width=16,
                          state='readonly')
        cb.pack(side='left', padx=4)
        cb.bind('<<ComboboxSelected>>', lambda e: self.apply_preset(self.preset.get()))
        self.auto_bright = tk.BooleanVar(value=self.s['display']['auto_bright'])
        ttk.Checkbutton(f, text='Auto brightness (dim beams fill the screen)',
                        variable=self.auto_bright, command=self.push_display).pack(anchor='w')
        # the camera sees 638 nm red ~5x weaker than the lasers look to the eye
        self.balance = tk.BooleanVar(value=bool(self.s['display'].get('balance', True)))
        ttk.Checkbutton(f, text='Balance lasers (each at full brightness)',
                        variable=self.balance, command=self.push_display).pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Boost (all)').pack(side='left')
        self.gain = tk.DoubleVar(value=self.s['display']['gain'])
        sc = tk.Scale(r, from_=0.2, to=10, resolution=0.1, orient='horizontal', variable=self.gain,
                      length=200, bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0,
                      command=lambda v: self.push_display())
        sc.pack(side='left')
        self._on_right_reset(sc, lambda: (self.gain.set(1.0), self.push_display()))
        # contrast: a brightness curve (display only). Left lifts the faint wings
        # of the beam, right darkens them so only the bright core shows.
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Contrast (all)').pack(side='left')
        self.contrast = tk.IntVar(value=int(self.s['display'].get('contrast', 0)))
        sc = tk.Scale(r, from_=-100, to=100, resolution=5, orient='horizontal', variable=self.contrast,
                      length=180, bg=PANEL, fg=FG, troughcolor='#444', highlightthickness=0,
                      command=lambda v: self.on_contrast())
        sc.pack(side='left')
        self._on_right_reset(sc, lambda: (self.contrast.set(0), self.on_contrast()))
        ttk.Button(r, text='0', width=3, command=lambda: (self.contrast.set(0), self.on_contrast())
                   ).pack(side='left', padx=(4, 0))
        ttk.Label(f, text='Contrast: left shows the faint edges of the beam, right only its '
                  'bright core. Display only.', style='Dim.TLabel', wraplength=290).pack(anchor='w')
        ttk.Label(f, text='Colours separate cleanly once each laser is power-calibrated; '
                  'before that, the camera\'s raw R/G/B pixels are used.',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')

    def pick_colour(self, i):
        c = colorchooser.askcolor(color=hexcol(self.s['display']['colors'][i]),
                                  title='Screen colour for the %s laser' % S.LASERS[i])
        if c and c[0]:
            self.s['display']['colors'][i] = [int(x) for x in c[0]]
            self.push_display()

    def apply_preset(self, name):
        self.s['display']['colors'] = [list(c) for c in COLOR_PRESETS[name]]
        self.push_display()

    def on_atten(self, i):
        v = int(self.atten_vars[i].get())
        self.atten_labels[i].configure(text='%d %%' % v)
        if getattr(self, '_atten_job', None):          # throttle while dragging
            self.root.after_cancel(self._atten_job)
        self._atten_job = self.root.after(50, self.push_display)

    # -------------------------------------------------- learned laser colours
    LEARN_FRAMES = 5

    def _colour_sigs(self):
        """The learned [r, g, b] signature of each laser on this camera (or None)."""
        sig = self.s.setdefault('colour_sigs', {}).get(self.serial or '', None) or [None, None, None]
        return list(sig) + [None] * (3 - len(sig))

    def learn_colour(self, i):
        self._learning = [i, [], self.LEARN_FRAMES]
        self.learn_labels[i].configure(text='learning...')
        self._refresh_blocked()

    def _learn_frame(self, res):
        """Called for every frame: collects the light of the laser being learned."""
        if self._learning is None:
            return
        i, got, want = self._learning
        name = S.LASERS[i]
        problem = None
        if not res.get('ok'):
            problem = 'No beam seen. Turn the %s laser on (only that one) and press Learn again.' % name
        elif max(res['peak_raw']) >= res.get('full', 255.0) * 254.0 / 255.0:
            problem = ('The sensor is saturated, so the colour would come out wrong. Lower the '
                       'exposure (or turn Auto exposure on) and press Learn again.')
        if problem:
            self._learning = None
            self.learn_labels[i].configure(text='')
            self._refresh_blocked()
            self.learn_reason.configure(text=problem)
            return
        got.append(np.asarray(res['rates'], float))
        if len(got) < want:
            return
        self._learning = None
        sig = np.mean(got, axis=0)
        sig = np.clip(sig, 0, None)
        if sig.max() <= 0:
            self.learn_reason.configure(text='No light measured. Turn the %s laser on and try again.' % name)
            self._update_learn_labels()
            return
        sig = sig / sig.max()
        main = int(np.argmax(sig))
        sigs = self._colour_sigs()
        sigs[i] = [round(float(v), 4) for v in sig]
        self.s.setdefault('colour_sigs', {})[self.serial or ''] = sigs
        S.save(self._collect())
        self.push_response()
        note = ''
        if main != i:
            note = ('Note: this light is mostly %s in the camera. If that is not the %s laser, '
                    'turn on only the %s laser and press Learn again.'
                    % (['red', 'green', 'blue'][main], name, name))
        self.learn_reason.configure(text=note)
        self._update_learn_labels()
        self._refresh_blocked()

    def _update_learn_labels(self):
        for i, sig in enumerate(self._colour_sigs()):
            if self._learning is not None and self._learning[0] == i:
                continue
            self.learn_labels[i].configure(text='learned' if sig else '')

    def forget_colours(self):
        self.s.setdefault('colour_sigs', {}).pop(self.serial or '', None)
        S.save(self._collect())
        self.push_response()
        self._update_learn_labels()
        self.learn_reason.configure(text='')
        self._refresh_blocked()

    def on_laser_tone(self, v, lab, fmt_):
        lab.configure(text=fmt_ % v.get())
        self.on_contrast()                              # same throttled push

    def reset_laser_tone(self):
        for i in range(len(self.atten_vars)):
            self.atten_vars[i].set(100)
            self.atten_labels[i].configure(text='100 %')
            self.boost_vars[i].set(1.0)
            self.boost_labels[i].configure(text='1.0x')
            self.lcon_vars[i].set(0)
            self.lcon_labels[i].configure(text='+0')
        self.push_display()

    def on_contrast(self):
        if getattr(self, '_atten_job', None):          # same throttle as the brightness sliders
            self.root.after_cancel(self._atten_job)
        self._atten_job = self.root.after(50, self.push_display)

    def push_display(self):
        self._atten_job = None
        d = self.s['display']
        d['atten'] = [max(0, min(100, int(v.get()))) / 100.0 for v in self.atten_vars]
        d['contrast'] = max(-100, min(100, int(self.contrast.get())))
        d['boost'] = [max(0.2, min(10.0, round(float(v.get()), 2))) for v in self.boost_vars]
        d['lcontrast'] = [max(-100, min(100, int(v.get()))) for v in self.lcon_vars]
        d['auto_bright'] = self.auto_bright.get()
        d['balance'] = self.balance.get()
        d['gain'] = float(self.gain.get())
        for i, b in enumerate(self.swatches):
            b.configure(bg=hexcol(d['colors'][i]), activebackground=hexcol(d['colors'][i]))
        self.send_set({'display': dict(d)})

    # ---------------------------------------------------------------- crosshair
    def _build_crosshair(self, p):
        f = self._box(p, 'Crosshair (alignment reference)')
        self.show_cross = tk.BooleanVar(value=self.s['show_crosshair'])
        ttk.Checkbutton(f, text='Show crosshair', variable=self.show_cross).pack(anchor='w')
        ttk.Label(f, text='Left-click the picture to put the crosshair there '
                  '(left-drag draws an ROI).', style='Dim.TLabel', wraplength=290).pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Button(r, text='Centre (0, 0)', command=self.cross_centre).pack(side='left')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(3, 0))
        ttk.Label(r, text='Snap to').pack(side='left')
        self.snap_to = tk.StringVar(value='Whole beam')
        ttk.Combobox(r, textvariable=self.snap_to, values=['Whole beam'] + S.LASERS, width=11,
                     state='readonly').pack(side='left', padx=4)
        self.btn_snapcross = ttk.Button(r, text='Snap', command=self.cross_snap)
        self.btn_snapcross.pack(side='left')
        self.cross_reason = self._reason(f)
        self.cross_reason.pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(3, 0))
        ttk.Label(r, text='"Aligned" within').pack(side='left')
        self.tol = ttk.Entry(r, width=6)
        self.tol.insert(0, str(self.s.get('align_tol_mrad', 0.1)))
        self.tol.pack(side='left', padx=4)
        ttk.Label(r, text='mrad').pack(side='left')
        self.cross_info = ttk.Label(f, text='', style='Dim.TLabel')
        self.cross_info.pack(anchor='w')
        self._update_cross_info()

    def _sensor_shape(self):
        """(h, w) of the analysis image, from the live frames (remembered for later)."""
        if self.last:
            self.s['sensor_shape'] = list(self.last['result']['shape'])
        return self.s.get('sensor_shape')

    def cross_coords(self):
        """Crosshair position with (0, 0) at the sensor centre, +x right, +y UP:
        (x_px, y_px) in analysis px and (x_mrad, y_mrad), or None before any frame."""
        c = self.s.get('crosshair_px')
        shape = self._sensor_shape()
        if not c or not shape:
            return (0.0, 0.0, 0.0, 0.0)          # no crosshair set = the centre = (0, 0)
        h, w = shape
        o = self.s['optics']
        k = 2 * o['pixel_pitch_um'] * 1e-3 / o['focal_mm'] * 1000
        x, y = c[0] - w / 2.0, -(c[1] - h / 2.0)
        return (x, y, x * k, y * k)

    def _update_cross_info(self):
        x, y, xm, ym = self.cross_coords()
        self.cross_info.configure(text='Crosshair at x %+.3f, y %+.3f mrad  (%+.1f, %+.1f px)'
                                  % (xm + 0.0, ym + 0.0, x + 0.0, y + 0.0))

    def _to_analysis(self, ex, ey):
        ox, oy, sc, x0, y0 = self._geom['live']
        return (ex - ox) / sc + x0, (ey - oy) / sc + y0

    def on_live_click(self, ev):
        """Put the crosshair where the picture was clicked."""
        if 'live' not in self._geom:
            return
        self.set_crosshair(list(self._to_analysis(ev.x, ev.y)))
        self.status.configure(text='Crosshair placed.')

    # left button: a click puts the crosshair at the cursor, a drag draws the ROI
    def on_live_press(self, ev):
        if self._fs:
            return                         # this click only wakes the window from full screen
        if 'live' in self._geom:
            self._roi_drag = (ev.x, ev.y)

    def on_live_drag(self, ev):
        if not self._roi_drag:
            return
        self.live.delete('roidrag')
        x, y = self._roi_drag
        self.live.create_rectangle(x, y, ev.x, ev.y, outline='#00ff00', dash=(4, 3), tags='roidrag')

    def on_live_release(self, ev):
        if not self._roi_drag:
            return
        sx, sy = self._roi_drag
        self._roi_drag = None
        self.live.delete('roidrag')
        if abs(ev.x - sx) < 6 and abs(ev.y - sy) < 6:
            # a plain click (not a drag): crosshair to where the button went down
            self.on_live_click(type('Click', (), {'x': sx, 'y': sy})())
            return
        if abs(ev.x - sx) < 6 or abs(ev.y - sy) < 6:
            return                                    # a thin line, not an ROI
        ax0, ay0 = self._to_analysis(min(sx, ev.x), min(sy, ev.y))
        ax1, ay1 = self._to_analysis(max(sx, ev.x), max(sy, ev.y))
        h, w = self.last['result']['shape'] if self.last else (1e9, 1e9)
        roi = [int(max(0, ax0)), int(max(0, ay0)), int(min(w, ax1 + 1)), int(min(h, ay1 + 1))]
        if roi[2] - roi[0] < 8 or roi[3] - roi[1] < 8:
            self.status.configure(text='ROI too small (at least 8 x 8 analysis px).')
            return
        self.set_roi(roi)
        self.status.configure(text='ROI set. Tick Auto ROI to follow the spot again.')

    def cross_centre(self):
        self.set_crosshair(None)

    def cross_snap(self):
        res = self.last['result'] if self.last else None
        if not res or not res.get('ok'):
            return
        which = self.snap_to.get()
        if which == 'Whole beam':
            self.set_crosshair([res['cx_px'], res['cy_px']])
            return
        li = S.LASERS.index(which)
        for j, idx in enumerate(res.get('laser_idx', [])):
            if idx == li and res['laser_cxy'][j]:
                self.set_crosshair(list(res['laser_cxy'][j]))
                return
        messagebox.showinfo('Crosshair', 'The %s laser is not visible right now.' % which)

    def set_crosshair(self, xy):
        self.s['crosshair_px'] = xy
        self.send_set({'zero_px': xy})
        self._update_cross_info()

    # ---------------------------------------------------------------- snapshot
    def _build_snapshot(self, p):
        f = self._box(p, 'Snapshot')
        r = ttk.Frame(f)
        r.pack(fill='x')
        self.btn_snap = ttk.Button(r, text='Save snapshot', command=self.save_snapshot)
        self.btn_snap.pack(side='left')
        ttk.Button(r, text='Open...', command=self.open_snapshot).pack(side='left', padx=4)
        self.btn_hidesnap = ttk.Button(r, text='Hide', command=self.hide_snapshot)
        self.btn_hidesnap.pack(side='left')
        self.snap_reason = self._reason(f)
        self.snap_reason.pack(anchor='w')
        # overlay: lay the snapshot over the live picture, so a beam you snapped
        # (and then turned off) stays on screen to line the next colour up with
        r = ttk.Frame(f)
        r.pack(fill='x', pady=(2, 0))
        self.overlay_on = tk.BooleanVar(value=False)
        self.chk_overlay = ttk.Checkbutton(r, text='Overlay on live', variable=self.overlay_on,
                                           command=self.on_overlay)
        self.chk_overlay.pack(side='left')
        self.overlay_amt = tk.IntVar(value=int(self.s.get('overlay_strength', 100)))
        self.scale_overlay = ttk.Scale(r, from_=10, to=100, orient='horizontal', length=110,
                                       value=self.overlay_amt.get(),
                                       command=lambda v: self.on_overlay())
        self.scale_overlay.pack(side='left', padx=(6, 2))
        self._on_right_reset(self.scale_overlay, lambda: (self.scale_overlay.set(100), self.on_overlay()))
        self.overlay_label = ttk.Label(r, text='%d %%' % self.overlay_amt.get(), width=5)
        self.overlay_label.pack(side='left')
        self.overlay_reason = self._reason(f)
        self.overlay_reason.pack(anchor='w')
        self.snap_info = ttk.Label(f, text='Saved snapshots appear next to the live view.',
                                   style='Dim.TLabel', wraplength=290)
        self.snap_info.pack(anchor='w')

    def _overlay(self, img, res, step, scale=1.0):
        """Draw crosshair + per-laser centre marks into a PIL image (preview px * scale)."""
        d = ImageDraw.Draw(img)
        w, h = img.size
        k = scale / step
        if self.show_cross.get():
            c = self.s.get('crosshair_px')
            cx, cy = (c[0] * k, c[1] * k) if c else (w / 2.0, h / 2.0)
            d.line([(0, cy), (w, cy)], fill=(255, 255, 255), width=1)
            d.line([(cx, 0), (cx, h)], fill=(255, 255, 255), width=1)
            d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(255, 255, 255))
        if res and res.get('ok'):
            for j, idx in enumerate(res.get('laser_idx', [])):
                xy = res['laser_cxy'][j] if j < len(res['laser_cxy']) else None
                if not xy:
                    continue
                col = tuple(self.s['display']['colors'][idx])
                x, y = xy[0] * k, xy[1] * k
                d.line([(x - 7, y - 7), (x + 7, y + 7)], fill=col, width=2)
                d.line([(x - 7, y + 7), (x + 7, y - 7)], fill=col, width=2)
        return img

    def save_snapshot(self):
        if not self.last:
            return
        msg = self.last
        clean = Image.fromarray(msg['preview'])
        os.makedirs(S.SNAP_DIR, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = os.path.join(S.SNAP_DIR, 'beam_%s.png' % stamp)
        # beam_X.png = with crosshair and marks (to share); beam_X_raw.png = clean (shown in the app)
        res = msg['result']
        meta = self._result_summary(res)
        meta['time'] = stamp
        annotated = self._overlay(clean.copy(), res, msg['step'])
        self._draw_text_block(annotated, self._snap_lines(meta))      # numbers in the shared PNG too
        annotated.save(path)
        clean.save(path[:-4] + '_raw.png')
        # the zoomed-in window at full resolution (so the snapshot is as sharp as live)
        det = msg.get('detail')
        detail = None
        if det:
            detail = {'img': Image.fromarray(det['img']).convert('RGB'), 'box': list(det['box']),
                      'step': det['step']}
            detail['img'].save(path[:-4] + '_detail.png')
        meta.update({'time': stamp, 'camera': self.cam_name, 'serial': self.serial,
                     'step': msg['step'], 'shape': list(res['shape']),
                     'crosshair_px': self.s.get('crosshair_px'),
                     'cx_px': res.get('cx_px'), 'cy_px': res.get('cy_px'),
                     'marks': [[list(xy), list(col)] for xy, col in self._live_marks(res)],
                     'boxes': [list(b) for b in (res.get('laser_box') or []) if b],
                     'view': list(self._view) if self._view else None,
                     'detail_box': detail['box'] if detail else None,
                     'detail_step': detail['step'] if detail else None})
        with open(path[:-4] + '.json', 'w') as fh:
            json.dump(meta, fh, indent=1)
        self.show_snapshot(clean.convert('RGB'), meta, path, detail)

    def open_snapshot(self):
        path = filedialog.askopenfilename(initialdir=S.SNAP_DIR, title='Open a beam snapshot',
                                          filetypes=[('PNG', '*.png'), ('All', '*.*')])
        if not path:
            return
        if path.endswith('_raw.png'):
            path = path[:-8] + '.png'
        meta = {}
        try:
            with open(path[:-4] + '.json') as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            pass
        raw = path[:-4] + '_raw.png'
        if not os.path.exists(raw):          # older snapshot: only the annotated picture exists
            raw = path
            meta['marks'] = []
        detail = None
        dpath = path[:-4] + '_detail.png'
        if meta.get('detail_box') and os.path.exists(dpath):
            try:
                detail = {'img': Image.open(dpath).convert('RGB'), 'box': list(meta['detail_box']),
                          'step': int(meta.get('detail_step') or 1)}
            except OSError:
                detail = None
        self.show_snapshot(Image.open(raw).convert('RGB'), meta, path, detail)

    def show_snapshot(self, img, meta, path, detail=None):
        self.snapshot = {'img': img, 'meta': meta, 'path': path, 'detail': detail}
        if not self.snapc.winfo_ismapped():
            self.snapc.pack(side='left', fill='both', expand=True, padx=(2, 4), pady=4)
        txt = os.path.basename(path)
        if meta.get('fwhm_x') is not None:
            txt += '\nFWHM %s x %s mrad, power %s W' % (fmt(meta.get('fwhm_x')), fmt(meta.get('fwhm_y')),
                                                         fmt(meta.get('power_total')))
        self.snap_info.configure(text=txt + '\nThe snapshot stays still; mouse wheel on it zooms it.')
        self._snap_view = self._snapshot_fixed_view(img, meta)
        self.root.after(50, self.draw_snapshot)
        self._refresh_blocked()

    def _snap_shape(self, img, meta):
        step = meta.get('step', 1)
        return tuple(meta.get('shape') or (img.height * step, img.width * step))

    def _snapshot_fixed_view(self, img, meta):
        """The snapshot's own window, set ONCE: the view the live picture had when it
        was saved (so it looks exactly as you saw it), else a fit around its spots.
        It never follows the live beam - it is the static reference."""
        h, w = self._snap_shape(img, meta)
        v = meta.get('view')
        if v and v[2] <= w + 1e-6 and v[3] <= h + 1e-6:
            return tuple(v)
        pts = [b for b in meta.get('boxes', [])]
        pts += [[xy[0], xy[1], xy[0], xy[1]] for xy, _ in meta.get('marks', [])]
        if meta.get('cx_px') is not None:
            pts.append([meta['cx_px'], meta['cy_px'], meta['cx_px'], meta['cy_px']])
        if not pts:
            return (0.0, 0.0, float(w), float(h))
        x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
        x1, y1 = max(p[2] for p in pts), max(p[3] for p in pts)
        pad = max(0.3 * max(x1 - x0, y1 - y0), 40.0)
        z = max(1.0, min(w / (x1 - x0 + 2 * pad), h / (y1 - y0 + 2 * pad), 8.0))
        return self._window_at(((x0 + x1) / 2.0, (y0 + y1) / 2.0), z, (h, w))

    @staticmethod
    def _window_at(centre, z, shape):
        h, w = shape
        vw, vh = w / z, h / z
        cx = min(max(centre[0], vw / 2.0), w - vw / 2.0)
        cy = min(max(centre[1], vh / 2.0), h - vh / 2.0)
        return (cx - vw / 2.0, cy - vh / 2.0, vw, vh)

    def on_snap_wheel(self, ev):
        """Zoom the snapshot by itself, about its own centre (it still does not move)."""
        if not self.snapshot or not self._snap_view:
            return 'break'
        h, w = self._snap_shape(self.snapshot['img'], self.snapshot['meta'])
        x0, y0, vw, vh = self._snap_view
        z = min(max((w / vw) * (1.25 if ev.delta > 0 else 0.8), 1.0), 16.0)
        self._snap_view = self._window_at((x0 + vw / 2.0, y0 + vh / 2.0), z, (h, w))
        self.draw_snapshot()
        return 'break'

    def on_overlay(self):
        v = int(round(float(self.scale_overlay.get())))
        self.overlay_amt.set(v)
        self.overlay_label.configure(text='%d %%' % v)
        self._refresh_blocked()
        if self.last:
            self.draw_views(self.last)

    def _overlay_problem(self, res):
        """Why the snapshot cannot be laid over this live picture, or None."""
        if not self.snapshot:
            return 'Save or open a snapshot to overlay it.'
        if res is None:
            return None
        sh = self.snapshot['meta'].get('shape')
        if sh and (abs(sh[0] - res['shape'][0]) > 2 or abs(sh[1] - res['shape'][1]) > 2):
            return 'This snapshot is from a different sensor size (%dx%d, live %dx%d); not overlaid.' % (
                sh[1], sh[0], res['shape'][1], res['shape'][0])
        return None

    def _overlay_active(self, res):
        return bool(self.overlay_on.get()) and self._overlay_problem(res) is None

    def hide_snapshot(self):
        self.snapshot = None
        self._snap_view = None
        self.overlay_on.set(False)
        self.snapc.pack_forget()
        self.snap_info.configure(text='Saved snapshots appear next to the live view.')
        self._refresh_blocked()

    def draw_snapshot(self):
        """Draw the snapshot in its own FIXED window (see _snapshot_fixed_view)."""
        if not self.snapshot:
            return
        img, meta = self.snapshot['img'], self.snapshot['meta']
        if self._snap_view is None:
            self._snap_view = self._snapshot_fixed_view(img, meta)
        h, w = self._snap_shape(img, meta)
        pic, step, origin = pick_detail(self._snap_view, img, meta.get('step', 1), self.snapshot.get('detail'))
        self._render(self.snapc, 'snap', pic, step, self._snap_view,
                     [(tuple(xy), col) for xy, col in meta.get('marks', [])],
                     zlabel=w / self._snap_view[2], shape=(h, w), text=self._snap_lines(meta),
                     origin=origin)

    # ---------------------------------------------------------------- power
    def _build_power(self, p):
        f = self._box(p, 'Power calibration')
        ttk.Label(f, text='Shoot ONE laser at a known power into the sensor, type that '
                  'power, press Calibrate. Repeat for each colour. Saved per camera.',
                  style='Dim.TLabel', wraplength=290).pack(anchor='w')
        self.pw_entries, self.pw_buttons, self.pw_state, self.wl_entries = [], [], [], []
        for i, name in enumerate(S.LASERS):
            r = ttk.Frame(f)
            r.pack(fill='x', pady=1)
            ttk.Label(r, text=name, width=6).pack(side='left')
            wl = ttk.Entry(r, width=5)
            wl.insert(0, str(self.s['wavelengths_nm'][i]))
            wl.pack(side='left')
            ttk.Label(r, text='nm').pack(side='left')
            e = ttk.Entry(r, width=7)
            e.pack(side='left', padx=(6, 0))
            e.bind('<KeyRelease>', lambda ev: self._refresh_blocked())
            ttk.Label(r, text='W').pack(side='left')
            b = ttk.Button(r, text='Calibrate', command=lambda i=i: self.calibrate(i))
            b.pack(side='left', padx=4)
            st = ttk.Label(f, text='', style='Dim.TLabel')
            st.pack(anchor='w')
            self.pw_entries.append(e)
            self.pw_buttons.append(b)
            self.pw_state.append(st)
            self.wl_entries.append(wl)
        r = ttk.Frame(f)
        r.pack(fill='x')
        self.btn_clearcal = ttk.Button(r, text='Clear calibration', command=self.clear_cal)
        self.btn_clearcal.pack(side='left')
        self.btn_zero = ttk.Button(r, text='Zero (laser off)', command=self.zero)
        self.btn_zero.pack(side='left', padx=4)
        self.zero_info = ttk.Label(f, text='Zeroes automatically when the camera connects '
                                   '(keep the laser off until then).', style='Dim.TLabel',
                                   wraplength=290)
        self.zero_info.pack(anchor='w')
        self.pw_reason = self._reason(f)
        self.pw_reason.pack(anchor='w')

    def _cal(self):
        return self.s['calibrations'].get(self.serial or '', {})

    def calibrate(self, i):
        try:
            w = float(self.pw_entries[i].get())
        except ValueError:
            return
        if not messagebox.askokcancel('Calibrate %s' % S.LASERS[i],
                                      'Shoot ONLY the %s laser (%s nm) at %g W into the sensor.\n\n'
                                      '- The ROI must contain the entire spot\n'
                                      '- No saturation (peak below 100 %%)\n'
                                      '- Keep the device and laser still\n\nStart now?'
                                      % (S.LASERS[i], self.wl_entries[i].get(), w)):
            return
        self.calibrating = i
        self.cmd_q.put(('calibrate', i, w, 15))
        self.status.configure(text='Calibrating %s laser...' % S.LASERS[i])
        self._refresh_blocked()

    def on_calibrated(self, msg):
        i = msg['laser']
        cal = self.s['calibrations'].setdefault(self.serial, {'response': [None, None, None]})
        cal['response'][i] = msg['rates_per_w']
        cal.setdefault('dates', [None, None, None])[i] = datetime.date.today().isoformat()
        cal.setdefault('wavelengths', [None, None, None])[i] = self.wl_entries[i].get()
        self.calibrating = None
        S.save(self._collect())
        self.push_response()
        self.status.configure(text='%s laser calibrated.' % S.LASERS[i])
        self._refresh_blocked()

    def zero(self):
        if not messagebox.askokcancel('Zero', 'Turn the laser OFF (nothing shining into the sensor), '
                                      'then press OK.'):
            return
        self.cmd_q.put(('zero', 10))
        self.zero_info.configure(text='Recording the dark frame...')

    def clear_cal(self):
        if not messagebox.askyesno('Clear calibration', 'Forget the power calibration of camera %s?'
                                   % self.serial):
            return
        self.s['calibrations'].pop(self.serial, None)
        S.save(self._collect())
        self.push_response()
        self._refresh_blocked()

    def push_response(self):
        resp, idx = S.response_matrix(self.s, self.serial)
        self.send_set({'response': resp, 'laser_idx': idx, 'colour_sig': self._colour_sigs()})
        self._update_learn_labels()

    # ---------------------------------------------------------------- optics
    def _build_optics(self, p):
        f = self._box(p, 'Optics (pixels -> mrad)')
        ttk.Label(f, text='angle = size on sensor / focal length. Use your LBA\'s values '
                  '(camera pixel size, lens focal length).', style='Dim.TLabel',
                  wraplength=290).pack(anchor='w')
        r = ttk.Frame(f)
        r.pack(fill='x')
        ttk.Label(r, text='Pixel').pack(side='left')
        self.pitch = ttk.Entry(r, width=6)
        self.pitch.insert(0, str(self.s['optics']['pixel_pitch_um']))
        self.pitch.pack(side='left', padx=2)
        ttk.Label(r, text='um   Focal').pack(side='left')
        self.focal = ttk.Entry(r, width=6)
        self.focal.insert(0, str(self.s['optics']['focal_mm']))
        self.focal.pack(side='left', padx=2)
        ttk.Label(r, text='mm').pack(side='left')
        ttk.Button(r, text='Apply', command=self.apply_optics).pack(side='left', padx=4)
        self.optics_info = ttk.Label(f, text='', style='Dim.TLabel')
        self.optics_info.pack(anchor='w')
        self._optics_text()

    def _optics_text(self):
        o = self.s['optics']
        k = 2 * o['pixel_pitch_um'] * 1e-3 / o['focal_mm'] * 1000
        self.optics_info.configure(text='%.4f mrad per analysis pixel (2x2 Bayer cell)' % k)

    def apply_optics(self):
        try:
            o = {'pixel_pitch_um': float(self.pitch.get()), 'focal_mm': float(self.focal.get())}
            assert o['pixel_pitch_um'] > 0 and o['focal_mm'] > 0
        except (ValueError, AssertionError):
            messagebox.showerror('Optics', 'Pixel size and focal length must be positive numbers.')
            return
        self.s['optics'] = o
        self._optics_text()
        self._update_cross_info()                # mrad scale changed
        self.send_set({'optics': o})

    # ---------------------------------------------------------------- CSV
    def _build_log(self, p):
        f = self._box(p, 'Log measurements')
        r = ttk.Frame(f)
        r.pack(fill='x')
        self.btn_log = ttk.Button(r, text='Start CSV log...', command=self.toggle_log)
        self.btn_log.pack(side='left')
        self.log_reason = self._reason(f)
        self.log_reason.pack(anchor='w')
        self.log_info = ttk.Label(f, text='', style='Dim.TLabel', wraplength=290)
        self.log_info.pack(anchor='w')

    CSV_COLS = ['time', 'exposure_us', 'peak_pct', 'fwhm_x', 'fwhm_y', 'fwhm_geo', 'e2_x', 'e2_y',
                'e2_geo', 'd4s_x', 'd4s_y', 'align_x', 'align_y',
                'red_dx', 'red_dy', 'green_dx', 'green_dy', 'blue_dx', 'blue_dy',
                'p_red', 'p_green', 'p_blue', 'power_total', 'dens_x', 'dens_y', 'dens_point',
                'cx_px', 'cy_px', 'avg_frames', 'roi_x0', 'roi_y0', 'roi_x1', 'roi_y1'] + \
        ['%s_%s' % (c, m) for c in ('red', 'green', 'blue')
         for m in ('fwhm_x', 'fwhm_y', 'e2_x', 'e2_y', 'd4s_x', 'd4s_y')]

    def toggle_log(self):
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = self.csv_writer = None
            self.btn_log.configure(text='Start CSV log...')
            self.log_info.configure(text='Log closed.')
            return
        path = getattr(self, '_auto_log_path', None)
        self._auto_log_path = None
        if not path:
            path = filedialog.asksaveasfilename(initialdir=self.s['csv_dir'], defaultextension='.csv',
                                                initialfile='beam_%s.csv' % datetime.datetime.now()
                                                .strftime('%Y-%m-%d_%H-%M'),
                                                filetypes=[('CSV', '*.csv')])
        if not path:
            return
        self.s['csv_dir'] = os.path.dirname(path)
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(self.CSV_COLS)
        self.csv_rows = 0
        self.btn_log.configure(text='Stop log')
        self.log_info.configure(text=path)

    def _result_summary(self, res):
        out = {}
        for k in ('exposure_us', 'fwhm_x', 'fwhm_y', 'fwhm_geo', 'e2_x', 'e2_y', 'e2_geo',
                  'd4s_x', 'd4s_y', 'align_x', 'align_y', 'power_total', 'dens_x', 'dens_y',
                  'dens_point'):
            v = res.get(k)
            out[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) else v
        out['peak_pct'] = 100.0 * max(res.get('peak_raw', (0,))) / res.get('full', 255.0)
        out['cx_px'], out['cy_px'] = res.get('cx_px'), res.get('cy_px')
        out['avg_frames'] = (res.get('avg') or (None,))[0]
        for i, d in enumerate(self._laser_divs(res)):
            for m in ('fwhm_x', 'fwhm_y', 'e2_x', 'e2_y', 'd4s_x', 'd4s_y'):
                v = d.get(m) if d else None
                out['%s_%s' % (S.LASERS[i].lower(), m)] = None if v is None or v != v else v
        roi = res.get('roi') or (None,) * 4
        out['roi_x0'], out['roi_y0'], out['roi_x1'], out['roi_y1'] = [None if v is None else int(v) for v in roi]
        offs = self._laser_offsets(res)
        pw = self._laser_powers(res)
        for i, name in enumerate(S.LASERS):
            n = name.lower()
            out[n + '_dx'], out[n + '_dy'] = offs[i] if offs[i] else (None, None)
            out['p_' + n] = pw[i]
        return out

    def _write_csv(self, res):
        if not self.csv_writer:
            return
        row = self._result_summary(res)
        row['time'] = datetime.datetime.now().isoformat(timespec='milliseconds')
        self.csv_writer.writerow(['' if row.get(c) is None else
                                  (('%.6g' % row[c]) if isinstance(row[c], float) else row[c])
                                  for c in self.CSV_COLS])
        self.csv_rows += 1
        if self.csv_rows % 30 == 0:
            self.csv_file.flush()

    # ================================================================ blocked-state reasons
    def _refresh_blocked(self):
        connected = self.proc is not None and self.serial is not None
        has_frame = connected and self.last is not None and self.last['result'].get('ok')

        def gate(btn, reason, label=None):
            btn.state(['disabled'] if reason else ['!disabled'])
            if label is not None:
                label.configure(text=reason or '')
        gate(self.btn_connect, 'Already connected' if self.proc else None)
        gate(self.btn_disconnect, None if self.proc else 'Not connected')
        gate(self.btn_snapcross, None if has_frame else 'Connect a camera and point a beam at it to snap.',
             self.cross_reason)
        gate(self.btn_snap, None if self.last else 'Connect a camera first.', self.snap_reason)
        gate(self.btn_hidesnap, None if self.snapshot else 'No snapshot shown')
        if self._learning is not None:
            lr = 'Learning the %s laser...' % S.LASERS[self._learning[0]]
        elif not has_frame:
            lr = 'Connect the camera and turn ONE laser on to learn its colour.'
        else:
            lr = None
        for b in self.learn_btns:
            gate(b, lr)
        if lr or not self.learn_reason.cget('text').startswith(('No ', 'The sensor', 'Note:')):
            self.learn_reason.configure(text=lr or '')
        gate(self.btn_forget, None if any(self._colour_sigs()) else 'Nothing learned yet')
        prob = self._overlay_problem(self.last['result'] if self.last else None)
        gate(self.chk_overlay, prob, self.overlay_reason)
        gate(self.btn_zero, None if connected and self.calibrating is None else 'Connect first')
        gate(self.btn_log, None if (connected or self.csv_file) else 'Connect a camera first.', self.log_reason)
        reason = None
        if not connected:
            reason = 'Connect the LBA (or the simulator) to calibrate.'
        elif self.calibrating is not None:
            reason = 'Calibrating the %s laser...' % S.LASERS[self.calibrating]
        # (auto exposure needs no turning off: the camera process holds the exposure
        # still while it calibrates or zeroes, then auto exposure carries on)
        for i, b in enumerate(self.pw_buttons):
            r = reason
            if r is None:
                try:
                    if float(self.pw_entries[i].get()) <= 0:
                        raise ValueError
                except ValueError:
                    r = 'x'
            b.state(['disabled'] if r else ['!disabled'])
        if reason is None and not any(str(b['state']) != 'disabled' for b in self.pw_buttons):
            reason = 'Type the laser power in W next to a colour to enable its Calibrate button.'
        self.pw_reason.configure(text=reason or '')
        cal = self._cal().get('response') or [None] * 3
        dates = self._cal().get('dates') or [None] * 3
        for i, st in enumerate(self.pw_state):
            st.configure(text=('   calibrated %s' % dates[i]) if cal[i] else '   not calibrated')
        gate(self.btn_clearcal, None if any(cal) else 'Nothing calibrated')

    # ================================================================ worker
    def _collect(self):
        s = self.s
        s['backend'] = self.backend.get()
        s['auto_exposure'] = self.auto_exp.get()
        s['stack'] = int(self.stack.get())
        s['auto_roi'] = self.auto_roi.get()
        s['show_crosshair'] = self.show_cross.get()
        s['spot'] = self.spot_key()
        s['overlay_strength'] = int(self.overlay_amt.get())
        s['show_profiles'] = self.show_profiles.get()
        s['zoom'] = float(self.zoom.get())
        s['zoom_follow'] = self.zoom_follow.get()
        s['auto_zoom_crosshair'] = self.auto_cross.get()
        try:
            s['auto_zoom_max'] = float(self.auto_zmax.get())
        except (ValueError, tk.TclError):
            pass
        s['fs_enabled'] = self.fs_enabled.get()
        try:
            s['fs_timeout_s'] = max(5, int(self.fs_timeout.get()))
        except (ValueError, tk.TclError):
            pass
        try:
            s['align_tol_mrad'] = float(self.tol.get())
        except ValueError:
            pass
        s['wavelengths_nm'] = [w.get() for w in self.wl_entries]
        return s

    def connect(self):
        if self.proc:
            return
        s = self._collect()
        cfg = {'backend': s['backend'], 'exposure_us': s['exposure_us'],
               'auto_exposure': s['auto_exposure'], 'stack': s['stack'], 'auto_roi': s['auto_roi'],
               'roi': s.get('roi'),
               'optics': s['optics'], 'display': s['display'], 'zero_px': s.get('crosshair_px'),
               'response': None, 'laser_idx': [], 'ae_target': s.get('ae_target', 0.85),
               'spot': s.get('spot', 'right')}
        self.cmd_q = mp.Queue()
        self.out_q = mp.Queue(maxsize=3)
        self.proc = mp.Process(target=worker.run, args=(self.cmd_q, self.out_q, cfg), daemon=True)
        self.proc.start()
        self._frame_no, self._last_frame_t, self._stall_reported = 0, None, False
        self._detail_req = None          # a new camera process has no zoom window yet
        log_error('Connect: backend %s, BeamLab %s' % (s['backend'], __version__))
        self.status.configure(text='Opening camera...')
        self._refresh_blocked()

    def disconnect(self):
        if self.csv_file:
            self.toggle_log()
        if self.proc:
            try:
                self.cmd_q.put(('quit',))
                self.proc.join(2.0)
            except Exception:  # noqa: BLE001
                pass
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
        self.serial = None
        self.last = None
        self.calibrating = None
        self.cam_info.configure(text='')
        self.status.configure(text='Not connected.')
        self._refresh_blocked()

    def send_set(self, d):
        if 'exposure_us' in d:
            self.s['exposure_us'] = d['exposure_us']
        if self.proc and self.cmd_q:
            self.cmd_q.put(('set', d))

    def poll(self):
        """Every 15 ms: take new messages from the camera process and draw. This loop
        must never stop (one uncaught error used to freeze the picture for good while
        the rest of the window kept working), so errors are logged + shown instead."""
        try:
            self._poll_body()
            self._watchdog()
        except Exception:  # noqa: BLE001
            self._on_tk_error(*sys.exc_info())
        finally:
            try:
                self.root.after(15, self.poll)
            except tk.TclError:
                pass                                    # window closed

    def _watchdog(self):
        """Tell the user when the camera process died or stopped sending pictures."""
        if not self.proc:
            return
        now = time.time()
        if not self.proc.is_alive():
            code = self.proc.exitcode
            self.proc = None
            self.serial = None
            msg = ('The camera process stopped unexpectedly (exit code %s).' % code)
            log_error(msg + ' Last frame #%d.' % self._frame_no)
            self.status.configure(text=msg)
            self._refresh_blocked()
            if messagebox.askyesno('Camera', msg + '\n\nDetails (if any) are in\n%s\n\nReconnect now?'
                                   % S.ERROR_LOG):
                self.disconnect()
                self.connect()
            return
        if self._last_frame_t is None:
            return
        exp_s = max(self.s.get('exposure_us', 5000.0), 1.0) / 1e6
        limit = max(5.0, 8 * exp_s * max(1, int(self.stack.get())))
        stalled = now - self._last_frame_t > limit
        if stalled and not self._stall_reported:
            self._stall_reported = True
            msg = 'No new picture from the camera for %.0f s (frame #%d).' % (now - self._last_frame_t, self._frame_no)
            log_error(msg + ' Camera process alive: yes.')
            self.status.configure(text=msg + ' Disconnect + Connect restarts it.')
        elif not stalled and self._stall_reported:
            self._stall_reported = False
            log_error('Pictures resumed at frame #%d.' % self._frame_no)

    def _on_tk_error(self, exc, val, tb):
        """Any error in the window: write the full traceback to errors.log and say so
        (once per distinct error, so a per-frame error does not flood pop-ups)."""
        import traceback
        text = ''.join(traceback.format_exception(exc, val, tb))
        log_error('Window error:\n' + text)
        key = (exc.__name__, str(val)[:120])
        if key in self._errors_shown:
            return
        self._errors_shown.add(key)
        try:
            self.status.configure(text='Error: %s: %s (details in %s)' % (exc.__name__, val, S.ERROR_LOG))
            messagebox.showerror('BeamLab error', '%s: %s\n\nThe full details were saved to\n%s\n\n'
                                 'BeamLab keeps running. Please send that file if this repeats.'
                                 % (exc.__name__, val, S.ERROR_LOG))
        except tk.TclError:
            pass

    def _poll_body(self):
        latest = None
        if self.out_q is not None:
            for _ in range(10):
                try:
                    m = self.out_q.get_nowait()
                except queue.Empty:
                    break
                t = m['type']
                if t == 'frame':
                    latest = m
                elif t == 'opened':
                    self.on_opened(m)
                elif t == 'calibrated':
                    self.on_calibrated(m)
                elif t == 'zeroed':
                    self.zero_info.configure(text='Zeroed at %s (dark level %.1f counts).'
                                             % (time.strftime('%H:%M:%S'), m['level']))
                elif t == 'error':
                    self.on_error(m['text'])
        if latest is not None:
            first = self.last is None
            self.last = latest
            self._frame_no += 1
            self._last_frame_t = time.time()
            self.draw(latest)
            # re-check the buttons whenever "a beam is seen" changes: buttons that need
            # a beam (Learn, crosshair Snap) used to stay grey for good when the first
            # frame had none (laser turned on after connecting, auto exposure settling)
            beam = bool(latest['result'].get('ok'))
            if (first or beam != getattr(self, '_beam_seen', None)
                    or latest.get('calibrating') != (self.calibrating is not None)):
                self._beam_seen = beam
                self._refresh_blocked()
            if first:
                self._update_cross_info()        # now the sensor size is known
            if not self._stall_reported:
                self.status.configure(text='Streaming: frame #%d, %.1f fps' % (self._frame_no, latest.get('fps', 0)))

    def on_opened(self, m):
        self.serial = m['serial']
        self.cam_name = '%s %s' % (m['name'], m['model'])
        lo, hi = m['exposure_range']
        self.exposure_range = (float(lo), float(min(hi, MAX_EXPOSURE_US)))
        self._exp_range_text()
        fmt_ = m.get('pixel_format') or ''
        bits = '12-bit' if fmt_.endswith('12') else '8-bit'
        self.cam_info.configure(text='%s\nmodel %s, serial %s\n%s raw pixels (%s)'
                                % (m['name'], m['model'], m['serial'], bits, fmt_))
        self.status.configure(text='Streaming.')
        self._set_exp_widgets(self.s['exposure_us'])
        self.push_response()
        self._refresh_blocked()

    def on_error(self, text):
        was_cal = self.calibrating is not None
        self.calibrating = None
        if self.proc and not self.proc.is_alive():
            self.proc = None
            self.serial = None
        self._refresh_blocked()
        if 'No camera opened' in text or 'not installed' in text or 'found' in text:
            if messagebox.askyesno('Camera', text + '\n\nFix: plug in the LBA, install its USB driver '
                                   '(Start menu > BeamLab > Install camera drivers) and the Python package '
                                   'named above.\n\nUse the built-in simulator instead for now?'):
                self.disconnect()
                self.backend.set('sim')
                self.connect()
            return
        messagebox.showerror('Calibration' if was_cal else 'Camera', text)

    # ================================================================ drawing
    def _snap_lines(self, meta):
        """Text block for a snapshot: its divergence (and when it was taken)."""
        lines = ['Snapshot %s' % meta.get('time', '').replace('_', ' ')]
        if meta.get('fwhm_x') is None and meta.get('e2_x') is None:
            lines.append('(no measurements saved with this snapshot)')
            return lines
        lines.append('Divergence (mrad)  FWHM X x Y     1/e2 X x Y     D4s X x Y')
        row = '  %-6s  %5s x %-5s  %5s x %-5s  %5s x %-5s'
        lines.append(row % ('Beam', fmt(meta.get('fwhm_x')), fmt(meta.get('fwhm_y')), fmt(meta.get('e2_x')),
                            fmt(meta.get('e2_y')), fmt(meta.get('d4s_x')), fmt(meta.get('d4s_y'))))
        for name in S.LASERS:
            n = name.lower()
            if meta.get(n + '_fwhm_x') is not None or meta.get(n + '_e2_x') is not None:
                lines.append(row % (name, fmt(meta.get(n + '_fwhm_x')), fmt(meta.get(n + '_fwhm_y')),
                                    fmt(meta.get(n + '_e2_x')), fmt(meta.get(n + '_e2_y')),
                                    fmt(meta.get(n + '_d4s_x')), fmt(meta.get(n + '_d4s_y'))))
        if meta.get('power_total') is not None:
            lines.append('Power %s W' % fmt(meta.get('power_total')))
        return lines

    @staticmethod
    def _draw_text_block(img, lines, xy=(6, 6)):
        """Readable text box (dark backing) drawn into a PIL image."""
        if not lines:
            return
        d = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype('consola.ttf', 13)
        except (OSError, ImportError):
            font = None
        lh = 16
        w = max(len(s) for s in lines) * 8 + 10
        x, y = xy
        d.rectangle([x, y, x + w, y + lh * len(lines) + 6], fill=(0, 0, 0))
        for i, s in enumerate(lines):
            d.text((x + 5, y + 3 + i * lh), s, fill=(235, 235, 235), font=font)

    def _render(self, canvas, key, img, step, view, marks, zlabel=None, shape=None, text=None,
                origin=(0, 0)):
        """Show the `view` window (analysis px) of `img` (preview px = analysis px / step;
        `img` starts at `origin` on the sensor, analysis px) fitted to `canvas`, with the
        crosshair and per-laser marks drawn on top.
        marks = [((x, y) analysis px, (r, g, b)[, 'ring']), ...]"""
        x0, y0, vw, vh = view
        cw, ch = max(canvas.winfo_width(), 50), max(canvas.winfo_height(), 50)
        sc = min(cw / vw, ch / vh)                          # display px per analysis px
        W, H = max(1, int(vw * sc)), max(1, int(vh * sc))
        ox0, oy0 = origin
        box = ((x0 - ox0) / step, (y0 - oy0) / step, (x0 + vw - ox0) / step, (y0 + vh - oy0) / step)
        # smooth at every zoom (a switch to sharp blocks used to kick in at ~10x);
        # bicubic: zoomed far in, bilinear still shows the facets between pixels
        im = img.resize((W, H), Image.BICUBIC, box=box)
        d = ImageDraw.Draw(im)

        def T(x, y):
            return (x - x0) * sc, (y - y0) * sc
        if self.show_cross.get():
            c = self.s.get('crosshair_px')
            if shape is None:
                shape = self.last['result']['shape'] if self.last else (img.height * step, img.width * step)
            cx, cy = T(*(c if c else (shape[1] / 2.0, shape[0] / 2.0)))
            d.line([(0, cy), (W, cy)], fill=(255, 255, 255), width=1)
            d.line([(cx, 0), (cx, H)], fill=(255, 255, 255), width=1)
            rr = max(12, 0.5 * sc)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 255, 255))
        for m in marks:
            xy, col = m[0], m[1]
            x, y = T(*xy)
            if len(m) > 2 and m[2] == 'ring':       # a snapshot's spot centre (overlay)
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=tuple(col), width=2)
                continue
            d.line([(x - 8, y - 8), (x + 8, y + 8)], fill=tuple(col), width=2)
            d.line([(x - 8, y + 8), (x + 8, y - 8)], fill=tuple(col), width=2)
        z = self.zoom.get() if zlabel is None else zlabel
        if z > 1.01:
            d.text((6, H - 16), '%.2gx' % z, fill=(200, 200, 200))
        if text:
            self._draw_text_block(im, text)
        ox, oy = (cw - W) // 2, (ch - H) // 2
        tkimg = ImageTk.PhotoImage(im)
        item = '_live_item' if key == 'live' else '_snap_item'
        if getattr(self, item) is None:
            setattr(self, item, canvas.create_image(ox, oy, anchor='nw', image=tkimg))
        else:
            canvas.coords(getattr(self, item), ox, oy)
            canvas.itemconfigure(getattr(self, item), image=tkimg)
        canvas._img = tkimg            # keep a reference or Tk drops the image
        self._geom[key] = (ox, oy, sc, x0, y0)

    def _request_detail(self, shape):
        """Ask the camera process for the zoomed-in window at full resolution: the
        view plus a margin (so a beam-following zoom does not need a new one every
        frame). Re-sent only when the view leaves it or the zoom changes a lot."""
        H, W = shape
        x0, y0, vw, vh = self._view
        want = None
        if vw < 0.6 * W:                                    # zoomed in enough to matter
            # a wide margin (up to one view each side) so zooming in further, or the
            # beam-follow drifting, still finds full detail until the next frame;
            # capped so the window stays <= 960 px and is sent at full resolution
            k = max(0.1, min(1.0, (worker.PREVIEW_MAX_W - 2.0 - vw) / (2.0 * vw)))
            mx, my = k * vw, k * vh
            want = [max(0.0, x0 - mx), max(0.0, y0 - my), min(float(W), x0 + vw + mx), min(float(H), y0 + vh + my)]
        cur = getattr(self, '_detail_req', None)
        if want is None:
            if cur is not None:
                self._detail_req = None
                self.send_set({'detail_view': None})
            return
        if cur is not None:
            inside = cur[0] <= x0 and cur[1] <= y0 and x0 + vw <= cur[2] and y0 + vh <= cur[3]
            size_ok = 0.7 < (cur[2] - cur[0]) / max(want[2] - want[0], 1e-6) < 1.4
            if inside and size_ok:
                return
        self._detail_req = want
        self.send_set({'detail_view': want})

    def _live_marks(self, res):
        out = []
        if res and res.get('ok'):
            for j, idx in enumerate(res.get('laser_idx', [])):
                xy = res['laser_cxy'][j] if j < len(res['laser_cxy']) else None
                if xy:
                    out.append((tuple(xy), self.s['display']['colors'][idx]))
        return out

    def draw_views(self, msg):
        res = msg['result']
        self._view = self._compute_view(res['shape'], res)
        self._request_detail(res['shape'])
        live, step, origin = pick_detail(self._view, msg['preview'], msg['step'], msg.get('detail'))
        marks = self._live_marks(res)
        if self._overlay_active(res):
            h, w = live.shape[:2]
            area = (origin[0], origin[1], origin[0] + w * step, origin[1] + h * step)
            sd = self.snapshot.get('detail')
            # the snapshot's own full-resolution window when it covers this area
            snap, ss, so = pick_detail((area[0], area[1], area[2] - area[0], area[3] - area[1]),
                                       self.snapshot['img'], self.snapshot['meta'].get('step', 1), sd)
            region = ((area[0] - so[0]) / ss, (area[1] - so[1]) / ss, (area[2] - so[0]) / ss, (area[3] - so[1]) / ss)
            live = overlay_snapshot(live, snap, self.overlay_amt.get() / 100.0, region)
            marks = [(tuple(xy), col, 'ring') for xy, col in self.snapshot['meta'].get('marks', [])] + marks
        self._render(self.live, 'live', Image.fromarray(live), step, self._view, marks, origin=origin)
        if self.snapshot:
            self.draw_snapshot()
        self.live.delete('hud')
        if self._fs:
            self.draw_hud(res)

    def draw_hud(self, res):
        """Compact readout on the live picture while in full screen."""
        lines = []
        if res.get('ok'):
            lines.append(('Divergence (mrad)   FWHM X x Y      1/e2 X x Y', DIM))
            lines.append(('%-6s %s x %s   %s x %s' % ('Beam', fmt(res['fwhm_x']), fmt(res['fwhm_y']),
                                                      fmt(res['e2_x']), fmt(res['e2_y'])), FG))
            hud_cols = ('#ff8a80', '#b9f6ca', '#82b1ff')
            for i, d in enumerate(self._laser_divs(res)):
                if d:
                    lines.append(('%-6s %s x %s   %s x %s' % (S.LASERS[i], fmt(d['fwhm_x']), fmt(d['fwhm_y']),
                                                              fmt(d['e2_x']), fmt(d['e2_y'])), hud_cols[i]))
            if res.get('power_total') is not None:
                lines.append(('Power %s W' % fmt(res['power_total']), FG))
        else:
            lines.append(('No beam found', WARN))
        y = 10
        for text, col in lines:
            self.live.create_text(12, y, text=text, anchor='nw', fill=col, tags='hud',
                                  font=('Consolas', 13, 'bold'))
            y += 22

    def draw(self, msg):
        res = msg['result']
        self.draw_views(msg)
        self._learn_frame(res)
        # follow the camera's exposure only when auto exposure drives it; while you
        # drag the slider, the slider is the truth (frames lag the setting)
        if self.auto_exp.get() and not self._dragging:
            self.s['exposure_us'] = res['exposure_us']
            self._set_exp_widgets(res['exposure_us'])
        self.draw_profiles(res)
        self.draw_readout(msg)
        if res.get('ok'):
            self._write_csv(res)

    def draw_profiles(self, res):
        if not self.show_profiles.get():
            return                                   # hidden: skip the drawing work too
        for cvs, key, cross, label in ((self.prof_x, 'profile_x', 'cross_x', 'X profile'),
                                       (self.prof_y, 'profile_y', 'cross_y', 'Y profile')):
            cvs.delete('all')
            w, h = max(cvs.winfo_width(), 50), max(cvs.winfo_height(), 50)
            cvs.create_text(6, 4, text=label, anchor='nw', fill=DIM, font=('Segoe UI', 8))
            p = res.get(key)
            if not p:
                continue
            p = np.asarray(p, dtype=float)
            top = p.max() if p.max() > 0 else 1.0
            n = len(p)
            # plot only around the beam: the 1/e2 edges plus half a 1/e2 width each side
            # (the measuring box is ~2.6x wider for the ISO method; the plot need not be)
            lo, hi = 0.0, float(n - 1)
            c = res.get(cross)
            if c and c[2] == c[2] and c[3] == c[3] and c[3] > c[2]:
                pad = 0.5 * (c[3] - c[2])
                lo, hi = max(0.0, c[2] - pad), min(float(n - 1), c[3] + pad)
            elif c and c[0] == c[0] and c[1] == c[1] and c[1] > c[0]:
                pad = 1.2 * (c[1] - c[0])
                lo, hi = max(0.0, c[0] - pad), min(float(n - 1), c[1] + pad)
            i0, i1 = int(math.floor(lo)), int(math.ceil(hi)) + 1
            span = max(hi - lo, 1e-6)

            def X(v):
                return 4 + (w - 8) * (v - lo) / span
            idx = np.arange(i0, min(i1, n))
            xs = X(idx.astype(float))
            ys = h - 6 - (h - 20) * p[idx] / top
            if len(idx) >= 2:
                cvs.create_line(*np.column_stack([xs, ys]).ravel().tolist(), fill='#4fc3f7', width=2)
            for lvl, col in ((0.5, '#ffd54f'), (0.135, '#ef9a9a')):
                y = h - 6 - (h - 20) * lvl
                cvs.create_line(4, y, w - 4, y, fill=col, dash=(2, 3))
            if c:
                for v, col in ((c[0], '#ffd54f'), (c[1], '#ffd54f'), (c[2], '#ef9a9a'), (c[3], '#ef9a9a')):
                    if v == v:
                        cvs.create_line(X(v), 14, X(v), h - 6, fill=col)

    def _laser_offsets(self, res):
        """Per laser (Red, Green, Blue): (dx, dy) mrad from the crosshair, or None."""
        out = [None, None, None]
        if not res.get('ok'):
            return out
        k = res['mrad_per_px']
        c = self.s.get('crosshair_px')
        zx, zy = c if c else (res['shape'][1] / 2.0, res['shape'][0] / 2.0)
        for j, idx in enumerate(res.get('laser_idx', [])):
            xy = res['laser_cxy'][j] if j < len(res['laser_cxy']) else None
            if xy:
                out[idx] = ((xy[0] - zx) * k, -(xy[1] - zy) * k)
        return out

    def _laser_divs(self, res):
        """Per laser (Red, Green, Blue): divergence dict in mrad, or None."""
        out = [None, None, None]
        for j, idx in enumerate(res.get('laser_idx', [])):
            dv = res.get('laser_div') or []
            if j < len(dv) and dv[j]:
                out[idx] = dv[j]
        return out

    def _laser_powers(self, res):
        out = [None, None, None]
        pw = res.get('powers')
        if pw:
            for j, idx in enumerate(res.get('laser_idx', [])):
                out[idx] = pw[j]
        return out

    def draw_readout(self, msg):
        res = msg['result']
        t = self.readout
        t.delete('1.0', 'end')
        peak = 100.0 * max(res['peak_raw']) / res.get('full', 255.0)
        t.insert('end', 'Exposure %s   peak %.0f %%   %.0f fps' %
                 (('%.2f ms' % (res['exposure_us'] / 1000)), peak, msg['fps']), 'dim')
        if msg.get('temp') is not None:
            t.insert('end', '   sensor %.0f C' % msg['temp'], 'bad' if msg['temp'] > 75 else 'dim')
        st = res.get('ae_state')
        if st and st != 'off':
            t.insert('end', '   auto: %s' % st, 'good' if st == 'settled' else 'dim')
        t.insert('end', '\n')
        ns = res.get('n_spots') or 0
        if ns > 1:
            t.insert('end', '%d spots seen: measuring the %s\n'
                     % (ns, self.SPOT_NAMES.get(self.spot_key(), 'right spot').lower()), 'dim')
        if st == 'at_min':
            t.insert('end', 'Shortest exposure and still too bright:\n'
                     '  add an ND filter or lower the laser power\n', 'bad')
        elif st == 'at_max':
            t.insert('end', 'Longest exposure reached and still dim\n', 'warn')
        avg = res.get('avg')
        if avg and avg[1] > 1:
            if avg[0] < avg[1]:
                t.insert('end', 'Averaging %d of %d frames (restarts while the beam moves)\n'
                         % tuple(avg), 'dim')
            else:
                t.insert('end', 'Averaging %d frames (beam still)\n' % avg[1], 'dim')
        if peak >= 99.5:
            t.insert('end', 'Exposure too high: the peak is clipped, widths read too WIDE.\n'
                     '  Lower the exposure or use Auto.\n', 'bad')
        elif peak < 25:
            t.insert('end', 'Exposure is low: the measure is not accurate\n', 'warn')
        if not res.get('ok'):
            t.insert('end', 'No beam on the sensor%s.\n' % (
                ' (recording the zero - laser off)' if res.get('dark') == 'recording' else ''), 'warn')
            return
        if math.isnan(res['fwhm_x']) or math.isnan(res['fwhm_y']):
            t.insert('end', 'The ROI must contain the entire spot.\n', 'warn')
        # divergence only (signed position lines were easy to mistake for divergence):
        # the whole beam, then every colour on its own, same method and units
        t.insert('end', 'DIVERGENCE (mrad)  FWHM X  FWHM Y  1/e2 X  1/e2 Y   D4s X   D4s Y\n', 'h')
        row = '  %-15s %7s %7s %7s %7s %7s %7s\n'      # 65 chars: fits the 80-wide box
        t.insert('end', row % ('Whole beam', fmt(res['fwhm_x']), fmt(res['fwhm_y']), fmt(res['e2_x']),
                               fmt(res['e2_y']), fmt(res['d4s_x']), fmt(res['d4s_y'])))
        pws = self._laser_powers(res)
        divs = self._laser_divs(res)
        for i, name in enumerate(S.LASERS):
            d = divs[i]
            if d:
                t.insert('end', row % (name, fmt(d['fwhm_x']), fmt(d['fwhm_y']), fmt(d['e2_x']),
                                       fmt(d['e2_y']), fmt(d['d4s_x']), fmt(d['d4s_y'])),
                         'c%d' % i)
            else:
                t.insert('end', '  %-15s not visible\n' % name, 'dim')
        if not any(self._cal().get('response') or []):
            t.insert('end', '  (colours = raw R/G/B pixels; power-calibrate each laser\n'
                     '   to separate overlapping colours cleanly)\n', 'dim')
        t.insert('end', '-' * 66 + '\n', 'dim')
        if any(p is not None for p in pws):
            t.insert('end', 'POWER  ' + '   '.join('%s %s W' % (S.LASERS[i], fmt(p, 3))
                                               for i, p in enumerate(pws) if p is not None) + '\n')
        if res.get('power_total') is not None:
            t.insert('end', 'Power %s W   density X %s  Y %s W/mrad   point %s W/mrad2\n'
                     % (fmt(res['power_total'], 3), fmt(res.get('dens_x'), 2), fmt(res.get('dens_y'), 2),
                        fmt(res.get('dens_point'), 2)), 'h')
        else:
            t.insert('end', 'Power: calibrate a laser to measure watts.\n', 'dim')

    # ================================================================ shutdown
    def on_close(self):
        try:
            S.save(self._collect())
        except OSError:
            pass
        self.disconnect()
        self.root.destroy()


def selftest(path):
    """For the installed / frozen build (no console): can it load both camera SDKs,
    see cameras, and run the camera process + analysis on the simulator?"""
    lines, ok = [], True
    try:
        from pypylon import pylon
        lines.append('pypylon OK, Basler cameras: %d' % len(pylon.TlFactory.GetInstance().EnumerateDevices()))
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append('pypylon FAILED: %r' % (e,))
    try:
        import imagingcontrol4 as ic4
        try:
            ic4.Library.init()
        except Exception:  # noqa: BLE001 - already initialised
            pass
        lines.append('imagingcontrol4 OK, Imaging Source cameras: %d' % len(ic4.DeviceEnum.devices()))
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append('imagingcontrol4 FAILED: %r' % (e,))
    try:                                     # the window itself: Tk, Pillow drawing + fonts
        root = tk.Tk()
        root.title('BeamLab self-test')
        App(root)
        img = Image.new('RGB', (200, 60))
        App._draw_text_block(img, ['self-test 1.234'])
        ImageTk.PhotoImage(img)
        root.update()
        root.destroy()
        lines.append('window + drawing OK')
    except Exception as e:  # noqa: BLE001
        ok = False
        lines.append('window FAILED: %r' % (e,))
    cmd_q, out_q = mp.Queue(), mp.Queue(maxsize=3)
    cfg = dict(S.DEFAULTS, backend='sim', response=None, laser_idx=[], zero_px=None, roi=None,
               exposure_us=400.0, auto_exposure=True)
    p = mp.Process(target=worker.run, args=(cmd_q, out_q, cfg), daemon=True)
    p.start()
    got = None
    t_end = time.time() + 60
    while time.time() < t_end:
        try:
            m = out_q.get(timeout=5)
        except queue.Empty:
            continue
        if m['type'] == 'error':
            break
        if m['type'] == 'frame' and m['result'].get('ok') and m['result'].get('ae_state') == 'settled':
            got = m['result']            # a frame after auto exposure settled (not saturated)
            break
    cmd_q.put(('quit',))
    p.join(5)
    if got:
        lines.append('camera process + analysis OK (simulator): FWHM %.3f x %.3f mrad'
                     % (got['fwhm_x'], got['fwhm_y']))
    else:
        ok = False
        lines.append('camera process FAILED (no measured frame from the simulator)')
    lines.append('RESULT: %s' % ('PASS' if ok else 'FAIL'))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return 0 if ok else 1


def main(argv=None):
    """python -m beamlab [--connect] [--log FILE.csv] [--quit-after SECONDS]
    --connect     connect to the saved camera source at start
    --log         start the CSV log into FILE once frames arrive (same log as the button)
    --quit-after  close normally (settings saved) after SECONDS"""
    import argparse
    mp.freeze_support()
    ap = argparse.ArgumentParser(prog='beamlab')
    ap.add_argument('--connect', action='store_true')
    ap.add_argument('--log')
    ap.add_argument('--quit-after', type=float)
    ap.add_argument('--selftest', metavar='FILE',
                    help='check the camera SDKs and a simulated measurement, write the report to FILE, exit')
    args = ap.parse_args(argv)
    if args.selftest:
        sys.exit(selftest(args.selftest))
    root = tk.Tk()
    app = App(root)
    if args.connect:
        root.after(300, app.connect)
    if args.log:
        def start_log():
            if app.last is None:
                root.after(200, start_log)
                return
            app._auto_log_path = os.path.abspath(args.log)
            app.toggle_log()
        root.after(500, start_log)
    if args.quit_after:
        root.after(int(args.quit_after * 1000), app.on_close)
    root.mainloop()
