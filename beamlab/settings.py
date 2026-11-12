"""Settings and per-camera calibrations, saved as JSON in %APPDATA%\\BeamLab."""
import copy
import json
import os

# BEAMLAB_DIR overrides the folder (tests use it; child processes inherit it)
DIR = os.environ.get('BEAMLAB_DIR') or os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'BeamLab')
FILE = os.path.join(DIR, 'settings.json')
SNAP_DIR = os.path.join(DIR, 'snapshots')
ERROR_LOG = os.path.join(DIR, 'errors.log')


def log_error(text):
    """Append a time-stamped entry to errors.log (window, camera process and crash
    handler all write here). Kept under ~1 MB: the old half moves to errors.log.1."""
    import datetime
    try:
        os.makedirs(DIR, exist_ok=True)
        if os.path.exists(ERROR_LOG) and os.path.getsize(ERROR_LOG) > 1000000:
            os.replace(ERROR_LOG, ERROR_LOG + '.1')
        with open(ERROR_LOG, 'a', encoding='utf-8') as fh:
            fh.write('[%s pid %d] %s\n' % (datetime.datetime.now().isoformat(timespec='milliseconds'),
                                           os.getpid(), text.rstrip()))
    except OSError:
        pass

LASERS = ['Red', 'Green', 'Blue']

DEFAULTS = {
    'backend': 'auto',
    'exposure_us': 5000.0,
    'auto_exposure': False,
    'ae_target': 0.85,
    'stack': 1,
    'auto_roi': True,
    'spot': 'right',                           # which spot is measured: right|middle|left|brightest
                                               # (the LBA always shows 3 spots side by side)
    'width_mode': 'fwhm',                      # fwhm | e2 | d4s
    'optics': {'pixel_pitch_um': 3.45, 'focal_mm': 50.0},
    'display': {'colors': [[255, 0, 0], [0, 255, 0], [0, 80, 255]],
                'auto_bright': True, 'gain': 1.0, 'roi_color': [0, 255, 0],
                'atten': [1.0, 1.0, 1.0],              # per-laser on-screen brightness 0..1
                'contrast': 0,                         # -100..100, display gamma 2^(c/50)
                'balance': True,                       # each laser's peak at full brightness
                'boost': [1.0, 1.0, 1.0],              # per-laser boost (x)
                'lcontrast': [0, 0, 0]},               # per-laser contrast -100..100
    'crosshair_px': None,                      # analysis px; None = image centre
    'show_crosshair': True,
    'overlay_strength': 100,                   # snapshot overlay on live, %
    'wavelengths_nm': [638, 520, 450],
    'calibrations': {},                        # serial -> {'response': 3x3 (None columns), 'date': str}
    'colour_sigs': {},                         # serial -> per laser learned [r, g, b] (display only)
    'csv_dir': os.path.join(os.path.expanduser('~'), 'Documents'),
}


def load():
    s = copy.deepcopy(DEFAULTS)
    try:
        with open(FILE, 'r') as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(s.get(k), dict) and k != 'calibrations':
                s[k].update(v)
            else:
                s[k] = v
    except (OSError, ValueError):
        pass
    return s


def save(s):
    os.makedirs(DIR, exist_ok=True)
    tmp = FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, FILE)


def response_matrix(s, serial):
    """3 x k matrix of calibrated lasers (columns), plus the laser indices, or (None, [])."""
    cal = s['calibrations'].get(serial or '', {})
    cols = cal.get('response') or [None, None, None]
    idx = [i for i, c in enumerate(cols) if c]
    if not idx:
        return None, []
    return [[cols[i][row] for i in idx] for row in range(3)], idx
