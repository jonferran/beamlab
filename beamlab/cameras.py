"""Camera backends. Each returns raw frames in BayerGR layout as numpy arrays:
uint16 0..4095 from a 12-bit camera, uint8 0..255 from an 8-bit one; the
camera's `full` (4095.0 or 255.0) is the saturation level every check uses.

The LBA-C-100 contains either a Basler (pylon) or an Imaging Source (IC4)
USB3 Vision colour camera. BeamLab uses the vendors' official Python packages,
sets a raw Bayer pixel format (12-bit if offered), turns ExposureAuto/GainAuto off and turns Basler's
colour space / light-source presets off (raw, linear pixel values):
    pip install pypylon            (Basler)
    pip install imagingcontrol4    (The Imaging Source)
Both need the vendor's USB driver (BeamLab's installer can install them, or
install pylon / IC4 from the vendor's site).
"""
import math
import time

import numpy as np


MAX_EXPOSURE_US = 400000.0   # 400 ms: the longest exposure offered (as the LBA's own software)


class CameraError(Exception):
    pass


class Camera(object):
    name = 'camera'
    serial = ''
    model = ''
    exposure_range = (20.0, 1000000.0)   # microseconds
    full = 255.0                         # saturation level of the raw values (4095 for 12-bit)
    bayer = 'BayerGR8'

    def set_exposure(self, us):
        raise NotImplementedError

    def grab(self):
        """-> (raw HxW BayerGR layout, uint8 or uint16 up to self.full,
        exposure_us actually used, temperature_C or None)"""
        raise NotImplementedError

    def close(self):
        pass


def _try(fn, *a):
    try:
        fn(*a)
        return True
    except Exception:
        return False


# The analysis expects the BayerGR layout. Cameras that only offer another Bayer
# order (the LBA's Basler daA3840-45uc is BayerRG only) are converted by
# dropping the first and last column and/or row: that shifts the colour phase
# by one pixel and keeps the sensor centre exactly where it was.
BAYER_ORDERS_8 = ('BayerGR8', 'BayerRG8', 'BayerGB8', 'BayerBG8')
# 12-bit first: 16x finer steps, so a dim beam (the LBA's reaches ~50 of 255 at the
# longest exposure) is no longer a few coarse levels that turn grainy when boosted.
# Unpacked 12-bit (two bytes per pixel); the packed '12p' formats are not used.
BAYER_ORDERS_12 = ('BayerGR12', 'BayerRG12', 'BayerGB12', 'BayerBG12')
BAYER_ORDERS = BAYER_ORDERS_12 + BAYER_ORDERS_8


def bayer_full(order):
    """Saturation level of a pixel format: 4095 (12-bit) or 255 (8-bit)."""
    return 4095.0 if order.endswith('12') else 255.0


def to_gr8(raw, order):
    """raw Bayer frame in `order` (8 or 12 bit) -> BayerGR layout (G R / B G)."""
    phase = order[:7]                    # 'BayerGR' / 'BayerRG' / ...
    if phase == 'BayerGR':
        return raw
    if phase == 'BayerRG':               # R G / G B: shift one column
        return raw[:, 1:-1]
    if phase == 'BayerBG':               # B G / G R: shift one row
        return raw[1:-1, :]
    if phase == 'BayerGB':               # G B / R G: shift row and column
        return raw[1:-1, 1:-1]
    raise CameraError('unsupported pixel format %s' % order)


# ------------------------------------------------------------------ Basler

class PylonCamera(Camera):
    name = 'Basler (pylon)'

    def __init__(self, serial=None):
        try:
            from pypylon import pylon
        except ImportError:
            raise CameraError('pypylon is not installed. Run: pip install pypylon')
        self.pylon = pylon
        tl = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        if not devs:
            raise CameraError('No Basler camera found. Is the LBA plugged in and the '
                              'pylon USB driver installed?')
        dev = devs[0]
        if serial:
            for d in devs:
                if d.GetSerialNumber() == serial:
                    dev = d
        self.cam = pylon.InstantCamera(tl.CreateDevice(dev))
        self.cam.Open()
        self.serial = dev.GetSerialNumber()
        self.model = dev.GetModelName()
        n = self.cam.GetNodeMap()

        def setv(name, v):
            node = n.GetNode(name)
            if node is not None:
                _try(setattr, node, 'Value', v)
        # a raw Bayer format the camera really accepts: 12-bit first, GR preferred
        pf = n.GetNode('PixelFormat')
        self.bayer = None
        for order in BAYER_ORDERS:
            if _try(setattr, pf, 'Value', order) and str(pf.Value) == order:
                self.bayer = order
                break
        if self.bayer is None:
            cur = str(pf.Value)
            _try(self.cam.Close)
            raise CameraError('This camera offers no raw Bayer pixel format (it is set to %s); '
                              'BeamLab needs the raw colour pixels.' % cur)
        self.full = bayer_full(self.bayer)
        self._dtype = np.uint16 if self.full > 255 else np.uint8
        setv('ExposureAuto', 'Off')
        setv('GainAuto', 'Off')
        setv('BalanceWhiteAuto', 'Off')
        setv('Gain', 0.0)
        setv('BslLightSourcePreset', 'Off')
        setv('BslColorSpace', 'Off')
        setv('TriggerMode', 'Off')
        e = n.GetNode('ExposureTime')
        self._exp = e
        try:
            self.exposure_range = (float(e.Min), float(e.Max))
        except Exception:
            pass
        self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def set_exposure(self, us):
        lo, hi = self.exposure_range
        self._exp.Value = float(min(max(us, lo), hi))

    def grab(self):
        res = self.cam.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException)
        try:
            if not res.GrabSucceeded():
                raise CameraError('grab failed: %s' % res.ErrorDescription)
            raw = np.ascontiguousarray(to_gr8(np.asarray(res.Array, dtype=self._dtype), self.bayer))
        finally:
            res.Release()
        temp = None
        try:
            temp = float(self.cam.GetNodeMap().GetNode('DeviceTemperature').Value)
        except Exception:
            pass
        return raw, float(self._exp.Value), temp

    def close(self):
        _try(self.cam.StopGrabbing)
        _try(self.cam.Close)


# ------------------------------------------------------------ Imaging Source

class IC4Camera(Camera):
    name = 'The Imaging Source (IC4)'

    def __init__(self, serial=None):
        try:
            import imagingcontrol4 as ic4
        except ImportError:
            raise CameraError('imagingcontrol4 is not installed. Run: pip install imagingcontrol4')
        self.ic4 = ic4
        _try(ic4.Library.init)
        devs = ic4.DeviceEnum.devices()
        if not devs:
            raise CameraError('No Imaging Source camera found. Is the LBA plugged in and '
                              'the IC4 USB3 Vision driver installed?')
        dev = devs[0]
        if serial:
            for d in devs:
                if d.serial == serial:
                    dev = d
        self.serial, self.model = dev.serial, dev.model_name
        g = ic4.Grabber()
        g.device_open(dev)
        pm = g.device_property_map
        self.pm = pm
        self.bayer = None
        # 8-bit only here: how IC4 cameras deliver 12-bit (packed / MSB-aligned) is
        # not tested on real hardware yet
        for order in BAYER_ORDERS_8:
            fmt = getattr(ic4.PixelFormat, order, None)
            if fmt is not None and _try(pm.set_value, ic4.PropId.PIXEL_FORMAT, fmt):
                self.bayer = order
                break
        if self.bayer is None:
            _try(g.device_close)
            raise CameraError('This camera offers no 8-bit Bayer pixel format; '
                              'BeamLab needs the raw colour pixels.')
        _try(pm.set_value, ic4.PropId.EXPOSURE_AUTO, 'Off')
        _try(pm.set_value, ic4.PropId.GAIN_AUTO, 'Off')
        _try(pm.set_value, ic4.PropId.GAIN, 0.0)
        try:
            p = pm.find_float(ic4.PropId.EXPOSURE_TIME)
            self.exposure_range = (float(p.minimum), float(p.maximum))
        except Exception:
            pass
        self.sink = ic4.SnapSink()
        g.stream_setup(self.sink)
        self.g = g

    def set_exposure(self, us):
        lo, hi = self.exposure_range
        self.pm.set_value(self.ic4.PropId.EXPOSURE_TIME, float(min(max(us, lo), hi)))

    def grab(self):
        timeout = int(self.pm.get_value_float(self.ic4.PropId.EXPOSURE_TIME) / 1000) + 2000
        buf = self.sink.snap_single(timeout)
        a = np.array(buf.numpy_wrap(), copy=True)
        buf.release()
        if a.ndim == 3:
            a = a[:, :, 0]
        temp = None
        try:
            temp = float(self.pm.get_value_float('DeviceTemperature'))
        except Exception:
            pass
        a = np.ascontiguousarray(to_gr8(a.astype(np.uint8), self.bayer))
        return a, self.pm.get_value_float(self.ic4.PropId.EXPOSURE_TIME), temp

    def close(self):
        _try(self.g.stream_stop)
        _try(self.g.device_close)


# --------------------------------------------------------------- simulator

class SimCamera(Camera):
    """Synthetic far-field camera for testing without the LBA: an elliptical
    Gaussian spot per laser colour, wandering slightly, on a BayerGR sensor with
    dark level, shot noise and saturation (bits=8: 0..255; bits=12: 0..4095, the
    same light 16x finer, like the LBA's Basler). Ground truth is in `self.truth`."""
    name = 'Simulator'
    serial = 'SIM0001'
    model = 'Simulated LBA'

    # how strongly each Bayer colour sees each laser (rows R,G,B sensor; cols lasers)
    CROSSTALK = np.array([[1.00, 0.08, 0.02],
                          [0.12, 1.00, 0.20],
                          [0.01, 0.10, 0.90]])

    def __init__(self, width=1440, height=1080, powers_w=(0.6, 0.4, 0.3),
                 sigma_px=(18.0, 12.0), counts_per_w_ms=3000.0, seed=1, wander=True,
                 fps=30.0, laser_on_after=0, bits=8):
        self.laser_on_after = laser_on_after     # frames of darkness first (for the zero)
        self.full = float(2 ** bits - 1)
        self.bayer = 'BayerGR%d' % bits
        self._u = self.full / 255.0              # 12-bit: same light, 16x more counts
        self._dtype = np.uint16 if bits > 8 else np.uint8
        self.w, self.h = width, height
        self.powers = np.array(powers_w, dtype=float)
        self.sigma = sigma_px            # in SENSOR pixels
        self.k = counts_per_w_ms
        self.exp = 5000.0
        self.exposure_range = (20.0, 400000.0)
        self.rng = np.random.default_rng(seed)
        self.wander = wander
        self.t0 = time.time()
        self.period = 1.0 / fps
        self._next = time.time()
        yy, xx = np.mgrid[0:height, 0:width]
        self.xx, self.yy = xx.astype(np.float32), yy.astype(np.float32)
        self.mask = np.zeros((3, height, width), np.float32)   # which sensor colour each pixel is
        self.mask[1, 0::2, 0::2] = 1
        self.mask[0, 0::2, 1::2] = 1
        self.mask[2, 1::2, 0::2] = 1
        self.mask[1, 1::2, 1::2] = 1
        self.truth = {}

    def set_exposure(self, us):
        lo, hi = self.exposure_range
        self.exp = float(min(max(us, lo), hi))

    def grab(self):
        now = time.time()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self.period, time.time())
        t = time.time() - self.t0
        cx = self.w / 2 + (40 * math.sin(t * 0.3) if self.wander else 0) + 60
        cy = self.h / 2 + (25 * math.cos(t * 0.23) if self.wander else 0) - 30
        sx, sy = self.sigma
        # self.k = total counts per W per ms over the whole spot -> peak counts per pixel
        norm = self.k * (self.exp / 1000.0) / (2 * math.pi * sx * sy) * 400
        self.frames = getattr(self, 'frames', 0) + 1
        # the laser is switched on only after the app has connected (so the zero works)
        on = self.frames > self.laser_on_after
        seen = self.CROSSTALK @ (self.powers if on else self.powers * 0)   # per sensor colour
        # dark level + read noise: a bank of pre-made noise frames (making 1.5 M
        # gaussian numbers per frame cost ~60 ms and capped the simulator at ~3 fps)
        if not hasattr(self, '_noise'):
            self._noise = [np.clip(np.floor((3.5 + self.rng.normal(0, 1.0, (self.h, self.w))) * self._u),
                                   0, self.full).astype(self._dtype) for _ in range(6)]
        # random even shift each frame, so the noise is NOT the same pattern frame to frame
        # (a repeating pattern is fixed-pattern noise, which the dark frame would remove)
        sy_, sx_ = 2 * int(self.rng.integers(self.h // 2)), 2 * int(self.rng.integers(self.w // 2))
        raw = np.roll(self._noise[int(self.rng.integers(len(self._noise)))], (sy_, sx_), axis=(0, 1))
        # the spot only where it is: a window of +-7 sigma (rounded to even = Bayer phase)
        hx, hy = int(7 * sx) + 2, int(7 * sy) + 2
        x0, x1 = max(0, (int(cx) - hx) // 2 * 2), min(self.w, int(cx) + hx)
        y0, y1 = max(0, (int(cy) - hy) // 2 * 2), min(self.h, int(cy) + hy)
        if seen.max() > 0 and x1 > x0 and y1 > y0:
            xx, yy = self.xx[y0:y1, x0:x1], self.yy[y0:y1, x0:x1]
            spot = np.exp(-((xx - cx) ** 2 / (2 * sx * sx) + (yy - cy) ** 2 / (2 * sy * sy)))
            sig = np.zeros_like(spot)
            for c in range(3):
                sig += self.mask[c, y0:y1, x0:x1] * (seen[c] * norm)
            sig *= spot
            sig += self.rng.normal(0, 1.0, sig.shape).astype(np.float32) * np.sqrt(sig * 0.3)  # shot noise
            sig *= np.float32(self._u)
            # round (not truncate): truncating lost 0.5 count/pixel inside the window only,
            # a step at the window edge that read as negative power on dim colours
            raw[y0:y1, x0:x1] = np.clip(np.floor(raw[y0:y1, x0:x1] + sig + 0.5), 0, self.full).astype(self._dtype)
        self.truth = {'cx_sensor': cx, 'cy_sensor': cy, 'sigma': self.sigma}
        return raw, self.exp, 35.0


BACKENDS = {'auto': None, 'pylon': PylonCamera, 'ic4': IC4Camera, 'sim': SimCamera}


def _cap(cam):
    """Longest exposure BeamLab uses (slider, typed value and auto exposure)."""
    lo, hi = cam.exposure_range
    cam.exposure_range = (float(lo), float(min(hi, MAX_EXPOSURE_US)))
    return cam


def open_camera(kind='auto', serial=None):
    """'auto' tries Basler then Imaging Source and reports why both failed."""
    if kind != 'auto':
        return _cap(BACKENDS[kind](serial) if kind != 'sim' else SimCamera(laser_on_after=15, bits=12))
    errs = []
    for cls in (PylonCamera, IC4Camera):
        try:
            return _cap(cls(serial))
        except Exception as e:     # noqa: BLE001 - report every backend's reason
            errs.append('%s: %s' % (cls.name, e))
    raise CameraError('No camera opened.\n' + '\n'.join(errs))
