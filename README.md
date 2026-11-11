# BeamLab

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

A free Windows app for measuring laser beams with the **LBA-C-100** far-field
beam analyzer: divergence (FWHM, 1/e² and ISO 11146 D4σ) for the whole beam
and for each colour of an RGB laser, alignment against a reference crosshair,
and optional power measurement. Made for aligning laser-show projectors, with
a display mode that lets you see a beam your safety goggles block.

BeamLab is an independent open-source project. It is not made, supported or
endorsed by Lightdiction, Basler or The Imaging Source; their names are used
only to say which hardware it works with.

> **Status:** tested with a built-in camera simulator and a real install on
> Windows 11. Not yet run against a real LBA-C-100: reports welcome.

## Install

**Windows installer:** download `BeamLab-x.y.z.msi` from the
[Releases](../../releases) page and run it. It installs BeamLab and, unless you
untick it, the camera USB drivers (Basler pylon USB and Imaging Source USB3
Vision). Pick **sim** as the source to try BeamLab without the hardware.

**From source** (Python 3.9 or newer):

```
pip install -r requirements.txt
python -m beamlab
```

The camera SDKs (`pypylon` for Basler, `imagingcontrol4` for The Imaging
Source) need the matching USB driver installed.

## What it measures

The LBA-C-100 is a camera behind a lens focused at infinity, so a spot's
position on the sensor is the beam's pointing angle and its size is the beam's
divergence: **angle = size on sensor / focal length**.

- **Divergence** (full angle, mrad), in X and Y, for the whole beam and for
  red, green and blue separately:
  - **FWHM**: width where the profile crosses 50 % of its peak
  - **1/e²**: width at 13.5 % of the peak
  - **D4σ**: ISO 11146 second-moment width, background-subtracted with no
    noise clipping, inside an iterated 3 × D4σ aperture
- **Spatial accuracy:** each colour's centre is corrected for its Bayer pixel
  offset. Frames are only averaged while the beam stands still, so a moving
  beam is never smeared. With no beam on the sensor, BeamLab says so instead of
  measuring noise.
- **Power** (optional): calibrate each laser once with a known power. A 3×3
  colour response then separates the lasers and gives watts per colour and
  power density (W/mrad, W/mrad²).

## Calibration

- **Optics:** enter the camera's pixel size and the lens focal length (Optics
  box). These set the mrad-per-pixel scale.
- **Zero:** a dark frame is recorded automatically when the camera connects, so
  keep the laser off until BeamLab is running. There is also a **Zero (laser
  off)** button.
- **Power:** shoot ONE laser at a known power into the sensor, type the watts,
  press Calibrate. Repeat for each colour. Calibrations are saved per camera
  serial number.

## Features

- Live view with **auto zoom**: fits all spots, and zooms out on the very next
  frame if a beam is adjusted too far. Also manual zoom (wheel), pan
  (right-drag), and centring on the beam or the crosshair.
- **Display colours for goggles:** show each laser in any colour, plus a
  brightness slider per laser. Measurements are not affected.
- **Crosshair** reference: left-click the picture to place it, or centre it, or snap it
  to the beam or one colour.
- **Snapshot:** a still reference picture shown next to the live view, with
  its divergence numbers, saved as PNG + JSON.
- **Exposure:** logarithmic slider, plus a robust auto exposure (ignores hot
  pixels, steps down by how saturated the image is, never hunts).
- **Idle full screen** (screensaver style) with a compact readout; any mouse
  movement or key brings the window back.
- **ROI:** automatic, or drag one on the picture.
- X/Y profile graphs (can be hidden), CSV logging, sensor temperature.

Settings and calibrations live in `%APPDATA%\BeamLab`.

## Command line

```
python -m beamlab --connect --log run.csv --quit-after 40
python tools/plot_log.py run.csv
```

`--connect` connects at start, `--log` writes every measured frame to a CSV,
and `--quit-after` closes normally after N seconds. `--selftest FILE` checks
the camera SDKs, the window and a simulated measurement.

## Tests

```
python tests/test_beamlab.py   # measurements against a simulator with known truth
python tests/test_ui.py        # the real window with simulated frames
```

They run on GitHub Actions for Python 3.9 and 3.12.

## Building the installer

Needs `pip install -r requirements-build.txt`, the
[WiX 3.14](https://github.com/wixtoolset/wix3/releases) binaries in
`%USERPROFILE%\tools\wix314`, and the two driver packages in
`installer\drivers\` (see the README there; they are not in this repository).

```
powershell -ExecutionPolicy Bypass -File installer\build.ps1
```

This freezes the app with PyInstaller, runs the frozen app's self-test, and
builds `dist\BeamLab-<version>.msi`.

- **Size:** parts BeamLab never uses (pypylon's AI plugins, GigE parts, IC4's
  GUI/video plugins) are left out, giving about 35 MB for the .msi and about
  100 MB installed.
- **Driver step:** Windows Installer can't run one .msi inside another, so the
  drivers are installed right after BeamLab finishes, with one admin prompt.
- **Skipping drivers:** `msiexec /i BeamLab-x.y.z.msi INSTALLDRIVERS=0`.

## Licence

GPL-3.0 (see [LICENSE](LICENSE)). The camera SDKs and drivers are their
makers' software under their own licences.
