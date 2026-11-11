# PyInstaller build file for BeamLab.exe (one folder).  Built by installer\build.ps1.
# pypylon's AI / data-processing plugins (~230 MB) are left out: BeamLab only uses
# the pylon camera core (InstantCamera + USB transport).
import os

from PyInstaller.utils.hooks import collect_all

HERE = os.path.abspath(SPECPATH)               # ...\beam-analyzer\installer
ROOT = os.path.dirname(HERE)                   # ...\beam-analyzer

datas, binaries, hiddenimports = [], [], []
for pkg in ('pypylon', 'imagingcontrol4'):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

SKIP = (
    # pypylon: AI / data processing (~230 MB), and everything not needed for a USB camera
    'pylondataprocessingplugins', 'dataprocessingpluginsb', 'pylondataprocessing',
    '_pylondataprocessing', 'pylonutilitypcl',
    'pylongige_', 'gxapi_', 'pylongigeconnectionguard', 'pylongtc_', 'pyloncamemu_',
    'pylonrdma_', 'pylonnebulamatrix_', 'pypylon/docs',   # (PylonGUI stays: _pylon.pyd links it)
    # imagingcontrol4: GUI dialogs, video encoders, display windows (BeamLab only grabs)
    'ic4gui', 'ic4plugin_video', 'ic4plugin_display',
    # Pillow: image formats / colour management BeamLab never uses
    '_avif', '_webp', '_imagingcms',
)


def keep(entry):
    src = entry[0].replace('\\', '/').lower()
    return not any(s in src for s in SKIP)


datas = [e for e in datas if keep(e)]
binaries = [e for e in binaries if keep(e)]
hiddenimports = [h for h in hiddenimports if 'dataprocessing' not in h.lower()]

a = Analysis(
    [os.path.join(HERE, 'launcher.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=['matplotlib', 'scipy', 'pandas', 'IPython', 'pytest', 'pypylon.pylondataprocessing',
              'ssl', '_ssl', 'PIL.AvifImagePlugin', 'PIL.WebPImagePlugin', 'PIL.ImageCms',
              'unittest', 'pydoc_data', 'lib2to3'],
    noarchive=False,
)
a.binaries = [e for e in a.binaries if keep(e)]
a.datas = [e for e in a.datas if keep(e)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='BeamLab', console=False,
          icon=os.path.join(HERE, 'beamlab.ico'), version=os.path.join(HERE, 'version_info.txt'))
coll = COLLECT(exe, a.binaries, a.datas, name='BeamLab')
