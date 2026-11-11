"""Plot a BeamLab CSV log: divergence, position, exposure and peak over time.
usage: python tools/plot_log.py logs/run.csv [out.png]"""
import csv
import datetime
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else path[:-4] + '.png'
rows = list(csv.DictReader(open(path)))


def col(name):
    return np.array([float(r[name]) if r.get(name) not in (None, '') else np.nan for r in rows])


t0 = datetime.datetime.fromisoformat(rows[0]['time'])
t = np.array([(datetime.datetime.fromisoformat(r['time']) - t0).total_seconds() for r in rows])
fig, ax = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
for name, c in (('fwhm_x', '#1f77b4'), ('fwhm_y', '#ff7f0e'), ('e2_x', '#2ca02c'), ('e2_y', '#d62728'),
                ('d4s_x', '#9467bd'), ('d4s_y', '#8c564b')):
    ax[0].plot(t, col(name), '.-', ms=3, lw=0.8, label=name, color=c)
ax[0].axhline(0, color='k', lw=0.6)
ax[0].set_ylabel('divergence (mrad)')
ax[0].legend(ncol=6, fontsize=8)
ax[0].set_title('%s  (%d frames, %.1f s, %.1f frames/s)' % (path, len(rows), t[-1], len(rows) / max(t[-1], 1e-6)))
ax[1].plot(t, col('cx_px'), label='beam x (px)')
ax[1].plot(t, col('cy_px'), label='beam y (px)')
ax[1].set_ylabel('beam centre (analysis px)')
ax[1].legend(fontsize=8)
ax[2].plot(t, col('peak_pct'), color='k', label='peak %')
ax[2].set_ylabel('peak (% of full scale)')
a2 = ax[2].twinx()
a2.plot(t, col('exposure_us'), color='#e377c2', label='exposure')
a2.set_ylabel('exposure (us)', color='#e377c2')
ax[3].plot(t, col('avg_frames'), label='frames averaged')
ax[3].plot(t, col('roi_x1') - col('roi_x0'), label='ROI width (px)')
ax[3].plot(t, col('roi_y1') - col('roi_y0'), label='ROI height (px)')
ax[3].set_ylabel('averaging / ROI')
ax[3].set_xlabel('time (s)')
ax[3].legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=90)
print(out)
