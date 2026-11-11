"""Draw BeamLab's icon (a beam spot on a crosshair) -> installer/beamlab.ico."""
import os

import numpy as np
from PIL import Image, ImageDraw

S = 256
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
r2 = ((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (2 * (S * 0.13) ** 2)
spot = np.exp(-r2)
img = np.zeros((S, S, 4), np.uint8)
img[..., 0] = np.clip(40 + 215 * spot, 0, 255)
img[..., 1] = np.clip(40 + 215 * spot ** 1.6, 0, 255)
img[..., 2] = np.clip(60 + 120 * spot ** 3, 0, 255)
img[..., 3] = 255
im = Image.fromarray(img, 'RGBA')
mask = Image.new('L', (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=44, fill=255)
im.putalpha(mask)
d = ImageDraw.Draw(im)
d.line([(S / 2, 18), (S / 2, S - 18)], fill=(255, 255, 255, 230), width=6)
d.line([(18, S / 2), (S - 18, S / 2)], fill=(255, 255, 255, 230), width=6)
d.ellipse([S / 2 - 70, S / 2 - 70, S / 2 + 70, S / 2 + 70], outline=(255, 255, 255, 230), width=6)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'beamlab.ico')
im.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(out)
