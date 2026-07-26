#!/usr/bin/env python3
"""
Animatic renderer for the Thonis-Heracleion 60-second Short.

Produces a 1080x1920 / 30fps cut with the exact timing of
scripts/thonis-heracleion-60s-short.md: procedural underwater art as
placeholder plates, camera moves, burned-in caption schedule, a synthesized
sound-design bed and a scratch espeak VO.

The placeholder plates exist to lock staging and pacing. Each one is meant to
be replaced by generated footage from the AI prompt for that scene.

Usage: python3 tools/render_short.py [outfile.mp4]
"""

import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
DUR = 60.0
NFRAMES = int(DUR * FPS)
SR = 44100

HW, HH = W // 2, H // 2  # half-res mask canvas

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

SCRATCH = os.environ.get(
    "SHORT_SCRATCH", "/tmp/claude-0/-home-user-Ystt/f2140de2-deb3-5931-bb2b-c92220781c0a/scratchpad/vid"
)
os.makedirs(SCRATCH, exist_ok=True)

xx = np.linspace(0.0, 1.0, W, dtype=np.float32).reshape(1, W)
yy = np.linspace(0.0, 1.0, H, dtype=np.float32).reshape(H, 1)


# ---------------------------------------------------------------- utilities

def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def seg(t, a, b):
    """Progress of t through [a, b], clamped."""
    if b <= a:
        return 0.0
    return float(np.clip((t - a) / (b - a), 0.0, 1.0))


def vgrad(c_top, c_bot, power=1.0):
    g = yy ** power
    c_top = np.asarray(c_top, dtype=np.float32).reshape(1, 1, 3)
    c_bot = np.asarray(c_bot, dtype=np.float32).reshape(1, 1, 3)
    return (c_top * (1.0 - g[..., None]) + c_bot * g[..., None]).repeat(W, axis=1)


def vignette(strength=0.75, radius=0.85):
    cx, cy = 0.5, 0.5
    d = np.sqrt(((xx - cx) * 1.0) ** 2 + ((yy - cy) * (H / W) * 0.55) ** 2)
    v = 1.0 - strength * smooth((d - radius * 0.35) / 0.7)
    return np.clip(v, 0.0, 1.0)[..., None]


VIGNETTE = vignette()


_RAY_CACHE = {}
RW, RH = W // 4, H // 4
_rx = np.linspace(0.0, 1.0, RW, dtype=np.float32).reshape(1, RW)
_ry = np.linspace(0.0, 1.0, RH, dtype=np.float32).reshape(RH, 1)


def god_rays(t, n=6, seed=3, tilt=0.22, width=0.055, top_fade=1.4):
    """Light shafts. Built at quarter res and upscaled -- this is per-frame and
    the full-res exp() dominated render time otherwise."""
    key = (n, seed)
    if key not in _RAY_CACHE:
        rng = np.random.default_rng(seed)
        _RAY_CACHE[key] = [
            (rng.uniform(-0.25, 1.25), rng.uniform(0.55, 1.7),
             rng.uniform(0.4, 1.5), rng.uniform(0.5, 1.0))
            for _ in range(n)
        ]
    acc = np.zeros((RH, RW), dtype=np.float32)
    for k, (x0, wmul, tmul, amul) in enumerate(_RAY_CACHE[key]):
        drift = math.sin(t * 0.18 + k * 1.7) * 0.03
        wdt = width * wmul
        cx = x0 + drift + _ry * tilt * tmul
        acc += np.exp(-((_rx - cx) ** 2) / (2 * wdt ** 2)) * amul
    acc *= np.exp(-_ry * top_fade)
    return np.asarray(
        Image.fromarray(acc).resize((W, H), Image.BILINEAR), dtype=np.float32
    )


def make_sprite(size, power=2.0):
    a = np.linspace(-1, 1, size, dtype=np.float32)
    gx, gy = np.meshgrid(a, a)
    d = np.sqrt(gx ** 2 + gy ** 2)
    s = np.clip(1.0 - d, 0.0, 1.0) ** power
    return s


SPRITE = make_sprite(15)
SPRITE_BIG = make_sprite(41, 2.2)


def add_sprite(img, cx, cy, sprite, color, alpha, scale=1.0):
    """Additively stamp a soft dot into img (H,W,3)."""
    s = sprite
    if scale != 1.0:
        n = max(3, int(s.shape[0] * scale))
        idx = np.linspace(0, s.shape[0] - 1, n)
        s = s[np.ix_(idx.astype(int), idx.astype(int))]
    sh = s.shape[0]
    x0 = int(cx) - sh // 2
    y0 = int(cy) - sh // 2
    x1, y1 = x0 + sh, y0 + sh
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = sh - max(0, x1 - W), sh - max(0, y1 - H)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    dx0, dy0 = max(0, x0), max(0, y0)
    patch = s[sy0:sy1, sx0:sx1, None] * alpha
    img[dy0:dy0 + patch.shape[0], dx0:dx0 + patch.shape[1]] += patch * np.asarray(
        color, dtype=np.float32
    )


class MaskCanvas:
    """Half-res Pillow canvas for silhouettes; blurred then upscaled."""

    def __init__(self):
        self.img = Image.new("L", (HW, HH), 0)
        self.d = ImageDraw.Draw(self.img)

    def poly(self, pts, fill=255):
        self.d.polygon([(p[0] * HW, p[1] * HH) for p in pts], fill=fill)

    def rect(self, x0, y0, x1, y1, fill=255):
        self.d.rectangle([x0 * HW, y0 * HH, x1 * HW, y1 * HH], fill=fill)

    def ellipse(self, x0, y0, x1, y1, fill=255):
        self.d.ellipse([x0 * HW, y0 * HH, x1 * HH * 0 + x1 * HW, y1 * HH], fill=fill)

    def line(self, pts, fill=255, width=3):
        self.d.line([(p[0] * HW, p[1] * HH) for p in pts], fill=fill, width=width)

    def out(self, blur=1.6):
        im = self.img
        if blur > 0:
            im = im.filter(ImageFilter.GaussianBlur(blur))
        im = im.resize((W, H), Image.BILINEAR)
        return (np.asarray(im, dtype=np.float32) / 255.0)[..., None]


def over(img, mask, color, alpha=1.0):
    """In-place alpha composite. These are 6M-element arrays; the extra
    temporaries from the naive form dominated the frame budget."""
    c = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    a = mask if alpha == 1.0 else mask * alpha
    img *= (1.0 - a)
    img += c * a
    return img


# ------------------------------------------------------------- shape helpers

def tform(pts, cx, cy, sx, sy=None, rot=0.0):
    """Map unit-box points into frame space with scale/rotation."""
    if sy is None:
        sy = sx
    ca, sa = math.cos(rot), math.sin(rot)
    out = []
    for px, py in pts:
        ux, uy = (px - 0.5) * sx, (py - 0.5) * sy
        rx = ux * ca - uy * sa
        ry = ux * sa + uy * ca
        out.append((cx + rx, cy + ry))
    return out


# Pharaonic head in left-facing profile, unit box, y down. The facial landmarks
# are deliberately exaggerated -- at animatic scale, under blur, a naturally
# proportioned profile just reads as a boulder.
HEAD = [
    (0.32, 0.000),   # nemes crown, back
    (0.74, 0.030),
    (0.88, 0.280),   # back of headdress
    (0.90, 0.600),
    (0.82, 0.860),
    (0.64, 0.970),   # nemes lappet, bottom
    (0.38, 0.970),
    (0.30, 0.900),   # jaw
    (0.215, 0.800),  # chin
    (0.245, 0.745),
    (0.195, 0.718),  # lower lip
    (0.230, 0.692),  # mouth line
    (0.190, 0.663),  # upper lip
    (0.238, 0.628),  # philtrum
    (0.070, 0.575),  # nose tip
    (0.205, 0.522),  # bridge
    (0.140, 0.472),  # brow ridge
    (0.216, 0.432),
    (0.200, 0.300),  # forehead
    (0.160, 0.160),  # front edge of the nemes
    (0.240, 0.045),
]

EYE_U = (0.300, 0.500)   # unit-space eye position on HEAD

PYLON = [(0.12, 1.0), (0.20, 0.18), (0.80, 0.18), (0.88, 1.0)]


def tpoint(p, cx, cy, sx, sy=None, rot=0.0):
    return tform([p], cx, cy, sx, sy, rot)[0]


def draw_head(mc, cx, cy, sx, sy=None, rot=0.0):
    mc.d.polygon([(p[0] * HW, p[1] * HH)
                  for p in tform(HEAD, cx, cy, sx, sy, rot)], fill=255)


def head_detail(cx, cy, sx, sy=None, rot=0.0):
    """Carved relief -- nemes stripes, eye, brow, lip. Returned as its own mask
    so it can be clipped to the silhouette and lit a shade lighter than it.
    The stripes do most of the work: they are what makes a dark polygon read as
    Egyptian statuary rather than a rock."""
    mc = MaskCanvas()

    def ln(pts, w):
        mc.d.line([(p[0] * HW, p[1] * HH) for p in tform(pts, cx, cy, sx, sy, rot)],
                  fill=255, width=w)

    for i in range(6):
        u = 0.105 + i * 0.072
        ln([(0.30, u + 0.018), (0.88, u)], 4)
    eye = [(0.248, 0.506), (0.300, 0.479), (0.362, 0.497), (0.300, 0.528)]
    mc.d.polygon([(p[0] * HW, p[1] * HH)
                  for p in tform(eye, cx, cy, sx, sy, rot)], fill=255)
    ln([(0.225, 0.455), (0.395, 0.432)], 5)
    ln([(0.198, 0.693), (0.302, 0.686)], 4)
    return mc


def draw_diver(mc, cx, cy, s, flip=False):
    d = -1 if flip else 1
    body = [
        (cx, cy - 0.055 * s), (cx + d * 0.030 * s, cy - 0.030 * s),
        (cx + d * 0.034 * s, cy + 0.030 * s), (cx + d * 0.012 * s, cy + 0.075 * s),
        (cx - d * 0.020 * s, cy + 0.070 * s), (cx - d * 0.026 * s, cy + 0.010 * s),
        (cx - d * 0.030 * s, cy - 0.030 * s),
    ]
    mc.d.polygon([(p[0] * HW, p[1] * HH) for p in body], fill=255)
    r = 0.020 * s
    mc.d.ellipse(
        [(cx - r) * HW, (cy - 0.075 * s - r) * HH, (cx + r) * HW, (cy - 0.075 * s + r) * HH],
        fill=255,
    )
    mc.d.line(
        [((cx - d * 0.02 * s) * HW, (cy + 0.075 * s) * HH),
         ((cx - d * 0.075 * s) * HW, (cy + 0.115 * s) * HH)],
        fill=255, width=max(2, int(3 * s)),
    )


def draw_columns(mc, items):
    for cx, cy, w, h in items:
        mc.rect(cx - w / 2, cy - h, cx + w / 2, cy)
        mc.rect(cx - w * 0.78, cy - h - 0.022, cx + w * 0.78, cy - h)


# --------------------------------------------------------------- particulate

RNG = np.random.default_rng(11)
NP_ = 260
P_X = RNG.uniform(-0.1, 1.1, NP_).astype(np.float32)
P_Y = RNG.uniform(0.0, 1.0, NP_).astype(np.float32)
P_L = RNG.uniform(0.25, 1.0, NP_).astype(np.float32)   # depth layer
P_S = RNG.uniform(0.35, 1.6, NP_).astype(np.float32)   # size
P_PH = RNG.uniform(0, 6.28, NP_).astype(np.float32)


def particulate(img, t, rise=0.012, color=(0.65, 0.85, 1.0), gain=0.10, drift=0.006):
    ys = (P_Y - t * rise * P_L) % 1.0
    xs = P_X + np.sin(t * 0.5 + P_PH) * drift * P_L
    for i in range(NP_):
        add_sprite(
            img, xs[i] * W, ys[i] * H, SPRITE, color,
            gain * P_L[i] ** 2, scale=P_S[i],
        )


# ------------------------------------------------------------------- captions

_font_cache = {}


def font(path, size):
    key = (path, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def draw_text(img, text, size, y_frac, alpha, tracking=0, color=(255, 255, 255),
              stroke=10, stroke_fill=(0, 0, 0), scale=1.0, rot=0.0, font_path=FONT_BOLD):
    if alpha <= 0.003:
        return img
    f = font(font_path, max(8, int(size * scale)))
    # Composite only the rows the text can touch -- a full-frame RGBA layer per
    # caption was costing more than the whole scene render.
    half = int(size * scale * 2.2 + abs(rot) * 12 + stroke * scale + 40)
    y_px = int(y_frac * H)
    b0, b1 = max(0, y_px - half), min(H, y_px + half)
    layer = Image.new("RGBA", (W, b1 - b0), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    ly = y_px - b0
    tr = int(tracking * scale)
    if tr:
        widths = [d.textlength(ch, font=f) + tr for ch in text]
        total = sum(widths) - tr
        x = (W - total) / 2
        for ch, wch in zip(text, widths):
            d.text((x, ly), ch, font=f, fill=color + (255,), anchor="lm",
                   stroke_width=int(stroke * scale), stroke_fill=stroke_fill + (255,))
            x += wch
    else:
        d.text((W / 2, ly), text, font=f, fill=color + (255,), anchor="mm",
               stroke_width=int(stroke * scale), stroke_fill=stroke_fill + (255,))
    if rot:
        layer = layer.rotate(rot, resample=Image.BILINEAR, center=(W / 2, ly))
    arr = np.asarray(layer, dtype=np.float32) / 255.0
    a = arr[..., 3:4] * alpha
    band = img[b0:b1]
    band *= (1.0 - a)
    band += arr[..., :3] * a
    return img


# text, size, y, t_in, t_out, tracking, punch(bool)
CAPTIONS = [
    ("ATLANTIS ISN'T",      96, 0.30, 0.10, 1.70, 0, True),
    ("THE STORY",           96, 0.355, 0.10, 1.70, 0, True),
    ("30 FEET DOWN",       104, 0.32, 2.05, 3.90, 0, True),
    ("THONIS-HERACLEION",   64, 0.31, 11.25, 13.60, 14, False),
    ("BEFORE ALEXANDRIA",   72, 0.32, 15.95, 17.90, 4, True),
    ("2000 AD",            132, 0.31, 26.20, 28.30, 8, True),
    ("16 FT GODS",         104, 0.31, 34.10, 35.95, 0, True),
    ("64 SHIPWRECKS",       96, 0.31, 36.10, 37.95, 0, True),
    ("THE NAME,",           88, 0.30, 40.10, 42.00, 0, True),
    ("CARVED IN STONE",     88, 0.355, 40.10, 42.00, 0, True),
    ("THE GROUND",          92, 0.235, 44.00, 47.60, 0, True),
    ("TURNED TO LIQUID",    92, 0.29, 44.00, 47.60, 0, True),
    ("95% STILL BURIED",    98, 0.31, 52.00, 54.35, 0, True),
]


def captions(img, t):
    for text, size, y, t0, t1, tr, punch in CAPTIONS:
        if not (t0 - 0.2 <= t <= t1 + 0.35):
            continue
        fin = seg(t, t0, t0 + 0.12)
        fout = 1.0 - seg(t, t1, t1 + 0.22)
        a = smooth(fin) * smooth(fout)
        s = 1.0
        if punch:
            s = lerp(1.10, 1.0, smooth(seg(t, t0, t0 + 0.16)))
        img = draw_text(img, text, size, y, a, tracking=tr, scale=s)
    return img


# ------------------------------------------------------------------- scenes

def sc_hook(u, t):
    """0-4  Atlantis flash, then the plunge."""
    if t < 0.42:
        # golden concentric-ring city, top down
        img = vgrad((0.95, 0.78, 0.42), (0.55, 0.36, 0.14), 1.2)
        d = np.sqrt((xx - 0.5) ** 2 + ((yy - 0.5) * (H / W)) ** 2)
        rings = 0.5 + 0.5 * np.sin(d * 95.0)
        img += (rings * np.exp(-d * 3.2))[..., None] * np.array(
            [0.55, 0.42, 0.16], dtype=np.float32
        )
        img *= VIGNETTE
        img += smooth(seg(t, 0.30, 0.42)) * 1.4  # blow out to white
        return img

    p = seg(t, 0.42, 4.0)
    depth = smooth(p)
    top = lerp(np.array([0.42, 0.80, 0.88]), np.array([0.05, 0.20, 0.34]), depth)
    bot = lerp(np.array([0.06, 0.28, 0.44]), np.array([0.005, 0.02, 0.06]), depth)
    img = vgrad(top, bot, lerp(0.9, 1.7, depth))

    # surface caustics band sliding up out of frame as we descend
    surf_y = -0.05 + (1.0 - depth) * 0.55
    band = np.exp(-((yy - surf_y) ** 2) / (2 * 0.10 ** 2))
    caus = 0.5 + 0.5 * np.sin(xx * 46.0 + t * 2.2) * np.sin(xx * 17.0 - t * 1.3)
    img += (band * caus * lerp(0.55, 0.02, depth))[..., None] * np.array(
        [0.55, 0.85, 0.95], dtype=np.float32
    )

    rays = god_rays(t, n=7, tilt=0.16, width=0.05, top_fade=lerp(1.1, 2.4, depth))
    img += rays[..., None] * np.array([0.16, 0.34, 0.42], dtype=np.float32) * lerp(
        0.85, 0.35, depth
    )

    # rising bubbles read as downward camera travel
    for i in range(NP_):
        ys = (P_Y[i] - t * (0.16 + 0.42 * P_L[i])) % 1.0
        add_sprite(img, (P_X[i] + math.sin(t + P_PH[i]) * 0.01) * W, ys * H,
                   SPRITE, (0.7, 0.92, 1.0), 0.16 * P_L[i] ** 2, scale=P_S[i] * 1.2)

    img *= VIGNETTE
    img += (1.0 - smooth(seg(t, 0.42, 0.62))) * 1.2  # tail of the white flash
    return img


def sc_reveal(u, t):
    """4-11  statue head emerges from silt."""
    img = vgrad((0.05, 0.22, 0.33), (0.004, 0.02, 0.05), 1.5)
    rays = god_rays(t, n=5, seed=9, tilt=0.10, width=0.075, top_fade=1.9)
    img += rays[..., None] * np.array([0.10, 0.24, 0.30], dtype=np.float32)

    z = lerp(1.0, 1.28, smooth(u))       # slow push in
    reveal = smooth(seg(t, 4.4, 8.6))

    mc = MaskCanvas()
    draw_columns(mc, [(0.13, 1.02, 0.085, 0.40), (0.90, 1.02, 0.075, 0.28)])
    mc.poly(tform([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
                  0.93, 0.70, 0.34 * z, 0.14 * z, -0.06))
    draw_head(mc, 0.44, 0.575, 0.78 * z, 0.48 * z, 0.14)
    silt = [(0.0, 0.80), (0.22, 0.755), (0.52, 0.795), (0.78, 0.75), (1.0, 0.785),
            (1.0, 1.05), (0.0, 1.05)]
    mc.poly(silt)
    m = mc.out(blur=2.0)
    img = over(img, m, (0.030, 0.075, 0.105), alpha=lerp(0.55, 1.0, reveal))
    hd = head_detail(0.44, 0.575, 0.78 * z, 0.48 * z, 0.14).out(blur=1.2) * m
    img = over(img, hd, (0.105, 0.195, 0.225), alpha=0.85 * reveal)

    # rim light along the head so it separates from the murk
    mr = MaskCanvas()
    mr.poly(tform(HEAD, 0.44 - 0.013, 0.575 - 0.011, 0.78 * z, 0.48 * z, 0.14))
    rim = np.clip(mr.out(blur=3.0) - m, 0.0, 1.0)
    img += rim * np.array([0.30, 0.52, 0.55], dtype=np.float32) * reveal

    # distant diver for scale
    if t > 5.4:
        md = MaskCanvas()
        dy = 0.235 + math.sin(t * 0.7) * 0.006
        draw_diver(md, 0.775, dy, 1.0)
        dm = md.out(blur=1.2)
        img = over(img, dm, (0.02, 0.05, 0.08), alpha=0.9 * smooth(seg(t, 5.4, 6.2)))
        add_sprite(img, 0.735 * W, (dy + 0.012) * H, SPRITE_BIG, (0.55, 0.80, 0.85),
                   0.30 * smooth(seg(t, 5.4, 6.2)), scale=0.9)

    particulate(img, t, gain=0.13)
    haze = (1.0 - reveal) * 0.42
    img = img * (1.0 - haze) + np.array([0.05, 0.17, 0.24], dtype=np.float32) * haze
    img *= VIGNETTE
    return img


def sc_name(u, t):
    """11-18  ruins, then the flip to the living golden city."""
    flip = smooth(seg(t, 12.9, 13.5))
    flash = math.exp(-((t - 13.15) ** 2) / (2 * 0.16 ** 2))

    cold = vgrad((0.05, 0.20, 0.30), (0.004, 0.02, 0.05), 1.5)
    warm = vgrad((0.99, 0.80, 0.48), (0.30, 0.20, 0.20), 1.25)

    mc = MaskCanvas()
    if flip < 0.5:
        mc.poly(tform(PYLON, 0.50, 0.66, 0.62, 0.40, 0.0))
        draw_columns(mc, [(0.16, 0.86, 0.07, 0.24), (0.85, 0.88, 0.065, 0.19)])
        mc.poly([(0.0, 0.86), (0.35, 0.83), (0.7, 0.865), (1.0, 0.84), (1.0, 1.05), (0.0, 1.05)])
        m = mc.out(blur=2.0)
        img = over(cold, m, (0.025, 0.065, 0.09), 1.0)
        rays = god_rays(t, n=5, seed=4, top_fade=1.8)
        img += rays[..., None] * np.array([0.09, 0.22, 0.28], dtype=np.float32)
        particulate(img, t, gain=0.11)
    else:
        # sun + harbour
        d = np.sqrt((xx - 0.5) ** 2 + ((yy - 0.20) * (H / W)) ** 2)
        img = warm + np.exp(-d * 7.0)[..., None] * np.array(
            [0.55, 0.40, 0.14], dtype=np.float32
        )
        mc.poly(tform(PYLON, 0.50, 0.60, 0.50, 0.34, 0.0))
        draw_columns(mc, [(0.12, 0.80, 0.055, 0.17), (0.88, 0.80, 0.055, 0.17)])
        # quay
        mc.poly([(0.0, 0.845), (1.0, 0.845), (1.0, 1.05), (0.0, 1.05)])
        # ships with triangular sails
        for i, (sx, sy, ss) in enumerate([(0.22, 0.80, 1.0), (0.62, 0.815, 0.85),
                                          (0.86, 0.79, 0.7)]):
            bob = math.sin(t * 1.4 + i) * 0.004
            hull = [(sx - 0.10 * ss, sy + bob), (sx + 0.10 * ss, sy + bob),
                    (sx + 0.06 * ss, sy + 0.035 * ss + bob),
                    (sx - 0.06 * ss, sy + 0.035 * ss + bob)]
            mc.d.polygon([(p[0] * HW, p[1] * HH) for p in hull], fill=255)
            sail = [(sx, sy - 0.115 * ss + bob), (sx + 0.062 * ss, sy - 0.005 * ss + bob),
                    (sx - 0.012 * ss, sy - 0.005 * ss + bob)]
            mc.d.polygon([(p[0] * HW, p[1] * HH) for p in sail], fill=255)
        # crowd on the quay
        for i in range(26):
            px = 0.03 + (i / 25.0) * 0.94
            ph = 0.021 + 0.006 * math.sin(i * 2.1)
            mc.rect(px - 0.006, 0.845 - ph, px + 0.006, 0.845)
        m = mc.out(blur=1.4)
        img = over(img, m, (0.20, 0.11, 0.07), 0.92)
        for i in range(90):
            add_sprite(img, ((i * 0.107 + t * 0.01) % 1.0) * W,
                       (0.30 + 0.55 * ((i * 0.31) % 1.0)) * H, SPRITE,
                       (1.0, 0.85, 0.5), 0.07, scale=0.8)

    img *= VIGNETTE
    img += flash * 1.5
    return img


def sc_myth(u, t):
    """18-26  manuscript, Helen, the MYTH stamp."""
    img = vgrad((0.86, 0.76, 0.56), (0.42, 0.33, 0.22), 1.35)
    grain = RNG.standard_normal((H // 8, W // 8)).astype(np.float32)
    grain = np.asarray(Image.fromarray(((grain * 0.5 + 0.5) * 255).clip(0, 255).astype(
        np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    img *= (0.86 + 0.28 * grain)[..., None]

    d = np.sqrt((xx - 0.5) ** 2 + ((yy - 0.42) * (H / W)) ** 2)
    img *= np.clip(1.25 - d * 1.5, 0.15, 1.25)[..., None]

    # ink lines writing on
    mc = MaskCanvas()
    lines = 13
    prog = seg(t, 18.15, 21.6) * lines
    for i in range(lines):
        f = np.clip(prog - i, 0.0, 1.0)
        if f <= 0:
            break
        y = 0.30 + i * 0.030
        wob = 0.004 * math.sin(i * 3.0)
        x1 = 0.20 + f * (0.58 + 0.06 * math.sin(i * 1.7))
        mc.line([(0.20, y + wob), (x1, y + wob)], width=4)

    if t > 21.3:
        a = smooth(seg(t, 21.3, 21.9))
        cx, cy = 0.50, 0.62
        # ship + staircase + Helen silhouette
        mc.poly([(cx - 0.30, cy + 0.10), (cx + 0.02, cy + 0.10),
                 (cx - 0.02, cy + 0.16), (cx - 0.26, cy + 0.16)])
        for s in range(5):
            mc.rect(cx + 0.04 + s * 0.014, cy + 0.10 - s * 0.026,
                    cx + 0.34, cy + 0.10 - s * 0.026 + 0.026)
        fig = [(cx - 0.005, cy - 0.075), (cx + 0.026, cy - 0.040),
               (cx + 0.034, cy + 0.045), (cx + 0.050, cy + 0.105),
               (cx - 0.052, cy + 0.105), (cx - 0.034, cy + 0.040),
               (cx - 0.030, cy - 0.040)]
        mc.d.polygon([(p[0] * HW, p[1] * HH) for p in fig], fill=int(255 * a))
        mc.d.ellipse([(cx - 0.020) * HW, (cy - 0.115) * HH,
                      (cx + 0.020) * HW, (cy - 0.062) * HH], fill=int(255 * a))

    m = mc.out(blur=1.1)
    img = over(img, m, (0.09, 0.06, 0.045), 0.95)

    # burning edges after the stamp
    if t > 22.15:
        b = smooth(seg(t, 22.15, 25.9))
        ex = np.broadcast_to(xx, (H, W))
        ey = np.broadcast_to(yy, (H, W))
        edge = np.maximum.reduce([
            1.0 - ey / 0.30, 1.0 - (1.0 - ey) / 0.30,
            1.0 - ex / 0.22, 1.0 - (1.0 - ex) / 0.22,
        ])
        edge = np.clip(edge, 0.0, 1.0) ** 1.4
        burn = np.clip(edge - (1.0 - b) * 1.15, 0.0, 1.0)
        img = img * (1.0 - burn[..., None]) + np.array(
            [0.05, 0.02, 0.01], dtype=np.float32) * burn[..., None]
        img += (np.clip(burn * 6.0, 0, 1) * np.clip(1.0 - burn * 4.5, 0, 1))[
            ..., None] * np.array([1.0, 0.32, 0.05], dtype=np.float32) * 1.5

    img *= VIGNETTE
    if 21.95 <= t <= 25.6:
        s = lerp(3.2, 1.0, smooth(seg(t, 21.98, 22.13)))
        a = smooth(seg(t, 21.95, 22.0)) * (1.0 - smooth(seg(t, 25.2, 25.6)))
        img = draw_text(img, "MYTH", 150, 0.44, a, tracking=12, scale=s, rot=-9,
                        color=(198, 26, 26), stroke=6, stroke_fill=(58, 6, 6))
        img += math.exp(-((t - 22.02) ** 2) / (2 * 0.05 ** 2)) * 0.55
    return img


def sc_discovery(u, t):
    """26-34  vessel, sonar contact, diver torch."""
    if t < 28.4:
        p = seg(t, 26.0, 28.4)
        img = vgrad((0.36, 0.42, 0.47), (0.05, 0.08, 0.12), 1.25)
        hz = 0.50
        img = np.where((yy > hz)[..., None],
                       vgrad((0.10, 0.15, 0.20), (0.02, 0.035, 0.06), 1.0), img)
        for i in range(70):
            wy = hz + 0.005 + ((i * 0.0134) % 0.47)
            amp = 0.5 + 0.5 * math.sin(i * 2.3 + t * 1.6)
            add_sprite(img, ((i * 0.0917 + math.sin(t * 0.6 + i) * 0.02) % 1.0) * W,
                       wy * H, SPRITE, (0.42, 0.50, 0.56), 0.06 * amp, scale=2.6)
        mc = MaskCanvas()
        sx = 0.5 + lerp(-0.04, 0.02, p)
        bob = math.sin(t * 1.1) * 0.004
        mc.poly([(sx - 0.16, hz - 0.012 + bob), (sx + 0.16, hz - 0.012 + bob),
                 (sx + 0.12, hz + 0.022 + bob), (sx - 0.12, hz + 0.022 + bob)])
        mc.rect(sx - 0.05, hz - 0.062 + bob, sx + 0.03, hz - 0.012 + bob)
        mc.line([(sx + 0.06, hz - 0.012 + bob), (sx + 0.06, hz - 0.10 + bob)], width=3)
        img = over(img, mc.out(blur=1.0), (0.015, 0.025, 0.04), 0.95)
        img *= VIGNETTE
        return img

    if t < 31.6:
        p = seg(t, 28.4, 31.6)
        img = np.zeros((H, W, 3), dtype=np.float32) + np.array(
            [0.004, 0.020, 0.010], dtype=np.float32)
        cx, cy = 0.5, 0.42
        r = np.sqrt((xx - cx) ** 2 + ((yy - cy) * (H / W)) ** 2)
        scope = (r < 0.46).astype(np.float32)
        img += (scope * 0.05)[..., None] * np.array([0.05, 0.5, 0.22], dtype=np.float32)
        for rr in (0.115, 0.23, 0.345, 0.46):
            img += (np.exp(-((r - rr) ** 2) / (2 * 0.0022 ** 2)) * 0.5)[
                ..., None] * np.array([0.06, 0.55, 0.25], dtype=np.float32)
        ang = np.arctan2((yy - cy) * (H / W), xx - cx)
        sweep = (t - 28.4) * 2.3
        rel = (ang + sweep * 2 * math.pi) % (2 * math.pi)
        trail = np.exp(-rel * 2.3) * scope
        img += trail[..., None] * np.array([0.10, 0.95, 0.42], dtype=np.float32) * 0.8
        # the contact resolves on the third pass
        if t > 30.0:
            a = smooth(seg(t, 30.0, 30.5))
            rect = ((np.abs(xx - 0.545) < 0.115) & (np.abs(yy - 0.455) < 0.052)).astype(
                np.float32)
            img += rect[..., None] * np.array(
                [0.25, 1.0, 0.45], dtype=np.float32) * (0.35 + 0.28 * math.sin(t * 14)) * a
        noise = RNG.standard_normal((H // 6, W // 6)).astype(np.float32)
        noise = np.asarray(Image.fromarray(((noise * .5 + .5) * 255).clip(0, 255).astype(
            np.uint8)).resize((W, H), Image.NEAREST), dtype=np.float32) / 255.0
        img += (noise * scope * 0.05)[..., None]
        img *= VIGNETTE
        return img

    p = seg(t, 31.6, 34.0)
    img = vgrad((0.012, 0.035, 0.055), (0.0, 0.005, 0.012), 1.6)
    bx = 0.36 + 0.16 * p
    by = 0.30 + 0.05 * math.sin(t * 1.2)
    ang = np.arctan2((yy - by) * (H / W), xx - bx)
    r = np.sqrt((xx - bx) ** 2 + ((yy - by) * (H / W)) ** 2)
    cone = np.exp(-((ang - 1.15) ** 2) / (2 * 0.30 ** 2)) * np.exp(-r * 2.0)
    img += cone[..., None] * np.array([0.35, 0.62, 0.72], dtype=np.float32) * 1.2
    mc = MaskCanvas()
    draw_diver(mc, bx, by, 2.6)
    img = over(img, mc.out(blur=1.4), (0.0, 0.01, 0.02), 0.95)
    add_sprite(img, bx * W, by * H, SPRITE_BIG, (0.7, 0.95, 1.0), 0.9, scale=1.6)
    particulate(img, t, gain=0.16)
    img *= VIGNETTE
    return img


def sc_haul(u, t):
    """34-42  four rapid artifact cuts."""
    img = vgrad((0.035, 0.13, 0.19), (0.0, 0.012, 0.03), 1.5)
    k = int((t - 34.0) // 2.0)
    lt = (t - 34.0) - k * 2.0
    orb = math.sin(lt * 0.9) * 0.03

    mc = MaskCanvas()
    if k == 0:      # colossus raised on cables
        rise = smooth(lt / 2.0)
        cy = 0.72 - rise * 0.12
        draw_head(mc, 0.5 + orb, cy - 0.20, 0.40, 0.26, 0.04)
        head_det = (0.5 + orb, cy - 0.20, 0.40, 0.26, 0.04)
        mc.rect(0.40 + orb, cy - 0.10, 0.60 + orb, cy + 0.28)
        mc.line([(0.44 + orb, cy - 0.16), (0.40, -0.05)], width=3)
        mc.line([(0.56 + orb, cy - 0.16), (0.60, -0.05)], width=3)
    elif k == 1:    # shipwreck ribs
        for i in range(9):
            x = 0.13 + i * 0.092 + orb
            h = 0.070 + 0.020 * math.sin(i * 1.3)
            mc.d.arc([(x - 0.062) * HW, (0.735 - h) * HH,
                      (x + 0.062) * HW, (0.735 + h) * HH],
                     start=180, end=360, fill=255, width=8)
        mc.poly([(0.0, 0.755), (1.0, 0.742), (1.0, 1.05), (0.0, 1.05)])
    elif k == 2:    # gold in a diver's glove
        for i in range(16):
            a = i * 0.9 + t
            cx = 0.5 + 0.15 * math.cos(a) + orb
            cy = 0.55 + 0.10 * math.sin(a * 1.3)
            rr = 0.020 + 0.008 * math.sin(i * 2.0)
            mc.d.ellipse([(cx - rr) * HW, (cy - rr * (W / H) * 1.9) * HH,
                          (cx + rr) * HW, (cy + rr * (W / H) * 1.9) * HH], fill=255)
        glove = MaskCanvas()
        glove.poly([(0.20, 0.80), (0.80, 0.78), (0.72, 1.05), (0.26, 1.05)])
    else:           # the stele
        z = lerp(1.0, 1.18, smooth(lt / 2.0))
        mc.poly(tform([(0.18, 0.06), (0.82, 0.06), (0.86, 1.0), (0.14, 1.0)],
                      0.5 + orb * 0.4, 0.55, 0.62 * z, 0.74 * z))
    m = mc.out(blur=1.5)

    if k == 2:
        img = over(img, m, (0.85, 0.66, 0.20), 0.95)
        img = over(img, glove.out(blur=1.6), (0.035, 0.065, 0.080), 0.95)
        for i in range(14):
            a = i * 0.9 + t
            add_sprite(img, (0.5 + 0.15 * math.cos(a) + orb) * W,
                       (0.55 + 0.10 * math.sin(a * 1.3)) * H, SPRITE_BIG,
                       (1.0, 0.85, 0.45),
                       0.35 + 0.35 * math.sin(t * 9 + i * 2.0) ** 2, scale=0.8)
    else:
        img = over(img, m, (0.045, 0.085, 0.10), 0.95)
        mr = MaskCanvas()
        if k == 0:
            mr.poly(tform(HEAD, 0.5 + orb - 0.011, 0.72 - smooth(lt / 2.0) * 0.12 - 0.21,
                          0.40, 0.26, 0.04))
        elif k == 3:
            mr.poly(tform([(0.18, 0.06), (0.82, 0.06), (0.86, 1.0), (0.14, 1.0)],
                          0.5 + orb * 0.4 - 0.012, 0.545, 0.62, 0.74))
        else:
            for i in range(9):
                x = 0.13 + i * 0.092 + orb - 0.006
                h = 0.070 + 0.020 * math.sin(i * 1.3)
                mr.d.arc([(x - 0.062) * HW, (0.730 - h) * HH,
                          (x + 0.062) * HW, (0.730 + h) * HH],
                         start=180, end=360, fill=255, width=8)
        img += np.clip(mr.out(blur=3.0) - m, 0, 1) * np.array(
            [0.40, 0.62, 0.68], dtype=np.float32) * 1.1
        if k == 3:   # hieroglyph grid
            gl = MaskCanvas()
            for r in range(9):
                for c in range(5):
                    if (r * 5 + c) % 3 == 0:
                        continue
                    gx = 0.30 + c * 0.10
                    gy = 0.22 + r * 0.065
                    gl.rect(gx - 0.018, gy - 0.012, gx + 0.018, gy + 0.012)
            img += gl.out(blur=1.0) * m * np.array(
                [0.30, 0.40, 0.42], dtype=np.float32) * 1.3
        if k == 0:
            img = over(img, head_detail(*head_det).out(blur=1.0) * m,
                       (0.11, 0.20, 0.23), 0.85)

    particulate(img, t, gain=0.14)
    img *= VIGNETTE
    flash = math.exp(-((lt - 0.02) ** 2) / (2 * 0.055 ** 2))
    img += flash * 0.85
    return img


def sc_sink(u, t):
    """42-50  liquefaction; the temple drops straight down."""
    sink = smooth(seg(t, 44.6, 48.6)) ** 1.25
    quake = seg(t, 44.0, 44.9) * (1.0 - seg(t, 48.6, 49.4))
    shake = math.sin(t * 47.0) * 0.010 * quake

    img = vgrad((0.16, 0.14, 0.18), (0.40, 0.30, 0.24), 0.85)
    img *= np.clip(1.15 - np.abs(yy - 0.25) * 0.8, 0.3, 1.15)[..., None]

    ground = 0.80
    mc = MaskCanvas()
    # rippling ground plane
    gpts = []
    for i in range(41):
        gx = i / 40.0
        ripple = math.sin(gx * 22.0 - t * 7.0) * 0.012 * quake
        gpts.append((gx, ground + ripple + sink * 0.02))
    gpts += [(1.0, 1.06), (0.0, 1.06)]
    mc.d.polygon([(p[0] * HW, p[1] * HH) for p in gpts], fill=255)
    gm = mc.out(blur=1.6)

    tm = MaskCanvas()
    ty = 0.60 + sink * 0.46 + shake
    tm.poly(tform(PYLON, 0.5 + shake, ty, 0.66, 0.44))
    draw_columns(tm, [(0.185 + shake, ty + 0.21, 0.055, 0.17),
                      (0.815 + shake, ty + 0.21, 0.055, 0.17)])
    tmm = tm.out(blur=1.4)

    img = over(img, tmm, (0.12, 0.10, 0.10), 1.0)
    img = over(img, gm, (0.20, 0.15, 0.12), 1.0)

    # cracks
    for ci, (ct, cx0) in enumerate([(45.0, 0.30), (46.2, 0.68), (47.4, 0.48)]):
        if t < ct:
            continue
        a = smooth(seg(t, ct, ct + 0.25))
        cm = MaskCanvas()
        pts = [(cx0, ground)]
        for s in range(6):
            pts.append((cx0 + (s + 1) * 0.035 * (1 if s % 2 else -1) * (1 + s * 0.2),
                        ground + (s + 1) * 0.028))
        cm.line(pts, width=max(2, int(6 * a)))
        img = over(img, cm.out(blur=1.0), (0.02, 0.015, 0.015), a * 0.9)

    # dust
    for i in range(120):
        ph = i * 0.7
        life = ((t - 44.4 + i * 0.02) * 0.45) % 1.0
        if t < 44.4:
            break
        px = 0.5 + math.cos(ph) * (0.10 + life * 0.55)
        py = ground - life * 0.28 + math.sin(ph * 2) * 0.03
        add_sprite(img, px * W, py * H, SPRITE_BIG, (0.50, 0.42, 0.34),
                   0.22 * (1.0 - life), scale=1.3 + life * 2.2)

    # water floods in
    if t > 47.6:
        wl = 1.06 - smooth(seg(t, 47.6, 50.0)) * 0.92
        wm = (yy > wl).astype(np.float32)[..., None]
        foam = np.exp(-((yy - wl) ** 2) / (2 * 0.012 ** 2))[..., None]
        img = img * (1.0 - wm * 0.88) + np.array(
            [0.02, 0.10, 0.18], dtype=np.float32) * wm * 0.88
        img += foam * np.array([0.45, 0.65, 0.75], dtype=np.float32) * 0.55

    img *= VIGNETTE
    return img


def sc_loop(u, t):
    """50-58  pull back into the dark, then the eye."""
    img = vgrad((0.012, 0.045, 0.070), (0.0, 0.004, 0.012), 1.7)

    if t < 56.6:
        p = seg(t, 50.0, 56.6)
        sc = lerp(1.0, 0.34, smooth(p))         # torch recedes
        bx, by = 0.5, 0.50
        r = np.sqrt((xx - bx) ** 2 + ((yy - by) * (H / W)) ** 2)
        ang = np.arctan2((yy - by) * (H / W), xx - bx)
        cone = np.exp(-((ang - 0.9) ** 2) / (2 * (0.34 * sc + 0.06) ** 2)) * np.exp(
            -r * (2.0 / sc))
        img += cone[..., None] * np.array([0.30, 0.55, 0.65], dtype=np.float32) * 1.15

        mc = MaskCanvas()
        draw_columns(mc, [(0.30, 0.74, 0.05 * sc, 0.16 * sc),
                          (0.66, 0.755, 0.045 * sc, 0.13 * sc)])
        draw_head(mc, 0.46, 0.70, 0.24 * sc, 0.155 * sc, 0.4)
        lit = np.exp(-((ang - 0.9) ** 2) / (2 * 0.40 ** 2)) * np.exp(-r * 1.6)
        img = over(img, mc.out(blur=1.6) * lit[..., None] * 1.6, (0.05, 0.11, 0.13), 0.9)

        md = MaskCanvas()
        draw_diver(md, bx, by, 1.9 * sc)
        img = over(img, md.out(blur=1.2), (0.0, 0.008, 0.015), 0.95)
        add_sprite(img, bx * W, by * H, SPRITE_BIG, (0.65, 0.9, 1.0), 0.75, scale=1.5 * sc)

        # survey grid: one filled corner against a vast unmapped field
        gm = MaskCanvas()
        for i in range(11):
            gy = 0.34 + i * 0.036
            gm.line([(0.10, gy), (0.90, gy)], width=1)
        for i in range(13):
            gx = 0.10 + i * 0.0667
            gm.line([(gx, 0.34), (gx, 0.70)], width=1)
        ga = 0.16 * smooth(seg(t, 51.6, 52.6)) * (1.0 - smooth(seg(t, 56.2, 56.6)))
        img += gm.out(blur=0.6) * np.array([0.20, 0.55, 0.65], dtype=np.float32) * ga
        fill = MaskCanvas()
        fill.rect(0.10, 0.34, 0.163, 0.376)
        img += fill.out(blur=0.8) * np.array(
            [0.30, 0.85, 0.95], dtype=np.float32) * ga * 2.6

        particulate(img, t, gain=0.10)
    else:
        p = smooth(seg(t, 56.6, 60.0))
        z = lerp(0.92, 1.06, p)
        mc = MaskCanvas()
        draw_head(mc, 0.56, 0.50, 0.95 * z, 0.62 * z, 0.06)
        m = mc.out(blur=2.4)
        img = over(img, m, (0.020, 0.045, 0.058), 1.0)
        mr = MaskCanvas()
        mr.poly(tform(HEAD, 0.56 - 0.013, 0.50 - 0.010, 0.95 * z, 0.62 * z, 0.06))
        img += np.clip(mr.out(blur=4.0) - m, 0, 1) * np.array(
            [0.22, 0.42, 0.48], dtype=np.float32) * lerp(1.0, 0.45, p)
        hd = head_detail(0.56, 0.50, 0.95 * z, 0.62 * z, 0.06).out(blur=1.4) * m
        img = over(img, hd, (0.085, 0.150, 0.175), 0.9)
        # the eye catches the light
        ex, ey = tpoint(EYE_U, 0.56, 0.50, 0.95 * z, 0.62 * z, 0.06)
        add_sprite(img, ex * W, ey * H, SPRITE_BIG, (0.85, 0.97, 1.0),
                   0.55 + 0.45 * smooth(seg(t, 56.8, 58.6)), scale=1.15)
        add_sprite(img, ex * W, ey * H, SPRITE_BIG, (0.55, 0.80, 0.95), 0.30, scale=3.4)
        particulate(img, t, gain=0.07)
        img *= lerp(1.0, 0.82, smooth(seg(t, 59.2, 60.0)))

    img *= VIGNETTE
    return img


SCENES = [
    (0.0, 4.0, sc_hook),
    (4.0, 11.0, sc_reveal),
    (11.0, 18.0, sc_name),
    (18.0, 26.0, sc_myth),
    (26.0, 34.0, sc_discovery),
    (34.0, 42.0, sc_haul),
    (42.0, 50.0, sc_sink),
    (50.0, 60.0, sc_loop),
]


def render_frame(i):
    t = i / FPS
    for a, b, fn in SCENES:
        if a <= t < b or (b >= DUR and t >= a):
            img = fn(seg(t, a, b), t)
            break
    else:
        img = np.zeros((H, W, 3), dtype=np.float32)

    img = captions(img, t)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------- audio

VO_LINES = [
    (0.30, 3.55, "Everyone calls Atlantis a fairy tale. Then divers found this. "
                 "Thirty feet under the Mediterranean."),
    (4.35, 6.30, "An entire Egyptian city. Temples, harbors, streets, "
                 "and giants, face down in the silt."),
    (11.30, 6.30, "This is Thonis Heracleion. Egypt's richest port, "
                  "centuries before Alexandria was even built."),
    (18.30, 7.20, "Herodotus wrote that Helen of Troy sheltered here. "
                  "For two thousand years, scholars called that a myth."),
    (26.35, 6.90, "Then in two thousand, Franck Goddio's sonar pinged "
                  "something impossible in Abu Qir Bay."),
    (34.20, 7.40, "Sixteen foot gods. Sixty four shipwrecks. Gold. "
                  "And a granite slab carved with the city's own name."),
    (42.30, 7.20, "It didn't drift under. The clay beneath it turned to liquid, "
                  "and the temples' own weight pulled them down."),
    (50.60, 4.05, "A quarter century of diving, and we've mapped "
                  "maybe five percent."),
    (54.90, 4.90, "Plato said a city vanished in a single night. "
                  "Egypt has the receipts."),
]


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        src = np.linspace(0.0, 1.0, len(raw))
        dst = np.linspace(0.0, 1.0, int(len(raw) * SR / sr))
        raw = np.interp(dst, src, raw).astype(np.float32)
    return raw


def synth_vo():
    """espeak-ng scratch VO, speed-fitted to each slot."""
    track = np.zeros(int(DUR * SR) + SR, dtype=np.float32)
    for idx, (t0, slot, text) in enumerate(VO_LINES):
        path = os.path.join(SCRATCH, f"vo_{idx}.wav")
        wpm = 168
        for _ in range(6):
            subprocess.run(
                ["espeak-ng", "-v", "en-us+m3", "-s", str(wpm), "-p", "22",
                 "-a", "170", "-g", "3", "-w", path, text],
                check=True, capture_output=True,
            )
            a = read_wav(path)
            dur = len(a) / SR
            if dur <= slot:
                break
            wpm = int(wpm * (dur / slot) * 1.02) + 1
        s = int(t0 * SR)
        a *= 1.0
        # short fades so espeak's hard edges don't click
        f = int(0.012 * SR)
        a[:f] *= np.linspace(0, 1, f)
        a[-f:] *= np.linspace(1, 0, f)
        track[s:s + len(a)] += a
    return track[:int(DUR * SR)]


def lp(x, taps):
    k = np.hanning(taps)
    k /= k.sum()
    return np.convolve(x, k, mode="same").astype(np.float32)


def synth_bed():
    n = int(DUR * SR)
    t = np.arange(n, dtype=np.float32) / SR
    bed = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)

    def at(t0, sig, gain=1.0):
        s = int(t0 * SR)
        e = min(n, s + len(sig))
        if s >= n:
            return
        bed[s:e] += sig[: e - s] * gain

    def env(ln, a=0.01, d=0.5, p=2.0):
        k = np.linspace(0, 1, ln, dtype=np.float32)
        atk = np.clip(k / max(a, 1e-6), 0, 1)
        dec = np.exp(-k * d * 10.0) ** (1 / p)
        return atk * dec

    def noise(dur, lp_taps=120, seedoff=0):
        ln = int(dur * SR)
        x = np.random.default_rng(7 + seedoff).standard_normal(ln).astype(np.float32)
        return lp(x, lp_taps)

    def padd(*sigs):
        """Sum signals of differing length, zero-padded to the longest."""
        ln = max(len(x) for x in sigs)
        acc = np.zeros(ln, dtype=np.float32)
        for x in sigs:
            acc[: len(x)] += x
        return acc

    def sine(f, dur, ph=0.0):
        ln = int(dur * SR)
        k = np.arange(ln, dtype=np.float32) / SR
        return np.sin(2 * math.pi * f * k + ph).astype(np.float32)

    def sweep(f0, f1, dur):
        ln = int(dur * SR)
        k = np.arange(ln, dtype=np.float32) / SR
        f = f0 * (f1 / f0) ** (k / max(dur, 1e-6))
        ph = 2 * math.pi * np.cumsum(f) / SR
        return np.sin(ph).astype(np.float32)

    # continuous drone bed
    drone = (0.34 * np.sin(2 * math.pi * 41.0 * t)
             + 0.20 * np.sin(2 * math.pi * 61.5 * t + 0.7)
             + 0.12 * np.sin(2 * math.pi * 82.0 * t + 1.9))
    lfo = 0.62 + 0.38 * np.sin(2 * math.pi * 0.09 * t)
    prof = np.interp(t, [0, 0.5, 4, 11, 18, 26, 31.5, 34, 42, 49.5, 50.0,
                         50.45, 55.5, 60],
                        [0, 0.55, 0.7, 0.62, 0.5, 0.62, 0.95, 0.7, 1.0, 1.0,
                         0.0, 0.0, 0.75, 0.95])
    bed += (drone * lfo * prof * 0.30).astype(np.float32)

    # water ambience
    amb = lp(rng.standard_normal(n).astype(np.float32), 700)
    amb_prof = np.interp(t, [0, 0.4, 4, 18, 26, 34, 42, 50, 60],
                            [0, 0.9, 1.0, 0.15, 0.5, 0.8, 0.6, 0.7, 0.7])
    bed += amb * amb_prof * 0.55

    # 0:00 whoosh into the plunge
    wh = noise(0.85, 40)
    wh *= np.linspace(0.05, 1.0, len(wh)) ** 2 * np.exp(
        -np.linspace(0, 1, len(wh)) * 1.2)
    at(0.02, wh, 0.85)
    at(0.42, noise(1.6, 900, 3) * env(int(1.6 * SR), 0.02, 0.35), 1.1)

    # bass drops
    for tt, g in ((0.44, 1.0), (2.55, 0.85)):
        d = sweep(140, 32, 1.5) * env(int(1.5 * SR), 0.004, 0.30)
        at(tt, d, 1.15 * g)

    # sonar ping on "giants"
    for tt, f, g in ((8.55, 880, 0.5), (26.55, 760, 0.55), (27.30, 900, 0.6),
                     (28.05, 1080, 0.7)):
        p = sine(f, 1.5) * env(int(1.5 * SR), 0.001, 0.55)
        p += 0.35 * np.pad(sine(f, 1.5) * env(int(1.5 * SR), 0.001, 0.7),
                           (int(0.22 * SR), 0))[: len(p)]
        at(tt, p, g)

    # oud-ish plucks + frame drum through the reconstruction
    scale_hz = [146.8, 155.6, 174.6, 196.0, 233.1, 261.6]
    for i, tt in enumerate(np.arange(13.35, 18.0, 0.42)):
        f0 = scale_hz[[0, 2, 3, 1, 4, 5, 3, 2, 0, 3, 4, 2][i % 12]]
        pl = sum((1.0 / (h + 1)) * sine(f0 * (h + 1), 1.1, ph=h)
                 for h in range(5)) * env(int(1.1 * SR), 0.002, 0.75)
        at(tt, pl, 0.20)
    for tt in np.arange(13.3, 18.0, 0.84):
        dr = (sine(58, 0.30) * env(int(0.30 * SR), 0.002, 1.6)
              + noise(0.30, 60, 11) * env(int(0.30 * SR), 0.001, 3.0) * 0.5)
        at(tt, dr, 0.42)

    # quill scratches
    for tt in np.arange(18.3, 21.5, 0.26):
        sc = noise(0.16, 12, int(tt * 7)) * env(int(0.16 * SR), 0.01, 2.2)
        at(tt, sc, 0.10)

    # stamp thud on "myth"
    at(22.00, sine(52, 1.0) * env(int(1.0 * SR), 0.001, 0.9), 1.0)
    at(22.00, noise(0.35, 25, 5) * env(int(0.35 * SR), 0.001, 2.5), 0.55)

    # riser into the discovery, hard hit on "impossible"
    rs = noise(5.0, 60, 8)
    k = np.linspace(0, 1, len(rs))
    rs *= (k ** 2.4)
    rs += sweep(180, 1500, 5.0) * (k ** 3) * 0.35
    at(28.10, rs, 0.55)
    at(31.65, sweep(160, 30, 1.8) * env(int(1.8 * SR), 0.003, 0.24), 1.3)
    at(31.65, noise(0.5, 30, 9) * env(int(0.5 * SR), 0.001, 2.0), 0.5)

    # percussive hit on each artifact cut
    for tt in (34.02, 36.02, 38.02, 40.02):
        hit = padd(sine(70, 0.5) * env(int(0.5 * SR), 0.001, 1.5),
                   noise(0.22, 18, int(tt)) * env(int(0.22 * SR), 0.001, 3.0) * 0.7)
        at(tt, hit, 0.85)
    # coin shimmer
    for i, tt in enumerate(np.arange(38.05, 40.0, 0.11)):
        sh = sine(1800 + (i * 337) % 1400, 0.45) * env(int(0.45 * SR), 0.001, 1.6)
        at(tt, sh, 0.10)
    # stone grind under the stele
    at(40.30, noise(1.7, 220, 12) * env(int(1.7 * SR), 0.08, 0.5), 0.55)

    # liquefaction: rumble, cracks, flood
    rum = noise(7.6, 1400, 13)
    kk = np.linspace(0, 1, len(rum))
    rum *= np.clip(kk * 2.2, 0, 1) * np.clip(1.6 - kk * 1.2, 0, 1)
    at(42.10, rum, 1.5)
    at(42.10, (sine(29, 7.6) * np.clip(np.linspace(0, 1, int(7.6 * SR)) * 2.0, 0, 1)),
       0.55)
    for tt in (45.0, 46.2, 47.4):
        cr = noise(0.7, 14, int(tt * 3)) * env(int(0.7 * SR), 0.001, 1.4)
        cr += sine(95, 0.7) * env(int(0.7 * SR), 0.002, 1.1) * 0.6
        at(tt, cr, 0.75)
    fl = noise(2.4, 90, 15)
    kf = np.linspace(0, 1, len(fl))
    fl *= np.clip(kf * 3.0, 0, 1) * np.clip(1.4 - kf * 1.4, 0, 1)
    at(47.60, fl, 1.0)

    # the 0:50 silence
    s0, s1 = int(49.88 * SR), int(50.45 * SR)
    bed[s0:s1] *= np.linspace(1.0, 0.0, s1 - s0) ** 2
    bed[int(50.0 * SR):s1] = 0.0

    # single low piano note out of the silence
    pn = sum((1.0 / (h + 1) ** 1.35) * sine(110.0 * (h + 1), 4.0, ph=h * 0.6)
             for h in range(7)) * env(int(4.0 * SR), 0.002, 0.30)
    at(50.48, pn, 0.85)

    # closing swell
    sw = sine(43.0, 4.6) + 0.5 * sine(64.5, 4.6)
    sw *= np.clip(np.linspace(0, 1, int(4.6 * SR)) * 1.6, 0, 1)
    at(54.70, sw, 0.55)
    at(54.70, noise(4.6, 500, 21) * np.clip(
        np.linspace(0, 1, int(4.6 * SR)) * 1.5, 0, 1), 0.35)

    return bed


def build_audio(path):
    vo = synth_vo()
    bed = synth_bed()
    n = min(len(vo), len(bed))
    vo, bed = vo[:n], bed[:n]

    # duck the bed under the voice
    envv = lp(np.abs(vo), 4000)
    envv /= max(envv.max(), 1e-6)
    duck = 1.0 - 0.55 * np.clip(envv * 3.0, 0, 1)
    mix = bed * duck * 0.85 + vo * 1.05

    mix = np.tanh(mix * 1.25) * 0.92
    mix /= max(np.abs(mix).max(), 1e-6)
    mix *= 0.94
    f = int(0.03 * SR)
    mix[:f] *= np.linspace(0, 1, f)
    mix[-f:] *= np.linspace(1, 0, f)

    st = np.stack([mix, mix], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    return path


# ----------------------------------------------------------------------- main

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "short.mp4")
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    apath = os.path.join(SCRATCH, "audio.wav")
    print("synthesising audio ...", flush=True)
    build_audio(apath)

    print(f"rendering {NFRAMES} frames ...", flush=True)
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", apath,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level", "4.2",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "-movflags", "+faststart", out,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NFRAMES):
        proc.stdin.write(render_frame(i).tobytes())
        if i % 150 == 0:
            print(f"  frame {i}/{NFRAMES}", flush=True)
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg exited {rc}")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
