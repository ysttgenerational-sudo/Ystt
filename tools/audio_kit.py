#!/usr/bin/env python3
"""
Sound design and voiceover synthesis for the Thonis-Heracleion Short.

Single source of truth for the audio: `render_short.py` imports the mix from
here, and `render_audio.py` exports the same material as separate one-shots and
stems. Every cue corresponds to a bracketed SFX marker in
scripts/thonis-heracleion-60s-short.md.

Nothing here samples a library -- each cue is synthesized from noise and
oscillators, so the whole bed is original and carries no licence obligations.
"""

import math
import os
import subprocess
import wave

import numpy as np

SR = 44100
DUR = 60.0

SCRATCH = os.environ.get(
    "SHORT_SCRATCH",
    "/tmp/claude-0/-home-user-Ystt/f2140de2-deb3-5931-bb2b-c92220781c0a/scratchpad/vid",
)

# espeak-ng voice. MBROLA diphone voices are markedly less robotic than
# espeak's own formant synthesis; mb-us2 is the male US diphone set.
VOICE = os.environ.get("SHORT_VOICE", "mb-us2")
VOICE_FALLBACK = "en-us+m3"
VO_SEMITONES = -1.5   # resample-based shift; adds weight without chipmunking


# ------------------------------------------------------------------ primitives

def env(ln, atk=0.01, dec=0.5, power=2.0):
    k = np.linspace(0, 1, int(ln), dtype=np.float32)
    a = np.clip(k / max(atk, 1e-6), 0, 1)
    d = np.exp(-k * dec * 10.0) ** (1 / power)
    return (a * d).astype(np.float32)


def sine(f, dur, ph=0.0):
    k = np.arange(int(dur * SR), dtype=np.float32) / SR
    return np.sin(2 * math.pi * f * k + ph).astype(np.float32)


def sweep(f0, f1, dur):
    k = np.arange(int(dur * SR), dtype=np.float32) / SR
    f = f0 * (f1 / f0) ** (k / max(dur, 1e-6))
    return np.sin(2 * math.pi * np.cumsum(f) / SR).astype(np.float32)


def lp(x, taps):
    k = np.hanning(max(3, int(taps)))
    k /= k.sum()
    return np.convolve(x, k, mode="same").astype(np.float32)


def hp(x, taps):
    return (x - lp(x, taps)).astype(np.float32)


def bp(x, lo_taps, hi_taps):
    return lp(hp(x, lo_taps), hi_taps)


def noise(dur, taps=120, seed=0):
    n = int(dur * SR)
    x = np.random.default_rng(1000 + seed).standard_normal(n).astype(np.float32)
    return lp(x, taps)


def padd(*sigs):
    """Sum signals of differing length, zero-padded to the longest."""
    ln = max(len(x) for x in sigs)
    acc = np.zeros(ln, dtype=np.float32)
    for x in sigs:
        acc[: len(x)] += x
    return acc


def fftconv(x, ir):
    n = len(x) + len(ir) - 1
    m = 1 << (n - 1).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]
    return y.astype(np.float32)


def plate_ir(dur=1.6, decay=5.5, seed=4, dark=140):
    """Synthetic plate-ish impulse response."""
    n = int(dur * SR)
    k = np.linspace(0, 1, n, dtype=np.float32)
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    ir = lp(x, dark) * np.exp(-k * decay)
    ir[: int(0.008 * SR)] = 0.0          # pre-delay
    ir /= np.abs(ir).max() + 1e-9
    return ir


def compress(x, thresh=0.25, ratio=4.0, atk=0.004, rel=0.12):
    """Glue compressor. The detector is a smoothed magnitude envelope rather
    than a true one-pole follower -- a per-sample IIR over millions of samples
    is far too slow in Python, and for narration levelling the difference is
    inaudible."""
    e = lp(np.abs(x), max(3, int(rel * SR) // 4))
    g = np.ones_like(e)
    over = e > thresh
    g[over] = (thresh + (e[over] - thresh) / ratio) / (e[over] + 1e-9)
    g = lp(g, max(3, int(atk * SR) * 4 + 3))
    return (x * g).astype(np.float32)


def normalize(x, peak=0.97):
    m = np.abs(x).max()
    return (x * (peak / m)).astype(np.float32) if m > 1e-9 else x


def fade(x, t_in=0.005, t_out=0.02):
    x = x.copy()
    fi, fo = int(t_in * SR), int(t_out * SR)
    if fi > 0 and len(x) > fi:
        x[:fi] *= np.linspace(0, 1, fi)
    if fo > 0 and len(x) > fo:
        x[-fo:] *= np.linspace(1, 0, fo)
    return x


def write_wav(path, x, stereo=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    data = np.stack([x, x], axis=1) if stereo and x.ndim == 1 else x
    with wave.open(path, "wb") as w:
        w.setnchannels(2 if stereo else 1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((data * 32767).astype(np.int16).tobytes())
    return path


def read_wav(path):
    with wave.open(path, "rb") as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        raw = np.interp(
            np.linspace(0.0, 1.0, int(len(raw) * SR / sr)),
            np.linspace(0.0, 1.0, len(raw)), raw,
        ).astype(np.float32)
    return raw


# ----------------------------------------------------------------- SFX one-shots

def sfx_whoosh():
    x = noise(0.85, 40, 1)
    k = np.linspace(0, 1, len(x), dtype=np.float32)
    return fade(x * (k ** 2) * np.exp(-k * 1.2) * 0.85)


def sfx_plunge():
    x = noise(1.6, 900, 3) * env(1.6 * SR, 0.02, 0.35)
    return fade(x * 1.1)


def sfx_bass_drop():
    return fade(sweep(140, 32, 1.5) * env(1.5 * SR, 0.004, 0.30) * 1.15)


def sfx_bass_drop_soft():
    return fade(sweep(140, 32, 1.5) * env(1.5 * SR, 0.004, 0.30) * 0.98)


def _ping(f):
    p = sine(f, 1.5) * env(1.5 * SR, 0.001, 0.55)
    echo = np.pad(sine(f, 1.5) * env(1.5 * SR, 0.001, 0.7), (int(0.22 * SR), 0))[: len(p)]
    return fade(p + 0.35 * echo)


def sfx_sonar_giants():
    return _ping(880) * 0.5


def sfx_sonar_1():
    return _ping(760) * 0.55


def sfx_sonar_2():
    return _ping(900) * 0.6


def sfx_sonar_3():
    return _ping(1080) * 0.7


def sfx_reverse_swell():
    """Reversed-cymbal lift into the ruins-to-city reality flip."""
    x = bp(noise(1.9, 8, 22), 900, 14)
    k = np.linspace(0, 1, len(x), dtype=np.float32)
    x = x * (k ** 3.0)
    x += sweep(400, 3200, 1.9) * (k ** 4) * 0.25
    return fade(x * 0.7, 0.05, 0.01)


def sfx_quill():
    return fade(noise(0.16, 12, 31) * env(0.16 * SR, 0.01, 2.2) * 0.55)


def sfx_stamp_thud():
    return fade(padd(sine(52, 1.0) * env(1.0 * SR, 0.001, 0.9),
                     noise(0.35, 25, 5) * env(0.35 * SR, 0.001, 2.5) * 0.55))


def sfx_riser():
    x = noise(5.0, 60, 8)
    k = np.linspace(0, 1, len(x), dtype=np.float32)
    x = x * (k ** 2.4) + sweep(180, 1500, 5.0) * (k ** 3) * 0.35
    return fade(x * 0.55, 0.2, 0.05)


def sfx_bass_hit():
    return fade(padd(sweep(160, 30, 1.8) * env(1.8 * SR, 0.003, 0.24) * 1.3,
                     noise(0.5, 30, 9) * env(0.5 * SR, 0.001, 2.0) * 0.5))


def sfx_perc_hit():
    return fade(padd(sine(70, 0.5) * env(0.5 * SR, 0.001, 1.5),
                     noise(0.22, 18, 17) * env(0.22 * SR, 0.001, 3.0) * 0.7) * 0.85)


def sfx_coin_shimmer():
    """Two seconds of gold glinting -- stacked high partials, random attacks."""
    out = np.zeros(int(2.0 * SR), dtype=np.float32)
    rng = np.random.default_rng(77)
    for i in range(20):
        t0 = int(rng.uniform(0, 1.55) * SR)
        f = rng.uniform(1700, 3400)
        s = sine(f, 0.45) * env(0.45 * SR, 0.001, 1.6)
        s = s + sine(f * 2.02, 0.45) * env(0.45 * SR, 0.001, 2.2) * 0.4
        out[t0:t0 + len(s)] += s[: len(out) - t0] * rng.uniform(0.5, 1.0)
    return fade(out * 0.22)


def sfx_stone_grind():
    return fade(noise(1.7, 220, 12) * env(1.7 * SR, 0.08, 0.5) * 0.55)


def sfx_earth_rumble():
    x = noise(7.6, 1400, 13)
    k = np.linspace(0, 1, len(x), dtype=np.float32)
    x = x * np.clip(k * 2.2, 0, 1) * np.clip(1.6 - k * 1.2, 0, 1) * 1.5
    x += sine(29, 7.6) * np.clip(k * 2.0, 0, 1) * 0.55
    return fade(x, 0.15, 0.3)


def sfx_stone_crack():
    return fade(padd(noise(0.7, 14, 41) * env(0.7 * SR, 0.001, 1.4),
                     sine(95, 0.7) * env(0.7 * SR, 0.002, 1.1) * 0.6) * 0.75)


def sfx_water_rush():
    x = noise(2.4, 90, 15)
    k = np.linspace(0, 1, len(x), dtype=np.float32)
    return fade(x * np.clip(k * 3.0, 0, 1) * np.clip(1.4 - k * 1.4, 0, 1))


def sfx_piano_note():
    x = sum((1.0 / (h + 1) ** 1.35) * sine(110.0 * (h + 1), 4.0, ph=h * 0.6)
            for h in range(7)) * env(4.0 * SR, 0.002, 0.30)
    return fade(x * 0.85, 0.002, 0.4)


def sfx_closing_swell():
    x = sine(43.0, 4.6) + 0.5 * sine(64.5, 4.6)
    k = np.clip(np.linspace(0, 1, len(x), dtype=np.float32) * 1.6, 0, 1)
    x = x * k * 0.55 + noise(4.6, 500, 21) * k * 0.35
    return fade(x, 0.3, 0.2)


def sfx_oud_pluck(f0=174.6):
    x = sum((1.0 / (h + 1)) * sine(f0 * (h + 1), 1.1, ph=h) for h in range(5))
    return fade(x * env(1.1 * SR, 0.002, 0.75) * 0.20)


def sfx_frame_drum():
    return fade(padd(sine(58, 0.30) * env(0.30 * SR, 0.002, 1.6),
                     noise(0.30, 60, 11) * env(0.30 * SR, 0.001, 3.0) * 0.5) * 0.42)


def sfx_harbour_crowd():
    """Distant murmur for the living-city reconstruction."""
    n = int(4.6 * SR)
    x = bp(np.random.default_rng(55).standard_normal(n).astype(np.float32), 260, 90)
    t = np.arange(n, dtype=np.float32) / SR
    mod = 0.55 + 0.45 * (np.sin(2 * math.pi * 0.7 * t) * np.sin(2 * math.pi * 0.23 * t))
    return fade(x * mod * 0.16, 0.4, 0.6)


def sfx_underwater_amb():
    n = int(6.0 * SR)
    x = lp(np.random.default_rng(91).standard_normal(n).astype(np.float32), 700)
    return fade(x * 0.55, 0.5, 0.5)


# time, id, builder, gain, note (the bracketed cue it implements)
CUES = [
    (0.02,  "whoosh_smash_cut",   sfx_whoosh,        0.85, "Sharp Air Whoosh"),
    (0.42,  "underwater_plunge",  sfx_plunge,        1.00, "Underwater Plunge / Muffle"),
    (0.44,  "bass_drop_main",     sfx_bass_drop,     1.00, "Deep Bass Drop"),
    (2.55,  "bass_drop_this",     sfx_bass_drop_soft, 0.87, "Bass Drop on 'this'"),
    (9.60,  "sonar_ping_giants",  sfx_sonar_giants,  1.00, "Sonar Ping on 'giants'"),
    (13.15, "reverse_swell_flip", sfx_reverse_swell, 1.00, "Reversed Cymbal Swell"),
    (13.50, "harbour_crowd",      sfx_harbour_crowd, 1.00, "Distant Harbor Crowd"),
    (22.00, "stamp_thud_myth",    sfx_stamp_thud,    1.00, "Heavy Stamp Thud on 'myth'"),
    (26.55, "sonar_ping_1",       sfx_sonar_1,       1.00, "Sonar Ping x3, rising"),
    (27.30, "sonar_ping_2",       sfx_sonar_2,       1.00, "Sonar Ping x3, rising"),
    (28.05, "sonar_ping_3",       sfx_sonar_3,       1.00, "Sonar Ping x3, rising"),
    (28.10, "synth_riser",        sfx_riser,         1.00, "Synth Riser"),
    (31.65, "bass_hit_impossible", sfx_bass_hit,     1.00, "Hard Bass Hit on 'impossible'"),
    (34.02, "perc_hit_1",         sfx_perc_hit,      1.00, "Percussive Hit on cut 1"),
    (36.02, "perc_hit_2",         sfx_perc_hit,      1.00, "Percussive Hit on cut 2"),
    (38.02, "perc_hit_3",         sfx_perc_hit,      1.00, "Percussive Hit on cut 3"),
    (38.05, "coin_shimmer",       sfx_coin_shimmer,  1.00, "Metallic Coin Shimmer"),
    (40.02, "perc_hit_4",         sfx_perc_hit,      1.00, "Percussive Hit on cut 4"),
    (40.30, "stone_grind",        sfx_stone_grind,   1.00, "Stone Grind on 'granite slab'"),
    (42.10, "earth_rumble",       sfx_earth_rumble,  1.00, "Deep Earth Rumble"),
    (45.00, "stone_crack_1",      sfx_stone_crack,   1.00, "Structural Stone Cracking"),
    (46.20, "stone_crack_2",      sfx_stone_crack,   1.00, "Structural Stone Cracking"),
    (47.40, "stone_crack_3",      sfx_stone_crack,   1.00, "Structural Stone Cracking"),
    (47.60, "water_rush",         sfx_water_rush,    1.00, "Massive Water Rush"),
    (50.48, "piano_note",         sfx_piano_note,    1.00, "Single Low Piano Note"),
    (54.70, "closing_swell",      sfx_closing_swell, 1.00, "Deep Bass Swell"),
]

# quill scratches and the oud/drum figure are patterns rather than single hits
QUILL_TIMES = list(np.arange(18.30, 21.50, 0.26))
OUD_TIMES = list(np.arange(13.35, 18.00, 0.42))
DRUM_TIMES = list(np.arange(13.30, 18.00, 0.84))
OUD_SCALE = [146.8, 155.6, 174.6, 196.0, 233.1, 261.6]
OUD_PATTERN = [0, 2, 3, 1, 4, 5, 3, 2, 0, 3, 4, 2]


# ------------------------------------------------------------------------ stems

def _place(track, t0, sig, gain=1.0):
    s = int(t0 * SR)
    if s >= len(track):
        return
    e = min(len(track), s + len(sig))
    track[s:e] += sig[: e - s] * gain


def render_sfx_stem():
    n = int(DUR * SR)
    tr = np.zeros(n, dtype=np.float32)
    for t0, _id, make, gain, _note in CUES:
        _place(tr, t0, make(), gain)
    for t0 in QUILL_TIMES:
        _place(tr, t0, sfx_quill(), 0.20)
    for i, t0 in enumerate(OUD_TIMES):
        _place(tr, t0, sfx_oud_pluck(OUD_SCALE[OUD_PATTERN[i % len(OUD_PATTERN)]]))
    for t0 in DRUM_TIMES:
        _place(tr, t0, sfx_frame_drum())
    return tr


def render_bed_stem():
    """Continuous drone and underwater ambience -- the tonal floor under the
    one-shots. Ducked to silence for the 0:50 beat."""
    n = int(DUR * SR)
    t = np.arange(n, dtype=np.float32) / SR
    drone = (0.34 * np.sin(2 * math.pi * 41.0 * t)
             + 0.20 * np.sin(2 * math.pi * 61.5 * t + 0.7)
             + 0.12 * np.sin(2 * math.pi * 82.0 * t + 1.9))
    lfo = 0.62 + 0.38 * np.sin(2 * math.pi * 0.09 * t)
    prof = np.interp(t, [0, 0.5, 4, 11, 18, 26, 31.5, 34, 42, 49.5, 50.0,
                         50.45, 55.5, 60],
                        [0, 0.55, 0.7, 0.62, 0.5, 0.62, 0.95, 0.7, 1.0, 1.0,
                         0.0, 0.0, 0.75, 0.95])
    bed = (drone * lfo * prof * 0.30).astype(np.float32)

    amb = lp(np.random.default_rng(7).standard_normal(n).astype(np.float32), 700)
    amb_prof = np.interp(t, [0, 0.4, 4, 18, 26, 34, 42, 50, 60],
                            [0, 0.9, 1.0, 0.15, 0.5, 0.8, 0.6, 0.7, 0.7])
    bed += amb * amb_prof * 0.55
    return bed


def silence_beat(x):
    """The scripted 0:50 cut to silence."""
    s0, s1 = int(49.88 * SR), int(50.45 * SR)
    x[s0:s1] *= np.linspace(1.0, 0.0, s1 - s0) ** 2
    x[int(50.0 * SR):s1] = 0.0
    return x


# --------------------------------------------------------------------- voiceover

VO_LINES = [
    (0.35, 5.20, "hook",        "Everyone calls Atlantis a fairy tale. Then divers found this. "
                                "Thirty feet down."),
    (5.90, 5.30, "reveal",      "An entire Egyptian city. Temples, streets, "
                                "and giants, face down in the silt."),
    (11.30, 6.30, "name",       "This is Thonis Heracleion. Egypt's richest port, "
                                "centuries before Alexandria was even built."),
    (18.30, 7.20, "myth",       "Herodotus wrote that Helen of Troy sheltered here. "
                                "For two thousand years, scholars called that a myth."),
    (26.35, 6.90, "discovery",  "Then in two thousand, Franck Goddio's sonar pinged "
                                "something impossible in Abu Qir Bay."),
    (34.20, 7.40, "haul",       "Sixteen foot gods. Sixty four shipwrecks. Gold. "
                                "And a granite slab carved with the city's own name."),
    (42.30, 7.20, "liquefy",    "It didn't drift under. The clay beneath it turned to liquid, "
                                "and the temples' own weight pulled them down."),
    (50.60, 4.05, "five-pct",   "A quarter century of diving, and we've mapped "
                                "maybe five percent."),
    (54.90, 4.90, "plato",      "Plato said a city vanished in a single night. "
                                "Egypt has the receipts."),
]


def _pitch_shift(a, semitones):
    """Resampling shift -- alters length as well as pitch, which the wpm fit
    loop then compensates for."""
    if abs(semitones) < 1e-3:
        return a
    ratio = 2.0 ** (semitones / 12.0)
    n = max(2, int(len(a) / ratio))
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a).astype(np.float32)


def _espeak(text, wpm, path, voice=None):
    v = voice or VOICE
    cmd = ["espeak-ng", "-v", v, "-s", str(wpm), "-a", "170", "-g", "4", "-w", path, text]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(path) or os.path.getsize(path) < 1000:
        subprocess.run(
            ["espeak-ng", "-v", VOICE_FALLBACK, "-s", str(wpm), "-p", "22",
             "-a", "170", "-g", "4", "-w", path, text],
            check=True, capture_output=True,
        )
    return read_wav(path)


def voice_line(text, slot, idx=0, voice=None):
    """Render one line, speed-fitted to its slot, then treated for broadcast."""
    path = os.path.join(SCRATCH, f"vo_raw_{idx}.wav")
    os.makedirs(SCRATCH, exist_ok=True)
    wpm = 165
    a = None
    for _ in range(7):
        a = _pitch_shift(_espeak(text, wpm, path, voice), VO_SEMITONES)
        d = len(a) / SR
        if d <= slot:
            break
        wpm = int(wpm * (d / slot) * 1.02) + 1
    return treat_voice(a), wpm


def treat_voice(a):
    """Narration chain: de-harsh, warm the low mids, compress, small plate."""
    a = hp(a, 900)                                   # roll off rumble
    a = a - 0.35 * bp(a, 26, 10)                     # tame 4-7k sibilance
    a = a + 0.22 * lp(a, 260)                        # low-mid body
    a = compress(a, thresh=0.20, ratio=3.5)
    a = a + 0.10 * fftconv(a, plate_ir(1.3, 6.5))    # subtle space
    a = normalize(a, 0.85)
    return fade(a, 0.012, 0.05)


def render_vo_stem(voice=None, per_line=None):
    n = int(DUR * SR) + SR
    tr = np.zeros(n, dtype=np.float32)
    for idx, (t0, slot, slug, text) in enumerate(VO_LINES):
        a, wpm = voice_line(text, slot, idx, voice)
        if per_line is not None:
            per_line.append((idx, t0, slug, wpm, len(a) / SR, a))
        _place(tr, t0, a)
    return tr[: int(DUR * SR)]


# -------------------------------------------------------------------- final mix

def render_mix(vo=None, bed=None, sfx=None):
    bed = render_bed_stem() if bed is None else bed
    sfx = render_sfx_stem() if sfx is None else sfx
    vo = render_vo_stem() if vo is None else vo
    n = min(len(bed), len(sfx), len(vo))
    bed, sfx, vo = bed[:n].copy(), sfx[:n].copy(), vo[:n].copy()

    silence_beat(bed)
    silence_beat(sfx)

    # duck the music bed under the voice; leave the SFX hits proud
    e = lp(np.abs(vo), 4000)
    e /= max(e.max(), 1e-6)
    duck = 1.0 - 0.55 * np.clip(e * 3.0, 0, 1)

    mix = bed * duck * 0.85 + sfx * 0.85 + vo * 1.05
    mix = np.tanh(mix * 1.25) * 0.92
    mix = normalize(mix, 0.94)
    return fade(mix, 0.03, 0.03)
