#!/usr/bin/env python3
"""
Export the Short's audio as separate, editor-ready assets.

Writes into audio/:
  sfx/     one WAV per cue, trimmed and peak-normalised for dropping on a
           timeline by hand
  vo/      one WAV per voiceover line
  stems/   bed / sfx / vo as full-length 60s stems that line up at 0:00
  mix.wav  the same mix used in the animatic
  cue-sheet.md  every cue with its timecode

Usage: python3 tools/render_audio.py [outdir]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_kit as A


def tc(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    sfx_dir = os.path.join(out, "sfx")
    vo_dir = os.path.join(out, "vo")
    stem_dir = os.path.join(out, "stems")

    print("rendering SFX one-shots ...", flush=True)
    rows = []
    for i, (t0, cid, make, gain, note) in enumerate(A.CUES):
        sig = A.normalize(make() * gain, 0.95)
        p = os.path.join(sfx_dir, f"{i:02d}_{cid}.wav")
        A.write_wav(p, sig)
        rows.append((t0, cid, len(sig) / A.SR, note, os.path.relpath(p, out)))

    # pattern cues exported as single representative hits
    for cid, sig, note, first in [
        ("quill_scratch", A.sfx_quill(), "Quill Scratch (repeats every 0.26s, 18.30-21.50)", 18.30),
        ("oud_pluck", A.sfx_oud_pluck(), "Oud Pluck (figure runs 13.35-18.00)", 13.35),
        ("frame_drum", A.sfx_frame_drum(), "Frame Drum (every 0.84s, 13.30-18.00)", 13.30),
        ("underwater_amb_loop", A.sfx_underwater_amb(), "Muffled Underwater Ambience (loopable)", 0.0),
    ]:
        p = os.path.join(sfx_dir, f"xx_{cid}.wav")
        A.write_wav(p, A.normalize(sig, 0.95))
        rows.append((first, cid, len(sig) / A.SR, note, os.path.relpath(p, out)))

    print("rendering voiceover ...", flush=True)
    per_line = []
    vo = A.render_vo_stem(per_line=per_line)
    vo_rows = []
    for idx, t0, slug, wpm, dur, a in per_line:
        p = os.path.join(vo_dir, f"{idx:02d}_{slug}.wav")
        A.write_wav(p, A.normalize(a, 0.92))
        _, slot, _, text = A.VO_LINES[idx]
        # delivered rate, not espeak's -s parameter: the two diverge badly
        # once word gaps and the pitch shift are applied
        rate = len(text.split()) / dur * 60.0
        vo_rows.append((t0, slug, dur, slot, rate, len(text.split()), text,
                        os.path.relpath(p, out)))

    print("rendering stems ...", flush=True)
    bed = A.render_bed_stem()
    sfx = A.render_sfx_stem()
    A.write_wav(os.path.join(stem_dir, "bed.wav"), A.normalize(A.silence_beat(bed.copy()), 0.9))
    A.write_wav(os.path.join(stem_dir, "sfx.wav"), A.normalize(A.silence_beat(sfx.copy()), 0.9))
    A.write_wav(os.path.join(stem_dir, "vo.wav"), A.normalize(vo, 0.9))
    mix = A.render_mix(vo=vo, bed=bed, sfx=sfx)
    A.write_wav(os.path.join(out, "mix.wav"), mix)

    # music-and-effects stem: everything except the voice, for dropping a real
    # VO on top
    me = A.render_mix(vo=np.zeros_like(vo), bed=A.render_bed_stem(), sfx=A.render_sfx_stem())
    A.write_wav(os.path.join(stem_dir, "bed_and_sfx_no_vo.wav"), me)

    print("writing cue sheet ...", flush=True)
    lines = [
        "# Cue sheet — Thonis-Heracleion 60s Short",
        "",
        "All timecodes are from 0:00 of the cut. Every asset is 44.1 kHz / 16-bit "
        "stereo WAV, synthesized from scratch — no sampled library material, "
        "nothing to clear.",
        "",
        "## Voiceover",
        "",
        "| In | Line | Words | Length | Slot | Delivered rate | File |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t0, slug, dur, slot, rate, nw, text, rel in vo_rows:
        lines.append(f"| {tc(t0)} | {slug} | {nw} | {dur:.2f}s | {slot:.2f}s | {rate:.0f} wpm | `{rel}` |")
    lines += ["", "Full text of each line:", ""]
    for t0, slug, dur, slot, rate, nw, text, rel in vo_rows:
        lines.append(f"- **{tc(t0)} {slug}** — \"{text}\"")

    lines += ["", "## Sound effects", "",
              "| In | Cue | Length | Script marker | File |",
              "| --- | --- | --- | --- | --- |"]
    for t0, cid, dur, note, rel in sorted(rows):
        lines.append(f"| {tc(t0)} | {cid} | {dur:.2f}s | {note} | `{rel}` |")

    lines += [
        "", "## Stems", "",
        "Each stem is the full 60 seconds and starts at 0:00, so they drop onto a "
        "timeline aligned with no offset.",
        "",
        "| Stem | Contents |",
        "| --- | --- |",
        "| `stems/bed.wav` | Drone and underwater ambience only |",
        "| `stems/sfx.wav` | All one-shots and pattern cues, no bed |",
        "| `stems/vo.wav` | Voiceover only |",
        "| `stems/bed_and_sfx_no_vo.wav` | Music and effects, **no voice** — use this "
        "under a real VO |",
        "| `mix.wav` | Full mix as heard in the animatic |",
        "",
        "## On the voiceover",
        "",
        "The VO is **scratch**, not final. It is espeak-ng driving MBROLA diphone "
        "voices, pitched down and run through a narration chain (de-essing, low-mid "
        "warmth, compression, a short plate). That is good enough to lock timing and "
        "to edit picture against, and it is genuinely not good enough to publish. "
        "For the final cut, record a human or use a commercial neural TTS, and keep "
        "each line inside the slot length in the table above — those slots are what "
        "the picture edit is cut to.",
    ]
    with open(os.path.join(out, "cue-sheet.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {len(rows)} SFX, {len(vo_rows)} VO lines, 5 stems -> {out}")


if __name__ == "__main__":
    main()
