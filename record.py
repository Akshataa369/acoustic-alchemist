"""Collect training data, fast.

Instead of recording one clip at a time, you perform a sound repeatedly for a
while (e.g. 30 s of snaps, with small pauses), and the onset detector cuts the take
into one clip per hit automatically.

    python record.py pop                    # 30 s from the mic
    python record.py click --seconds 45
    python record.py noise                  # talk, type, breathe, cough: things that must NOT trigger
    python record.py hiss --file phone.wav  # slice an existing recording instead

The sounds: pop (lip pop), click (tongue click), hiss ("ts"), hum ("mm"), tap (if you can),
noise. Make YOUR version of each sound; the model learns it. Vary distance and loudness on
purpose: variety in the data is what makes it work in a new room, on a tired day.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from alchemist import config as C
from alchemist.dataset import read_wav, write_wav
from alchemist.onset import detect_onsets, slice_clips


def record(seconds):
    import sounddevice as sd
    for i in (3, 2, 1):
        print(f"  starting in {i}...", flush=True)
        time.sleep(1)
    print(f"  RECORDING for {seconds} s. Go!", flush=True)
    y = sd.rec(int(seconds * C.SAMPLE_RATE), samplerate=C.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    print("  done.")
    return y[:, 0]


def noise_windows(y, every=0.15):
    """For the noise class: cut windows all along the take, not just at onsets."""
    step = int(every * C.SAMPLE_RATE)
    return [y[i:i + C.CLIP_SAMPLES] for i in range(0, len(y) - C.CLIP_SAMPLES, step)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label", choices=C.CLASSES)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--file", help="slice an existing .wav instead of recording")
    p.add_argument("--who", default="me", help="your name, so sessions are traceable")
    p.add_argument("--data", default="data")
    args = p.parse_args()

    session = f"{args.who}-{datetime.now():%m%d-%H%M%S}"
    if not args.file:
        print(f"\n  Sound: {args.label}. {C.DESCRIPTIONS.get(args.label, '')}")
        print("  Repeat it with short gaps until recording stops.")
    y = read_wav(args.file) if args.file else record(args.seconds)

    raw = Path(args.data, "raw", f"{args.label}__{session}.wav")
    write_wav(raw, y)                                     # keep the full take for later

    onsets = detect_onsets(y)
    clips = slice_clips(y, onsets)
    if args.label == "noise":
        clips += noise_windows(y)

    out_dir = Path(args.data, args.label)
    for i, c in enumerate(clips):
        write_wav(out_dir / f"{session}__{i:03d}.wav", c)

    total = len(list(out_dir.glob("*.wav")))
    print(f"\n  {len(clips)} '{args.label}' clips saved to {out_dir}/ ({total} total)")
    print(f"  full take kept at {raw}")
    if args.label != "noise" and len(onsets) and len(y) / C.SAMPLE_RATE / len(onsets) < 0.12:
        print("  ! Lots of onsets very close together: the room may be noisy, "
              "or leave a little more space between hits.")
    if args.label != "noise" and len(clips) < 15:
        print("  ! Few hits detected. Hit louder/closer, or record for longer.")


if __name__ == "__main__":
    main()
