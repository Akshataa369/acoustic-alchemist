"""Generate FAKE mouth sounds to test the pipeline without a microphone.

This is only for checking that record -> train -> live works end to end. The real
model must be trained on your own recordings: these synthetic sounds are far
cleaner than a real beatbox.

    python tools/make_test_data.py            # writes data_synthetic/ and a test take
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alchemist import config as C  # noqa: E402
from alchemist.dataset import write_wav  # noqa: E402
from alchemist.onset import detect_onsets, slice_clips  # noqa: E402

SR = C.SAMPLE_RATE


def _t(sec):
    return np.arange(int(sec * SR)) / SR


def make(label, rng):
    v = lambda a, b: rng.uniform(a, b)
    if label == "pop":        # lip pop: low thump with a tiny click
        t = _t(0.12)
        f = v(60, 110) + v(60, 140) * np.exp(-t / 0.02)
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / v(0.02, 0.04)) \
            + 0.15 * rng.normal(0, 1, len(t)) * np.exp(-t / 0.002)
    elif label == "click":    # tongue click: sharp knock with a resonance ~1-2 kHz
        t = _t(0.08)
        y = np.sin(2 * np.pi * v(900, 2000) * t) * np.exp(-t / v(0.004, 0.01)) \
            + rng.normal(0, 0.3, len(t)) * np.exp(-t / 0.0015)
    elif label == "hiss":     # "ts": short high hiss
        t = _t(0.12)
        y = np.diff(rng.normal(0, 1, len(t) + 1)) * np.exp(-t / v(0.02, 0.05))
    elif label == "hum":      # "mm": low voiced tone, quick start, slower decay
        t = _t(0.25)
        f0 = v(110, 220)
        y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3)) * np.minimum(1, t / 0.008) \
            * np.exp(-t / v(0.1, 0.2))
    elif label == "tap":      # knuckle on table: woody mid tone
        t = _t(0.12)
        y = np.sin(2 * np.pi * v(300, 600) * t) * np.exp(-t / v(0.01, 0.025)) \
            + 0.2 * rng.normal(0, 1, len(t)) * np.exp(-t / 0.003)
    else:                     # noise: speech-like tone, typing, rustle
        t = _t(0.3)
        kind = rng.integers(3)
        if kind == 0:
            y = np.sin(2 * np.pi * v(100, 220) * t * (1 + 0.05 * np.sin(2 * np.pi * 4 * t))) * 0.3
        elif kind == 1:
            y = rng.normal(0, 0.05, len(t))
            for _ in range(3):
                k = rng.integers(0, len(t) - 200)
                y[k:k + 200] += rng.normal(0, 0.3, 200) * np.exp(-np.arange(200) / 40)
        else:
            y = np.convolve(rng.normal(0, 0.1, len(t)), np.ones(30) / 30, "same")
    return (y / (np.max(np.abs(y)) + 1e-9) * v(0.2, 0.9)).astype(np.float32)


def take(label, rng, seconds=8.0, room=0.003):
    """A continuous 'recording session': hits with gaps, over room noise."""
    y = rng.normal(0, room, int(seconds * SR)).astype(np.float32)
    truth = []
    pos = int(0.4 * SR)
    while pos < len(y) - int(0.4 * SR):
        s = make(label, rng)
        y[pos:pos + len(s)] += s
        truth.append(pos)
        pos += int(rng.uniform(0.18, 0.4) * SR)
    return y, truth


def main():
    rng = np.random.default_rng(42)
    root = Path("data_synthetic")
    for label in C.CLASSES:
        n = 0
        for session in range(4):                      # 4 "sessions" per class
            y, _ = take(label, rng)
            if label == "noise":
                clips = [y[i:i + C.CLIP_SAMPLES] for i in range(0, len(y) - C.CLIP_SAMPLES, 2400)]
                clips += slice_clips(y, detect_onsets(y))
            else:
                clips = slice_clips(y, detect_onsets(y))
            for i, c in enumerate(clips):
                write_wav(root / label / f"synth{session}__{i:03d}.wav", c)
            n += len(clips)
        print(f"{label:>6}: {n} clips")

    # A mixed 'performance' with known ground truth, for replay testing
    perf = rng.normal(0, 0.003, int(12 * SR)).astype(np.float32)
    truth = []
    pos = int(0.5 * SR)
    pattern = ["pop", "click", "hiss", "pop", "hum", "pop", "click", "tap", "hiss", "hum"]
    i = 0
    while pos < len(perf) - SR // 2:
        lab = pattern[i % len(pattern)]
        s = make(lab, rng)
        perf[pos:pos + len(s)] += s
        truth.append((pos / SR, lab))
        pos += int(rng.uniform(0.2, 0.35) * SR)
        i += 1
    write_wav("test_performance.wav", perf)
    Path("test_performance_truth.txt").write_text("\n".join(f"{t:.3f} {l}" for t, l in truth))
    print(f"test_performance.wav: {len(truth)} hits")


if __name__ == "__main__":
    main()
