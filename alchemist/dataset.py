"""Load recorded clips and augment them.

Folder layout (created by record.py):
    data/<class_name>/<anything>.wav     each file = one ~100 ms clip around one hit

Augmentation matters more than model size here: with ~100 recordings per class, the
model must still work in a different room, at a different volume, on another laptop.
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

from . import config as C
from .features import clip_features, fix_length


def read_wav(path):
    sr, y = wavfile.read(path)
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2 ** 31
    y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        y = resample(y, int(len(y) * C.SAMPLE_RATE / sr)).astype(np.float32)
    return y


def write_wav(path, y, sr=C.SAMPLE_RATE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def load_dataset(root="data", classes=C.CLASSES):
    clips, labels, sources = [], [], []
    for ci, name in enumerate(classes):
        for f in sorted(Path(root, name).glob("*.wav")):
            clips.append(fix_length(read_wav(f)))
            labels.append(ci)
            # Recording session = file prefix before "__". Used to split train/test
            # by session, so the test set is a take the model has never heard.
            sources.append(f.stem.split("__")[0])
    return clips, np.array(labels), np.array(sources)


def augment(clip, rng, noise_bank=None):
    y = clip.copy()
    # 1. Volume: closer/further from the mic
    y *= rng.uniform(0.3, 1.5)
    # 2. Timing jitter: the onset detector isn't sample-perfect
    shift = int(rng.integers(-160, 161))   # +/- 10 ms
    y = np.roll(y, shift)
    if shift > 0:
        y[:shift] = 0
    elif shift < 0:
        y[shift:] = 0
    # 3. Speed/pitch: different mouths, hands and tables
    if rng.random() < 0.5:
        factor = rng.uniform(0.9, 1.1)
        y = fix_length(resample(y, int(len(y) * factor)))
    # 4. Background: mix in recorded room noise, or white noise
    if noise_bank is not None and len(noise_bank) and rng.random() < 0.6:
        bg = noise_bank[rng.integers(len(noise_bank))]
        y = y + bg * rng.uniform(0.05, 0.4) * np.max(np.abs(y))
    if rng.random() < 0.5:
        y = y + rng.normal(0, rng.uniform(0.001, 0.02), len(y)).astype(np.float32)
    # 5. Cheap "different microphone": random tilt in the spectrum
    if rng.random() < 0.3:
        a = rng.uniform(-0.6, 0.6)
        y = np.concatenate([[y[0]], y[1:] - a * y[:-1]]).astype(np.float32)
    return y.astype(np.float32)


def spec_augment(feat, rng):
    """Mask a random band of mel bins (SpecAugment-style)."""
    f = feat.copy()
    if rng.random() < 0.5:
        w = int(rng.integers(1, 6))
        s = int(rng.integers(0, f.shape[0] - w))
        f[s:s + w] = f.mean()
    return f


def featurize(clips, rng=None, noise_bank=None, n_aug=0):
    """Features for every clip, plus n_aug augmented copies each."""
    X, idx = [], []
    for i, c in enumerate(clips):
        X.append(clip_features(c))
        idx.append(i)
        for _ in range(n_aug):
            X.append(spec_augment(clip_features(augment(c, rng, noise_bank)), rng))
            idx.append(i)
    return np.stack(X), np.array(idx)
