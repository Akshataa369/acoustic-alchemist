"""Whistle pitch tracking with the YIN algorithm (de Cheveigné & Kawahara, 2002).

YIN finds the period of a waveform by asking: "how far do I shift the signal before
it best matches itself?" It's robust and cheap enough to run on every audio block.
"""
import numpy as np

from . import config as C


def yin(frame, sr=C.SAMPLE_RATE, fmin=C.PITCH_MIN_HZ, fmax=C.PITCH_MAX_HZ, threshold=0.15):
    """Return (frequency_hz, confidence 0..1). frequency is None when no clear pitch."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    tau_min = max(2, int(sr / fmax))
    tau_max = min(len(x) // 2, int(sr / fmin) + 1)
    if tau_max <= tau_min + 2 or np.sqrt(np.mean(x ** 2)) < 0.01:
        return None, 0.0

    w = len(x) - tau_max
    # Step 1-2: difference function d(tau) = sum (x[j] - x[j+tau])^2
    d = np.array([np.sum((x[:w] - x[tau:tau + w]) ** 2) for tau in range(tau_max + 1)])
    # Step 3: cumulative mean normalised difference
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(running, 1e-12)

    # Step 4: first dip below the threshold, then walk to its local minimum
    tau = None
    for t in range(tau_min, tau_max):
        if cmnd[t] < threshold:
            while t + 1 < tau_max and cmnd[t + 1] < cmnd[t]:
                t += 1
            tau = t
            break
    if tau is None:
        return None, 0.0

    # Step 5: parabolic interpolation for sub-sample accuracy
    if 1 <= tau < len(cmnd) - 1:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    else:
        shift = 0.0
    return sr / (tau + shift), float(max(0.0, 1.0 - cmnd[tau]))


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_hz(m):
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def snap_to_scale(midi, root=60, scale=C.SCALE):
    """Snap a (fractional) MIDI note to the nearest note in the scale."""
    best, best_dist = None, 1e9
    for octave in range(-2, 6):
        for step in scale:
            cand = root + 12 * octave + step
            dist = abs(cand - midi)
            if dist < best_dist:
                best, best_dist = cand, dist
    return best


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi):
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"
