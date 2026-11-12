"""Procedural instrument sounds + a real-time mixer.

Every sound is synthesised from noise and sine waves, so there are no sample files
to download or license, and you can tweak each one in code.
"""
import threading

import numpy as np
from scipy.signal import lfilter

OUT_SR = 44100


def _env(n, attack=0.002, decay=0.2, sr=OUT_SR):
    t = np.arange(n) / sr
    a = np.clip(t / attack, 0, 1)
    return a * np.exp(-t / decay)


def _noise(n, rng=np.random.default_rng(0)):
    return rng.uniform(-1, 1, n)


def _highpass(x, alpha=0.9):
    """One-pole high-pass filter: y[i] = alpha * (y[i-1] + x[i] - x[i-1])."""
    return lfilter([alpha, -alpha], [1.0, -alpha], x)


def kick(sr=OUT_SR):
    n = int(0.45 * sr)
    t = np.arange(n) / sr
    freq = 45 + 110 * np.exp(-t / 0.04)                 # pitch sweep = the "thump"
    phase = 2 * np.pi * np.cumsum(freq) / sr
    click = _noise(n) * np.exp(-t / 0.003) * 0.3
    return (np.sin(phase) * _env(n, decay=0.18) + click) * 0.9


def snare(sr=OUT_SR):
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    rattle = _highpass(_noise(n), 0.7) * np.exp(-t / 0.09)
    return (0.5 * body + 0.8 * rattle) * 0.7


def hat(sr=OUT_SR):
    n = int(0.08 * sr)
    t = np.arange(n) / sr
    return _highpass(_noise(n), 0.97) * np.exp(-t / 0.018) * 0.45


def clap(sr=OUT_SR):
    n = int(0.3 * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for off in (0.0, 0.011, 0.022):                      # three quick bursts = a clap
        k = int(off * sr)
        env[k:] += np.exp(-(t[: n - k]) / 0.012)
    env += np.exp(-t / 0.12) * 0.5
    return _highpass(_noise(n), 0.85) * env * 0.4


def rim(sr=OUT_SR):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    tone = np.sin(2 * np.pi * 1700 * t) + 0.6 * np.sin(2 * np.pi * 820 * t)
    return (tone * np.exp(-t / 0.02) + _noise(n) * np.exp(-t / 0.004) * 0.4) * 0.45


def tom(sr=OUT_SR):
    n = int(0.4 * sr)
    t = np.arange(n) / sr
    freq = 110 + 60 * np.exp(-t / 0.08)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * _env(n, decay=0.2) * 0.75


def pluck(freq, sr=OUT_SR, length=0.6):
    """Marimba-ish note for whistle mode."""
    n = int(length * sr)
    t = np.arange(n) / sr
    y = (np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * 4 * freq * t) * np.exp(-t / 0.03))
    return y * _env(n, attack=0.003, decay=0.25) * 0.5


def metronome_tick(accent=False, sr=OUT_SR):
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    f = 1500 if accent else 1000
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.008) * 0.25


KIT = {name: fn().astype(np.float32) for name, fn in
       {"kick": kick, "snare": snare, "hat": hat, "clap": clap, "rim": rim, "tom": tom}.items()}


class Mixer:
    """Plays overlapping sounds. play() is safe to call from any thread."""

    def __init__(self, sr=OUT_SR, max_voices=32):
        self.sr = sr
        self.voices = []          # [samples, position]
        self.lock = threading.Lock()
        self.max_voices = max_voices
        self.master = 0.8

    def play(self, samples, gain=1.0):
        with self.lock:
            if len(self.voices) >= self.max_voices:
                self.voices.pop(0)
            self.voices.append([np.asarray(samples, dtype=np.float32) * gain, 0])

    def render(self, n):
        out = np.zeros(n, dtype=np.float32)
        with self.lock:
            alive = []
            for v in self.voices:
                s, pos = v
                chunk = s[pos:pos + n]
                out[: len(chunk)] += chunk
                v[1] = pos + n
                if v[1] < len(s):
                    alive.append(v)
            self.voices = alive
        return np.tanh(out * self.master)    # soft clip so stacked hits don't distort harshly
