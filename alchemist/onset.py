"""Streaming onset detection with spectral flux.

An "onset" is the start of a hit: the moment energy suddenly rises across
frequencies. We compare each new short spectrum to the previous one and sum the
increases (spectral flux). A hit is a flux spike that clears an adaptive threshold
AND is louder than the background noise floor.

The same class is used live (block by block) and offline (to slice recordings into
training clips), so training data is cut exactly the way live audio is.
"""
from collections import deque

import numpy as np

from . import config as C
from .features import mel_filterbank

_N = 512
_WIN = np.hanning(_N).astype(np.float32)
_ONSET_FB = mel_filterbank(n_fft=_N, n_mels=24)


class OnsetDetector:
    def __init__(self, sr=C.SAMPLE_RATE, hop=C.BLOCK_SIZE, sensitivity=1.0,
                 refractory=C.REFRACTORY_SECONDS):
        self.sr = sr
        self.hop = hop
        self.buf = np.zeros(_N, dtype=np.float32)
        self.prev = None
        self.flux_hist = deque(maxlen=30)                     # ~0.5 s of recent flux values
        self.rms_hist = deque(maxlen=int(2.0 * sr / hop))     # ~2 s of loudness history
        self.noise_rms = 1e-3                                 # estimate of the room's background
        self.sensitivity = sensitivity
        self.refractory = int(refractory * sr)
        self.samples_seen = 0
        self.last_onset = -10 ** 9
        self.last_flux = 0.0

    def process(self, block):
        """Feed one block of audio. Returns the onset's absolute sample index, or None."""
        block = np.asarray(block, dtype=np.float32)
        n = len(block)
        self.buf = np.concatenate([self.buf[n:], block]) if n < _N else block[-_N:].copy()
        self.samples_seen += n

        # Flux over 24 mel bands rather than 257 raw FFT bins. With raw bins, broadband
        # noise and hi-hats swamp the few low bins a kick drum lives in, and kicks get
        # missed (v1 caught only 31/41 test hits; mel-band flux catches 41/41).
        power = np.abs(np.fft.rfft(self.buf * _WIN)) ** 2
        mel = np.log1p(100.0 * (power @ _ONSET_FB.T))
        flux = 0.0 if self.prev is None else float(np.sum(np.maximum(0.0, mel - self.prev)))
        self.prev = mel
        self.last_flux = flux

        rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
        hist = np.array(self.flux_hist) if self.flux_hist else np.array([flux])
        threshold = (np.median(hist) * 2.0 + 2.0) / self.sensitivity
        self.flux_hist.append(flux)

        # Noise floor = a low percentile of recent loudness, so it adapts to the room
        self.rms_hist.append(rms)
        self.noise_rms = max(float(np.percentile(self.rms_hist, 20)), 1e-4)

        loud_enough = rms > self.noise_rms * 4.0 and rms > 0.005
        is_onset = (flux > threshold and loud_enough
                    and self.samples_seen - self.last_onset > self.refractory)

        if is_onset:
            self.last_onset = self.samples_seen
            return self.samples_seen - n   # start of the block that contained the hit
        return None


def detect_onsets(y, **kwargs):
    """Offline helper: run the streaming detector over a whole recording."""
    det = OnsetDetector(**kwargs)
    hop = det.hop
    out = []
    for i in range(0, len(y) - hop + 1, hop):
        o = det.process(y[i:i + hop])
        if o is not None:
            out.append(o)
    return out


def slice_clips(y, onsets):
    """Cut a training clip around each onset, aligned exactly as the live engine does."""
    clips = []
    for o in onsets:
        start = max(0, o - C.PRE_ONSET_SAMPLES)
        clip = y[start:start + C.CLIP_SAMPLES]
        if len(clip) == C.CLIP_SAMPLES:
            clips.append(clip)
    return clips
