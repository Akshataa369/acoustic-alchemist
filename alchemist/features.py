"""Log-mel spectrogram, written from scratch in NumPy.

No librosa dependency: it keeps the live path fast and means every step of the
pipeline is ours to explain to the jury.
"""
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import config as C


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr=C.SAMPLE_RATE, n_fft=C.N_FFT, n_mels=C.N_MELS, fmin=30.0, fmax=None):
    """Triangular filters spaced evenly on the mel scale. Shape: (n_mels, n_fft//2 + 1)."""
    fmax = fmax or sr / 2
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    # Normalise each filter by its width so wide high-frequency bands don't dominate
    fb *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(C.N_FFT).astype(np.float32)


def stft_mag(y, n_fft=C.N_FFT, hop=C.HOP):
    """Magnitude spectrogram. Shape: (frames, n_fft//2 + 1)."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    n_frames = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = y[idx] * _WINDOW
    return np.abs(np.fft.rfft(frames, axis=1))


def log_mel(y):
    """Log-mel spectrogram of a clip. Shape: (n_mels, frames)."""
    power = stft_mag(y) ** 2
    mel = power @ _FB.T
    return np.log(mel + 1e-6).T.astype(np.float32)


def fix_length(y, n=C.CLIP_SAMPLES):
    y = np.asarray(y, dtype=np.float32)
    return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))


def normalize_clip(y):
    """Peak-normalise so loud and quiet hits look alike to the model."""
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y))
    return y / peak if peak > 1e-5 else y


def clip_features(y):
    """The full feature step used by BOTH training and live inference."""
    return log_mel(fix_length(normalize_clip(y)))


N_FRAMES = log_mel(np.zeros(C.CLIP_SAMPLES)).shape[1]


# 4th-order 150 Hz high-pass before onset detection: removes rumble, mains hum and desk bumps.
# On a test recording with hum and 25 desk bumps added, false triggers fell from 6 to 1 and
# all 39 real sounds were still found (70 Hz, 2nd order did not help: 6 false).
# The website applies the same filter (two cascaded biquads), so both see the same signal.
_HP = butter(4, 150, btype="highpass", fs=C.SAMPLE_RATE, output="sos")


def highpass(y):
    """Filter a whole recording."""
    return sosfilt(_HP, np.asarray(y, dtype=np.float64)).astype(np.float32)


class StreamingHighpass:
    """Filter block by block, keeping state between blocks (for live audio)."""

    def __init__(self):
        self.zi = sosfilt_zi(_HP) * 0.0

    def __call__(self, block):
        out, self.zi = sosfilt(_HP, np.asarray(block, dtype=np.float64), zi=self.zi)
        return out.astype(np.float32)
