"""A small CNN that classifies one log-mel window (~100 ms) into a sound class.

Tiny on purpose (~19k parameters): it trains in a minute on a laptop CPU and runs
in about a millisecond, which keeps live latency down.
"""
import numpy as np
import torch
import torch.nn as nn

from . import config as C
from .features import clip_features


class SoundNet(nn.Module):
    def __init__(self, n_classes, n_mels=C.N_MELS):
        super().__init__()

        def block(cin, cout, pool):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, 16, (2, 2)),
            block(16, 32, (2, 2)),
            block(32, 48, (2, 1)),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(48, n_classes))

    def forward(self, x):            # x: (batch, 1, n_mels, frames)
        return self.head(self.features(x))


class Classifier:
    """Wraps a trained SoundNet for live use: raw audio clip in, (label, confidence) out."""

    def __init__(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = ckpt["classes"]
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.model = SoundNet(len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)     # one thread is faster for a single tiny input

    def predict(self, clip):
        x = (clip_features(clip) - self.mean) / self.std
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x[None, None]).float())
            probs = torch.softmax(logits, dim=1)[0].numpy()
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i]), probs
