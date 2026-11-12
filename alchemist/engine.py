r"""The live engine, kept separate from audio I/O so it can be tested offline.

    mic block -> onset detector -> wait ~80 ms -> CNN -> control: keyboard/mouse action
                                                     \-> music: drum sound + looper
    mic block -> YIN pitch tracker -> note snapped to scale -> pluck   (whistle mode)
"""
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import synth
from .features import StreamingHighpass
from .onset import OnsetDetector
from .pitch import hz_to_midi, midi_to_hz, note_name, snap_to_scale, yin


@dataclass
class Event:
    t: float                 # seconds since start (input clock)
    kind: str                # "hit" or "note"
    label: str
    confidence: float
    latency_ms: float = 0.0
    played: bool = True
    action: str = ""          # control mode: the action performed ("" if none)


class Looper:
    """A step sequencer that records hits quantised to a 16th-note grid and loops them."""

    def __init__(self, bpm=C.DEFAULT_BPM, bars=C.LOOP_BARS, out_sr=synth.OUT_SR):
        self.out_sr = out_sr
        self.bars = bars
        self.lock = threading.Lock()
        self.set_bpm(bpm)
        self.pattern = {}            # step -> set of sound names
        self.pos = 0                 # output samples elapsed
        self.armed = False           # waiting for the next loop start to record
        self.recording = False
        self.metronome = True
        self.history = []            # undo stack of patterns

    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = max(50, min(200, bpm))
            self.steps = self.bars * C.BEATS_PER_BAR * C.QUANTIZE_STEPS_PER_BEAT
            self.step_len = self.out_sr * 60.0 / self.bpm / C.QUANTIZE_STEPS_PER_BEAT

    def step_at(self, pos):
        return int(round(pos / self.step_len)) % self.steps

    def add_hit(self, sound, seconds_ago=0.0):
        """Called when a hit is detected. seconds_ago corrects for detection delay."""
        with self.lock:
            if not self.recording:
                return
            pos = self.pos - seconds_ago * self.out_sr
            self.pattern.setdefault(self.step_at(pos), set()).add(sound)

    def toggle_record(self):
        with self.lock:
            if self.recording:
                self.recording = False
            else:
                self.history.append({k: set(v) for k, v in self.pattern.items()})
                self.armed = True

    def undo(self):
        with self.lock:
            self.recording = self.armed = False
            self.pattern = self.history.pop() if self.history else {}

    def clear(self):
        with self.lock:
            self.history.append(self.pattern)
            self.pattern = {}
            self.recording = self.armed = False

    def tick(self, n, mixer):
        """Advance by n output samples, firing any steps crossed. Called from the audio thread."""
        with self.lock:
            start, end = self.pos, self.pos + n
            first = int(np.ceil(start / self.step_len))
            last = int(np.floor((end - 1) / self.step_len))
            for s in range(first, last + 1):
                step = s % self.steps
                if step == 0:
                    if self.armed:                    # start recording on the loop's downbeat
                        self.armed, self.recording = False, True
                    elif self.recording:              # one full pass recorded: stop
                        self.recording = False
                if self.metronome and step % C.QUANTIZE_STEPS_PER_BEAT == 0:
                    mixer.play(synth.metronome_tick(accent=step % (C.QUANTIZE_STEPS_PER_BEAT *
                                                                   C.BEATS_PER_BAR) == 0), 0.6)
                for sound in self.pattern.get(step, ()):
                    mixer.play(synth.KIT[sound], 0.8)
            self.pos = end

    def grid(self):
        """Text view of the pattern, for the terminal display."""
        with self.lock:
            sounds = sorted({s for v in self.pattern.values() for s in v})
            cur = int(self.pos / self.step_len) % self.steps
            rows = []
            for snd in sounds:
                row = "".join(("X" if snd in self.pattern.get(i, ()) else
                               ("|" if i % C.QUANTIZE_STEPS_PER_BEAT == 0 else "."))
                              for i in range(self.steps))
                rows.append(f"{snd:>5} {row}")
            marker = " " * (6 + cur) + "^"
            return rows, marker


class Engine:
    def __init__(self, classifier, mixer, looper=None, mode="control", on_event=None, sensitivity=1.0,
                 actions=None):
        self.clf = classifier
        self.mixer = mixer
        self.looper = looper
        self.mode = mode                         # "control", "drums", "whistle" or "both"
        self.actions = actions                   # ActionController, used in control mode
        self.listening = True                    # control mode: paused/resumed by a sound
        self.last_action_t = -1e9
        self.onsets = OnsetDetector(sensitivity=sensitivity)
        self.highpass = StreamingHighpass()
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_start = 0                       # absolute sample index of buf[0]
        self.pending = []                        # onset sample indices awaiting classification
        self.on_event = on_event or (lambda e: None)
        self.events = []
        self.pitch_hist = []
        self.last_note = None
        self.muted = set()

    def process(self, block):
        """Feed one input block. Returns the events it produced."""
        block = self.highpass(np.asarray(block, dtype=np.float32))
        self.buf = np.concatenate([self.buf, block])
        produced = []

        if self.mode in ("control", "drums", "both"):
            o = self.onsets.process(block)
            if o is not None:
                self.pending.append(o)
            produced += self._classify_ready()
        else:
            self.onsets.process(block)           # keep the detector's history warm

        if self.mode in ("whistle", "both"):
            produced += self._track_pitch()

        # Keep about 1 s of audio
        keep = C.SAMPLE_RATE
        if len(self.buf) > keep:
            drop = len(self.buf) - keep
            self.buf = self.buf[drop:]
            self.buf_start += drop
        return produced

    def _classify_ready(self):
        out = []
        end = self.buf_start + len(self.buf)
        ready = [o for o in self.pending if o + C.POST_ONSET_SAMPLES <= end]
        self.pending = [o for o in self.pending if o + C.POST_ONSET_SAMPLES > end]
        for o in ready:
            s = o - C.PRE_ONSET_SAMPLES - self.buf_start
            clip = self.buf[max(0, s):max(0, s) + C.CLIP_SAMPLES]
            t0 = time.perf_counter()
            label, conf, _ = self.clf.predict(clip)
            infer = (time.perf_counter() - t0) * 1000
            wait_ms = (end - o) / C.SAMPLE_RATE * 1000
            ev = Event(t=o / C.SAMPLE_RATE, kind="hit", label=label, confidence=conf,
                       latency_ms=wait_ms + infer)
            if self.mode == "control":
                self._control(ev, o)
            else:
                self._music(ev)
            out.append(ev)
            self.events.append(ev)
            self.on_event(ev)
        return out

    def _control(self, ev, onset):
        """Turn a confident hit into an action, with a cooldown and a pause switch."""
        action = C.ACTION_MAP.get(ev.label, "none")
        t = onset / C.SAMPLE_RATE
        ok = (ev.label != "noise" and ev.confidence >= C.CONTROL_CONFIDENCE and action != "none"
              and t - self.last_action_t >= C.ACTION_COOLDOWN)
        # While paused, only the pause/resume sound is obeyed
        if ok and not self.listening and action != "toggle_listening":
            ok = False
        if not ok:
            ev.played = False
            return
        self.last_action_t = t
        if action == "toggle_listening":
            self.listening = not self.listening
            ev.action = "resume listening" if self.listening else "pause listening"
            self.mixer.play(synth.metronome_tick(accent=self.listening), 1.0)
        else:
            ev.action = action
            if self.actions:
                self.actions.perform(action)
            self.mixer.play(synth.metronome_tick(accent=True), 0.5)   # quiet "done" beep

    def _music(self, ev):
        label, conf = ev.label, ev.confidence
        sound = C.SOUND_MAP.get(label)
        if label == "noise" or conf < C.CONFIDENCE_THRESHOLD or sound is None or label in self.muted:
            ev.played = False
        else:
            self.mixer.play(synth.KIT[sound], 0.5 + 0.5 * conf)
            if self.looper:
                self.looper.add_hit(sound, seconds_ago=ev.latency_ms / 1000)

    def _track_pitch(self):
        frame = self.buf[-1024:]
        if len(frame) < 1024:
            return []
        f, conf = yin(frame)
        if f is None or conf < 0.8:
            self.pitch_hist.clear()
            self.last_note = None
            return []
        note = snap_to_scale(hz_to_midi(f))
        self.pitch_hist = (self.pitch_hist + [note])[-3:]
        # Require the same note on 3 consecutive blocks (~48 ms) before playing it
        if len(self.pitch_hist) == 3 and len(set(self.pitch_hist)) == 1 and note != self.last_note:
            self.last_note = note
            play_note = note - 12                   # whistles are high: drop an octave
            self.mixer.play(synth.pluck(midi_to_hz(play_note)), 0.8)
            ev = Event(t=(self.buf_start + len(self.buf)) / C.SAMPLE_RATE, kind="note",
                       label=note_name(play_note), confidence=conf, latency_ms=3 * C.BLOCK_SIZE /
                       C.SAMPLE_RATE * 1000)
            self.events.append(ev)
            self.on_event(ev)
            return [ev]
        return []
