"""Shared settings. Change things here, not scattered through the code."""

SAMPLE_RATE = 16000          # Hz. 16 kHz is plenty for percussive sounds and keeps the model fast.
BLOCK_SIZE = 256             # Samples per audio callback (16 ms). Smaller = lower latency, more CPU.

# Feature extraction (log-mel spectrogram)
N_FFT = 512
HOP = 128                    # 8 ms hop
N_MELS = 40
# The window classified around each onset. This is THE latency/accuracy trade-off:
# we must wait (CLIP_SECONDS - PRE_ONSET_SECONDS) after a hit before we can classify it.
# 0.10 s total -> ~80 ms of waiting. Try 0.06 / 0.10 / 0.15 and report the results.
CLIP_SECONDS = 0.10
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
PRE_ONSET_SECONDS = 0.02     # Keep a little audio before the detected onset (attack transient)
PRE_ONSET_SAMPLES = int(SAMPLE_RATE * PRE_ONSET_SECONDS)
POST_ONSET_SAMPLES = CLIP_SAMPLES - PRE_ONSET_SAMPLES

# The sound classes the model learns. They are deliberately small mouth sounds that most
# people can make without their hands or clear speech. Each person records THEIR OWN
# version, so an unusual-sounding "click" is fine. "noise" = anything that must NOT trigger.
CLASSES = ["pop", "click", "hiss", "hum", "tap", "noise"]

DESCRIPTIONS = {
    "pop": "Lip pop: press your lips together and release ('p')",
    "click": "Tongue click: 'tsk' or a cluck against the roof of the mouth",
    "hiss": "Short hiss: 'ts' or 'sss'",
    "hum": "Short hum: 'mm', mouth closed",
    "tap": "Tap: knuckle, finger, foot or a switch, if you can",
    "noise": "Talking, typing, breathing, coughing: things to IGNORE",
}

# CONTROL MODE (the main use): what each sound does on the computer.
# Available actions: next, previous, select, enter, scroll_down, scroll_up,
#                    click, toggle_listening, none
ACTION_MAP = {
    "pop": "next",                 # Right arrow: next slide / next photo / skip forward in a video
    "click": "previous",           # Left arrow: previous slide / back
    "hiss": "select",              # Space: play/pause video, select, page down in a browser
    "hum": "toggle_listening",     # Pause/resume, so a cough in a meeting can't flip slides
    "tap": "enter",                # Enter: open / confirm
}
CONTROL_CONFIDENCE = 0.70    # Control mode is stricter than music: a wrong action costs more
ACTION_COOLDOWN = 0.25       # Seconds between actions, so one long sound can't fire twice

# MUSIC MODE (the fun demo): which drum each sound plays.
SOUND_MAP = {
    "pop": "kick",
    "click": "rim",
    "hiss": "hat",
    "hum": "tom",
    "tap": "snare",
}

# Live engine
CONFIDENCE_THRESHOLD = 0.55  # Music mode: below this, an onset is ignored (treated as noise)
REFRACTORY_SECONDS = 0.07    # Ignore new onsets this soon after the last one

# Looper
DEFAULT_BPM = 100
BEATS_PER_BAR = 4
LOOP_BARS = 2
QUANTIZE_STEPS_PER_BEAT = 4  # 16th notes

# Whistle mode
PITCH_MIN_HZ = 500           # Whistles typically sit around 500-3000 Hz
PITCH_MAX_HZ = 2500
SCALE = [0, 2, 4, 7, 9]      # Major pentatonic: hard to play a wrong note
