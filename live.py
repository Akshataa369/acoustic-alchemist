"""Acoustic-Alchemist: control a computer with any sound you can make.

    python live.py                      # CONTROL mode: your sounds press keys (next slide, play/pause...)
    python live.py --dry-run            # control mode, but only print the actions (safe for testing)
    python live.py --mode drums         # MUSIC mode: your sounds play drums
    python live.py --mode both          # drums + whistle melodies
    python live.py --list-devices       # pick a mic / speaker if the default is wrong
    python live.py --replay take.wav --out result.wav    # run offline on a recording

Control mode needs permission on a Mac, once:
    System Settings -> Privacy & Security -> Accessibility -> turn on Terminal (or VS Code)
Edit ACTION_MAP in alchemist/config.py to choose what each sound does.

While running, type a command and press Enter:
    p   pause/resume listening (a sound can do this too: "hum" by default)
    r   record a loop (starts on the next downbeat, records 2 bars, then loops)
    u   undo the last loop layer        c   clear the loop
    m   cycle mode: control / drums / whistle / both
    t   metronome on/off                +/- tempo up/down by 5 BPM
    g   show the loop grid               s   latency + accuracy stats
    q   quit

Use headphones: otherwise the mic hears the speakers and triggers itself.
"""
import argparse
import queue
import sys
import threading

import numpy as np

from alchemist import config as C
from alchemist import synth
from alchemist.actions import ACTIONS, ActionController
from alchemist.engine import Engine, Looper
from alchemist.model import Classifier

COLORS = {"pop": 31, "click": 35, "hiss": 36, "hum": 33, "tap": 34, "noise": 90}


def fmt_event(e):
    if e.kind == "note":
        return f"  \033[95m♪ {e.label:<6}\033[0m whistle  conf {e.confidence:.2f}"
    bar = "█" * int(e.confidence * 10) + "░" * (10 - int(e.confidence * 10))
    color = COLORS.get(e.label, 37) if e.played else 90
    status = f"  -> {e.action}" if e.action else ("" if e.played else "  (ignored)")
    return f"  \033[{color}m{e.label.upper():<6}\033[0m {bar} {e.confidence:.2f}  {e.latency_ms:5.0f} ms{status}"


def stats(engine):
    hits = [e for e in engine.events if e.kind == "hit"]
    if not hits:
        return "  no hits yet"
    lat = np.array([e.latency_ms for e in hits])
    counts = {}
    for e in hits:
        counts[e.label] = counts.get(e.label, 0) + 1
    played = sum(e.played for e in hits)
    return (f"  hits: {len(hits)} ({played} played)   latency: mean {lat.mean():.0f} ms, "
            f"p95 {np.percentile(lat, 95):.0f} ms\n  by class: {counts}")


def run_live(args):
    import sounddevice as sd

    clf = Classifier(args.model)
    mixer = synth.Mixer()
    looper = Looper(bpm=args.bpm)
    looper.metronome = False
    actions = ActionController(dry_run=args.dry_run)
    engine = Engine(clf, mixer, looper, mode=args.mode, sensitivity=args.sensitivity, actions=actions,
                    on_event=lambda e: print(fmt_event(e), flush=True))

    blocks = queue.Queue()

    def in_cb(indata, frames, t, status):
        blocks.put(indata[:, 0].copy())       # keep the callback tiny; work happens elsewhere

    def out_cb(outdata, frames, t, status):
        looper.tick(frames, mixer)
        outdata[:, 0] = mixer.render(frames)

    def worker():
        while True:
            b = blocks.get()
            if b is None:
                break
            engine.process(b)

    threading.Thread(target=worker, daemon=True).start()
    in_stream = sd.InputStream(samplerate=C.SAMPLE_RATE, blocksize=C.BLOCK_SIZE, channels=1,
                               dtype="float32", callback=in_cb, device=args.input, latency="low")
    out_stream = sd.OutputStream(samplerate=synth.OUT_SR, blocksize=256, channels=1, dtype="float32",
                                 callback=out_cb, device=args.output, latency="low")

    print(f"\n  Acoustic-Alchemist is listening. Mode: {engine.mode}. Classes: {clf.classes}")
    if engine.mode == "control":
        print("  What each sound does:")
        for snd, act in C.ACTION_MAP.items():
            if snd in clf.classes:
                print(f"    {snd:<6} -> {ACTIONS.get(act, act)}")
        if actions.dry_run:
            print("  (dry run: actions are printed, not performed)")
    print("  Commands: p pause | m mode | s stats | r record loop | u undo | c clear | t metronome | "
          "+/- tempo | g grid | q quit\n")
    with in_stream, out_stream:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            elif cmd == "r":
                looper.toggle_record()
                print(f"  loop: {'armed, starts on the next downbeat' if looper.armed else 'stopped'}")
            elif cmd == "u":
                looper.undo(); print("  loop: undone")
            elif cmd == "c":
                looper.clear(); print("  loop: cleared")
            elif cmd == "p":
                engine.listening = not engine.listening
                print(f"  listening: {'on' if engine.listening else 'PAUSED'}")
            elif cmd == "m":
                modes = ["control", "drums", "whistle", "both"]
                engine.mode = modes[(modes.index(engine.mode) + 1) % 4]
                print(f"  mode: {engine.mode}")
            elif cmd == "t":
                looper.metronome = not looper.metronome
                print(f"  metronome: {'on' if looper.metronome else 'off'}")
            elif cmd in ("+", "-"):
                looper.set_bpm(looper.bpm + (5 if cmd == "+" else -5)); print(f"  tempo: {looper.bpm} BPM")
            elif cmd == "g":
                rows, marker = looper.grid()
                print("\n".join(rows) + "\n" + marker if rows else "  loop is empty")
            elif cmd == "s":
                print(stats(engine))
    blocks.put(None)
    print("\n" + stats(engine))


def run_replay(args):
    """Feed a recording through the engine and render what it would have played."""
    from alchemist.dataset import read_wav, write_wav
    from scipy.signal import resample

    y = read_wav(args.replay)
    clf = Classifier(args.model)
    mixer = synth.Mixer()
    engine = Engine(clf, mixer, None, mode=args.mode, sensitivity=args.sensitivity,
                    actions=ActionController(dry_run=True), on_event=lambda e: print(fmt_event(e)))

    ratio = synth.OUT_SR / C.SAMPLE_RATE
    out = []
    for i in range(0, len(y) - C.BLOCK_SIZE + 1, C.BLOCK_SIZE):
        engine.process(y[i:i + C.BLOCK_SIZE])
        out.append(mixer.render(int(round((i + C.BLOCK_SIZE) * ratio)) - sum(len(o) for o in out)))
    rendered = np.concatenate(out)
    original = resample(y, int(len(y) * ratio))[:len(rendered)]
    mix = 0.35 * original / (np.max(np.abs(original)) + 1e-9) + rendered
    print("\n" + stats(engine))
    if args.out:
        write_wav(args.out, mix / max(1.0, np.max(np.abs(mix))), sr=synth.OUT_SR)
        print(f"  wrote {args.out} (your take, quietly, + what the system played)")
    return engine


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/model.pt")
    p.add_argument("--mode", choices=["control", "drums", "whistle", "both"], default="control")
    p.add_argument("--dry-run", action="store_true", help="control mode: print actions instead of pressing keys")
    p.add_argument("--bpm", type=int, default=C.DEFAULT_BPM)
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="onset sensitivity: raise (e.g. 1.5) if soft hits are missed, lower in a noisy room")
    p.add_argument("--input", type=int, help="input device index")
    p.add_argument("--output", type=int, help="output device index")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--replay", help="process a .wav file instead of the mic")
    p.add_argument("--out", help="with --replay: write the result here")
    args = p.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
    elif args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
