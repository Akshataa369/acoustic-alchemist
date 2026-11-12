# Acoustic-Alchemist

**A computer you can control with any sound you can make.** Acoustic-Alchemist learns the small sounds a person can make, like a lip pop, a tongue click or a hum, and turns each one into an action on the computer: next slide, play or pause, scroll, open. It runs on an ordinary laptop microphone and learns *each person's own version* of each sound.

> *Independence is made of small moments: turning a page, pausing a video, answering a message without having to ask. We want to give a few of them back.*

**Live demo:** https://akshataa369.github.io/acoustic-alchemist/ (open in Chrome and allow the microphone)

Built by **Team Winners** for *Hack on Holiday 2026*, Reinforce Club, Scaler School of Technology.
Problem statement: *"Acoustic-Alchemist": Non-Verbal Audio & Environmental Trigger System*.

---

## The problem

Computers assume you can use your hands. Voice assistants assume you can speak clearly. Many people can do neither easily.

- **1 in 6 people** in the world, about 1.3 billion, live with a significant disability ([WHO, 2023][who-disability]).
- **Over 15 million people** live with a spinal cord injury ([WHO, 2024][who-sci]).
- **2.5 billion people** need at least one assistive product, and **nearly 1 billion** are denied access. Access ranges from **3%** in poorer countries to **90%** in wealthy ones ([WHO & UNICEF, 2022][who-at]).
- In India, **26.8 million people** (2.21% of the population) were counted with a disability in the 2011 Census ([Factly][india]).

The conditions that affect movement, such as ALS, cerebral palsy and stroke, often affect speech as well. ALS, for example, progressively weakens the muscles used to speak ([JSLHR][als]). That leaves people out of both hands-based and voice-based computing, and specialist devices like eye trackers are out of reach for most.

## Our answer

Most systems expect people to fit them. Acoustic-Alchemist does it the other way round:

- **Any sound, any voice.** The person records a few examples of sounds that are comfortable for them, and a small neural network learns exactly those sounds. An unusual-sounding click is fine.
- **No special hardware.** A normal laptop microphone.
- **Private.** Everything runs on the computer. No audio is uploaded.
- **Safe by design.** It acts only when it's at least 70% sure, waits 0.25 s between actions, and one sound pauses and resumes it, so a cough in a meeting can't flip a slide.
- **Fun too.** Music mode turns the same sounds into drums, which makes practising feel like a game and makes the demo instant to understand.

It could help people with spinal cord injury, ALS, cerebral palsy, after a stroke, with arthritis or tremor, with a temporary injury, or anyone whose hands are busy.

> This is a student prototype, not a medical device. The next step is testing it with the people it's meant for.

## The sounds

| Sound | How to make it | Control mode (default) | Music mode |
|---|---|---|---|
| `pop` | Lips together, then release ("p") | Right arrow: next slide, next photo, skip forward | Kick drum |
| `click` | Tongue click ("tsk") | Left arrow: previous slide, back | Rimshot |
| `hiss` | Short "ts" or "sss" | Space: play/pause, select | Hi-hat |
| `hum` | Short "mm", mouth closed | Pause / resume listening | Tom |
| `tap` | Knuckle, finger, foot or switch, if possible | Enter: open, confirm | Snare |
| `noise` | Talking, breathing, typing, coughing | Nothing (it learns to ignore these) | Nothing |

Change what each sound does in `ACTION_MAP` in `alchemist/config.py`. Available actions: `next`, `previous`, `select`, `enter`, `scroll_down`, `scroll_up`, `click`, `toggle_listening`, `none`.

## Architecture

```
                 ┌──────────────── every 16 ms audio block ────────────────┐
 mic ──► Onset detector ──► wait 80 ms ──► log-mel ──► SoundNet CNN ──► sound + confidence
 16 kHz  (mel-band spectral   (hear enough    (40 bands    (~19k params,          │
          flux + adaptive      of the sound)   x 9 frames,  ~1 ms on CPU)          ▼
          threshold + gate)                    NumPy)                     noise, or not sure enough?
                                                                            │yes        │no
                                                                         ignore   ┌─────┴──────┐
                                                                                  ▼            ▼
                                                                        CONTROL: key press   MUSIC: drum
                                                                        (cooldown, pause)    + looper
 mic ──► YIN pitch tracker ──► snap to scale ──► marimba note     (whistle, music mode only)
```

**Why it's built this way**

- **Onset first, classify second.** The onset detector finds *when* a sound happened; the CNN only decides *what*. Running the CNN on every block would be wasteful and trigger-happy.
- **The 80 ms trade-off.** We must hear enough of a sound before judging it. A longer window is more accurate but slower. This is `CLIP_SECONDS` in `config.py` and our main experiment.
- **Stricter in control mode.** A wrong action costs more than a wrong drum, so control mode needs 70% confidence (music mode 55%), plus a cooldown and a pause sound.
- **Mel-band spectral flux.** Our first onset detector measured flux across raw FFT bins and missed 10 of 41 low "kick-like" sounds in a test recording, because high-frequency hiss dominated. Measuring flux across 24 mel bands caught 41/41 with no false triggers.
- **Same code for training and live use.** Clips are cut with the same detector and window in both, so the model trains on what it will actually see.
- **A "noise" class.** The model learns what *not* to react to, which matters when it's listening all day.
- **Built from scratch where it matters.** Log-mel features, onset detection and YIN pitch tracking are written in NumPy. No pretrained audio model: one trained on dogs and sirens doesn't know one person's tongue click.
- **Heavy augmentation.** With ~100 recordings per sound, the model must still work in another room, at another volume, on a tired day. We randomise volume, timing (±10 ms), speed (±10 %), background noise, microphone colour and mel-band masking.

## Setup

Python 3.10–3.12 is required (macOS's built-in 3.9 is too old). On a Mac, install 3.12 from python.org, then:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Control mode on a Mac needs one permission:** System Settings → Privacy & Security → **Accessibility** → turn on Terminal (or VS Code). Without it, macOS silently blocks the key presses. The first recording also asks for **Microphone** access.

## Easiest way to try it: the website

`tester.html` explains the problem and lets you try everything in the browser: record your sounds, check how reliable they are, control a hands-free reader with them, play drums, and download the recordings for training. It uses the same onset detector and features as the Python code (verified to match exactly).

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000/tester.html** in Chrome and allow the microphone. Opening the file directly from Finder won't work, because browsers only allow the mic on `localhost` or `https`.

**Easiest start:** press **Quick start** on the website. It speaks each instruction, records by itself, checks the result and starts listening, so after that one click nothing else needs pressing. One sound (a lip pop) plus a few seconds of everyday noise is enough, and **one-sound mode** lets that single sound reach every action by picking from options that light up in turn. Next time, "Welcome back" skips setup entirely.

The website uses a quick nearest-neighbour classifier so it learns instantly. The real model is the CNN in `train.py`.

## Quick start (Python)

**1. Record your sounds** (30 s each; vary distance and loudness on purpose):

```bash
python record.py pop   --who akshata
python record.py click --who akshata
python record.py hiss  --who akshata
python record.py hum   --who akshata
python record.py tap   --who akshata
python record.py noise --who akshata     # talk, breathe, type, cough: things to ignore
```

Or use the website and download `data.zip`. Record each sound at least 3 times, ideally both of you, in different rooms. Training holds out whole recordings as the test set, so the score reflects sessions the model has never heard.

**2. Train** (about a minute):

```bash
python train.py
```

This saves `models/model.pt` and writes `reports/report.txt` with accuracy, a confusion matrix and a baseline comparison.

**3. Use it:**

```bash
python live.py --dry-run       # control mode, printing actions only: test this first
python live.py                 # control mode: your sounds press real keys
python live.py --mode drums    # music mode
python live.py --mode both     # drums + whistle melodies
```

Try control mode with a slideshow open in Keynote, PowerPoint or Google Slides, or a YouTube video. Commands while running (type, then Enter): `p` pause · `m` mode · `s` stats · `q` quit. In music mode: `r` record loop · `u` undo · `c` clear · `t` metronome · `+`/`-` tempo · `g` grid.

**Offline replay** (for testing and a reliable demo video):

```bash
python live.py --replay my_take.wav --out result.wav --mode drums
```

**No mic? Check the pipeline with synthetic sounds:**

```bash
python tools/make_test_data.py
python train.py --data data_synthetic
python live.py --replay test_performance.wav --out result.wav --mode drums
```

Synthetic sounds are far cleaner than real ones, so their scores say nothing about real accuracy. They only prove the pipeline runs.

## Project layout

```
tester.html      the website: the problem, the facts, and a working demo
alchemist/
  config.py      all settings: sounds, actions, window size, thresholds
  features.py    log-mel spectrogram in NumPy
  onset.py       streaming onset detector + clip slicing
  model.py       SoundNet CNN + live classifier wrapper
  dataset.py     loading, augmentation
  actions.py     turns sounds into key presses and mouse actions (pynput)
  engine.py      live engine: control mode, music mode, looper (testable without audio)
  pitch.py       YIN pitch tracker, scale snapping
  synth.py       procedural drum sounds + mixer
record.py        collect and auto-slice training data
train.py         train + evaluate + baseline
live.py          real-time app and offline replay
tools/make_test_data.py   synthetic data for testing without a mic
```

## Experiments to run (and log)

| Experiment | How | What to report |
|---|---|---|
| Window length vs accuracy vs delay | `CLIP_SECONDS` = 0.06 / 0.10 / 0.15, retrain | Accuracy and delay for each; which we chose and why |
| CNN vs baseline | In `reports/report.txt` | Does the CNN earn its complexity? |
| Augmentation | `python train.py --aug 0` vs `--aug 8` | Test accuracy with and without |
| A new person | Train on one of us, test on the other | How well it works for someone it hasn't heard |
| False actions | Talk and type for 5 minutes in control mode | How many actions fired by mistake (target: 0) |
| Noisy room | Record with a fan or music on | Missed sounds, false triggers; tune `--sensitivity` |
| Onset detector | Raw-bin flux vs mel-band flux | 31/41 vs 41/41 on our test take (already measured) |

## Results

*(Fill in from `reports/report.txt` after training on real recordings.)*

| Metric | Value |
|---|---|
| Test accuracy (unseen recordings) | |
| Baseline (logistic regression) | |
| Accuracy on a person it wasn't trained on | |
| False actions in 5 minutes of talking and typing | |
| Mean delay from sound to action | |
| Model size / inference time | ~19k parameters / |

## Roadmap

- Test with people who have motor or speech impairments, and with occupational therapists
- Quick personal setup: 5 examples per sound, fine-tuned in under a minute
- More actions: mouse movement, dwell clicking, launching apps
- Run fully in the browser so it works on any device with no install
- Combine with existing tools like on-screen keyboards and switch-access scanning

## Sources

1. [WHO, Disability fact sheet][who-disability] (7 March 2023)
2. [WHO & UNICEF, Global report on assistive technology, news release][who-at] (16 May 2022)
3. [WHO, Spinal cord injury fact sheet][who-sci] (16 April 2024)
4. [Factly, Census 2011 data on persons with disabilities][india]
5. [Speech intelligibility decline in ALS, Journal of Speech, Language, and Hearing Research][als]

[who-disability]: https://www.who.int/news-room/fact-sheets/detail/disability-and-health
[who-at]: https://www.who.int/news-room/releases/16-05-2022-almost-one-billion-children-and-adults-with-disabilities-and-older-persons-in-need-of-assistive-technology-denied-access--according-to-new-report
[who-sci]: https://www.who.int/news-room/fact-sheets/detail/spinal-cord-injury
[india]: https://factly.in/data-persons-with-disabilities-make-up-more-than-2-of-indias-population-with-some-states-reporting-a-higher-than-national-average
[als]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6693567

## Use of AI tools

We used AI assistants for boilerplate and debugging. Data collection, model training and evaluation, experiments and design decisions are our own, and are documented in our progress logs.
