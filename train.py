"""Train the sound classifier and write an evaluation report.

    python train.py
    python train.py --epochs 40 --aug 10

Outputs:
    models/model.pt          the trained model (used by live.py)
    reports/metrics.json     accuracy, per-class scores, confusion matrix, baseline
    reports/report.txt       the same, readable: paste it into your progress log
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from alchemist import config as C
from alchemist.dataset import featurize, load_dataset
from alchemist.model import SoundNet


def split(labels, sources, rng_seed):
    """Hold out whole recording sessions where possible, so test = unseen takes."""
    test = np.zeros(len(labels), dtype=bool)
    rng = np.random.default_rng(rng_seed)
    for c in np.unique(labels):
        sess = np.unique(sources[labels == c])
        if len(sess) >= 3:
            held = rng.choice(sess, size=max(1, len(sess) // 4), replace=False)
            test |= (labels == c) & np.isin(sources, held)
        else:
            idx = np.where(labels == c)[0]
            if len(idx) >= 5:
                _, te = train_test_split(idx, test_size=0.2, random_state=rng_seed)
                test[te] = True
    return ~test, test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--aug", type=int, default=8, help="augmented copies per clip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/model.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    clips, labels, sources = load_dataset(args.data)
    present = [C.CLASSES[i] for i in sorted(set(labels.tolist()))]
    print(f"Loaded {len(clips)} clips: " +
          ", ".join(f"{C.CLASSES[c]}={int((labels == c).sum())}" for c in sorted(set(labels.tolist()))))
    missing = [c for c in C.CLASSES if c not in present]
    if missing:
        print(f"  ! No data yet for: {missing}. Record them with record.py.")
    # Re-index to the classes that actually have data
    remap = {C.CLASSES.index(n): i for i, n in enumerate(present)}
    y_all = np.array([remap[l] for l in labels])

    tr, te = split(y_all, sources, args.seed)
    train_clips = [clips[i] for i in np.where(tr)[0]]
    test_clips = [clips[i] for i in np.where(te)[0]]
    noise_bank = [clips[i] for i in np.where(tr)[0] if present[y_all[i]] == "noise"]

    t0 = time.time()
    Xtr, itr = featurize(train_clips, rng, noise_bank, n_aug=args.aug)
    ytr = y_all[tr][itr]
    Xte, _ = featurize(test_clips)
    yte = y_all[te]
    print(f"Features: train {Xtr.shape}, test {Xte.shape} ({time.time() - t0:.1f}s)")

    mean = Xtr.mean(axis=(0, 2), keepdims=True)[0]
    std = Xtr.std(axis=(0, 2), keepdims=True)[0] + 1e-5
    norm = lambda X: torch.from_numpy(((X - mean) / std)[:, None]).float()
    Xtr_t, Xte_t = norm(Xtr), norm(Xte)
    ytr_t = torch.from_numpy(ytr).long()

    model = SoundNet(len(present))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.epochs *
                                                ((len(ytr) + 63) // 64))
    # Class weights, so a class with fewer recordings isn't ignored
    counts = np.bincount(ytr, minlength=len(present)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(counts.sum() / (len(counts) * np.maximum(counts, 1))),
                                  label_smoothing=0.05)

    print(f"Training SoundNet ({n_params:,} params) for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr_t))
        total = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"  epoch {epoch + 1:3d}  loss {total / len(perm):.3f}"
            if len(yte):
                model.eval()
                with torch.no_grad():
                    acc = (model(Xte_t).argmax(1).numpy() == yte).mean()
                msg += f"  test acc {acc:.3f}"
            print(msg)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": present, "mean": mean, "std": std,
                "config": {"clip_seconds": C.CLIP_SECONDS, "n_mels": C.N_MELS}}, args.out)
    print(f"\nSaved model to {args.out}")

    if not len(yte):
        print("No held-out test data (record more takes per class to get a test score).")
        return

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).numpy()
    acc = float((pred == yte).mean())

    # Baseline: logistic regression on the average spectrum. If the CNN doesn't
    # clearly beat this, the extra complexity isn't earning its keep.
    base = LogisticRegression(max_iter=2000).fit(Xtr.mean(axis=2), ytr)
    base_acc = float((base.predict(Xte.mean(axis=2)) == yte).mean())

    # Inference speed for one clip
    x1 = Xte_t[:1]
    with torch.no_grad():
        for _ in range(10):
            model(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            model(x1)
    infer_ms = (time.perf_counter() - t0) / 200 * 1000

    labels_idx = list(range(len(present)))
    cm = confusion_matrix(yte, pred, labels=labels_idx)
    rep = classification_report(yte, pred, labels=labels_idx, target_names=present, zero_division=0)
    metrics = {"test_accuracy": acc, "baseline_logreg_accuracy": base_acc, "n_train_clips": len(train_clips),
               "n_test_clips": len(test_clips), "params": n_params, "inference_ms": infer_ms,
               "classes": present, "confusion_matrix": cm.tolist(),
               "window_seconds": C.CLIP_SECONDS}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))

    w = max(len(c) for c in present) + 2
    cm_txt = " " * w + "".join(f"{c[:6]:>7}" for c in present) + "   <- predicted\n"
    for i, row in enumerate(cm):
        cm_txt += f"{present[i]:<{w}}" + "".join(f"{v:7d}" for v in row) + "\n"
    report = (f"Test accuracy (CNN):            {acc:.3f}\n"
              f"Baseline (logistic regression): {base_acc:.3f}\n"
              f"Model size: {n_params:,} params   Inference: {infer_ms:.2f} ms/clip\n"
              f"Window: {C.CLIP_SECONDS * 1000:.0f} ms   Train clips: {len(train_clips)}   "
              f"Test clips: {len(test_clips)}\n\n{rep}\nConfusion matrix (rows = true)\n{cm_txt}")
    Path("reports/report.txt").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
