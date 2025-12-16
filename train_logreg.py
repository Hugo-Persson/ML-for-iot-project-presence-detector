from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
CHUNK_DURATION_S = 0.5
SEGMENT_DURATION_S = 2.0
SEGMENT_HOP_CHUNKS = 1

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)  # 8000
CHUNK_FEATURES = 207
SEGMENT_CHUNKS = int(SEGMENT_DURATION_S / CHUNK_DURATION_S)  # 4
MODEL_INPUTS = SEGMENT_CHUNKS * CHUNK_FEATURES  # 828


def iter_chunks(signal: np.ndarray, n: int):
    for i in range(0, len(signal), n):
        if i + n > len(signal):
            return
        yield signal[i : i + n]


def chunk_fft_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    chunked = np.asarray(list(iter_chunks(signal, CHUNK_SAMPLES)), dtype=np.float32)
    if chunked.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    fft_mag = np.abs(np.fft.rfft(chunked, axis=1))
    n_chunks, n_bins = fft_mag.shape

    bin_width = sample_rate / CHUNK_SAMPLES  # 2 Hz

    def hz_to_bin(hz: float) -> int:
        return int(np.floor(hz / bin_width))

    bin_320 = min(hz_to_bin(320.0), n_bins)  # 0–320 Hz
    bin_3200 = min(hz_to_bin(3200.0), n_bins)  # 320–3200 Hz

    def merge_region(region: np.ndarray, target_bw_hz: float) -> np.ndarray:
        if region.shape[1] == 0:
            return region[:, :0]

        bins_per = max(1, int(round(target_bw_hz / bin_width)))
        usable = (region.shape[1] // bins_per) * bins_per
        if usable == 0:
            return region[:, :0]

        region = region[:, :usable]
        return region.reshape(n_chunks, -1, bins_per).mean(axis=2)

    low_raw = fft_mag[:, :bin_320]
    mid_raw = fft_mag[:, bin_320:bin_3200]
    high_raw = fft_mag[:, bin_3200:]

    low = merge_region(low_raw, 4.0)
    mid = merge_region(mid_raw, 32.0)
    high = merge_region(high_raw, 128.0)

    features = np.concatenate([low, mid, high], axis=1).astype(np.float32)
    if features.shape[1] != CHUNK_FEATURES:
        raise RuntimeError(f"Expected {CHUNK_FEATURES} features, got {features.shape[1]}")
    return features


def make_segments(feature_array: np.ndarray) -> np.ndarray:
    total_chunks = feature_array.shape[0]
    if total_chunks < SEGMENT_CHUNKS:
        return np.empty((0, SEGMENT_CHUNKS, feature_array.shape[1]), dtype=feature_array.dtype)

    segments = []
    for start in range(0, total_chunks - SEGMENT_CHUNKS + 1, max(1, SEGMENT_HOP_CHUNKS)):
        end = start + SEGMENT_CHUNKS
        segments.append(feature_array[start:end])
    return np.stack(segments, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    l2: float,
    epochs: int,
    batch_size: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    n = x.shape[0]

    # Shuffle / split
    indices = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx]
    y_val = y[val_idx]

    w = np.zeros((x.shape[1],), dtype=np.float32)
    b = np.float32(0.0)
    best_w = w.copy()
    best_b = float(b)
    best_val_loss = float("inf")

    def eval_metrics(x_eval: np.ndarray, y_eval: np.ndarray):
        p = sigmoid(x_eval @ w + b)
        eps = 1e-6
        p_clip = np.clip(p, eps, 1.0 - eps)
        loss = float(-(y_eval * np.log(p_clip) + (1.0 - y_eval) * np.log(1.0 - p_clip)).mean())
        acc = float(((p >= 0.5).astype(np.float32) == y_eval).mean())
        return loss, acc

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(x_train.shape[0])
        x_train = x_train[perm]
        y_train = y_train[perm]

        for start in range(0, x_train.shape[0], batch_size):
            xb = x_train[start : start + batch_size]
            yb = y_train[start : start + batch_size]

            z = xb @ w + b
            p = sigmoid(z)
            grad = (p - yb).astype(np.float32)

            grad_w = (xb.T @ grad) / np.float32(len(xb)) + np.float32(l2) * w
            grad_b = grad.mean()

            w -= np.float32(learning_rate) * grad_w
            b -= np.float32(learning_rate) * grad_b

        tr_loss, tr_acc = eval_metrics(x_train, y_train)
        va_loss, va_acc = eval_metrics(x_val, y_val)
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_w = w.copy()
            best_b = float(b)
        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val loss={va_loss:.4f} acc={va_acc:.3f}  "
            f"best_val_loss={best_val_loss:.4f}"
        )

    return best_w, best_b


def export_header(weights: np.ndarray, bias: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt_f(x: float) -> str:
        # Compact but stable output for Arduino compilation.
        return f"{x:.8g}f"

    lines = []
    lines.append("#pragma once")
    lines.append("")
    lines.append("// Auto-generated by train_logreg.py")
    lines.append(f"constexpr int LR_MODEL_INPUTS = {weights.shape[0]};")
    lines.append("")
    lines.append("static const float LR_WEIGHTS_INIT[LR_MODEL_INPUTS] = {")
    for i, v in enumerate(weights.tolist()):
        sep = "," if i + 1 < weights.shape[0] else ""
        lines.append(f"  {fmt_f(float(v))}{sep}")
    lines.append("};")
    lines.append("")
    lines.append(f"static const float LR_BIAS_INIT = {fmt_f(float(bias))};")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} ({weights.shape[0]} weights).")


def main():
    parser = argparse.ArgumentParser(description="Train tiny logistic regression and export Arduino header.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("arduino/ghost_detector/lr_weights.h"),
        help="Output header path.",
    )
    args = parser.parse_args()

    wav_files = sorted(Path(".").glob("*.wav"))
    labeled = []
    for wav_path in wav_files:
        stem = wav_path.stem
        if "no_presence" in stem:
            label = 0.0
        elif "presence" in stem:
            label = 1.0
        else:
            continue
        labeled.append((wav_path, np.float32(label)))

    if not labeled:
        raise SystemExit("No labeled wav files found (filenames should contain 'presence' or 'no_presence').")

    segments_all = []
    labels_all = []
    for wav_path, label in labeled:
        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            raise SystemExit(f"Sample rate mismatch for {wav_path}: {sr} != {SAMPLE_RATE}")

        feats = chunk_fft_features(audio, sr)
        segs = make_segments(feats)
        if len(segs) == 0:
            continue
        segments_all.append(segs)
        labels_all.append(np.full((len(segs),), label, dtype=np.float32))
        print(f"{wav_path.name}: segments={len(segs)}")

    x = np.concatenate(segments_all, axis=0).reshape(-1, MODEL_INPUTS).astype(np.float32)
    y = np.concatenate(labels_all, axis=0).astype(np.float32)
    print("dataset", x.shape, y.shape, "pos_rate", float(y.mean()))

    # Match the Arduino on-device trainer: scale by 1/mean(feature) per sample.
    x = x / (x.mean(axis=1, keepdims=True) + np.float32(1e-6))

    w, b = train_logreg(
        x,
        y,
        learning_rate=args.lr,
        l2=args.l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    export_header(w, b, args.out)


if __name__ == "__main__":
    main()
