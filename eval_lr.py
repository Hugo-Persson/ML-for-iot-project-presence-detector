"""
Evaluation script for the logistic regression presence detector.
Loads all WAV files from test_data directory and outputs aggregate confusion matrix.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


SAMPLE_RATE = 16000
CHUNK_DURATION_S = 0.5
SEGMENT_DURATION_S = 2.0
SEGMENT_HOP_CHUNKS = 1

CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_S)  # 8000
CHUNK_FEATURES = 207
SEGMENT_CHUNKS = int(SEGMENT_DURATION_S / CHUNK_DURATION_S)  # 4
MODEL_INPUTS = SEGMENT_CHUNKS * CHUNK_FEATURES  # 828


def parse_weights_header(path: Path) -> tuple[np.ndarray, float]:
    """Parse LR weights and bias from the Arduino C header file."""
    text = path.read_text()

    weight_pattern = r"static const float LR_WEIGHTS_INIT\[.*?\]\s*=\s*\{([^}]+)\}"
    weight_match = re.search(weight_pattern, text, re.DOTALL)
    if not weight_match:
        raise ValueError(f"Could not parse weights from {path}")

    weight_str = weight_match.group(1)
    weights = []
    for val in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?", weight_str):
        weights.append(float(val.rstrip("f")))
    weights = np.array(weights, dtype=np.float32)

    bias_pattern = r"static const float LR_BIAS_INIT\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?f?)"
    bias_match = re.search(bias_pattern, text)
    if not bias_match:
        raise ValueError(f"Could not parse bias from {path}")
    bias = float(bias_match.group(1).rstrip("f"))

    return weights, bias


def iter_chunks(signal: np.ndarray, n: int):
    """Iterate over non-overlapping chunks of size n."""
    for i in range(0, len(signal), n):
        if i + n > len(signal):
            return
        yield signal[i : i + n]


def chunk_fft_features(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Extract FFT-based features from audio signal."""
    chunked = np.asarray(list(iter_chunks(signal, CHUNK_SAMPLES)), dtype=np.float32)
    if chunked.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    fft_mag = np.abs(np.fft.rfft(chunked, axis=1))
    n_chunks, n_bins = fft_mag.shape

    bin_width = sample_rate / CHUNK_SAMPLES  # 2 Hz

    def hz_to_bin(hz: float) -> int:
        return int(np.floor(hz / bin_width))

    bin_320 = min(hz_to_bin(320.0), n_bins)
    bin_3200 = min(hz_to_bin(3200.0), n_bins)

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
    """Create sliding window segments of 4 chunks each."""
    total_chunks = feature_array.shape[0]
    if total_chunks < SEGMENT_CHUNKS:
        return np.empty((0, SEGMENT_CHUNKS, feature_array.shape[1]), dtype=feature_array.dtype)

    segments = []
    for start in range(0, total_chunks - SEGMENT_CHUNKS + 1, max(1, SEGMENT_HOP_CHUNKS)):
        end = start + SEGMENT_CHUNKS
        segments.append(feature_array[start:end])
    return np.stack(segments, axis=0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def predict(
    segments: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on segments. Returns (probabilities, binary_predictions)."""
    x = segments.reshape(-1, MODEL_INPUTS).astype(np.float32)
    x = x / (x.mean(axis=1, keepdims=True) + np.float32(1e-6))

    z = x @ weights + bias
    probs = sigmoid(z)
    preds = (probs >= threshold).astype(np.int32)

    return probs, preds


def label_from_filename(filename: str) -> int | None:
    """Extract label from filename. Returns 1 for presence, 0 for no_presence, None if unknown."""
    name = filename.lower()
    if "no_presence" in name:
        return 0
    elif "presence" in name:
        return 1
    return None


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path | None = None) -> None:
    """Print confusion matrix and metrics, optionally save as PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print()
    print("                  Predicted")
    print("                  no_presence  presence")
    print(f"Actual no_presence    {cm[0, 0]:5d}       {cm[0, 1]:5d}")
    print(f"       presence       {cm[1, 0]:5d}       {cm[1, 1]:5d}")
    print()

    print("=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["no_presence", "presence"],
            zero_division=0,
        )
    )

    if output_path is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["no_presence", "presence"],
            yticklabels=["no_presence", "presence"],
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix - LR Presence Detector", fontsize=14)

        accuracy = (y_pred == y_true).mean()
        plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.2%}", ha="center", fontsize=11)

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nConfusion matrix saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LR model on all WAV files in test_data directory."
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("test_data"),
        help="Directory containing test WAV files (default: test_data)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("arduino/ghost_detector/lr_weights.h"),
        help="Path to weights header file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("confusion_matrix.png"),
        help="Output path for confusion matrix PNG (default: confusion_matrix.png)",
    )
    args = parser.parse_args()

    if not args.test_dir.exists():
        raise SystemExit(f"Test directory not found: {args.test_dir}")
    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    weights, bias = parse_weights_header(args.weights)
    print(f"Loaded {len(weights)} weights, bias={bias:.6f}")

    wav_files = sorted(args.test_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.test_dir}")

    print(f"Found {len(wav_files)} WAV files in {args.test_dir}")
    print()

    all_y_true = []
    all_y_pred = []
    all_probs = []

    for wav_path in wav_files:
        label = label_from_filename(wav_path.name)
        if label is None:
            print(f"Skipping {wav_path.name}: cannot determine label from filename")
            continue

        audio, sr = sf.read(wav_path, dtype="float32")
        if sr != SAMPLE_RATE:
            print(f"Skipping {wav_path.name}: sample rate {sr} != {SAMPLE_RATE}")
            continue

        features = chunk_fft_features(audio, sr)
        segments = make_segments(features)

        if len(segments) == 0:
            print(f"Skipping {wav_path.name}: too short")
            continue

        probs, preds = predict(segments, weights, bias, args.threshold)

        y_true = np.full(len(segments), label, dtype=np.int32)
        all_y_true.append(y_true)
        all_y_pred.append(preds)
        all_probs.append(probs)

        if args.verbose:
            label_str = "presence" if label == 1 else "no_presence"
            correct = (preds == label).sum()
            total = len(preds)
            acc = correct / total
            print(f"{wav_path.name}: {total} segments, {correct}/{total} correct ({acc:.1%}), label={label_str}")

    if not all_y_true:
        raise SystemExit("No valid WAV files processed")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    print(f"\nTotal segments: {len(y_true)}")
    print(f"  - no_presence: {(y_true == 0).sum()}")
    print(f"  - presence: {(y_true == 1).sum()}")

    print_confusion_matrix(y_true, y_pred, args.output)

    accuracy = (y_pred == y_true).mean()
    print(f"Overall accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
