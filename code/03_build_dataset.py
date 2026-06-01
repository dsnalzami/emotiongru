"""
03_build_dataset.py  (REBUILD)
===============================

author: Dr. Eng. Farrikh Alzami, Sri Winarno Ph.D, M. Naufal M.Eng, Dewi Agustini Santoso MCS.

Build final dataset arrays from landmark extraction results (02_extract_landmarks).

CHANGES FROM PREVIOUS VERSION:
  - Correct smoothing order: full N frames BEFORE windowing (per framework Sec 3.1)
  - Three output arrays: X_f0 (none), X_f1 (SG), X_f2 (OEF)
  - X_RAW_NPY alias retained for backward compatibility (points to X_f0)
  - New figure: fig_smoothing_effect.png

Pipeline order per sequence (locked):
  [1] Load full (T, 468, 3) from .npy  -- truly raw MediaPipe output
  [2] Apply smoothing on ALL T frames  -- BEFORE windowing
  [3] Apply last-P=10 windowing        -- reflect-pad if T < P
  [4] Stack into X array               -- no normalization, no augmentation

Design decisions (locked):
  - X_f0/f1/f2: truly raw MediaPipe (no normalization, no augmentation)
  - Normalization (nose-center + XY scale + Z fold-scale): in training loop
  - Z scaling: per training fold only  (leakage prevention)
  - Augmentation: training only, in training loop

INPUT (Kaggle dataset from 02_extract_landmarks):
  /kaggle/input/.../landmark_manifest.csv
  /kaggle/input/.../landmarks/*.npy  -- full T-frame sequences

OUTPUT (/kaggle/working):
  X_f0.npy           (309, 10, 468, 3) float32  -- F0 no smoothing
  X_f1.npy           (309, 10, 468, 3) float32  -- F1 Savitzky-Golay
  X_f2.npy           (309, 10, 468, 3) float32  -- F2 One Euro Filter
  X_raw.npy          alias for X_f0 (backward compat)
  y_6cls.npy         (309,) int32
  subjects_6cls.npy  (309,) object
  sequence_info.csv
  dataset_summary.txt
  fig_smoothing_effect.png  (NEW)
  fig_class_distribution.png
  fig_sequence_length.png
"""

import gc
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

# ===========================================================================
# CONFIG -- copy from 00_config.py
# If any parameter changes, update 00_config.py first, then copy here.
# ===========================================================================

LANDMARK_DATASET = "/kaggle/input/datasets/udnyahyaalzami/02-extract-landmarks-results"
MANIFEST_CSV_IN  = os.path.join(LANDMARK_DATASET, "landmark_manifest.csv")
LANDMARK_DIR_IN  = os.path.join(LANDMARK_DATASET, "landmarks")

OUTPUT_DIR        = "/kaggle/working"
X_F0_NPY          = os.path.join(OUTPUT_DIR, "X_f0.npy")
X_F1_NPY          = os.path.join(OUTPUT_DIR, "X_f1.npy")
X_F2_NPY          = os.path.join(OUTPUT_DIR, "X_f2.npy")
X_RAW_NPY         = X_F0_NPY   # backward compatibility alias
Y_6CLS_NPY        = os.path.join(OUTPUT_DIR, "y_6cls.npy")
SUBJECTS_NPY      = os.path.join(OUTPUT_DIR, "subjects_6cls.npy")
SEQUENCE_INFO_CSV = os.path.join(OUTPUT_DIR, "sequence_info.csv")

CKPLUS_TO_IDX = {
    1: 0,   # anger
    3: 1,   # disgust
    4: 2,   # fear
    5: 3,   # happiness
    6: 4,   # sadness
    7: 5,   # surprise
}

EMOTION_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
]

N_CLASSES    = 6
N_LANDMARKS  = 468
N_COORDS     = 3
NOSE_TIP_IDX = 4
CHIN_IDX     = 152
P            = 10
PAD_MODE     = "reflect"

# Smoothing parameters (from 00_config.py)
SG_WINDOW    = 5
SG_POLYORDER = 2
OEF_FC_MIN   = 1.0
OEF_BETA     = 0.1
OEF_D_CUTOFF = 1.0

SMOOTH_VARIANTS  = ["F0_none", "F1_savgol", "F2_oef"]
SMOOTH_OUT_PATHS = {
    "F0_none"   : X_F0_NPY,
    "F1_savgol" : X_F1_NPY,
    "F2_oef"    : X_F2_NPY,
}

# ===========================================================================
# WINDOWING (unchanged from previous version)
# ===========================================================================

def apply_last_p_window(arr: np.ndarray, p: int) -> np.ndarray:
    """
    Take last-P frames from a sequence. Reflect-pad at START if N < P.

    Args:
        arr: (N_frames, 468, 3) float32
        p:   window size

    Returns:
        (P, 468, 3) float32

    Reflect-pad note:
      numpy reflect mode requires pad_width <= N-1.
      CK+ 6-class min N=6. Max pad needed = P-6=4, 4 <= 5=N-1. Safe.
      Fallback to 'edge' if N < 2.
    """
    n = arr.shape[0]
    if n >= p:
        return arr[-p:].copy()

    pad_size = p - n
    if n >= 2 and pad_size <= n - 1:
        padded = np.pad(
            arr,
            pad_width=((pad_size, 0), (0, 0), (0, 0)),
            mode=PAD_MODE,
        )
    else:
        padded = np.pad(
            arr,
            pad_width=((pad_size, 0), (0, 0), (0, 0)),
            mode="edge",
        )

    assert padded.shape == (p, N_LANDMARKS, N_COORDS), (
        "Padding error: got {} expected ({}, {}, {})".format(
            padded.shape, p, N_LANDMARKS, N_COORDS)
    )
    return padded


# ===========================================================================
# PATH RESOLUTION (unchanged from previous version)
# IMPORTANT: session must be read as str (zero-padding preserved).
# ===========================================================================

def resolve_npy_path(subject: str, session: str, emotion_name: str) -> str:
    """
    Return path to .npy if it exists in LANDMARK_DIR_IN, else empty string.

    subject      : e.g. "S005"
    session      : e.g. "001" (string, zero-padded)
    emotion_name : e.g. "disgust"
    """
    fname    = "{}_{}_{}.npy".format(subject, session, emotion_name)
    fullpath = os.path.join(LANDMARK_DIR_IN, fname)
    return fullpath if os.path.exists(fullpath) else ""


# ===========================================================================
# SMOOTHING FUNCTIONS
# All functions operate on a SINGLE full sequence: (T, N_LANDMARKS, N_COORDS)
# ===========================================================================

def _sg_smooth_seq(arr: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golay filter on one full sequence.

    arr: (T, N_LANDMARKS, N_COORDS) float32
    Returns: (T, N_LANDMARKS, N_COORDS) float32

    Guard: if T < SG_WINDOW, return arr unchanged (no-op).
    Filter applied along axis=0 (time axis).
    """
    T = arr.shape[0]
    if T < SG_WINDOW:
        # Sequence too short for this window length -- no-op
        return arr.copy().astype(np.float32)

    smoothed = scipy_signal.savgol_filter(
        arr.astype(np.float64),
        window_length=SG_WINDOW,
        polyorder=SG_POLYORDER,
        axis=0,
    )
    return smoothed.astype(np.float32)


def _oef_smooth_seq(arr: np.ndarray) -> np.ndarray:
    """
    One Euro Filter on one full sequence.

    Vectorized over all M = N_LANDMARKS * N_COORDS = 1404 coordinate series
    simultaneously. Sequential loop runs only over T time steps.

    Reference: Casiez et al. 2012 (CHI). DOI: 10.1145/2207676.2208639
    Parameters from config: fc_min=1.0, beta=0.1, d_cutoff=1.0, fs=1.0.

    Algorithm per time step (vectorized over M=1404):
      1. dx      = (x[t] - x_filt[t-1]) * fs
      2. dx_filt = alpha_d * dx + (1-alpha_d) * dx_filt   [derivative filter]
      3. fc      = fc_min + beta * |dx_filt|               [adaptive cutoff]
      4. alpha   = te / (te + 1/(2*pi*fc))                 [signal alpha]
      5. x_filt[t] = alpha * x[t] + (1-alpha) * x_filt[t-1]

    arr: (T, N_LANDMARKS, N_COORDS) float32
    Returns: (T, N_LANDMARKS, N_COORDS) float32
    """
    T = arr.shape[0]
    if T <= 1:
        return arr.copy().astype(np.float32)

    fs  = 1.0        # unit frame rate
    te  = 1.0 / fs   # = 1.0

    # Derivative filter: fixed alpha (constant d_cutoff)
    tau_d   = 1.0 / (2.0 * np.pi * OEF_D_CUTOFF)
    alpha_d = te / (te + tau_d)   # scalar

    # Reshape to (T, M) for vectorized computation over all coordinates
    M       = arr.shape[1] * arr.shape[2]   # 468 * 3 = 1404
    x       = arr.reshape(T, M).astype(np.float64)
    x_filt  = x.copy()                      # (T, M)
    dx_filt = np.zeros(M, dtype=np.float64) # (M,)

    for t in range(1, T):
        # Derivative estimate: vectorized over M
        dx      = (x[t] - x_filt[t - 1]) * fs

        # Low-pass filter on derivative
        dx_filt = alpha_d * dx + (1.0 - alpha_d) * dx_filt

        # Adaptive signal cutoff (per coordinate)
        fc     = OEF_FC_MIN + OEF_BETA * np.abs(dx_filt)   # (M,)
        tau_fc = 1.0 / (2.0 * np.pi * fc)                  # (M,)
        alpha  = te / (te + tau_fc)                         # (M,)

        # Filter signal
        x_filt[t] = alpha * x[t] + (1.0 - alpha) * x_filt[t - 1]

    return x_filt.reshape(arr.shape).astype(np.float32)


def apply_smoothing(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Dispatcher for smoothing modes. Operates on one full sequence.

    arr:  (T, N_LANDMARKS, N_COORDS)
    mode: "F0_none" | "F1_savgol" | "F2_oef"
    Returns: (T, N_LANDMARKS, N_COORDS) float32
    """
    if mode == "F0_none":
        return arr.copy().astype(np.float32)
    elif mode == "F1_savgol":
        return _sg_smooth_seq(arr)
    elif mode == "F2_oef":
        return _oef_smooth_seq(arr)
    else:
        raise ValueError("Unknown smoothing mode: {}".format(mode))


# ===========================================================================
# FIGURES
# ===========================================================================

def save_fig_smoothing_effect(
    sequences: List[np.ndarray],
    info_rows: List[dict],
) -> None:
    """
    Plot X coordinate of three key landmarks for one representative sequence,
    comparing F0/F1/F2 smoothing applied to ALL T frames (before windowing).

    Key landmarks plotted:
      idx 4   -- nose tip       (stable reference)
      idx 61  -- left lip corner (expressive during happiness/surprise)
      idx 291 -- right lip corner
    """
    # Select a happiness or surprise sequence with the most frames
    best_idx    = 0
    best_frames = 0
    for i, row in enumerate(info_rows):
        nf = row["n_frames_orig"]
        if row["emotion_name"] in ("happiness", "surprise") and nf > best_frames:
            best_frames = nf
            best_idx    = i

    arr_raw = sequences[best_idx]           # (T, 468, 3), full N frames
    T       = arr_raw.shape[0]
    arr_sg  = _sg_smooth_seq(arr_raw)
    arr_oef = _oef_smooth_seq(arr_raw)

    lm_indices = [4,    61,                    291]
    lm_labels  = ["Nose tip (idx 4)",
                  "Left lip corner (idx 61)",
                  "Right lip corner (idx 291)"]

    t      = np.arange(T)
    c_raw  = "#888888"
    c_sg   = "#4C72B0"
    c_oef  = "#C44E52"

    fig, axes = plt.subplots(len(lm_indices), 1, figsize=(10, 8), sharex=True)

    for ai, (lm_idx, lm_label) in enumerate(zip(lm_indices, lm_labels)):
        ax = axes[ai]
        ax.plot(t, arr_raw[:, lm_idx, 0],
                color=c_raw, linewidth=1.8, alpha=0.85,
                label="F0 raw (none)")
        ax.plot(t, arr_sg[:, lm_idx, 0],
                color=c_sg,  linewidth=1.5, linestyle="--",
                label="F1 Savitzky-Golay")
        ax.plot(t, arr_oef[:, lm_idx, 0],
                color=c_oef, linewidth=1.5, linestyle=":",
                label="F2 One Euro Filter")
        ax.set_ylabel("X coord (raw)", fontsize=8)
        ax.set_title(
            "{} -- T={} frames".format(lm_label, T),
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Frame index (full sequence, before windowing)", fontsize=9)

    info = info_rows[best_idx]
    plt.suptitle(
        "Smoothing Effect -- Subject {} Session {} Emotion: {}  (T={})".format(
            info["subject"], info["session"], info["emotion_name"], T,
        ),
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_smoothing_effect.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


def save_fig_class_distribution(y: np.ndarray) -> None:
    counts = np.bincount(y, minlength=N_CLASSES)
    total  = len(y)
    max_c  = counts.max()
    ir     = [(total - int(c)) / max(int(c), 1) for c in counts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax   = axes[0]
    bars = ax.bar(EMOTION_NAMES, counts,
                  color="#4C72B0", edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, str(val),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Class Distribution (CK+ 6-class, N={})".format(total), fontsize=12)
    ax.set_xlabel("Emotion", fontsize=10)
    ax.set_ylabel("Number of Sequences", fontsize=10)
    ax.set_ylim(0, max_c + 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2    = axes[1]
    colors = ["#C44E52" if r > 2.0 else "#55A868" for r in ir]
    bars2  = ax2.bar(EMOTION_NAMES, ir,
                     color=colors, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars2, ir):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02, "{:.2f}".format(val),
            ha="center", va="bottom", fontsize=9,
        )
    ax2.axhline(y=2.0, color="red", linestyle="--", linewidth=1,
                label="IR=2.0 threshold")
    ax2.set_title("Imbalance Ratio per Class  IR=(N-Nc)/Nc", fontsize=12)
    ax2.set_xlabel("Emotion", fontsize=10)
    ax2.set_ylabel("Imbalance Ratio", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_class_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


def save_fig_sequence_length(info_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes      = axes.flatten()
    colors    = ["#4C72B0", "#DD8452", "#55A868",
                 "#C44E52", "#8172B2", "#937860"]

    for idx, (name, color) in enumerate(zip(EMOTION_NAMES, colors)):
        ax     = axes[idx]
        subset = info_df[info_df["emotion_name"] == name]["n_frames_orig"].values
        n_pad  = int((subset < P).sum())

        ax.hist(subset, bins=15, color=color, edgecolor="black",
                linewidth=0.6, alpha=0.8)
        ax.axvline(x=P, color="red", linestyle="--", linewidth=1.5,
                   label="P={}".format(P))
        ax.set_title(
            "{} (n={}, pad={})".format(name, len(subset), n_pad),
            fontsize=10,
        )
        ax.set_xlabel("n_frames", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Original Frame Count Distribution per Emotion (red = P=10)",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig_sequence_length.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path))


# ===========================================================================
# SUMMARY TEXT
# ===========================================================================

def save_summary(
    info_df:   pd.DataFrame,
    y:         np.ndarray,
    n_padded:  int,
    n_skipped: int,
) -> None:
    counts = np.bincount(y, minlength=N_CLASSES)
    total  = len(y)
    lines  = []
    sep    = "=" * 60

    lines.append(sep)
    lines.append("03_build_dataset.py (REBUILD) -- DATASET SUMMARY")
    lines.append(sep)
    lines.append("")
    lines.append("OUTPUT ARRAYS (no normalization -- windowed only):")
    lines.append("  X_f0.npy  F0 none    : ({}, {}, {}, {}) float32".format(
        total, P, N_LANDMARKS, N_COORDS))
    lines.append("  X_f1.npy  F1 SavGol  : ({}, {}, {}, {}) float32".format(
        total, P, N_LANDMARKS, N_COORDS))
    lines.append("  X_f2.npy  F2 OEF     : ({}, {}, {}, {}) float32".format(
        total, P, N_LANDMARKS, N_COORDS))
    lines.append("  X_raw.npy alias      : X_f0.npy")
    lines.append("  y_6cls.npy           : ({},) int32".format(total))
    lines.append("  subjects_6cls.npy    : ({},) object".format(total))
    lines.append("")
    lines.append("PIPELINE ORDER (per sequence):")
    lines.append("  [1] Load full (T, 468, 3) from .npy (02 output)")
    lines.append("  [2] Apply smoothing on ALL T frames (BEFORE windowing)")
    lines.append("  [3] Apply last-P=10 windowing (reflect-pad if T<P)")
    lines.append("  Normalization (nose-center + XY/Z): in training loop")
    lines.append("  Augmentation: in training loop (training only)")
    lines.append("")
    lines.append("SMOOTHING PARAMETERS:")
    lines.append("  SG  : window={}, polyorder={}".format(SG_WINDOW, SG_POLYORDER))
    lines.append("  OEF : fc_min={}, beta={}, d_cutoff={}, fs=1.0".format(
        OEF_FC_MIN, OEF_BETA, OEF_D_CUTOFF))
    lines.append("")
    lines.append("  Windowing     : last-P={}, PAD_MODE={}".format(P, PAD_MODE))
    lines.append("  Padded (T<P)  : {}".format(n_padded))
    lines.append("  Skipped       : {}".format(n_skipped))
    lines.append("")
    lines.append("CLASS DISTRIBUTION:")
    lines.append("  {:<12} {:>5} {:>8} {:>10}".format(
        "Emotion", "Idx", "Count", "IR"))
    lines.append("  " + "-" * 38)
    for i, name in enumerate(EMOTION_NAMES):
        n  = int(counts[i])
        ir = (total - n) / max(n, 1)
        lines.append("  {:<12} {:>5} {:>8} {:>10.2f}".format(
            name, i, n, ir))
    lines.append("  Total              : {}".format(total))
    max_ir = max(
        [(total - int(counts[i])) / max(int(counts[i]), 1)
         for i in range(N_CLASSES)]
    )
    lines.append("  Max imbalance ratio: {:.2f}".format(max_ir))
    lines.append("")
    lines.append("FRAME COUNT STATS (original, before windowing):")
    fc = info_df["n_frames_orig"].values
    lines.append("  Min    : {}".format(int(fc.min())))
    lines.append("  Max    : {}".format(int(fc.max())))
    lines.append("  Mean   : {:.2f}".format(float(fc.mean())))
    lines.append("  Median : {:.1f}".format(float(np.median(fc))))
    lines.append("  Std    : {:.2f}".format(float(fc.std())))
    lines.append("")
    lines.append(sep)

    text = "\n".join(lines)
    print(text)

    path = os.path.join(OUTPUT_DIR, "dataset_summary.txt")
    with open(path, "w") as f:
        f.write(text)
    print("  Saved: {}".format(path))


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    print("[STEP 1] Load manifest ...")
    if not os.path.exists(MANIFEST_CSV_IN):
        raise FileNotFoundError(
            "Manifest not found: {}\n"
            "Ensure LANDMARK_DATASET is correct:\n  {}".format(
                MANIFEST_CSV_IN, LANDMARK_DATASET)
        )

    # dtype={"session": str} -- preserve zero-padding in session IDs
    manifest = pd.read_csv(MANIFEST_CSV_IN, dtype={"session": str})
    ok_mask  = manifest["status"].isin(["ok", "partial", "ok_resumed"])
    manifest = manifest[ok_mask].reset_index(drop=True)
    print("  Valid sequences (ok/partial/ok_resumed): {}".format(len(manifest)))

    print("  [DEBUG] Sample rows:")
    for _, r in manifest.head(3).iterrows():
        print("    subject={} session={!r} emotion={}".format(
            r["subject"], r["session"], r["emotion_name"]))

    if len(manifest) == 0:
        raise RuntimeError("No valid sequences in manifest.")

    # ------------------------------------------------------------------
    print("\n[STEP 2] Load full sequences from .npy files ...")

    raw_sequences: List[np.ndarray] = []
    y_list:        List[int]        = []
    subjects_list: List[str]        = []
    info_rows:     List[dict]       = []
    n_padded  = 0
    n_skipped = 0

    for _, row in manifest.iterrows():
        subject      = str(row["subject"])
        session      = str(row["session"])
        emotion_code = int(row["emotion_code"])
        emotion_name = str(row["emotion_name"])

        npy_path = resolve_npy_path(subject, session, emotion_name)
        if not npy_path:
            print("  WARN: not found, skip -> {}_{}_{}.npy".format(
                subject, session, emotion_name))
            n_skipped += 1
            continue

        arr = np.load(npy_path)  # (T, 468, 3)

        if (arr.ndim != 3
                or arr.shape[1] != N_LANDMARKS
                or arr.shape[2] != N_COORDS):
            print("  WARN: invalid shape {}, skip -> {}".format(
                arr.shape, npy_path))
            n_skipped += 1
            continue

        n_frames_orig = arr.shape[0]
        padded        = n_frames_orig < P
        if padded:
            n_padded += 1

        class_idx = CKPLUS_TO_IDX[emotion_code]

        raw_sequences.append(arr.astype(np.float32))
        y_list.append(class_idx)
        subjects_list.append(subject)
        info_rows.append({
            "sample_idx":    len(raw_sequences) - 1,
            "subject":       subject,
            "session":       session,
            "emotion_code":  emotion_code,
            "emotion_name":  emotion_name,
            "class_idx":     class_idx,
            "n_frames_orig": n_frames_orig,
            "padded":        padded,
            "npy_path":      npy_path,
        })

    n_total = len(raw_sequences)
    print("  Sequences loaded : {}".format(n_total))
    print("  Sequences padded : {}".format(n_padded))
    print("  Sequences skipped: {}".format(n_skipped))

    if n_total == 0:
        raise RuntimeError(
            "No sequences loaded. "
            "Check LANDMARK_DIR_IN: {}".format(LANDMARK_DIR_IN)
        )

    # Build shared arrays (identical for all smooth variants)
    y        = np.array(y_list, dtype=np.int32)
    subjects = np.array(subjects_list, dtype=object)
    info_df  = pd.DataFrame(info_rows)

    # ------------------------------------------------------------------
    print("\n[STEP 3] Apply smoothing + windowing for each variant ...")
    print("  Pipeline per sequence: load (T,468,3) -> smooth -> window -> stack")

    for mode in SMOOTH_VARIANTS:
        print("\n  [{}] ...".format(mode))
        windowed_list: List[np.ndarray] = []

        for i, arr in enumerate(raw_sequences):
            # [2] Smooth on FULL T frames
            smoothed = apply_smoothing(arr, mode)
            # [3] Last-P windowing
            windowed = apply_last_p_window(smoothed, P)
            windowed_list.append(windowed)

            if (i + 1) % 100 == 0 or (i + 1) == n_total:
                print("    [{}/{}]".format(i + 1, n_total))

        X = np.stack(windowed_list, axis=0).astype(np.float32)

        # Validate immediately before saving
        assert X.shape == (n_total, P, N_LANDMARKS, N_COORDS), \
            "Shape error for {}: {}".format(mode, X.shape)
        assert not np.isnan(X).any(), \
            "NaN detected in {} -- check source .npy files".format(mode)
        assert not np.isinf(X).any(), \
            "Inf detected in {}".format(mode)

        out_path = SMOOTH_OUT_PATHS[mode]
        np.save(out_path, X)
        print("  Saved: {}  shape={}  dtype={}".format(
            out_path, X.shape, X.dtype))

        del X, windowed_list
        gc.collect()

    # X_raw.npy is the same file as X_f0.npy (alias via shared path constant)
    # If for any reason paths differ, copy here
    if X_RAW_NPY != X_F0_NPY:
        import shutil
        shutil.copy2(X_F0_NPY, X_RAW_NPY)
        print("  Alias: {} -> {}".format(X_F0_NPY, X_RAW_NPY))

    # ------------------------------------------------------------------
    print("\n[STEP 4] Save y, subjects, sequence_info ...")

    np.save(Y_6CLS_NPY, y)
    print("  Saved: {}".format(Y_6CLS_NPY))

    np.save(SUBJECTS_NPY, subjects)
    print("  Saved: {}".format(SUBJECTS_NPY))

    info_df.to_csv(SEQUENCE_INFO_CSV, index=False)
    print("  Saved: {}".format(SEQUENCE_INFO_CSV))

    # ------------------------------------------------------------------
    print("\n[STEP 5] Final validation ...")

    X_f0_check = np.load(X_F0_NPY)
    assert X_f0_check.shape == (n_total, P, N_LANDMARKS, N_COORDS), \
        "X_f0 shape mismatch"
    assert len(np.unique(y)) == N_CLASSES, \
        "Expected {} classes, got {}".format(N_CLASSES, len(np.unique(y)))
    assert y.shape == (n_total,)
    assert subjects.shape == (n_total,)
    del X_f0_check
    print("  Validation PASSED")

    # ------------------------------------------------------------------
    print("\n[STEP 6] Generate figures ...")
    save_fig_smoothing_effect(raw_sequences, info_rows)
    save_fig_class_distribution(y)
    save_fig_sequence_length(info_df)

    # ------------------------------------------------------------------
    print("\n[STEP 7] Save dataset summary ...")
    save_summary(info_df, y, n_padded, n_skipped)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("BUILD DATASET COMPLETE (REBUILD)")
    print("=" * 60)
    counts = np.bincount(y, minlength=N_CLASSES)
    print("  Total samples: {}".format(n_total))
    print("  Subjects     : {}".format(len(np.unique(subjects))))
    print("  Per-class:")
    for i, name in enumerate(EMOTION_NAMES):
        print("    {:<12}: {}".format(name, counts[i]))
    print()
    print("  OUTPUT FILES:")
    for path in [
        X_F0_NPY, X_F1_NPY, X_F2_NPY,
        Y_6CLS_NPY, SUBJECTS_NPY, SEQUENCE_INFO_CSV,
        os.path.join(OUTPUT_DIR, "dataset_summary.txt"),
        os.path.join(OUTPUT_DIR, "fig_smoothing_effect.png"),
        os.path.join(OUTPUT_DIR, "fig_class_distribution.png"),
        os.path.join(OUTPUT_DIR, "fig_sequence_length.png"),
    ]:
        print("    {}".format(path))
    print("=" * 60)

    gc.collect()


if __name__ == "__main__":
    main()
