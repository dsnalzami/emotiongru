"""
04_model_factory.py
====================
Model factory: preprocessing utilities and Keras model builders for the
FER pipeline (CK+ 6-class, MediaPipe 468 3D landmarks).

USAGE:
  This file is SELF-CONTAINED. It is meant to be COPIED (not imported) into
  training scripts 05, 06, 07, 08. If any parameter changes, update
  00_config.py first, then copy the CONFIG block here.

  All models returned are UNCOMPILED. Compilation (loss + optimizer) is
  done in the training script (05/06/07/08) to support loss ablation.

CONTAINS:
  Preprocessing:
    normalize_sequence(X, z_scale)   -- nose-center + XY scale + optional Z
    compute_z_scale(X_train_normed)  -- median z-range from training fold only
    apply_delta_coords(X)            -- frame-to-frame velocity (A3 variant)

  Model builders:
    build_main_model()               -- shortcut for A2+B3+C1 (main model)
    build_variant_model(A, B, C)     -- all coord/extractor/temporal variants

  Loss and class weights:
    FocalLoss(gamma, class_weights)  -- focal loss for D3/D4 ablation
    get_class_weights(y_train)       -- balanced weights for D2 ablation

  Output:
    save_model_summary(model, dir)   -- text + optional PNG diagram
    main()                           -- self-test (no data files required)

SMOOTHING NOTE:
  Smoothing (F0/F1/F2) is applied in 03_build_dataset.py on full T frames
  BEFORE windowing. Training scripts load the appropriate X_f0/f1/f2.npy.

DATASET LOADING HELPER (for training scripts):
  SMOOTH_VARIANT_TO_FILE maps variant name to the .npy path.
  Example:
    X = np.load(SMOOTH_VARIANT_TO_FILE["F1_savgol"])
    y = np.load(Y_6CLS_NPY)

NORMALIZATION LEAKAGE PREVENTION:
  Z-scale must be computed from training fold ONLY:
    X_train_norm = normalize_sequence(X_train)          # XY only
    z_scale      = compute_z_scale(X_train_norm)        # training fold
    X_train_full = normalize_sequence(X_train, z_scale) # XY + Z
    X_test_full  = normalize_sequence(X_test,  z_scale) # same z_scale
"""

import gc
import os
from typing import Optional

import numpy as np
import tensorflow as tf

# ===========================================================================
# CONFIG -- copy from 00_config.py
# If any parameter changes, update 00_config.py first, then copy here.
# ===========================================================================

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATASET_ROOT      = "/kaggle/input/datasets/udnyahyaalzami/03-build-dataset-results"
X_F0_NPY          = os.path.join(DATASET_ROOT, "X_f0.npy")
X_F1_NPY          = os.path.join(DATASET_ROOT, "X_f1.npy")
X_F2_NPY          = os.path.join(DATASET_ROOT, "X_f2.npy")
X_RAW_NPY         = X_F0_NPY   # backward compat alias
Y_6CLS_NPY        = os.path.join(DATASET_ROOT, "y_6cls.npy")
SUBJECTS_NPY      = os.path.join(DATASET_ROOT, "subjects_6cls.npy")
SEQUENCE_INFO_CSV = os.path.join(DATASET_ROOT, "sequence_info.csv")
OUTPUT_DIR        = "/kaggle/working"

# ---------------------------------------------------------------------------
# EMOTION LABEL MAPPING
# ---------------------------------------------------------------------------
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
N_CLASSES = 6

# ---------------------------------------------------------------------------
# LANDMARK GEOMETRY
# ---------------------------------------------------------------------------
N_LANDMARKS  = 468
N_COORDS     = 3
NOSE_TIP_IDX = 4    # MediaPipe FaceMesh nose tip index (verified in 02)
CHIN_IDX     = 152  # MediaPipe FaceMesh chin bottom index (verified in 02)

# ---------------------------------------------------------------------------
# WINDOWING
# ---------------------------------------------------------------------------
P        = 10
PAD_MODE = "reflect"

# ---------------------------------------------------------------------------
# SMOOTHING (reference -- actual smoothing done in 03_build_dataset.py)
# ---------------------------------------------------------------------------
SG_WINDOW    = 5
SG_POLYORDER = 2
OEF_FC_MIN   = 1.0
OEF_BETA     = 0.1
OEF_D_CUTOFF = 1.0

SMOOTH_VARIANTS = ["F0_none", "F1_savgol", "F2_oef"]

# Mapping from smooth variant name to .npy input path
SMOOTH_VARIANT_TO_FILE = {
    "F0_none"   : X_F0_NPY,
    "F1_savgol" : X_F1_NPY,
    "F2_oef"    : X_F2_NPY,
}

# ---------------------------------------------------------------------------
# AUGMENTATION (reference -- actual augmentation done in training loop)
# ---------------------------------------------------------------------------
AUG_NOISE_SIGMA     = 0.002
MAJORITY_MULTIPLIER = 2
MINORITY_MULTIPLIER = 4
MINORITY_CLASS_INDICES = [2, 4]   # fear (idx 2), sadness (idx 4)

AUG_VARIANTS = ["G0_none", "G1_flip", "G2_targeted", "G3_flip_noise"]

# ---------------------------------------------------------------------------
# ARCHITECTURE
# ---------------------------------------------------------------------------
CNN_FILTERS           = [64, 128, 256, 128]
CNN_KERNELS           = [9,  7,   5,   3  ]
GRU_UNITS             = 64
GRU_RECURRENT_DROPOUT = 0.2
DROPOUT_RATE          = 0.4
DENSE_HEAD_UNITS      = 32

# ---------------------------------------------------------------------------
# TRAINING PROTOCOL (reference for training scripts)
# ---------------------------------------------------------------------------
SEEDS          = list(range(10))
ABLATION_SEEDS = [0, 1, 2]
BATCH_SIZE     = 16
MAX_EPOCHS     = 100
LR_INIT        = 1e-3
LR_FACTOR      = 0.5
LR_PATIENCE    = 10
LR_MIN         = 1e-5
ES_PATIENCE    = 20
CLIP_NORM      = 1.0
FOCAL_GAMMA_D3 = 1.0
FOCAL_GAMMA_D4 = 2.0

# ---------------------------------------------------------------------------
# ABLATION VARIANT IDENTIFIERS
# ---------------------------------------------------------------------------
COORD_VARIANTS     = ["A1_2d", "A2_3d", "A3_delta"]
EXTRACTOR_VARIANTS = ["B1_dense", "B2_cnn_small", "B3_cnn_main", "B4_cnn2d"]
TEMP_VARIANTS      = ["C1_gru", "C2_lstm", "C3_bigru"]
LOSS_VARIANTS      = ["D1_ce", "D2_weighted_ce", "D3_focal_g1", "D4_focal_g2"]
NORM_VARIANTS      = ["E1_raw", "E2_normalized"]


# ===========================================================================
# PREPROCESSING UTILITIES
# ===========================================================================

def normalize_sequence(
    X:       np.ndarray,
    z_scale: Optional[float] = None,
) -> np.ndarray:
    """
    Nose-center normalization + XY scale normalization.
    Optional per-fold Z scale normalization.

    Steps:
      [1] Subtract nose tip (NOSE_TIP_IDX=4) from all 468 landmarks.
          After this step, nose tip is at the origin.
      [2] Compute face height proxy: Euclidean distance from origin (nose)
          to chin (CHIN_IDX=152) in XY only (not Z).
      [3] Divide all XY coordinates by face height proxy.
      [4] (Optional) Divide Z coordinates by z_scale.

    Args:
        X:       (N, P, 468, 3) batch  OR  (P, 468, 3) single sequence
        z_scale: scalar from compute_z_scale(); if None, Z is not changed.
                 MUST be computed from training fold only (leakage prevention).

    Returns:
        Same shape as input, float32.
        After normalization: nose at (0,0,0), face height in XY ~ 1.0.
    """
    single = (X.ndim == 3)
    if single:
        X = X[np.newaxis]   # (1, P, 468, 3)

    out = X.copy().astype(np.float32)
    # out: (N, T, 468, 3)

    # [1] Center: subtract nose tip from all 468 landmarks
    nose = out[:, :, NOSE_TIP_IDX, :]        # (N, T, 3)
    out  = out - nose[:, :, np.newaxis, :]   # broadcast (N, T, 1, 3)

    # [2] Face height proxy: distance nose -> chin in XY
    #     After step [1], nose is at origin, so chin_centered = chin - nose
    chin_xy = out[:, :, CHIN_IDX, :2]               # (N, T, 2)
    scale   = np.linalg.norm(chin_xy, axis=-1)      # (N, T)
    scale   = np.maximum(scale, 1e-8)               # guard: avoid div by zero
    scale   = scale[:, :, np.newaxis, np.newaxis]   # (N, T, 1, 1)

    # [3] Normalize XY only
    out[:, :, :, :2] /= scale   # (N, T, 468, 2) / (N, T, 1, 1)

    # [4] Z normalization (optional, per training fold)
    if z_scale is not None:
        z_s = float(z_scale)
        if z_s > 0.0:
            out[:, :, :, 2] /= z_s
        else:
            print("  WARN: z_scale={:.6f} <= 0, skipping Z norm".format(z_s))

    if single:
        out = out[0]

    return out


def compute_z_scale(X_train_normed: np.ndarray) -> float:
    """
    Compute median Z-range from XY-normalized training data.

    Call AFTER normalize_sequence (XY only, z_scale=None) and BEFORE
    applying Z normalization. Use training fold data only.

    Formula: median over all training samples of (z_max - z_min),
    where max/min are taken over all frames and landmarks per sample.

    Args:
        X_train_normed: (N_train, P, 468, 3) after XY normalization

    Returns:
        float scalar -- median z-range (> 0); fallback 1.0 if <= 0.
    """
    z       = X_train_normed[:, :, :, 2]          # (N, P, 468)
    flat_z  = z.reshape(len(z), -1)               # (N, P*468)
    z_range = flat_z.max(axis=1) - flat_z.min(axis=1)  # (N,)
    z_scale = float(np.median(z_range))
    if z_scale <= 0.0:
        print("  WARN: compute_z_scale={:.6f}, using 1.0 fallback".format(z_scale))
        z_scale = 1.0
    return z_scale


def apply_delta_coords(X: np.ndarray) -> np.ndarray:
    """
    Compute frame-to-frame velocity (delta) features. Used for A3 variant.

    delta[t] = X[t] - X[t-1]
    delta[0] = 0 (first frame has no predecessor)

    Args:
        X: (N, P, 468, 3) or (P, 468, 3)

    Returns:
        Same shape as X, float32. delta[:, 0, :, :] is all zeros.
    """
    single = (X.ndim == 3)
    if single:
        X = X[np.newaxis]

    delta             = np.zeros_like(X, dtype=np.float32)
    delta[:, 1:, :, :] = (X[:, 1:, :, :] - X[:, :-1, :, :]).astype(np.float32)

    if single:
        delta = delta[0]

    return delta


# ===========================================================================
# AUGMENTATION UTILITIES
# (Applied in training loop, training split only. Included here for reference
#  and to keep training scripts self-contained when copied from this factory.)
# ===========================================================================

def augment_flip(X: np.ndarray) -> np.ndarray:
    """
    Horizontal flip augmentation (G1 component).
    Mirrors X coordinate: x_flipped = -x.
    Valid ONLY after normalize_sequence (centered coordinates).

    Args:
        X: (N, P, 468, 3) normalized sequences

    Returns:
        (N, P, 468, 3) with X coordinates negated.
    """
    X_flip = X.copy()
    X_flip[:, :, :, 0] = -X_flip[:, :, :, 0]
    return X_flip


def augment_noise(X: np.ndarray, sigma: float = AUG_NOISE_SIGMA,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Additive Gaussian noise augmentation (G3 component).
    Applied AFTER flip (if combined).

    Args:
        X:     (N, P, 468, 3) normalized sequences
        sigma: noise standard deviation (default AUG_NOISE_SIGMA=0.002)
        rng:   optional numpy Generator for reproducibility

    Returns:
        (N, P, 468, 3) with noise added.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=X.shape).astype(np.float32)
    return X + noise


def build_augmented_training_set(
    X_train:   np.ndarray,
    y_train:   np.ndarray,
    aug_mode:  str,
    seed:      int = 0,
) -> tuple:
    """
    Build augmented training set. Call AFTER normalize_sequence.

    aug_mode options:
      "G0_none"       -- no augmentation
      "G1_flip"       -- all classes x2 (original + flip)
      "G2_targeted"   -- minority (fear, sadness) x4, majority x2
      "G3_flip_noise" -- all classes x3 (original + flip + flip+noise)

    Args:
        X_train: (N, P, 468, 3) normalized training sequences
        y_train: (N,) integer class labels
        aug_mode: one of G0/G1/G2/G3
        seed:    random seed for noise

    Returns:
        (X_aug, y_aug) -- augmented and shuffled arrays
    """
    rng = np.random.default_rng(seed=seed)

    if aug_mode == "G0_none":
        return X_train.copy(), y_train.copy()

    X_list = [X_train]
    y_list = [y_train]

    if aug_mode in ("G1_flip", "G2_targeted", "G3_flip_noise"):
        # All classes: add flipped version
        X_flip = augment_flip(X_train)
        X_list.append(X_flip)
        y_list.append(y_train)

    if aug_mode == "G3_flip_noise":
        # All classes: add flipped + noisy version
        X_flip_noise = augment_noise(augment_flip(X_train), rng=rng)
        X_list.append(X_flip_noise)
        y_list.append(y_train)

    if aug_mode == "G2_targeted":
        # Minority classes: add 2 more copies (flip + flip+noise)
        for cls_idx in MINORITY_CLASS_INDICES:
            mask  = (y_train == cls_idx)
            X_cls = X_train[mask]
            y_cls = y_train[mask]
            if len(X_cls) == 0:
                continue
            # Extra flip copy
            X_list.append(augment_flip(X_cls))
            y_list.append(y_cls)
            # Extra flip + noise copy
            X_list.append(augment_noise(augment_flip(X_cls), rng=rng))
            y_list.append(y_cls)

    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)

    # Shuffle
    idx   = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ===========================================================================
# INTERNAL MODEL BLOCK BUILDERS
# Each returns a tf.keras.Model taking (N_LANDMARKS, n_coords) as input.
# Wrapped in TimeDistributed inside build_variant_model.
# ===========================================================================

def _build_dense_extractor(
    n_landmarks: int,
    n_coords:    int,
    name:        str = "dense_extractor",
) -> tf.keras.Model:
    """
    B1: TimeDistributed Dense -- Winarno 2025 baseline.
    Flattens all landmarks and processes with Dense(128).
    Equivalent to Conv1D(kernel_size=1) applied per landmark independently
    after flattening -- no inter-landmark locality modeling.

    Architecture: Flatten(1404) -> Dense(128) -> BN -> ReLU
    Output: (128,) per frame
    """
    inp = tf.keras.Input(shape=(n_landmarks, n_coords), name="frame_input")
    x   = tf.keras.layers.Flatten(name="flatten")(inp)
    x   = tf.keras.layers.Dense(128, name="dense")(x)
    x   = tf.keras.layers.BatchNormalization(name="bn")(x)
    x   = tf.keras.layers.ReLU(name="relu")(x)
    return tf.keras.Model(inp, x, name=name)


def _build_cnn1d_small_extractor(
    n_landmarks: int,
    n_coords:    int,
    name:        str = "cnn1d_small_extractor",
) -> tf.keras.Model:
    """
    B2: Small 1D-CNN spatial extractor.
    Single Conv1D layer with kernel_size=5.
    Input: (n_landmarks, n_coords) -- landmark-index as spatial dimension,
           coordinates as channels.
    Output: (64,) per frame via GlobalAveragePooling1D.
    """
    inp = tf.keras.Input(shape=(n_landmarks, n_coords), name="frame_input")
    x   = tf.keras.layers.Conv1D(
        64, 5, padding="same", use_bias=False, name="conv"
    )(inp)
    x   = tf.keras.layers.BatchNormalization(name="bn")(x)
    x   = tf.keras.layers.ReLU(name="relu")(x)
    x   = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    return tf.keras.Model(inp, x, name=name)


def _build_cnn1d_main_extractor(
    n_landmarks: int,
    n_coords:    int,
    name:        str = "cnn1d_main_extractor",
) -> tf.keras.Model:
    """
    B3: Main 1D-CNN spatial extractor (PROPOSAL -- main model component).

    Operates over landmark-index axis (468 positions, n_coords channels).
    Conv1D with kernel_size=9 captures 9 anatomically adjacent landmarks.
    MediaPipe FaceMesh has consistent anatomical ordering, so local
    convolutions over landmark index are anatomically meaningful.

    Architecture (per-frame, shared weights via TimeDistributed):
      Conv1D(64,  9, same) -> BN -> ReLU
      Conv1D(128, 7, same) -> BN -> ReLU -> MaxPool1D(2)  [468 -> 234]
      Conv1D(256, 5, same) -> BN -> ReLU -> MaxPool1D(2)  [234 -> 117]
      Conv1D(128, 3, same) -> BN -> ReLU
      GlobalAveragePooling1D                               [117 -> 128]

    Output: (128,) per frame
    Parameters: ~322,000 (shared via TimeDistributed across P=10 frames)
    """
    inp = tf.keras.Input(shape=(n_landmarks, n_coords), name="frame_input")

    # Block 1
    x = tf.keras.layers.Conv1D(
        CNN_FILTERS[0], CNN_KERNELS[0],
        padding="same", use_bias=False, name="conv1"
    )(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)

    # Block 2
    x = tf.keras.layers.Conv1D(
        CNN_FILTERS[1], CNN_KERNELS[1],
        padding="same", use_bias=False, name="conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)   # 468 -> 234

    # Block 3
    x = tf.keras.layers.Conv1D(
        CNN_FILTERS[2], CNN_KERNELS[2],
        padding="same", use_bias=False, name="conv3"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)   # 234 -> 117

    # Block 4
    x = tf.keras.layers.Conv1D(
        CNN_FILTERS[3], CNN_KERNELS[3],
        padding="same", use_bias=False, name="conv4"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.ReLU(name="relu4")(x)

    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)   # -> (128,)

    return tf.keras.Model(inp, x, name=name)


def _build_cnn2d_extractor(
    n_landmarks: int,
    n_coords:    int,
    name:        str = "cnn2d_extractor",
) -> tf.keras.Model:
    """
    B4: 2D-CNN per-frame extractor (Di Luzio 2023 style, approximate).

    Treats the frame (n_landmarks, n_coords) as a pseudo-2D image with
    shape (n_landmarks, n_coords, 1). Conv2D with kernel (k, 1) applies
    local convolution over the landmark-index axis only.

    NOTE: This is an approximation of Di Luzio et al. 2023, not identical.
    Di Luzio uses a pixel-space image representation; here we directly
    use raw coordinate arrays. Flagged in paper as B4 approximation.

    Architecture:
      Reshape(n_landmarks, n_coords, 1)
      Conv2D(32,  (9,1), same) -> BN -> ReLU
      Conv2D(64,  (7,1), same) -> BN -> ReLU -> MaxPool2D((2,1))
      Conv2D(128, (5,1), same) -> BN -> ReLU
      GlobalAveragePooling2D                   -> (128,)

    Output: (128,) per frame
    """
    inp = tf.keras.Input(shape=(n_landmarks, n_coords), name="frame_input")

    # Reshape to (n_landmarks, n_coords, 1) for Conv2D
    x = tf.keras.layers.Reshape(
        (n_landmarks, n_coords, 1), name="reshape"
    )(inp)

    x = tf.keras.layers.Conv2D(
        32, (9, 1), padding="same", use_bias=False, name="conv2d_1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)

    x = tf.keras.layers.Conv2D(
        64, (7, 1), padding="same", use_bias=False, name="conv2d_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 1), name="pool")(x)   # n_lm/2

    x = tf.keras.layers.Conv2D(
        128, (5, 1), padding="same", use_bias=False, name="conv2d_3"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)   # -> (128,)

    return tf.keras.Model(inp, x, name=name)


# ===========================================================================
# MODEL BUILDERS
# ===========================================================================

_EXTRACTOR_REGISTRY = {
    "B1": _build_dense_extractor,
    "B2": _build_cnn1d_small_extractor,
    "B3": _build_cnn1d_main_extractor,
    "B4": _build_cnn2d_extractor,
}


def build_variant_model(
    coord_variant:     str,
    extractor_variant: str,
    temporal_variant:  str,
    name:              Optional[str] = None,
) -> tf.keras.Model:
    """
    Build an uncompiled Keras model for a given ablation combination.

    Args:
        coord_variant:
            "A1" -- (P, 468, 2), drop Z (2D baseline, Di Luzio / Winarno style)
            "A2" -- (P, 468, 3), full xyz (MAIN MODEL)
            "A3" -- (P, 468, 3), delta coords (preprocessing via apply_delta_coords)

        extractor_variant:
            "B1" -- TimeDistributed(Dense) -- Winarno 2025 baseline
            "B2" -- TimeDistributed(small 1D-CNN)
            "B3" -- TimeDistributed(main 1D-CNN) -- MAIN MODEL
            "B4" -- TimeDistributed(2D-CNN approx.) -- Di Luzio style baseline

        temporal_variant:
            "C1" -- GRU(64, recurrent_dropout=0.2) -- MAIN MODEL
            "C2" -- LSTM(64, recurrent_dropout=0.2)
            "C3" -- Bidirectional(GRU(64, recurrent_dropout=0.2))

    Returns:
        Uncompiled tf.keras.Model.
        Input shape: (batch, P, N_LANDMARKS, n_coords)
        Output shape: (batch, N_CLASSES) -- softmax probabilities

    Note on A3:
        The model input shape is identical to A2 (P, 468, 3). Delta
        computation must be done by the caller via apply_delta_coords()
        before passing data to the model.
    """
    # Determine coordinate dimension
    n_coords = 2 if coord_variant == "A1" else 3

    # Input
    input_shape = (P, N_LANDMARKS, n_coords)
    seq_inp     = tf.keras.Input(shape=input_shape, name="sequence_input")

    # Spatial extractor sub-model
    if extractor_variant not in _EXTRACTOR_REGISTRY:
        raise ValueError(
            "Unknown extractor_variant: '{}'. "
            "Choose from {}.".format(extractor_variant, list(_EXTRACTOR_REGISTRY))
        )
    extractor = _EXTRACTOR_REGISTRY[extractor_variant](N_LANDMARKS, n_coords)

    # TimeDistributed wrapper: applies extractor to each of P frames
    x = tf.keras.layers.TimeDistributed(
        extractor, name="td_extractor"
    )(seq_inp)
    # x: (batch, P, feature_dim)

    # Temporal aggregator
    if temporal_variant == "C1":
        x = tf.keras.layers.GRU(
            GRU_UNITS,
            return_sequences=False,
            recurrent_dropout=GRU_RECURRENT_DROPOUT,
            reset_after=True,
            name="gru",
        )(x)
    elif temporal_variant == "C2":
        x = tf.keras.layers.LSTM(
            GRU_UNITS,
            return_sequences=False,
            recurrent_dropout=GRU_RECURRENT_DROPOUT,
            name="lstm",
        )(x)
    elif temporal_variant == "C3":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                GRU_UNITS,
                return_sequences=False,
                recurrent_dropout=GRU_RECURRENT_DROPOUT,
                reset_after=True,
            ),
            name="bigru",
        )(x)
    else:
        raise ValueError(
            "Unknown temporal_variant: '{}'. "
            "Choose from ['C1', 'C2', 'C3'].".format(temporal_variant)
        )

    # Classification head (fixed for all variants)
    x   = tf.keras.layers.Dropout(DROPOUT_RATE, name="head_dropout")(x)
    x   = tf.keras.layers.Dense(
        DENSE_HEAD_UNITS, activation="relu", name="head_dense"
    )(x)
    x   = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    out = tf.keras.layers.Dense(
        N_CLASSES, activation="softmax", name="output"
    )(x)

    # Model name
    if name is None:
        name = "{}_{}_{}_model".format(
            coord_variant, extractor_variant, temporal_variant
        )

    return tf.keras.Model(inputs=seq_inp, outputs=out, name=name)


def build_main_model() -> tf.keras.Model:
    """
    Shortcut: build the main model (A2 + B3 + C1, uncompiled).

    MAIN MODEL configuration:
      A2: full xyz input (P, 468, 3)
      B3: 1D-CNN spatial extractor (trained, kernel>1)
      C1: GRU temporal aggregator

    ~361,500 total parameters (~0.36M).
    """
    return build_variant_model("A2", "B3", "C1", name="main_model")


# ===========================================================================
# LOSS FUNCTION
# ===========================================================================

class FocalLoss(tf.keras.losses.Loss):
    """
    Multi-class focal loss for imbalanced classification.

    Reference:
      Lin T.Y. et al. "Focal Loss for Dense Object Detection."
      IEEE ICCV 2017. DOI: 10.1109/ICCV.2017.324

    Formula per sample:
      FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    where p_t = predicted probability of the TRUE class.

    Optional per-class weighting:
      FL_w(p_t) = -w_c * (1 - p_t)^gamma * log(p_t)
    where w_c = class weight for the true class of that sample.

    Args:
        gamma:         Focusing parameter (>=0).
                       D3: gamma=1.0, D4: gamma=2.0.
        class_weights: Optional numpy array of shape (N_CLASSES,).
                       If provided, multiplied per-sample by true-class weight.
        name:          Loss name.
    """

    def __init__(
        self,
        gamma:         float,
        class_weights: Optional[np.ndarray] = None,
        name:          str = "focal_loss",
    ):
        # reduction="none": we reduce manually inside call() for clarity
        super().__init__(name=name, reduction="none")
        self.gamma         = float(gamma)
        self.class_weights = (
            np.array(class_weights, dtype=np.float32)
            if class_weights is not None else None
        )

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute mean focal loss over a batch.

        Args:
            y_true: (batch,) integer class labels (sparse format)
            y_pred: (batch, N_CLASSES) float softmax probabilities

        Returns:
            Scalar mean loss.
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred  = tf.cast(y_pred, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Flatten and convert sparse labels to one-hot
        y_true_flat = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        y_true_oh   = tf.one_hot(y_true_flat, N_CLASSES)   # (batch, N_CLASSES)

        # p_t: predicted probability of the true class (per sample)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)   # (batch,)

        # Focal weight: down-weights easy examples
        focal_weight = tf.pow(1.0 - p_t, self.gamma)       # (batch,)

        # Cross-entropy for the true class
        log_p_t = tf.math.log(p_t)                         # (batch,)

        # Apply per-class weights if provided
        if self.class_weights is not None:
            cw_tensor = tf.constant(self.class_weights, dtype=tf.float32)
            cw_per    = tf.gather(cw_tensor, y_true_flat)  # (batch,)
            loss      = -cw_per * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t                 # (batch,)

        return tf.reduce_mean(loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "gamma"        : self.gamma,
            "class_weights": (
                self.class_weights.tolist()
                if self.class_weights is not None else None
            ),
        })
        return config


# ===========================================================================
# CLASS WEIGHTS
# ===========================================================================

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute balanced class weights for Keras model.fit(class_weight=...).

    Formula: w_c = N_total / (N_classes * N_c)
    This gives higher weight to minority classes.

    Args:
        y_train: (N,) integer class labels from the training fold

    Returns:
        dict {class_idx (int): weight (float)} with N_CLASSES entries.

    Guard: if a class is absent in the training fold (possible in small LOSO
    folds), assigns w_c = N_total as a large but finite weight and logs a
    warning. This avoids ZeroDivisionError and flags the issue.
    """
    y_arr   = np.asarray(y_train).flatten()
    n_total = len(y_arr)
    weights = {}
    for c in range(N_CLASSES):
        n_c = int((y_arr == c).sum())
        if n_c == 0:
            print(
                "  WARN: class {} ({}) absent in training fold. "
                "Assigning max weight = {}.".format(
                    c, EMOTION_NAMES[c], n_total)
            )
            weights[c] = float(n_total)
        else:
            weights[c] = float(n_total) / (N_CLASSES * n_c)
    return weights


# ===========================================================================
# OUTPUT UTILITIES
# ===========================================================================

def save_model_summary(
    model:      tf.keras.Model,
    output_dir: str,
) -> None:
    """
    Save model summary as text file. Optionally save architecture PNG.

    Text summary is always saved. Architecture PNG requires pydot and
    graphviz; if unavailable, a warning is printed and only text is saved.

    Args:
        model:      Keras model (compiled or uncompiled)
        output_dir: directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Text summary
    txt_path = os.path.join(
        output_dir, "model_summary_{}.txt".format(model.name)
    )
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print("  Saved: {}".format(txt_path))

    # Architecture diagram (optional -- requires pydot + graphviz)
    try:
        import pydot  # noqa: F401 -- just check availability
        png_path = os.path.join(
            output_dir, "model_arch_{}.png".format(model.name)
        )
        tf.keras.utils.plot_model(
            model,
            to_file=png_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            dpi=300,
        )
        print("  Saved: {}".format(png_path))
    except Exception as exc:
        print("  INFO: plot_model skipped ({}: {}). "
              "Text summary saved.".format(type(exc).__name__, exc))


# ===========================================================================
# SELF-TEST (no data files required -- uses dummy arrays)
# ===========================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("TensorFlow version : {}".format(tf.__version__))
    print("NumPy version      : {}".format(np.__version__))
    print()

    # ------------------------------------------------------------------
    print("[TEST 1] Model builds and forward pass ...")

    dummy_xyz = np.random.randn(2, P, N_LANDMARKS, 3).astype(np.float32)
    dummy_xy  = dummy_xyz[:, :, :, :2].copy()

    test_cases = [
        ("A2_B3_C1 (main)",  build_main_model(),                       dummy_xyz),
        ("A1_B1_C2",         build_variant_model("A1", "B1", "C2"),    dummy_xy),
        ("A2_B2_C3",         build_variant_model("A2", "B2", "C3"),    dummy_xyz),
        ("A2_B4_C1",         build_variant_model("A2", "B4", "C1"),    dummy_xyz),
        ("A3_B3_C2",         build_variant_model("A3", "B3", "C2"),    dummy_xyz),
        ("A2_B3_C3",         build_variant_model("A2", "B3", "C3"),    dummy_xyz),
    ]

    for tag, model, dummy in test_cases:
        out      = model(dummy, training=False)
        out_np   = out.numpy()
        prob_sum = out_np.sum(axis=-1)

        assert out_np.shape == (2, N_CLASSES), \
            "Shape mismatch for {}: got {}".format(tag, out_np.shape)
        assert np.allclose(prob_sum, 1.0, atol=1e-5), \
            "Softmax sum != 1 for {}: {}".format(tag, prob_sum)

        print("  [OK] {:25s}  params={:>8,}  output_shape={}".format(
            tag, model.count_params(), tuple(out_np.shape)))

        save_model_summary(model, OUTPUT_DIR)

        # Free memory after each model test
        tf.keras.backend.clear_session()
        gc.collect()

    # ------------------------------------------------------------------
    print("\n[TEST 2] normalize_sequence ...")

    # Realistic MediaPipe range: x,y in [0.1, 0.9], z small
    rng_np  = np.random.default_rng(seed=0)
    X_dummy = (rng_np.random((4, P, N_LANDMARKS, 3)).astype(np.float32)
               * 0.6 + 0.2)
    # Ensure nose and chin are distinct (avoid scale=0)
    X_dummy[:, :, NOSE_TIP_IDX, :2] = 0.5
    X_dummy[:, :, CHIN_IDX,     :2] = 0.5 + rng_np.random(
        (4, P, 2)).astype(np.float32) * 0.2 + 0.05

    X_norm = normalize_sequence(X_dummy)

    assert X_norm.shape == X_dummy.shape, \
        "Shape mismatch: {}".format(X_norm.shape)
    assert not np.isnan(X_norm).any(), "NaN in normalize output"
    assert not np.isinf(X_norm).any(), "Inf in normalize output"
    assert np.allclose(X_norm[:, :, NOSE_TIP_IDX, :], 0.0, atol=1e-5), \
        "Nose tip not at origin after normalization"

    print("  [OK] batch input: shape={}, nose_at_origin=True".format(
        X_norm.shape))

    # Single-sequence input
    X_single_out = normalize_sequence(X_dummy[0])
    assert X_single_out.shape == (P, N_LANDMARKS, 3), \
        "Single shape error: {}".format(X_single_out.shape)
    print("  [OK] single input: shape={}".format(X_single_out.shape))

    # ------------------------------------------------------------------
    print("\n[TEST 3] compute_z_scale ...")

    z_scale = compute_z_scale(X_norm)
    assert z_scale > 0.0, "z_scale <= 0: {}".format(z_scale)
    print("  [OK] z_scale = {:.6f}".format(z_scale))

    X_norm_z = normalize_sequence(X_dummy, z_scale=z_scale)
    assert not np.isnan(X_norm_z).any(), "NaN after Z normalization"
    print("  [OK] normalize_sequence with z_scale: no NaN, no Inf")

    # ------------------------------------------------------------------
    print("\n[TEST 4] apply_delta_coords ...")

    X_delta = apply_delta_coords(X_dummy)

    assert X_delta.shape == X_dummy.shape, \
        "Shape mismatch: {}".format(X_delta.shape)
    assert np.allclose(X_delta[:, 0, :, :], 0.0, atol=1e-8), \
        "delta[:, 0, :, :] must be all zeros"
    # Verify delta for frame 1
    expected_d1 = X_dummy[:, 1, :, :] - X_dummy[:, 0, :, :]
    assert np.allclose(X_delta[:, 1, :, :], expected_d1, atol=1e-6), \
        "delta[:, 1, :, :] mismatch"

    print("  [OK] shape={}, delta[0]=0, delta[1]=X[1]-X[0]".format(
        X_delta.shape))

    # Single-sequence
    X_delta_s = apply_delta_coords(X_dummy[0])
    assert X_delta_s.shape == (P, N_LANDMARKS, 3)
    print("  [OK] single input: shape={}".format(X_delta_s.shape))

    # ------------------------------------------------------------------
    print("\n[TEST 5] get_class_weights ...")

    # CK+ 6-class distribution from dataset_summary.txt
    y_ck = np.array(
        [0] * 45 + [1] * 59 + [2] * 25 + [3] * 69 + [4] * 28 + [5] * 83,
        dtype=np.int32,
    )
    cw = get_class_weights(y_ck)

    assert len(cw) == N_CLASSES, \
        "Expected {} keys, got {}".format(N_CLASSES, len(cw))
    assert all(v > 0 for v in cw.values()), "All weights must be > 0"
    # Minority class (fear, idx 2) should have higher weight than majority (surprise, idx 5)
    assert cw[2] > cw[5], \
        "fear weight ({:.3f}) should > surprise weight ({:.3f})".format(
            cw[2], cw[5])

    print("  [OK] Class weights:")
    for k in sorted(cw):
        print("       class {} ({:<10}): {:.4f}".format(
            k, EMOTION_NAMES[k], cw[k]))

    # ------------------------------------------------------------------
    print("\n[TEST 6] FocalLoss ...")

    y_true_t = tf.constant(list(range(N_CLASSES)), dtype=tf.int32)
    y_pred_t = tf.nn.softmax(
        tf.random.normal((N_CLASSES, N_CLASSES), seed=0)
    )

    for gamma in [FOCAL_GAMMA_D3, FOCAL_GAMMA_D4]:
        fl  = FocalLoss(gamma=gamma)
        lv  = float(fl(y_true_t, y_pred_t).numpy())
        assert lv > 0.0, "Loss must be positive"
        assert not np.isnan(lv), "Loss is NaN"
        print("  [OK] FocalLoss gamma={:.1f}: loss={:.6f}".format(gamma, lv))

    # FocalLoss with class weights
    cw_arr = np.array(list(cw.values()), dtype=np.float32)
    fl_cw  = FocalLoss(gamma=FOCAL_GAMMA_D4, class_weights=cw_arr)
    lv_cw  = float(fl_cw(y_true_t, y_pred_t).numpy())
    assert lv_cw > 0.0, "Weighted focal loss must be positive"
    print("  [OK] FocalLoss gamma={:.1f} + class_weights: "
          "loss={:.6f}".format(FOCAL_GAMMA_D4, lv_cw))

    # Verify get_config roundtrip
    cfg       = fl_cw.get_config()
    assert "gamma" in cfg and "class_weights" in cfg
    print("  [OK] get_config: keys={}".format(list(cfg.keys())))

    # ------------------------------------------------------------------
    print("\n[TEST 7] build_augmented_training_set ...")

    X_small = normalize_sequence(
        (rng_np.random((30, P, N_LANDMARKS, 3)).astype(np.float32) * 0.6 + 0.2)
    )
    y_small = np.array([0]*5 + [1]*5 + [2]*3 + [3]*7 + [4]*4 + [5]*6,
                       dtype=np.int32)

    for aug_mode in AUG_VARIANTS:
        X_aug, y_aug = build_augmented_training_set(
            X_small, y_small, aug_mode, seed=42
        )
        assert X_aug.ndim == 4, "X_aug must be 4D"
        assert X_aug.shape[0] == y_aug.shape[0], "X/y length mismatch"
        assert X_aug.shape[1:] == (P, N_LANDMARKS, 3), \
            "Shape mismatch: {}".format(X_aug.shape[1:])
        print("  [OK] {}: {} -> {} samples".format(
            aug_mode, len(y_small), len(y_aug)))

    # ------------------------------------------------------------------
    gc.collect()
    print()
    print("=" * 60)
    print("04_model_factory.py -- ALL SELF-TESTS PASSED")
    print("=" * 60)
    print("  Summaries saved to: {}".format(OUTPUT_DIR))
    print()
    print("  USAGE IN TRAINING SCRIPTS:")
    print("    from 04_model_factory import (")
    print("        normalize_sequence, compute_z_scale, apply_delta_coords,")
    print("        build_augmented_training_set,")
    print("        build_main_model, build_variant_model,")
    print("        FocalLoss, get_class_weights,")
    print("        SMOOTH_VARIANT_TO_FILE, Y_6CLS_NPY, SUBJECTS_NPY,")
    print("    )")
    print("    (or copy this file into each training script)")


if __name__ == "__main__":
    main()
