"""
02_extract_landmarks.py
=======================

author: Dr. Eng. Farrikh Alzami, Sri Winarno Ph.D, M. Naufal M.Eng, Dewi Agustini Santoso MCS.

Extract MediaPipe FaceLandmarker 468 3D landmarks (x, y, z) untuk setiap
frame PNG di setiap sequence CK+ yang berlabel (6 kelas, contempt dibuang).

INPUT (Kaggle datasets):
  /kaggle/input/datasets/udnyahyaalzami/dataset-ckplus/   -- dataset CK+

OUTPUT (di /kaggle/working, bisa diunduh):
  landmarks/<subject>_<session>_<emotion>.npy  -- shape (n_frames, 468, 3)
  landmark_manifest.csv                        -- satu baris per sequence
  missing_faces.csv                            -- frame yang gagal dideteksi
  fig_extraction_summary.png                   -- bar chart sequences per emosi
  fig_frame_distribution.png                   -- distribusi frame count

Missing-frame strategy:
  Forward-fill dari frame sebelumnya.
  Jika frame pertama gagal: backward-fill dari frame valid pertama.
  Jika SEMUA frame gagal: status "failed", tidak ada .npy.

Manifest di-checkpoint setiap CHECKPOINT_EVERY sequences.

mediapipe==0.10.35 (LOCKED -- jangan upgrade)
"""

import gc
import inspect
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ===========================================================================
# CONFIG -- copy dari 00_config.py
# Jika ada perubahan parameter, update 00_config.py dulu, lalu copy ke sini.
# ===========================================================================

DATASET_ROOT = "/kaggle/input/datasets/udnyahyaalzami/dataset-ckplus"
IMAGES_PATH  = os.path.join(
    DATASET_ROOT,
    "extended-cohn-kanade-images",
    "cohn-kanade-images",
)
EMOTION_PATH = os.path.join(DATASET_ROOT, "Emotion_labels", "Emotion")
OUTPUT_DIR   = "/kaggle/working"
LANDMARK_DIR = os.path.join(OUTPUT_DIR, "landmarks")

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(OUTPUT_DIR, "face_landmarker.task")

MANIFEST_CSV      = os.path.join(OUTPUT_DIR, "landmark_manifest.csv")
MISSING_FACES_CSV = os.path.join(OUTPUT_DIR, "missing_faces.csv")

CKPLUS_TO_IDX = {
    1: 0,   # anger
    3: 1,   # disgust
    4: 2,   # fear
    5: 3,   # happiness
    6: 4,   # sadness
    7: 5,   # surprise
}

EMOTION_NAMES = [
    "anger",      # index 0
    "disgust",    # index 1
    "fear",       # index 2
    "happiness",  # index 3
    "sadness",    # index 4
    "surprise",   # index 5
]

N_CLASSES           = 6
VALID_EMOTION_CODES = set(CKPLUS_TO_IDX.keys())   # {1, 3, 4, 5, 6, 7}

N_LANDMARKS  = 468
N_COORDS     = 3
NOSE_TIP_IDX = 4
CHIN_IDX     = 152

# ===========================================================================
# RUNTIME CONSTANTS
# ===========================================================================

MIN_DETECTION_CONFIDENCE = 0.5
CHECKPOINT_EVERY         = 25
GC_EVERY                 = 50
LOG_EVERY                = 25

# ===========================================================================
# MODEL DOWNLOAD
# ===========================================================================

def download_model() -> str:
    """Download FaceLandmarker .task jika belum ada. Return path."""
    candidates = [
        MODEL_PATH,
        "/kaggle/input/mediapipe-face-landmarker/face_landmarker.task",
    ]
    for path in candidates:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print("  Model ditemukan: {} ({:.1f} MB)".format(path, size_mb))
            return path
    print("  Downloading FaceLandmarker model ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size_mb = os.path.getsize(MODEL_PATH) / 1e6
    print("  Tersimpan: {} ({:.1f} MB)".format(MODEL_PATH, size_mb))
    return MODEL_PATH


# ===========================================================================
# MEDIAPIPE LANDMARKER
# Gunakan inspect() untuk skip kwargs yang tidak didukung versi ini.
# Referensi: 01_data_loading_v5.py create_face_landmarker()
# ===========================================================================

def build_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    """
    Instantiate MediaPipe FaceLandmarker untuk IMAGE running mode.
    Menggunakan inspect() untuk skip kwarg yang tidak valid di versi patch ini.
    mediapipe==0.10.35 (LOCKED).
    """
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)

    desired = {
        "running_mode"                          : mp_vision.RunningMode.IMAGE,
        "num_faces"                             : 1,
        "min_face_detection_confidence"         : MIN_DETECTION_CONFIDENCE,
        "min_face_presence_score"               : MIN_DETECTION_CONFIDENCE,
        "min_tracking_confidence"               : MIN_DETECTION_CONFIDENCE,
        "output_face_blendshapes"               : False,
        "output_facial_transformation_matrixes" : False,
    }

    try:
        valid = set(
            inspect.signature(
                mp_vision.FaceLandmarkerOptions.__init__
            ).parameters.keys()
        ) - {"self", "base_options"}
    except (ValueError, TypeError):
        # Fallback: hanya parameter yang pasti ada di semua versi 0.10.x
        valid = {
            "running_mode",
            "num_faces",
            "min_face_detection_confidence",
            "output_face_blendshapes",
            "output_facial_transformation_matrixes",
        }

    accepted = {k: v for k, v in desired.items() if k in valid}
    skipped  = set(desired) - set(accepted)
    if skipped:
        print("  [INFO] FaceLandmarkerOptions skipping: {}".format(sorted(skipped)))

    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_opts,
        **accepted,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ===========================================================================
# VERIFIKASI INDEKS LANDMARK
# Panggil sekali sebelum loop utama.
# Konfirmasi: nose.y < chin.y (hidung di atas dagu dalam image coords).
# ===========================================================================

def verify_landmark_indices(landmarker: mp_vision.FaceLandmarker,
                            frame_path: Path) -> None:
    bgr    = cv2.imread(str(frame_path))
    if bgr is None:
        print("  WARN: cv2 gagal baca {}".format(frame_path.name))
        return
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(image)

    if not result.face_landmarks:
        print("  WARN: tidak ada wajah di {}".format(frame_path.name))
        return

    lms  = result.face_landmarks[0]
    nose = lms[NOSE_TIP_IDX]
    chin = lms[CHIN_IDX]

    print("  [VERIFY] frame     : {}".format(frame_path.name))
    print("  [VERIFY] nose tip  (idx {}): x={:.4f}  y={:.4f}  z={:.4f}".format(
        NOSE_TIP_IDX, nose.x, nose.y, nose.z))
    print("  [VERIFY] chin      (idx {}): x={:.4f}  y={:.4f}  z={:.4f}".format(
        CHIN_IDX, chin.x, chin.y, chin.z))

    assert nose.y < chin.y, (
        "INDEKS SALAH: nose.y={:.4f} harus < chin.y={:.4f}. "
        "Periksa NOSE_TIP_IDX dan CHIN_IDX.".format(nose.y, chin.y)
    )
    print("  [VERIFY] PASSED: nose.y < chin.y")


# ===========================================================================
# LOAD EMOTION LABELS
# Identik dengan referensi 01_data_loading_v5.py, filter 6-class.
# ===========================================================================

def load_emotion_labels(emotion_path: str) -> Dict[Tuple[str, str], int]:
    """
    Scan Emotion_labels directory.
    Return {(subject, session): emotion_code} untuk VALID_EMOTION_CODES saja.
    """
    base  = Path(emotion_path)
    labels: Dict[Tuple[str, str], int] = {}

    # DEBUG: verifikasi path dan jumlah .txt yang ditemukan
    all_txt = list(base.rglob("*.txt"))
    print("  [DEBUG] Emotion path   : {}".format(emotion_path))
    print("  [DEBUG] Path exists    : {}".format(base.exists()))
    print("  [DEBUG] Total .txt     : {}".format(len(all_txt)))
    if all_txt:
        # Tampilkan 3 file pertama beserta isinya
        for f in all_txt[:3]:
            try:
                content = f.read_text().strip()
            except Exception as e:
                content = "ERROR: {}".format(e)
            print("  [DEBUG]   {} -> '{}'".format(
                f.relative_to(base), content))

    for txt_file in sorted(all_txt):
        rel   = txt_file.relative_to(base)
        parts = rel.parts
        if len(parts) < 3:
            continue
        subject, session = parts[0], parts[1]
        try:
            code = int(float(txt_file.read_text().strip()))
        except (ValueError, OSError):
            continue
        if code in VALID_EMOTION_CODES:
            labels[(subject, session)] = code

    return labels


# ===========================================================================
# EKSTRAKSI SATU SEQUENCE
# Menggunakan cv2 untuk load image, bukan mp.Image.create_from_file().
# Referensi: 01_data_loading_v5.py extract_raw_sequence()
# ===========================================================================

def extract_sequence(
    img_folder: Path,
    landmarker: mp_vision.FaceLandmarker,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Ekstrak (N_frames, 468, 3) float32 untuk satu sequence.
    Mengembalikan (array, n_missing). Array=None jika semua frame gagal.

    Missing-frame strategy:
      - Forward-fill dari frame valid sebelumnya
      - Backward-fill jika frame awal gagal
      - Return None jika semua frame gagal
    """
    png_files = sorted(img_folder.glob("*.png"))
    n_frames  = len(png_files)
    if n_frames == 0:
        return None, 0

    arr       = np.full((n_frames, N_LANDMARKS, N_COORDS), np.nan,
                        dtype=np.float32)
    n_missing = 0

    for fi, png_path in enumerate(png_files):
        bgr = cv2.imread(str(png_path))
        if bgr is None:
            n_missing += 1
            continue
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)

        if not result.face_landmarks:
            n_missing += 1
            continue

        lm_list = result.face_landmarks[0]
        if len(lm_list) < N_LANDMARKS:
            n_missing += 1
            continue

        for li in range(N_LANDMARKS):
            arr[fi, li, 0] = lm_list[li].x
            arr[fi, li, 1] = lm_list[li].y
            arr[fi, li, 2] = lm_list[li].z

    # Semua frame gagal
    if n_missing == n_frames:
        return None, n_missing

    # Forward-fill: frame NaN -> frame valid sebelumnya
    last_valid: Optional[int] = None
    for fi in range(n_frames):
        if not np.isnan(arr[fi, 0, 0]):
            last_valid = fi
        elif last_valid is not None:
            arr[fi] = arr[last_valid]

    # Backward-fill: frame awal NaN -> frame valid pertama
    first_valid: Optional[int] = None
    for fi in range(n_frames):
        if not np.isnan(arr[fi, 0, 0]):
            first_valid = fi
            break
    if first_valid is not None and first_valid > 0:
        arr[:first_valid] = arr[first_valid]

    return arr, n_missing


# ===========================================================================
# MISSING FRAME LOGGER
# ===========================================================================

def collect_missing_frames(
    subject:      str,
    session:      str,
    emotion_name: str,
    img_folder:   Path,
    landmarker:   mp_vision.FaceLandmarker,
) -> List[dict]:
    rows = []
    for fi, png_path in enumerate(sorted(img_folder.glob("*.png"))):
        bgr = cv2.imread(str(png_path))
        if bgr is None:
            rows.append({
                "subject": subject, "session": session,
                "emotion": emotion_name, "frame_idx": fi,
                "frame_file": png_path.name, "reason": "cv2_read_failed",
            })
            continue
        rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)
        if not result.face_landmarks:
            rows.append({
                "subject": subject, "session": session,
                "emotion": emotion_name, "frame_idx": fi,
                "frame_file": png_path.name, "reason": "no_detection",
            })
    return rows


# ===========================================================================
# CHECKPOINT SAVE
# ===========================================================================

def save_checkpoint(manifest_rows: List[dict],
                    missing_rows:  List[dict]) -> None:
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_CSV, index=False)
    pd.DataFrame(missing_rows).to_csv(MISSING_FACES_CSV, index=False)


# ===========================================================================
# FIGURES (DPI=300, simpan sebelum plt.show())
# ===========================================================================

def save_figures(manifest_df: pd.DataFrame) -> None:
    ok_df = manifest_df[
        manifest_df["status"].isin(["ok", "partial", "ok_resumed"])
    ]

    # ------------------------------------------------------------------
    # Figure 1: Bar chart -- jumlah sequence berhasil diekstrak per emosi
    # ------------------------------------------------------------------
    counts = (
        ok_df["emotion_name"]
        .value_counts()
        .reindex(EMOTION_NAMES, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        counts.index, counts.values,
        color="#4C72B0", edgecolor="black", linewidth=0.7,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(val),
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_title("Extracted Sequences per Emotion (CK+ 6-class)", fontsize=13)
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Number of Sequences", fontsize=11)
    ax.set_ylim(0, counts.max() + 8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "fig_extraction_summary.png")
    fig.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path1))

    # ------------------------------------------------------------------
    # Figure 2: Box + strip plot -- distribusi frame count per emosi
    # ------------------------------------------------------------------
    data_per_emotion = [
        ok_df[ok_df["emotion_name"] == name]["n_frames"].values
        for name in EMOTION_NAMES
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(
        data_per_emotion,
        labels=EMOTION_NAMES,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(seed=0)
    for xi, (vals, color) in enumerate(
        zip(data_per_emotion, colors), start=1
    ):
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), xi) + jitter, vals,
            color=color, alpha=0.55, s=20, zorder=3,
        )

    ax.set_title("Frame Count Distribution per Emotion (CK+ 6-class)",
                 fontsize=13)
    ax.set_xlabel("Emotion", fontsize=11)
    ax.set_ylabel("Number of Frames", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "fig_frame_distribution.png")
    fig.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}".format(path2))


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LANDMARK_DIR, exist_ok=True)

    print("mediapipe version : {}".format(mp.__version__))
    print("VALID_EMOTION_CODES: {}".format(sorted(VALID_EMOTION_CODES)))

    # ------------------------------------------------------------------
    print("\n[STEP 1] Download / verifikasi model MediaPipe ...")
    model_path = download_model()

    # ------------------------------------------------------------------
    print("\n[STEP 2] Load emotion labels (6-class) ...")
    labels = load_emotion_labels(EMOTION_PATH)
    print("  Ditemukan {} labeled sequences (contempt dibuang).".format(
        len(labels)))

    if len(labels) == 0:
        raise RuntimeError(
            "Tidak ada labeled sequence ditemukan. "
            "Periksa EMOTION_PATH dan debug output di atas."
        )

    # ------------------------------------------------------------------
    print("\n[STEP 3] Inisialisasi FaceLandmarker ...")
    landmarker = build_landmarker(model_path)
    print("  FaceLandmarker berhasil dibuat.")

    # ------------------------------------------------------------------
    print("\n[STEP 4] Verifikasi NOSE_TIP_IDX dan CHIN_IDX ...")
    images_base = Path(IMAGES_PATH)
    verified    = False
    for (subject, session), _ in sorted(labels.items()):
        folder = images_base / subject / session
        pngs   = sorted(folder.glob("*.png"))
        if pngs:
            verify_landmark_indices(landmarker, pngs[0])
            verified = True
            break
    if not verified:
        print("  WARN: tidak ada PNG untuk verifikasi.")

    # ------------------------------------------------------------------
    print("\n[STEP 5] Ekstraksi landmarks per sequence ...")

    manifest_rows: List[dict] = []
    missing_rows:  List[dict] = []
    n_ok      = 0
    n_partial = 0
    n_failed  = 0

    sorted_labels = sorted(labels.items())
    n_total       = len(sorted_labels)

    for seq_idx, ((subject, session), code) in enumerate(sorted_labels):
        emotion_name = EMOTION_NAMES[CKPLUS_TO_IDX[code]]
        folder       = images_base / subject / session
        npy_fname    = "{}_{}_{}.npy".format(subject, session, emotion_name)
        npy_path     = os.path.join(LANDMARK_DIR, npy_fname)

        # Resumable: skip jika sudah ada
        if os.path.exists(npy_path):
            existing = np.load(npy_path)
            manifest_rows.append({
                "subject":          subject,
                "session":          session,
                "emotion_code":     code,
                "emotion_name":     emotion_name,
                "n_frames":         existing.shape[0],
                "npy_path":         npy_path,
                "n_missing_frames": 0,
                "status":           "ok_resumed",
            })
            n_ok += 1
            continue

        # Folder tidak ada
        if not folder.exists():
            manifest_rows.append({
                "subject":          subject,
                "session":          session,
                "emotion_code":     code,
                "emotion_name":     emotion_name,
                "n_frames":         0,
                "npy_path":         "",
                "n_missing_frames": 0,
                "status":           "no_folder",
            })
            n_failed += 1
            continue

        # Ekstraksi
        arr, n_missing = extract_sequence(folder, landmarker)

        if arr is None:
            n_failed += 1
            manifest_rows.append({
                "subject":          subject,
                "session":          session,
                "emotion_code":     code,
                "emotion_name":     emotion_name,
                "n_frames":         len(sorted(folder.glob("*.png"))),
                "npy_path":         "",
                "n_missing_frames": n_missing,
                "status":           "failed",
            })
        else:
            status = "partial" if n_missing > 0 else "ok"
            if status == "ok":
                n_ok += 1
            else:
                n_partial += 1

            assert arr.shape[1] == N_LANDMARKS and arr.shape[2] == N_COORDS, (
                "Shape tidak valid {} untuk {}/{}".format(
                    arr.shape, subject, session)
            )
            np.save(npy_path, arr)

            manifest_rows.append({
                "subject":          subject,
                "session":          session,
                "emotion_code":     code,
                "emotion_name":     emotion_name,
                "n_frames":         arr.shape[0],
                "npy_path":         npy_path,
                "n_missing_frames": n_missing,
                "status":           status,
            })

            if n_missing > 0:
                missing_rows.extend(
                    collect_missing_frames(
                        subject, session, emotion_name, folder, landmarker
                    )
                )

        # Progress
        if (seq_idx + 1) % LOG_EVERY == 0 or (seq_idx + 1) == n_total:
            print("  [{:>3}/{}]  ok={}  partial={}  failed={}".format(
                seq_idx + 1, n_total, n_ok, n_partial, n_failed))

        # Checkpoint
        if (seq_idx + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(manifest_rows, missing_rows)

        # Memory
        if (seq_idx + 1) % GC_EVERY == 0:
            gc.collect()

    # ------------------------------------------------------------------
    print("\n[STEP 6] Simpan manifest dan missing-faces log ...")
    save_checkpoint(manifest_rows, missing_rows)
    print("  Saved: {}".format(MANIFEST_CSV))
    print("  Saved: {}".format(MISSING_FACES_CSV))

    # ------------------------------------------------------------------
    print("\n[STEP 7] Generate dan simpan figures ...")
    manifest_df = pd.DataFrame(manifest_rows)
    save_figures(manifest_df)

    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print("  Total sequences  : {}".format(n_total))
    print("  ok               : {}".format(n_ok))
    print("  partial          : {}".format(n_partial))
    print("  failed           : {}".format(n_failed))
    print("  Missing frames   : {}".format(len(missing_rows)))
    print()

    ok_df = manifest_df[
        manifest_df["status"].isin(["ok", "partial", "ok_resumed"])
    ]
    print("  Per-emotion (extractable sequences):")
    for name in EMOTION_NAMES:
        cnt = (ok_df["emotion_name"] == name).sum()
        print("    {:<12}: {}".format(name, cnt))

    print()
    print("  OUTPUT FILES:")
    print("    {}".format(MANIFEST_CSV))
    print("    {}".format(MISSING_FACES_CSV))
    print("    {}".format(os.path.join(OUTPUT_DIR, "fig_extraction_summary.png")))
    print("    {}".format(os.path.join(OUTPUT_DIR, "fig_frame_distribution.png")))
    print("    {}/".format(LANDMARK_DIR))
    print("=" * 60)

    gc.collect()


if __name__ == "__main__":
    main()
