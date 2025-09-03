# collect_knight_diagonals_prompted_aug.py
import csv
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============== CONFIG ==============
ROOT            = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
OUT_DIR         = ROOT / "data_refs"
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"

WEBCAM_INDEX    = 0
BOARD_SIDE      = 900
PATTERN_SIZE    = (7, 7)
MODEL_NAME      = "openai/clip-vit-base-patch32"

# Augmentations
AUG_ROTATIONS       = True
AUG_FLIPS           = True
AUG_JITTER_COUNT    = 2       # number of brightness/contrast jitters per geometric variant
JITTER_ALPHA_RANGE  = (0.90, 1.12)  # contrast multiplier
JITTER_BETA_RANGE   = (-18, 18)     # brightness shift (added after contrast)
RANDOM_SEED         = 1337
# ====================================

random.seed(RANDOM_SEED)

# ---------- CLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained(MODEL_NAME)

@torch.no_grad()
def embed_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp = clip_proc(images=pil, return_tensors="pt").to(device)
    z = clip_model.get_image_features(**inp)
    return F.normalize(z, p=2, dim=-1)  # [1, D]

# ---------- IO helpers ----------
def ensure_meta_exists():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EMB_META_PATH.exists():
        with open(EMB_META_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp","filepath","label","color","piece","square_name","square_index","h8_path","g8_path"]
            )

def append_embedding(embedding_1xD: torch.Tensor):
    if EMB_TENSOR_PATH.exists():
        old = torch.load(EMB_TENSOR_PATH, map_location="cpu")
        new = torch.cat([old, embedding_1xD.cpu()], dim=0)
    else:
        new = embedding_1xD.cpu()
    torch.save(new, EMB_TENSOR_PATH)

def append_meta_row(row):
    newfile = not EMB_META_PATH.exists()
    with open(EMB_META_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["timestamp","filepath","label","color","piece","square_name","square_index","h8_path","g8_path"])
        w.writerow([
            row["timestamp"], row["filepath"], row["label"], row["color"], row["piece"],
            row["square_name"], row["square_index"], "", ""
        ])

# ---------- Board warp ----------
def gray_world_white_balance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3.0
    eps = 1e-6
    b *= (m / (b.mean() + eps)); g *= (m / (g.mean() + eps)); r *= (m / (r.mean() + eps))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def compute_homography(frame_bgr):
    wb = gray_world_white_balance(frame_bgr)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)

    ok, corners = False, None
    try:
        res = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if isinstance(res, tuple):
            ok, pts = res; corners = pts if ok else None
        else:
            ok = res is not None; corners = res
    except Exception:
        ok = False

    if not ok:
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                 cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)
        ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
        if ok:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), term)

    if not ok or corners is None:
        raise RuntimeError("Could not find chessboard inner corners (7x7).")

    tl, tr, bl, br = corners[0,0], corners[6,0], corners[42,0], corners[48,0]
    src_inner = np.array([tl, tr, br, bl], dtype=np.float32)

    S = BOARD_SIDE / 8.0
    dst_inner = np.array([[1*S,1*S],[7*S,1*S],[7*S,7*S],[1*S,7*S]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_inner, dst_inner)

def capture_frame(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    # Try to reduce drift (may be ignored by driver)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture image.")
    return frame

def split_grid(warp_bgr):
    """Return dict: square name -> (crop, index), index in A1..H8 (row-major)."""
    h, w = warp_bgr.shape[:2]
    Xs = np.round(np.linspace(0, w-1, 9)).astype(int)
    Ys = np.round(np.linspace(0, h-1, 9)).astype(int)
    ranks = [chr(ord('A') + i) for i in range(8)]  # A..H (top→bottom)
    files = [str(i) for i in range(1, 9)]          # 1..8 (left→right)

    squares = {}
    idx = 0
    for r in range(8):
        for c in range(8):
            x1, x2, y1, y2 = Xs[c], Xs[c+1], Ys[r], Ys[r+1]
            crop = warp_bgr[y1:y2, x1:x2].copy()
            name = f"{ranks[r]}{files[c]}"
            squares[name] = (crop, idx)
            idx += 1
    return squares

# ---------- Augmentations ----------
def rot90k(img, k):
    k = int(k) % 4
    if k == 0: return img
    if k == 1: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2: return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def flip_h(img): return cv2.flip(img, 1)
def flip_v(img): return cv2.flip(img, 0)

def jitter_bc(img, alpha=None, beta=None):
    # img: uint8 BGR
    if alpha is None:
        alpha = random.uniform(*JITTER_ALPHA_RANGE)  # contrast
    if beta is None:
        beta = random.uniform(*JITTER_BETA_RANGE)    # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out, alpha, beta

def generate_augmented_variants(img_bgr):
    """
    Returns list of (variant_bgr, suffix) including:
      - rotations: 0, 90, 180, 270
      - flips: h, v
      - rotated flips (90 + h, 90 + v) for variety
      - plus JITTER for each variant (AUG_JITTER_COUNT copies)
    """
    variants = []

    base = [("orig", img_bgr)]
    if AUG_ROTATIONS:
        base.extend([
            ("rot90",  rot90k(img_bgr,1)),
            ("rot180", rot90k(img_bgr,2)),
            ("rot270", rot90k(img_bgr,3)),
        ])
    if AUG_FLIPS:
        base.extend([
            ("flipH", flip_h(img_bgr)),
            ("flipV", flip_v(img_bgr)),
        ])
        if AUG_ROTATIONS:
            base.extend([
                ("rot90_flipH", flip_h(rot90k(img_bgr,1))),
                ("rot90_flipV", flip_v(rot90k(img_bgr,1))),
            ])

    # Deduplicate by shape/content hash if needed (skip here for speed)
    for tag, im in base:
        variants.append((im, tag))
        # Jitters
        for j in range(AUG_JITTER_COUNT):
            jit, a, b = jitter_bc(im)
            variants.append((jit, f"{tag}_jit{j+1}"))

    return variants

# ---------- Main ----------
def main():
    ensure_meta_exists()
    print("=== Knight Collector (Prompted Diagonals + Augmentations) ===")
    diag_main = [f"{chr(ord('A')+i)}{i+1}" for i in range(8)]   # A1..H8
    diag_anti = [f"{chr(ord('A')+i)}{8-i}" for i in range(8)]   # A8..H1
    squares = diag_main + diag_anti
    print("Order:", squares)

    for sq_name in squares:
        ans = input(f"\nNext square: {sq_name}. Is there a piece on this square? [y/n] ").strip().lower()
        if ans not in {"y","yes"}:
            print(f"Skipping {sq_name}.")
            continue

        frame = capture_frame(WEBCAM_INDEX)
        try:
            H = compute_homography(frame)
        except Exception as e:
            print(f"[WARN] Board corners not found for {sq_name}: {e}")
            continue

        warp = cv2.warpPerspective(frame, H, (BOARD_SIDE, BOARD_SIDE))
        sq_dict = split_grid(warp)
        if sq_name not in sq_dict:
            print(f"[WARN] {sq_name} not recognized; skipping.")
            continue

        crop, sq_idx = sq_dict[sq_name]
        # Preview
        vis = crop.copy()
        cv2.putText(vis, f"{sq_name} piece (aug xN)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Pawn Capture (Augmented)", vis)
        cv2.waitKey(1)

        # Color
        color = input(f"{sq_name}: Color [w/b] ").strip().lower()
        if color in {"w","white"}:
            color = "white"
        elif color in {"b","black"}:
            color = "black"
        else:
            print("Invalid color; skipping.")
            continue

        label = f"{color}_knight"
        ts_base = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cls_dir = OUT_DIR / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save all variants
        variants = generate_augmented_variants(crop)
        saved = 0
        for im, tag in variants:
            fname = f"{sq_name}_{ts_base}_{tag}.png"
            out_path = cls_dir / fname
            cv2.imwrite(str(out_path), im)

            emb = embed_bgr(im)
            append_embedding(emb)
            append_meta_row({
                "timestamp": ts_base,
                "filepath": str(out_path),
                "label": label,
                "color": color,
                "piece": "N",            
                "square_name": sq_name,
                "square_index": sq_idx,
            })
            saved += 1

        print(f"Saved {saved} augmented samples for {sq_name} → {cls_dir}")

        # ESC to abort quickly
        if cv2.waitKey(300) == 27:
            break

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
