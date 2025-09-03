#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# perception_api.py
# - JSON-backed memory
# - Embeddings under ROOT / "embeddings"
# - --first wipes memory file
# - Calibrated empties re-integrated (+ fallback heuristic)
# - --calibrate captures per-square empty embeddings

from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import argparse

import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from memory_store import Memory  # JSON-backed memory

# ====== CONFIG ======
ROOT = Path("./")

# Embeddings live under ROOT / "embeddings"
EMB_DIR = ROOT / "embeddings"
EMB_TENSOR_PATH = EMB_DIR / "embeddings_master.pt"
EMB_META_PATH   = EMB_DIR / "embeddings_master_meta.csv"

# Per-square empty calibration file (from --calibrate)
EMPTY_FILE = EMB_DIR / "empty_grid_embeddings.pt"

MODEL_NAME = "openai/clip-vit-base-patch32"
WEBCAM_INDEX = 0
BOARD_SIDE = 900
PATTERN_SIZE = (7, 7)

# Empty handling:
# - Primary path when EMPTY_FILE exists:
EMPTY_SIM_THRESH   = 0.95  # if s_empty >= this → "empty"
# - Fallback if no calibration file:
EMPTY_IF_COS_BELOW = 0.30  # if max piece cosine < this → "empty"

OUT_DIR = ROOT / "live_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = OUT_DIR / "board_memory.json"

# Returned token for empty squares in get_board()
EMPTY_TOKEN = None

# Priors / smoothing
TAU = 0.10         # temperature for softmax over similarities
BIAS_START = 1.6   # bias added to expected starting piece (logit space)
BIAS_MEM_K = 2.0   # memory bias magnitude ~ k * last_confidence
INIT_LOCK_THRESH = 0.80  # fraction of initial layout needed to lock start bias

# Cosine comparison margin for same-color morph handling
EPS_COS = 0.01

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# ---------- utilities ----------------------------------------------

def _label_to_piece_color(label: str) -> Optional[Tuple[str, str]]:
    if not label:
        return None
    low = label.lower().strip()
    if low == "empty":
        return None

    color = None
    if "white" in low or (low.startswith("w") and "white" not in low):
        color = "White"
    elif "black" in low or (low.startswith("b") and "black" not in low):
        color = "Black"

    piece = None
    for k, v in {"king": "K", "queen": "Q", "rook": "R", "bishop": "B", "knight": "N", "pawn": "P"}.items():
        if k in low:
            piece = v
            break
    if piece is None:
        up = label.upper()
        for ch in ("K", "Q", "R", "B", "N", "P"):
            if ch in up:
                piece = ch
                break

    if piece and color:
        return (piece, color)
    return None


def _initial_expected_board() -> Dict[str, Tuple[str, str]]:
    board: Dict[str, Tuple[str, str]] = {}
    files = ["A", "B", "C", "D", "E", "F", "G", "H"]
    back_white = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    back_black = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    for i, f in enumerate(files):
        board[f + "1"] = (back_white[i], "White")
        board[f + "2"] = ("P", "White")
        board[f + "7"] = ("P", "Black")
        board[f + "8"] = (back_black[i], "Black")
    return board


def _expected_match_fraction(
    board_map: Dict[str, Optional[Tuple[str, str]]],
    expected: Dict[str, Tuple[str, str]],
) -> float:
    total = 0
    correct = 0
    for sq, exp in expected.items():
        total += 1
        if board_map.get(sq) == exp:
            correct += 1
    return (correct / max(total, 1))


def _square_sort_key(sq: str):
    return (int(sq[1:]), sq[0])


def _board_list_from_map(
    board_map: Dict[str, Optional[Tuple[str, str]]]
) -> List[Tuple[Optional[Tuple[str, str]], str, str]]:
    items: List[Tuple[Optional[Tuple[str, str]], str, str]] = []
    for sq in sorted(board_map.keys(), key=_square_sort_key):
        pc = board_map[sq] if board_map[sq] is not None else EMPTY_TOKEN
        file_letter, rank_str = sq[0], sq[1:]
        items.append((pc, rank_str, file_letter))
    return items

# ---------- vision + geometry ----------------------------------------------

def compute_homography(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    variants = [
        gray,
        cv2.equalizeHist(gray),
        cv2.GaussianBlur(gray, (5, 5), 0),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 31, 5),
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 31, 5),
    ]
    cols, rows = PATTERN_SIZE  # (7,7) inner corners
    need = cols * rows

    def rect_from_corners(corners, cols, rows):
        idx = lambda r, c: r * cols + c
        tl = corners[idx(0,        0       ), 0]
        tr = corners[idx(0,        cols - 1), 0]
        bl = corners[idx(rows - 1, 0       ), 0]
        br = corners[idx(rows - 1, cols - 1), 0]
        return tl, tr, br, bl

    for img in variants:
        ok, corners = cv2.findChessboardCornersSB(
            img, PATTERN_SIZE,
            flags=(cv2.CALIB_CB_EXHAUSTIVE |
                   cv2.CALIB_CB_ACCURACY  |
                   cv2.CALIB_CB_LARGER)
        )
        if not ok:
            flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            ok, corners = cv2.findChessboardCorners(img, PATTERN_SIZE, flags)
            if ok and corners is not None:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        if ok and corners is not None and len(corners) >= need:
            tl, tr, br, bl = rect_from_corners(corners, cols, rows)
            src_inner = np.array([tl, tr, br, bl], dtype=np.float32)

            S = BOARD_SIDE / 8.0
            dst_inner = np.array(
                [[1 * S, 1 * S],
                 [7 * S, 1 * S],
                 [7 * S, 7 * S],
                 [1 * S, 7 * S]], dtype=np.float32
            )
            return cv2.getPerspectiveTransform(src_inner, dst_inner)

    raise RuntimeError("Could not find chessboard inner corners (7x7). Check lighting/contrast and board size in frame.")


# def capture_frame(index=0):
#     cap = cv2.VideoCapture(index)
#     if not cap.isOpened():
#         raise RuntimeError("Could not open camera.")
#     cap.set(cv2.CAP_PROP_AUTO_WB, 0)
#     cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
#     ok, frame = cap.read()
#     cap.release()
#     if not ok:
#         raise RuntimeError("Failed to capture image.")
#     return frame

def capture_frame(index=0):
    return cv2.imread(ROOT/"test.png")

def map_rc_to_name(c, r):
    return f"{'ABCDEFGH'[c]}{'87654321'[r]}"


@torch.no_grad()
def embed_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inp = processor(images=pil, return_tensors="pt").to(device)
    z = model.get_image_features(**inp)
    return F.normalize(z, p=2, dim=-1)  # [1,D]


def build_piece_prototypes(MIN_SAMPLES_PER_CLASS: int = 3):
    if not EMB_TENSOR_PATH.exists() or not EMB_META_PATH.exists():
        raise FileNotFoundError("Missing master embeddings or meta CSV in 'embeddings' folder")
    embs = torch.load(EMB_TENSOR_PATH, map_location="cpu")
    meta = pd.read_csv(EMB_META_PATH)
    n = min(len(meta), embs.shape[0])
    if n != len(meta) or n != embs.shape[0]:
        print(f"[WARN] Aligning to first {n} rows due to mismatch.")
        meta = meta.iloc[:n].reset_index(drop=True)
        embs = embs[:n]

    protos, names = [], []
    for lbl, grp in meta.groupby("label"):
        if len(grp) < MIN_SAMPLES_PER_CLASS:
            continue
        m = F.normalize(embs[grp.index.to_list()].mean(0, keepdim=True), p=2, dim=-1)
        protos.append(m)
        names.append(lbl)

    if not protos:
        raise RuntimeError("No class has enough samples to prototype.")

    P = torch.cat(protos, 0).to(device)  # [C,D]
    parsed = [_label_to_piece_color(lbl) for lbl in names]
    return P, names, parsed


def split_grid(warp_bgr):
    h, w = warp_bgr.shape[:2]
    Xs = np.round(np.linspace(0, w - 1, 9)).astype(int)
    Ys = np.round(np.linspace(0, h - 1, 9)).astype(int)
    for r in range(0,8):                 # image rows  (top→bottom)
        for c in reversed(range(0,8)):   # image cols  (right→left)
            x1, x2, y1, y2 = Xs[c], Xs[c + 1], Ys[r], Ys[r + 1]
            crop = warp_bgr[y1:y2, x1:x2].copy()
            name = map_rc_to_name(c, r)
            yield name, (r, c), crop, (x1, y1, x2, y2)

# ---------- calibrated empties (load) ----------------------------------------

def load_empty_per_square():
    """
    Load per-square empty embeddings saved by calibration.
    Returns:
      name_to_idx: dict {"A1": idx, ...} or None
      Z:           (N, D) normalized tensor (device) or None
    """
    if not EMPTY_FILE.exists():
        return None, None
    d = torch.load(EMPTY_FILE, map_location="cpu")  # expects { "A1": tensor(D,), ... }
    # Stable order: sort by rank (1..8), then file (A..H)
    names = sorted(d.keys(), key=lambda s: (int(s[1:]), s[0]))
    Z = torch.stack([d[n] for n in names], dim=0)  # (N, D)
    Z = F.normalize(Z, p=2, dim=-1).to(device)
    name_to_idx = {n: i for i, n in enumerate(names)}
    return name_to_idx, Z

# ---------- biasing + classification ----------------------------------------

def _apply_biases_and_decide(
    sims: torch.Tensor,  # [C] RAW cosine sims
    class_names: List[str],
    class_parsed: List[Optional[Tuple[str, str]]],
    square: str,
    mem: Memory,
):
    # Convert sims to logits
    logits = (sims / TAU).clone()

    # Memory prior (bias toward last committed piece on this square)
    last_board, _last_conf, _last_cos = mem.last_committed()
    prev = last_board.get(square)
    if prev is not None:
        try:
            j = next(i for i, pc in enumerate(class_parsed) if pc == prev)
            logits[j] += BIAS_MEM_K * float(mem.state.last_confidence.get(square, 1.0))
        except StopIteration:
            pass

    # Initial start-position bias if locked
    if mem.start_bias_locked:
        exp = _initial_expected_board().get(square)
        if exp is not None:
            try:
                j = next(i for i, pc in enumerate(class_parsed) if pc == exp)
                logits[j] += BIAS_START
            except StopIteration:
                pass

    probs = torch.softmax(logits, dim=0)
    k = int(torch.argmax(probs).item())
    return k, float(probs[k].item())  # class index, posterior confidence

# ---------- Capture + classify pipeline -------------------------------------

def _capture_and_classify(mem: Memory, apply_start_bias: Optional[bool] = None):
    """
    Returns:
      board_map { "A1": ("R","White") or None },
      conf_map { "A1": 0.93 }  # posterior (after biases)
      cos_map  { "A1": 0.37 }  # RAW cosine sim of chosen label (or empty sim)
      annotated_img_path
    """
    piece_proto, class_names, class_parsed = build_piece_prototypes()

    # Load calibrated empties (if available)
    empty_name_to_idx, empty_Z = load_empty_per_square()
    have_calibrated_empty = (empty_name_to_idx is not None) and (empty_Z is not None)

    # Optionally toggle start bias just for this pass (temporarily)
    original_lock = mem.start_bias_locked
    if apply_start_bias is not None and apply_start_bias != original_lock:
        mem.state.start_bias_locked = apply_start_bias  # temporary

    frame = capture_frame(WEBCAM_INDEX)
    H = compute_homography(frame)
    warp = cv2.warpPerspective(frame, H, (BOARD_SIDE, BOARD_SIDE))

    # orientation adjustments (your current sequence)
    warp = cv2.flip(warp, 1)
    warp = cv2.flip(warp, 0)
    warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    warp = cv2.flip(warp, 1)
    warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

    annotated = warp.copy()
    board_map: Dict[str, Optional[Tuple[str, str]]] = {}
    conf_map: Dict[str, float] = {}
    cos_map: Dict[str, float] = {}

    for sq_name, (r, c), crop, (x1, y1, x2, y2) in split_grid(warp):
        z = embed_bgr(crop)                           # [1,D]
        sims = (z @ piece_proto.T).squeeze(0)         # [C]
        max_cos = float(torch.max(sims).item())

        # --- Calibrated empty check (per-square) ---
        if have_calibrated_empty and (sq_name in empty_name_to_idx):
            ei = empty_name_to_idx[sq_name]
            s_empty = float((z @ empty_Z[ei:ei+1].T).item())
            if s_empty >= EMPTY_SIM_THRESH:
                label = "empty"
                pc = None
                conf = s_empty          # store similarity as "confidence" for empty
                cosn = s_empty          # store raw empty similarity
            else:
                k, conf = _apply_biases_and_decide(sims, class_names, class_parsed, sq_name, mem)
                label = class_names[k]
                pc = _label_to_piece_color(label)
                cosn = float(sims[k].item())
        else:
            # Fallback (no calibration file)
            if max_cos < EMPTY_IF_COS_BELOW:
                label = "empty"
                pc = None
                conf = 1.0
                cosn = max_cos
            else:
                k, conf = _apply_biases_and_decide(sims, class_names, class_parsed, sq_name, mem)
                label = class_names[k]
                pc = _label_to_piece_color(label)
                cosn = float(sims[k].item())

        # draw
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(annotated, sq_name, (cx - 18, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"{label}", (x1 + 5, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        board_map[sq_name] = pc
        conf_map[sq_name] = float(conf)
        cos_map[sq_name] = float(cosn)

    # grid overlay
    h, w = warp.shape[:2]
    Xs = np.round(np.linspace(0, w - 1, 9)).astype(int)
    Ys = np.round(np.linspace(0, h - 1, 9)).astype(int)
    for x in Xs:
        cv2.line(annotated, (x, 0), (x, h - 1), (0, 0, 255), 2)
    for y in Ys:
        cv2.line(annotated, (0, y), (w - 1, y), (0, 0, 255), 2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_img = OUT_DIR / f"board_detect_{ts}.png"
    cv2.imwrite(str(out_img), annotated)
    print(f"Saved annotated PNG: {out_img}")
    if have_calibrated_empty:
        print(f"Using calibrated empties (EMPTY_SIM_THRESH={EMPTY_SIM_THRESH})")
    else:
        print(f"Using fallback empty rule (max piece cos < {EMPTY_IF_COS_BELOW})")

    # restore original lock state
    mem.state.start_bias_locked = original_lock

    return board_map, conf_map, cos_map, out_img

# ---------- Calibration routine ---------------------------------------------

@torch.no_grad()
def calibrate_empty_grid(num_frames: int = 5):
    """
    Capture an EMPTY chessboard and compute per-square empty embeddings.
    Averages across `num_frames` frames. Saves to EMPTY_FILE.
    """
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    acc: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for i in range(num_frames):
        frame = capture_frame(WEBCAM_INDEX)
        H = compute_homography(frame)
        warp = cv2.warpPerspective(frame, H, (BOARD_SIDE, BOARD_SIDE))


        warp = cv2.flip(warp, 1)
        warp = cv2.flip(warp, 0)
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
        warp = cv2.flip(warp, 1)
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

        for sq_name, (_r, _c), crop, _rect in split_grid(warp):
            z = embed_bgr(crop).squeeze(0).cpu()  # (D,)
            if sq_name not in acc:
                acc[sq_name] = z
                counts[sq_name] = 1
            else:
                acc[sq_name] += z
                counts[sq_name] += 1

        print(f"[calibrate] captured frame {i+1}/{num_frames}")

    # Average + L2 normalize
    out: Dict[str, torch.Tensor] = {}
    for sq_name, zsum in acc.items():
        zmean = zsum / max(counts[sq_name], 1)
        zmean = F.normalize(zmean.unsqueeze(0), p=2, dim=-1).squeeze(0)  # (D,)
        out[sq_name] = zmean

    torch.save(out, EMPTY_FILE)
    print(f"[calibrate] wrote per-square empty embeddings → {EMPTY_FILE}")
    print(f"[calibrate] squares: {len(out)} (expected 64)")

# ---------- Public API -------------------------------------------------------

def get_board():
    """
    Classifies and saves PNG, writes a *pending* board (with posterior + cosine)
    to JSON-backed memory, and returns 64 items:
      [ (piece_or_EMPTY_TOKEN, rank, file), ... ]
    where piece_or_EMPTY_TOKEN is (piece, Color) or EMPTY_TOKEN for empty.
    """
    mem = Memory(MEMORY_FILE)
    board_map, conf_map, cos_map, _ = _capture_and_classify(mem, apply_start_bias=None)

    # Lock start bias if initial setup detected
    if not mem.start_bias_locked:
        frac = _expected_match_fraction(board_map, _initial_expected_board())
        if frac >= INIT_LOCK_THRESH:
            mem.lock_start_bias()
            # Re-run with start bias applied immediately for this call
            board_map, conf_map, cos_map, _ = _capture_and_classify(mem, apply_start_bias=True)

    mem.set_pending(board_map, conf_map, cos_map)
    return _board_list_from_map(board_map)


def change_detected() -> bool:
    """
    Compare *pending* vs last committed.
    If a same-color different-type flip occurs, compare RAW cosine similarity scores
    and keep the higher-cosine piece using a margin: cos_curr > cos_prev + EPS_COS
    This updates memory but DOES NOT flag a change.
    Any empty<->occupied or color change flags a real change.
    Commits the resolved board to JSON-backed memory.
    """
    mem = Memory(MEMORY_FILE)
    pend_board, pend_conf, pend_cos = mem.pending()
    if pend_board is None:
        _ = get_board()
        pend_board, pend_conf, pend_cos = mem.pending()
    if pend_board is None:
        return False

    last_board, last_conf, last_cos = mem.last_committed()

    real_change = False
    resolved_board = dict(pend_board)
    resolved_conf = dict(pend_conf or {})
    resolved_cos = dict(pend_cos or {})

    squares = sorted(set(list(pend_board.keys()) + list(last_board.keys())))
    for sq in squares:
        prev = last_board.get(sq)  # None or (piece,color)
        curr = pend_board.get(sq)  # None or (piece,color)
        if prev == curr:
            continue

        # Same-color, different-type -> treat as potential noise
        if prev is not None and curr is not None and prev[1] == curr[1] and prev[0] != curr[0]:
            cos_prev = float((last_cos or {}).get(sq, 0.0))
            cos_curr = float((pend_cos or {}).get(sq, 0.0))
            if cos_curr > cos_prev + EPS_COS:
                # keep current prediction (higher RAW cosine by margin)
                resolved_board[sq] = curr
                resolved_cos[sq] = cos_curr
                resolved_conf[sq] = float((pend_conf or {}).get(sq, 0.0))
            else:
                # revert to previous
                resolved_board[sq] = prev
                resolved_cos[sq] = cos_prev
                resolved_conf[sq] = float((last_conf or {}).get(sq, 0.0))
            # do NOT mark as real change
        else:
            # Any empty<->occupied or color change is a real change
            real_change = True

    mem.commit_pending(
        updated_board=resolved_board,
        updated_conf=resolved_conf,
        updated_cos=resolved_cos,
    )
    return real_change


# ---------- CLI ----------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Perception API")
    parser.add_argument("--first", action="store_true", help="Wipe previous memory file and start fresh")
    parser.add_argument("--calibrate", action="store_true", help="Capture EMPTY board and save per-square embeddings")
    parser.add_argument("--calibrate-frames", type=int, default=5, help="Frames to average during calibration")
    args = parser.parse_args()

    if args.first and MEMORY_FILE.exists():
        print(f"--first specified: removing {MEMORY_FILE}")
        try:
            MEMORY_FILE.unlink()
        except Exception as e:
            print(f"[WARN] Could not remove memory file: {e}")

    # Run calibration and exit
    if args.calibrate:
        calibrate_empty_grid(num_frames=args.calibrate_frames)
        return

    lst = get_board()
    print("Board (64):", lst)
    print("Change detected:", change_detected())


if __name__ == "__main__":
    main()
