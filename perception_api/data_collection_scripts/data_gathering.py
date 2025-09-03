import os
import csv
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
OUT_DIR         = ROOT / "data_refs"                      # where labeled crops are stored
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"           # growing tensor of image embeddings
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"     # growing CSV metadata
WEBCAM_INDEX    = 0
BOARD_SIDE      = 900
PATTERN_SIZE    = (7, 7)                                  # 7x7 inner corners
MODEL_NAME      = "openai/clip-vit-base-patch32"
# ====================================

# ---------- Utilities ----------
def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not EMB_META_PATH.exists():
        with open(EMB_META_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "filepath", "label", "color", "piece",
                "square_name", "square_index", "h8_path", "g8_path"
            ])

def gray_world_white_balance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3.0
    eps = 1e-6
    b *= (m / (b.mean() + eps)); g *= (m / (g.mean() + eps)); r *= (m / (r.mean() + eps))
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def compute_homography(frame_bgr):
    """Corner finding uses WB+CLAHE only to get stable corners; we will warp ORIGINAL colors."""
    wb = gray_world_white_balance(frame_bgr)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    ok = False; corners = None
    try:
        res = cv2.findChessboardCornersSB(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if isinstance(res, tuple):
            ok, pts = res
            corners = pts if ok else None
        else:
            ok = res is not None
            corners = res
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
        raise RuntimeError("Could not find chessboard inner corners (7x7). Ensure the grid is visible and evenly lit.")

    # TL=0, TR=6, BL=42, BR=48 (for 7x7 inner)
    tl = corners[0, 0]
    tr = corners[6, 0]
    bl = corners[42, 0]
    br = corners[48, 0]

    src_inner = np.array([tl, tr, br, bl], dtype=np.float32)  # TL, TR, BR, BL

    S = BOARD_SIDE / 8.0
    dst_inner = np.array([
        [1*S, 1*S],
        [7*S, 1*S],
        [7*S, 7*S],
        [1*S, 7*S],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_inner, dst_inner)
    return H

def capture_frame(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    # (Optional) Try to lock auto settings to stabilize color
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # backend-specific
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture image.")
    return frame

def split_all_squares(warp_bgr):
    """
    Return list of (square_name, (r,c), crop_bgr) over the whole 8x8.
    Naming: A1..H8 (A=top row, H=bottom row; 1=left column, 8=right column)
    """
    h, w = warp_bgr.shape[:2]
    Xs = np.round(np.linspace(0, w-1, 9)).astype(int)
    Ys = np.round(np.linspace(0, h-1, 9)).astype(int)

    ranks = [chr(ord('A') + i) for i in range(8)]   # A..H (top to bottom)
    files = [str(i) for i in range(1, 9)]           # 1..8 (left to right)

    squares = []
    for r in range(8):           # 0 (A/top) .. 7 (H/bottom)
        for c in range(8):       # 0 (1/left) .. 7 (8/right)
            x1, x2 = Xs[c], Xs[c+1]
            y1, y2 = Ys[r], Ys[r+1]
            crop = warp_bgr[y1:y2, x1:x2].copy()
            name = f"{ranks[r]}{files[c]}"
            squares.append((name, (r, c), crop))
    return squares

# ---------- CLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained(MODEL_NAME)

@torch.no_grad()
def embed_image_bgr(img_bgr: np.ndarray):
    """Return L2-normalized CLIP image embedding (1xD tensor)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=pil, return_tensors="pt").to(device)
    feats  = clip_model.get_image_features(**inputs)
    feats  = F.normalize(feats, p=2, dim=-1)
    return feats  # [1, D]

def safe_imwrite(path: Path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)

def append_embedding(embedding_1xD: torch.Tensor):
    """Append a single 1xD embedding to the master .pt tensor (create if missing)."""
    if EMB_TENSOR_PATH.exists():
        old = torch.load(EMB_TENSOR_PATH, map_location="cpu")
        new = torch.cat([old, embedding_1xD.cpu()], dim=0)
    else:
        new = embedding_1xD.cpu()
    torch.save(new, EMB_TENSOR_PATH)

def append_meta_row(row_dict: dict):
    newfile = not EMB_META_PATH.exists()
    with open(EMB_META_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow([
                "timestamp", "filepath", "label", "color", "piece",
                "square_name", "square_index", "h8_path", "g8_path"
            ])
        w.writerow([
            row_dict.get("timestamp", ""),
            row_dict.get("filepath", ""),
            row_dict.get("label", ""),
            row_dict.get("color", ""),
            row_dict.get("piece", ""),
            row_dict.get("square_name", ""),
            row_dict.get("square_index", ""),
            "",  # h8_path (unused in 64-square mode)
            "",  # g8_path (unused in 64-square mode)
        ])

def prompt_label_for_square(square_name: str):
    """
    Ask for color/piece for THIS square.
    color: white/black/empty
    piece: P/N/B/R/Q/K or empty
    Returns (color, piece, label)
    """
    print(f"\nSquare: {square_name}")
    while True:
        color = input("Color [white/black/empty] (default=empty): ").strip().lower()
        if color == "":
            color = "none"
            piece = "empty"
            label = "empty"
            return color, piece, label
        if color in {"white","black","empty","none"}:
            break
        print("Please type 'white', 'black', or 'empty'.")

    if color in {"empty", "none"}:
        piece = "empty"
        label = "empty"
        return "none", piece, label

    while True:
        piece = input("Piece [P,N,B,R,Q,K] (default=P): ").strip().upper()
        if piece == "": piece = "P"
        if piece in {"P","N","B","R","Q","K"}:
            break
        print("Please type one of: P N B R Q K (or press ENTER for P).")

    piece_map = {"P":"pawn","N":"knight","B":"bishop","R":"rook","Q":"queen","K":"king"}
    label = f"{color}_{piece_map[piece]}"
    return color, piece, label

def show_square(crop_bgr, square_name):
    vis = crop_bgr.copy()
    cv2.putText(vis, square_name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Square", vis)
    cv2.waitKey(1)

def main():
    ensure_dirs()
    print("=== 64-Square Collector ===")
    input("Clear the board or arrange it as desired. Press ENTER to capture...")

    frame = capture_frame(WEBCAM_INDEX)

    # Compute homography from the SAME frame; warp ORIGINAL colors for crops
    try:
        H = compute_homography(frame)
    except Exception as e:
        print(f"[ERROR] Could not find board corners: {e}")
        return

    warp = cv2.warpPerspective(frame, H, (BOARD_SIDE, BOARD_SIDE))

    # Build list of all 64 squares (A1..H8)
    squares = split_all_squares(warp)

    # Iterate and ask for each
    for idx, (sq_name, (r, c), crop) in enumerate(squares):
        show_square(crop, sq_name)
        color, piece, label = prompt_label_for_square(sq_name)

        # Save crop & embedding
                # ---- Save crop & embeddings (with rotations) ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cls_dir = OUT_DIR / label
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Prepare rotations: 0 (original), 90, 180, 270 degrees
        rotations = [
            (0,   crop),
            (90,  cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(crop, cv2.ROTATE_180)),
            (270, cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]

        for angle, img in rotations:
            # Build filename; only add suffix for rotated versions
            suffix = "" if angle == 0 else f"_rot{angle}"
            out_path = cls_dir / f"{sq_name}_{ts}{suffix}.png"

            safe_imwrite(out_path, img)

            emb = embed_image_bgr(img)  # [1, D]
            append_embedding(emb)
            append_meta_row({
                "timestamp": ts,
                "filepath": str(out_path),
                "label": label,
                "color": color,
                "piece": piece,
                "square_name": sq_name,
                "square_index": idx  # 0..63 in A1..H8 order
            })

            print(f"Saved: {out_path}")


        # ESC to abort early
        key = cv2.waitKey(100)
        if key == 27:
            print("Aborted by user.")
            break

    cv2.destroyAllWindows()
    print("\nDone. Embeddings →", EMB_TENSOR_PATH)
    print("Metadata       →", EMB_META_PATH)

if __name__ == "__main__":
    main()
