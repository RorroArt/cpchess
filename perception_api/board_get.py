# label_with_embeddings.py
import os
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ============== CONFIG ==============
ROOT            = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
DATA_REFS_DIR   = ROOT / "data_refs"                       # fallback gallery if no .pt/.csv
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"            # optional prebuilt embeddings
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"      # optional meta for .pt
OUT_DIR         = ROOT                                     # where CSV + annotated image go
WEBCAM_INDEX    = 0
BOARD_SIDE      = 900
PATTERN_SIZE    = (7, 7)                                   # 7x7 inner corners
MODEL_NAME      = "openai/clip-vit-base-patch32"
CONF_UNKNOWN_TH = 0.2   # if max cosine < this, label as "empty"
# ====================================

# ====== CLIP ======
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
clip_proc  = CLIPProcessor.from_pretrained(MODEL_NAME)

@torch.no_grad()
def embed_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """Return L2-normalized CLIP image embedding (1xD)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = clip_proc(images=pil, return_tensors="pt").to(device)
    feats  = clip_model.get_image_features(**inputs)
    return F.normalize(feats, p=2, dim=-1)  # [1, D]

@torch.no_grad()
def embed_paths(paths: list[Path]) -> torch.Tensor:
    """Batch-embed image paths → NxD (L2-normalized)."""
    if not paths:
        return torch.empty(0, clip_model.config.projection_dim)
    embs = []
    batch = []
    for p in paths:
        try:
            pil = Image.open(p).convert("RGB")
            batch.append(pil)
            if len(batch) == 16:
                inputs = clip_proc(images=batch, return_tensors="pt").to(device)
                feats  = clip_model.get_image_features(**inputs)
                embs.append(F.normalize(feats, p=2, dim=-1))
                batch = []
        except Exception:
            pass
    if batch:
        inputs = clip_proc(images=batch, return_tensors="pt").to(device)
        feats  = clip_model.get_image_features(**inputs)
        embs.append(F.normalize(feats, p=2, dim=-1))
    return torch.cat(embs, dim=0).cpu()

# ====== Board helpers ======
def gray_world_white_balance(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    m = (b.mean() + g.mean() + r.mean()) / 3.0
    eps = 1e-6
    b *= (m / (b.mean() + eps)); g *= (m / (g.mean() + eps)); r *= (m / (r.mean() + eps))
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)

def compute_homography(frame_bgr):
    """Use WB+CLAHE for stable corner detection; warp ORIGINAL frame colors."""
    wb = gray_world_white_balance(frame_bgr)
    gray = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)

    ok, corners = False, None
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
        raise RuntimeError("Could not find chessboard inner corners (7x7).")

    tl = corners[0, 0]; tr = corners[6, 0]; bl = corners[42, 0]; br = corners[48, 0]
    src_inner = np.array([tl, tr, br, bl], dtype=np.float32)

    S = BOARD_SIDE / 8.0
    dst_inner = np.array([[1*S,1*S],[7*S,1*S],[7*S,7*S],[1*S,7*S]], dtype=np.float32)
    return cv2.getPerspectiveTransform(src_inner, dst_inner)

def capture_frame(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    # Try to stabilize camera (driver dependent; safe to keep)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to capture image.")
    return frame

def split_grid(warp_bgr):
    """Return list of (name,(x1,y1,x2,y2),crop_bgr) for all 64 squares A1..H8 (top→bottom, left→right)."""
    h, w = warp_bgr.shape[:2]
    Xs = np.round(np.linspace(0, w-1, 9)).astype(int)
    Ys = np.round(np.linspace(0, h-1, 9)).astype(int)
    ranks = [chr(ord('A') + i) for i in range(8)]  # A..H
    files = [str(i) for i in range(1, 9)]          # 1..8

    squares = []
    for r in range(8):
        for c in range(8):
            x1, x2 = Xs[c], Xs[c+1]
            y1, y2 = Ys[r], Ys[r+1]
            crop = warp_bgr[y1:y2, x1:x2].copy()
            name = f"{ranks[r]}{files[c]}"
            squares.append((name, (x1,y1,x2,y2), crop))
    return squares

# ====== Gallery / Prototypes ======
def label_to_color_piece(label: str):
    """Map 'white_pawn' -> ('White','P'), 'empty' -> ('none','empty')."""
    if label == "empty":
        return ("none", "empty")
    parts = label.split("_", 1)
    if len(parts) != 2:
        return ("none", "empty")
    color = "White" if parts[0].lower() == "white" else "Black"
    piece_map = {"pawn":"P","knight":"N","bishop":"B","rook":"R","queen":"Q","king":"K"}
    piece = piece_map.get(parts[1].lower(), "P")
    return (color, piece)

def build_prototypes_from_pt_csv():
    """Load embeddings_master.pt and meta CSV → dict[label] = prototype (1xD tensor)."""
    if not (EMB_TENSOR_PATH.exists() and EMB_META_PATH.exists()):
        return None

    # Load embeddings (NxD) and meta
    E = torch.load(EMB_TENSOR_PATH, map_location="cpu")  # NxD
    labels = []
    with open(EMB_META_PATH, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            labels.append(row.get("label","empty"))
    if len(labels) != E.shape[0]:
        print("[WARN] Embedding/meta count mismatch — falling back to data_refs.")
        return None

    buckets = defaultdict(list)
    for i, lab in enumerate(labels):
        buckets[lab].append(E[i:i+1])

    prototypes = {}
    for lab, tensors in buckets.items():
        X = torch.cat(tensors, dim=0)         # k x D
        mu = F.normalize(X.mean(dim=0, keepdim=True), p=2, dim=-1)  # 1 x D
        prototypes[lab] = mu
    return prototypes

def build_prototypes_from_data_refs():
    """Walk data_refs/<label>/*.png → embed → mean per label."""
    if not DATA_REFS_DIR.exists():
        return {}

    prototypes = {}
    for lab_dir in sorted(DATA_REFS_DIR.iterdir()):
        if not lab_dir.is_dir():
            continue
        imgs = sorted([p for p in lab_dir.glob("*.png")] + [p for p in lab_dir.glob("*.jpg")] + [p for p in lab_dir.glob("*.jpeg")])
        if not imgs:
            continue
        embs = embed_paths(imgs)  # NxD
        if embs.shape[0] == 0:
            continue
        mu = F.normalize(embs.mean(dim=0, keepdim=True), p=2, dim=-1)  # 1xD
        prototypes[lab_dir.name] = mu
    return prototypes

def load_prototypes():
    protos = build_prototypes_from_pt_csv()
    if protos is not None and len(protos) > 0:
        print(f"[INFO] Loaded prototypes from {EMB_TENSOR_PATH.name} + {EMB_META_PATH.name} ({len(protos)} classes).")
        return protos
    protos = build_prototypes_from_data_refs()
    print(f"[INFO] Built prototypes from {DATA_REFS_DIR} ({len(protos)} classes).")
    return protos

# ====== Classification ======
def classify_roi(roi_bgr, prototypes: dict[str, torch.Tensor]):
    """Return (label, cosine) by nearest prototype; apply CONF_UNKNOWN_TH to map to 'empty' if low."""
    if roi_bgr.size == 0:
        return "empty", 0.0
    x = embed_bgr(roi_bgr).cpu()  # 1xD
    best_lab, best_cos = "empty", -1.0
    for lab, proto in prototypes.items():
        cos = float((x @ proto.T).item())  # cosine (since both are normalized)
        if cos > best_cos:
            best_cos, best_lab = cos, lab

    # Unknown handling
    if best_cos < CONF_UNKNOWN_TH:
        return "empty", best_cos
    return best_lab, best_cos

def label_to_drawables(label: str):
    color, piece = label_to_color_piece(label)
    if color == "none":
        return "empty", color, piece
    return f"{color} {piece}", color, piece

# ====== Main ======
def main():
    # 1) Load prototypes
    prototypes = load_prototypes()
    if not prototypes:
        print("[ERROR] No prototypes found. Collect data in data_refs/ or via embeddings_master first.")
        return

    # 2) Capture + warp
    frame = capture_frame(WEBCAM_INDEX)
    try:
        H = compute_homography(frame)
    except Exception as e:
        print(f"[ERROR] Could not find board corners: {e}")
        return
    warp = cv2.warpPerspective(frame, H, (BOARD_SIDE, BOARD_SIDE))

    # 3) Split squares
    squares = split_grid(warp)

    # 4) Classify each square
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"labels_{ts}.csv"
    board_png = OUT_DIR / f"labels_board_{ts}.png"

    # draw + write
    vis = warp.copy()
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["square_name", "color", "piece_code", "label", "cosine"])

        for name, (x1,y1,x2,y2), crop in squares:
            lab, cos = classify_roi(crop, prototypes)
            draw_text, color, piece_code = label_to_drawables(lab)

            # CSV row
            w.writerow([name, color, piece_code, lab, f"{cos:.4f}"])

            # Draw
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 1)
            cv2.putText(vis, name, (x1+6, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            if lab != "empty":
                cv2.putText(vis, f"{draw_text} ({cos:.2f})", (cx-60, cy+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

    cv2.imwrite(str(board_png), vis)
    print(f"\nSaved labels CSV → {csv_path}")
    print(f"Saved annotated board → {board_png}")
    print(f"Classes used: {sorted(prototypes.keys())}")
    print(f"(CONF_UNKNOWN_TH={CONF_UNKNOWN_TH:.2f}; increase to be stricter, decrease to be more permissive.)")

if __name__ == "__main__":
    main()
