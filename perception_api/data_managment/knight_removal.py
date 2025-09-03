import csv, shutil
from pathlib import Path
import re

# === EDIT THESE IF NEEDED ===
ROOT = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
DATA_REFS_DIR = ROOT / "data_refs"
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"          # optional
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"    # optional
# If you have a dataset manifest CSV to prune (the one with the rows you pasted), set it here:
MANIFEST_CSV    = ROOT / "data_refs_manifest.csv"        # change or set to None if not used

# Target selector (the stuff you want gone) — tuned to your example rows
LABEL_DIR_NAME = "white_queen"
SQUARE         = "H1"
TIMESTAMP      = "20250901_234906_968362"

# Regexes for robust matching inside CSV/meta rows
RE_LABEL_DIR   = re.compile(r"/data_refs/" + re.escape(LABEL_DIR_NAME) + r"/")
RE_FILE_STEM   = re.compile(r"/" + re.escape(LABEL_DIR_NAME) + r"/" + re.escape(SQUARE) + r"_" + re.escape(TIMESTAMP))
RE_TIMESTAMP   = re.compile(re.escape(TIMESTAMP))

# ========= helpers =========

def delete_images(base_dir: Path, label_dir: str, square: str, timestamp: str) -> int:
    """Delete all image files under data_refs/<label_dir>/ matching '<square>_<timestamp>*'."""
    cls_dir = base_dir / label_dir
    if not cls_dir.exists():
        print(f"[INFO] Label dir not found, skipping images: {cls_dir}")
        return 0

    patt = f"{square}_{timestamp}*"
    hits = list(cls_dir.glob(patt))
    if not hits:
        print(f"[INFO] No images matching {patt} in {cls_dir}")
        return 0

    deleted = 0
    for p in sorted(set(hits)):
        try:
            p.unlink()
            deleted += 1
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")

    # Clean up empty directory
    try:
        if not any(cls_dir.iterdir()):
            cls_dir.rmdir()
    except Exception:
        pass

    print(f"[OK] Deleted {deleted} file(s) from {cls_dir} matching {square}_{timestamp}*")
    return deleted


def prune_csv(csv_path: Path) -> tuple[int, int]:
    if not csv_path or not csv_path.exists():
        return 0, 0
    backup = csv_path.with_suffix(csv_path.suffix + ".bak")
    shutil.copy2(csv_path, backup)

    kept, dropped = [], 0
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, 0

    # Try to preserve header if it looks like one
    header = None
    start_idx = 0
    if any(any(k in (cell or "").lower() for k in ("label","path","square","timestamp","cosine")) for cell in rows[0]):
        header = rows[0]
        start_idx = 1

    def row_matches(row):
        s = ",".join([c or "" for c in row])
        return bool(RE_LABEL_DIR.search(s) and RE_FILE_STEM.search(s))

    for row in rows[start_idx:]:
        if row_matches(row):
            dropped += 1
        else:
            kept.append(row)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(kept)

    print(f"[OK] Pruned {dropped} row(s) from {csv_path.name}; backup → {backup.name}")
    return len(kept), dropped


def prune_embeddings(emb_pt: Path, emb_meta_csv: Path) -> None:
    """Drop rows from embeddings (.pt) and meta CSV that match our target selection."""
    if not emb_pt.exists() or not emb_meta_csv.exists():
        print("[INFO] Skipping .pt pruning (missing .pt or meta csv).")
        return

    import torch  # lazy import
    import numpy as np

    # Read meta rows
    with open(emb_meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("[INFO] Meta CSV empty; nothing to prune.")
        return

    def row_str(d):
        return ",".join(str(d.get(k, "")) for k in fieldnames)

    keep_mask = []
    dropped = 0
    for r in rows:
        s = row_str(r)
        is_match = bool(RE_LABEL_DIR.search(s) and RE_FILE_STEM.search(s))
        keep_mask.append(not is_match)
        if is_match:
            dropped += 1

    if dropped == 0:
        print("[INFO] No matching rows found in meta; .pt unchanged.")
        return

    # Load embeddings tensor (NxD)
    E = torch.load(emb_pt, map_location="cpu")
    if not hasattr(E, "shape") or len(E.shape) != 2:
        raise RuntimeError(f"{emb_pt} is not a 2D tensor (NxD). Got type={type(E)} shape={getattr(E, 'shape', None)}")

    N = E.shape[0]
    if N != len(rows):
        print(f"[WARN] embeddings/meta count mismatch (E={N}, meta={len(rows)}). Not modifying .pt to avoid misalignment.")
        # Still write filtered meta (safe), but skip .pt
        kept_rows = [r for r, k in zip(rows, keep_mask) if k]
        backup_meta = emb_meta_csv.with_suffix(emb_meta_csv.suffix + ".bak")
        shutil.copy2(emb_meta_csv, backup_meta)
        with open(emb_meta_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept_rows)
        print(f"[OK] Wrote filtered meta CSV only; backup → {backup_meta.name}")
        return

    keep_idx = np.nonzero(keep_mask)[0]

    # Backups
    emb_pt_backup = emb_pt.with_suffix(emb_pt.suffix + ".bak")
    emb_meta_backup = emb_meta_csv.with_suffix(emb_meta_csv.suffix + ".bak")
    shutil.copy2(emb_pt, emb_pt_backup)
    shutil.copy2(emb_meta_csv, emb_meta_backup)

    # Save filtered
    E2 = E[keep_idx]
    torch.save(E2, emb_pt)

    kept_rows = [r for r, k in zip(rows, keep_mask) if k]
    with open(emb_meta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)

    print(f"[OK] Dropped {dropped} row(s) from embeddings ({emb_pt.name}) and meta ({emb_meta_csv.name}); backups → {emb_pt_backup.name}, {emb_meta_backup.name}")


if __name__ == "__main__":
    # 1) delete images from data_refs
    delete_images(DATA_REFS_DIR, LABEL_DIR_NAME, SQUARE, TIMESTAMP)

    # 2) prune manifest (if you use one)
    if MANIFEST_CSV and MANIFEST_CSV.exists():
        prune_csv(MANIFEST_CSV)

    # 3) prune meta csv & embeddings
    if EMB_META_PATH.exists():
        prune_embeddings(EMB_TENSOR_PATH, EMB_META_PATH)
