# cleanup_none_pawn.py
import csv, shutil, sys
from pathlib import Path
import re

# === EDIT THESE IF NEEDED ===
ROOT = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
DATA_REFS_DIR = ROOT / "data_refs"
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"          # optional
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"    # optional
# If you have a dataset manifest CSV to prune (the one with the rows you pasted), set it here:
MANIFEST_CSV    = ROOT / "data_refs_manifest.csv"        # change or set to None if not used

# Target selector (the stuff you want gone)
RE_LABEL_DIR   = re.compile(r"/data_refs/None_pawn/")
RE_TIMESTAMP   = re.compile(r"20250827_051645")

# ===== helpers =====

def prune_csv(csv_path: Path):
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
        # Convert to a single string for robust substring matching across unknown schemas
        s = ",".join([c or "" for c in row])
        return bool(RE_LABEL_DIR.search(s) and RE_TIMESTAMP.search(s))

    for i, row in enumerate(rows[start_idx:], start=start_idx):
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

def prune_embeddings(emb_pt: Path, emb_meta_csv: Path):
    """
    Keep embeddings (.pt) in sync with meta csv by dropping rows matching the target.
    Works when the meta CSV has either a 'path' column OR we just match across the whole row string.
    """
    if not emb_pt.exists() or not emb_meta_csv.exists():
        print("[INFO] Skipping .pt pruning (missing .pt or meta csv).")
        return

    import torch  # lazy import

    # Read meta rows
    with open(emb_meta_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        print("[INFO] Meta CSV empty; nothing to prune.")
        return

    # Build mask: keep rows NOT matching our target
    def row_str(d):
        return ",".join(str(d.get(k,"")) for k in fieldnames)

    keep_mask = []
    dropped = 0
    for r in rows:
        s = row_str(r)
        is_match = bool(RE_LABEL_DIR.search(s) and RE_TIMESTAMP.search(s))
        keep_mask.append(not is_match)
        if is_match:
            dropped += 1

    if dropped == 0:
        print("[INFO] No matching rows found in meta; .pt unchanged.")
        return

    # Load embeddings tensor (NxD)
    E = torch.load(emb_pt, map_location="cpu")
    if hasattr(E, "shape"):
        N = E.shape[0]
    else:
        raise RuntimeError(f"{emb_pt} is not a tensor. Expected NxD torch tensor.")

    if N != len(rows):
        print(f"[WARN] embeddings/meta count mismatch (E={N}, meta={len(rows)}). Not modifying .pt to avoid misalignment.")
        # Still write filtered meta (safe), but skip .pt
        kept_rows = [r for r,k in zip(rows, keep_mask) if k]
        backup_meta = emb_meta_csv.with_suffix(emb_meta_csv.suffix + ".bak")
        shutil.copy2(emb_meta_csv, backup_meta)
        with open(emb_meta_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(kept_rows)
        print(f"[OK] Wrote filtered meta CSV only; backup → {backup_meta.name}")
        return

    import numpy as np
    keep_idx = np.nonzero(keep_mask)[0]
    E2 = E[keep_idx]

    # Backups
    emb_pt_backup = emb_pt.with_suffix(emb_pt.suffix + ".bak")
    emb_meta_backup = emb_meta_csv.with_suffix(emb_meta_csv.suffix + ".bak")
    shutil.copy2(emb_pt, emb_pt_backup)
    shutil.copy2(emb_meta_csv, emb_meta_backup)

    # Save filtered
    torch.save(E2, emb_pt)
    kept_rows = [r for r,k in zip(rows, keep_mask) if k]
    with open(emb_meta_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)

    print(f"[OK] Dropped {dropped} row(s) from embeddings ({emb_pt.name}) and meta ({emb_meta_csv.name}); backups → {emb_pt_backup.name}, {emb_meta_backup.name}")

if __name__ == "__main__":
    # 1) delete images
    # 2) prune manifest (if you use one)
    if MANIFEST_CSV and MANIFEST_CSV.exists():
        prune_csv(MANIFEST_CSV)
    # 3) prune meta csv & embeddings
    if EMB_META_PATH.exists():
        prune_embeddings(EMB_TENSOR_PATH, EMB_META_PATH)
