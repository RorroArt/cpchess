# remove_white_queen_embeddings.py
import csv
from pathlib import Path
import torch
import shutil

# ===== CONFIG =====
ROOT            = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
DATA_REFS_DIR   = ROOT / "data_refs"
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"
# ==================

def main():
    if not EMB_TENSOR_PATH.exists() or not EMB_META_PATH.exists():
        print("No embeddings/meta file found.")
        return

    # --- Backup originals before touching ---
    shutil.copy(EMB_TENSOR_PATH, EMB_TENSOR_PATH.with_suffix(".pt.bak"))
    shutil.copy(EMB_META_PATH, EMB_META_PATH.with_suffix(".csv.bak"))
    print(f"[INFO] Backups created: {EMB_TENSOR_PATH}.bak, {EMB_META_PATH}.bak")

    # Load embeddings
    E = torch.load(EMB_TENSOR_PATH, map_location="cpu")  # NxD
    print(f"[INFO] Loaded embeddings: {E.shape}")

    # Read metadata
    with open(EMB_META_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != E.shape[0]:
        print("[WARN] Metadata rows != embeddings count — aborting for safety.")
        return

    # Filter out ONLY white queen rows
    keep_rows = []
    keep_embs = []
    removed_paths = []
    for emb, row in zip(E, rows):
        label = row.get("label", "").lower()
        if label == "white_queen":
            removed_paths.append(row.get("filepath", ""))
            continue
        keep_rows.append(row)
        keep_embs.append(emb.unsqueeze(0))

    if not keep_rows:
        print("Everything was white_queen! Nothing left.")
        return

    newE = torch.cat(keep_embs, dim=0)

    # Save new embeddings + metadata
    torch.save(newE, EMB_TENSOR_PATH)
    with open(EMB_META_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(keep_rows)

    print(f"[INFO] Kept {len(keep_rows)} embeddings, removed {len(rows)-len(keep_rows)} white queens.")

    # Delete white queen image files
    deleted = 0
    for path in removed_paths:
        if path and Path(path).exists():
            try:
                Path(path).unlink()
                deleted += 1
            except Exception as e:
                print(f"Could not delete {path}: {e}")
    print(f"[INFO] Deleted {deleted} white queen image files.")

    # Also remove leftover white_queen folder
    d = DATA_REFS_DIR / "white_queen"
    if d.exists():
        for img in d.glob("*.png"): img.unlink()
        for img in d.glob("*.jpg"): img.unlink()
        try:
            d.rmdir()
            print(f"[INFO] Removed folder {d}")
        except OSError:
            print(f"[INFO] Folder {d} not empty, left intact.")

if __name__ == "__main__":
    main()
