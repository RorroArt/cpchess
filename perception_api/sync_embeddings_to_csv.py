# sync_embeddings_to_csv.py
from pathlib import Path
import pandas as pd
import torch
import sys

ROOT            = Path("/Users/andredelacruz/Documents/GitHub/cpchess/perception_api")
EMB_TENSOR_PATH = ROOT / "embeddings_master.pt"
EMB_META_PATH   = ROOT / "embeddings_master_meta.csv"

# Optional: a prior snapshot of the CSV to diff against, if available
# e.g. created earlier as a safety copy; if present we can auto-map deletions.
BACKUP_META_CANDIDATES = [
    ROOT / "embeddings_master_meta.backup.csv",
    ROOT / "embeddings_master_meta.prev.csv",
]

def load_backup_meta():
    for p in BACKUP_META_CANDIDATES:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"[info] Using backup CSV to diff: {p.name}")
                return df
            except Exception as e:
                print(f"[warn] Could not read {p}: {e}")
    return None

def backup_files():
    # Make safety copies before changing anything
    if EMB_TENSOR_PATH.exists():
        torch.save(torch.load(EMB_TENSOR_PATH, map_location="cpu"), ROOT / "embeddings_master.pt.bak")
        print(f"[backup] Saved {ROOT/'embeddings_master.pt.bak'}")
    if EMB_META_PATH.exists():
        pd.read_csv(EMB_META_PATH).to_csv(ROOT / "embeddings_master_meta.csv.bak", index=False)
        print(f"[backup] Saved {ROOT/'embeddings_master_meta.csv.bak'}")

def main():
    if not EMB_TENSOR_PATH.exists() or not EMB_META_PATH.exists():
        print("[error] Missing embeddings tensor or CSV.")
        sys.exit(1)

    meta = pd.read_csv(EMB_META_PATH)
    embs = torch.load(EMB_TENSOR_PATH, map_location="cpu")
    n_meta, n_emb = len(meta), embs.shape[0]
    print(f"[info] current: meta={n_meta}, emb={n_emb}")

    if n_meta == n_emb:
        print("[ok] Already in sync. Nothing to do.")
        return

    if n_meta > n_emb:
        print("[error] CSV has more rows than embeddings. "
              "You likely trimmed tensor already or CSV was edited. "
              "Consider rebuilding embeddings from CSV.")
        sys.exit(2)

    # Now we know: n_emb > n_meta  (extra embeddings to drop)
    num_extra = n_emb - n_meta
    print(f"[info] Embeddings have {num_extra} extra row(s) to remove.")

    backup_meta = load_backup_meta()

    # Strategy A: if we have a backup CSV, detect which filepaths were deleted
    if backup_meta is not None:
        # The embeddings order historically followed the CSV append order.
        # We find which rows existed before but no longer exist now.
        if "filepath" not in meta.columns or "filepath" not in backup_meta.columns:
            print("[warn] CSV missing 'filepath' column; cannot auto-diff. Falling back to interactive mode.")
        else:
            prev_paths = backup_meta["filepath"].tolist()
            curr_paths = set(meta["filepath"].tolist())

            deleted_paths = [p for p in prev_paths if p not in curr_paths]
            if not deleted_paths:
                print("[warn] Backup CSV exists but no deleted filepaths found. "
                      "Maybe rows were edited rather than removed. Falling back to interactive mode.")
            else:
                # map deleted rows to indices in the *old* CSV, which correspond to embedding rows
                delete_indices = [i for i, p in enumerate(prev_paths) if p in deleted_paths]
                print(f"[info] Will remove {len(delete_indices)} row(s) from embeddings at indices: {delete_indices}")

                # Safety: ensure counts match expectation
                if len(delete_indices) != num_extra:
                    print(f"[warn] Number of diffs ({len(delete_indices)}) != extra embeddings ({num_extra}). "
                          f"Proceeding anyway, removing the mapped indices.")

                backup_files()
                keep_mask = torch.ones(n_emb, dtype=torch.bool)
                for idx in delete_indices:
                    if 0 <= idx < n_emb:
                        keep_mask[idx] = False
                embs_fixed = embs[keep_mask]
                torch.save(embs_fixed, EMB_TENSOR_PATH)
                print(f"[done] Wrote synced embeddings with shape {embs_fixed.shape}. CSV unchanged.")
                return

    # Strategy B: Interactive (no backup CSV or ambiguous)
    print("\n[interactive] No usable backup to auto-map deletions.")
    print("The embeddings tensor has more rows than the CSV.")
    print("You must choose which embedding row index/indices to drop (0-based).")
    print("Tip: if you remember the approximate capture time, check the CSV timestamps near that period to infer the index.")
    print(f"Enter indices to delete, comma-separated (e.g., '37' or '37, 102'). Need to remove {num_extra} row(s).")
    s = input("Indices to remove: ").strip()
    if not s:
        print("[abort] No indices provided.")
        sys.exit(3)
    try:
        to_remove = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except Exception:
        print("[error] Could not parse indices.")
        sys.exit(4)

    if len(to_remove) != num_extra:
        print(f"[warn] You entered {len(to_remove)} index/indices but {num_extra} extra rows exist.")

    backup_files()
    keep_mask = torch.ones(n_emb, dtype=torch.bool)
    for idx in to_remove:
        if 0 <= idx < n_emb:
            keep_mask[idx] = False
        else:
            print(f"[warn] index {idx} out of range; ignored.")
    embs_fixed = embs[keep_mask]
    torch.save(embs_fixed, EMB_TENSOR_PATH)
    print(f"[done] Wrote synced embeddings with shape {embs_fixed.shape}. CSV unchanged.")

if __name__ == "__main__":
    main()
