import os, glob, time, torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from datetime import datetime

# ---- Config ----
REF_DIR   = "refs"
QUERY_DIR = "queries"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16
POLL_INTERVAL = 5   # seconds between directory checks

# ---- Setup ----
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def load_images(paths):
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
    return imgs

@torch.no_grad()
def embed_images(paths):
    embs = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        batch_imgs = load_images(batch_paths)
        if not batch_imgs:
            continue
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        out = model.get_image_features(**inputs)
        out = F.normalize(out, p=2, dim=-1)
        embs.append(out)
    if not embs:
        return torch.empty(0, model.config.projection_dim, device=device)
    return torch.cat(embs, dim=0)

def run_clip_for_queries(ref_paths, query_paths):
    if not ref_paths or not query_paths:
        return

    print(f"Processing {len(query_paths)} new query(ies)...")
    ref_embs = embed_images(ref_paths)
    query_embs = embed_images(query_paths)

    sims = query_embs @ ref_embs.T
    sims_cpu = sims.detach().cpu().numpy()

    df = pd.DataFrame(
        sims_cpu,
        index=[os.path.basename(p) for p in query_paths],
        columns=[os.path.basename(p) for p in ref_paths]
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"clip_similarities_{timestamp}.csv"
    df.to_csv(out_csv, float_format="%.6f")
    print(f"Saved similarities to {out_csv}")

def main():
    ref_paths = sorted([p for p in glob.glob(os.path.join(REF_DIR, "*")) if os.path.isfile(p)])
    if not ref_paths:
        print(f"No reference images found in {REF_DIR}/")
        return

    seen = set()
    print("Watching for new query images...")

    while True:
        query_paths = sorted([p for p in glob.glob(os.path.join(QUERY_DIR, "*")) if os.path.isfile(p)])
        new_files = [p for p in query_paths if p not in seen]

        if new_files:
            run_clip_for_queries(ref_paths, new_files)
            seen.update(new_files)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
