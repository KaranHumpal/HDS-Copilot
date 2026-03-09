import os
import re
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
IN_TRAIN_CSV = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_TRAIN_READY.csv"
OUT_DIR      = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"

MODEL = "text-embedding-3-large"   # strong default
BATCH_SIZE = 64                   # 32-128 is usually fine
SLEEP_ON_RETRY_SEC = 2

# If jobcard_text ever gets huge, we lightly truncate by characters.
# Embedding inputs have a max token limit; docs note 8192 tokens for embedding models. :contentReference[oaicite:2]{index=2}
MAX_CHARS = 12000

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_for_embedding(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("\t", " ")
    s = s.replace("±", "+/-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > MAX_CHARS:
        s = s[:MAX_CHARS]
    return s

def embed_texts(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    """
    Returns float32 numpy array shape (len(texts), dim)
    Uses array input API. :contentReference[oaicite:3]{index=3}
    """
    resp = client.embeddings.create(
        model=model,
        input=texts,
        encoding_format="float",
    )
    embs = [d.embedding for d in resp.data]
    return np.array(embs, dtype=np.float32)

def with_retries(fn, max_tries=6):
    last_err = None
    for t in range(max_tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(SLEEP_ON_RETRY_SEC * (t + 1))
    raise last_err

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(IN_TRAIN_CSV)

    if "jobcard_text" not in df.columns:
        raise ValueError("Missing column 'jobcard_text'. Run Step 2 best cleaner first.")

    # Minimal metadata to keep aligned with embeddings
    keep_cols = [c for c in [
        "Company", "Part_Number", "PDF_FileName", "Process_Class",
        "Units_norm", "Material_norm", "Finish_norm",
        "Qty_num", "Unit_Price_num"
    ] if c in df.columns]

    meta = df[keep_cols].copy()
    meta["row_id"] = np.arange(len(df), dtype=np.int64)

    texts = [clean_for_embedding(x) for x in df["jobcard_text"].tolist()]

    # OpenAI client reads OPENAI_API_KEY automatically. :contentReference[oaicite:4]{index=4}
    client = OpenAI()

    all_embs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch_texts = texts[i:i+BATCH_SIZE]

        def _call():
            return embed_texts(client, batch_texts, MODEL)

        embs = with_retries(_call)
        all_embs.append(embs)

    X = np.vstack(all_embs).astype(np.float32)

    # L2 normalize (recommended for cosine similarity search)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms

    # Save artifacts
    emb_path = os.path.join(OUT_DIR, "embeddings_l2.npy")
    meta_path = os.path.join(OUT_DIR, "meta.csv")
    cfg_path  = os.path.join(OUT_DIR, "index_config.json")
    idx_path  = os.path.join(OUT_DIR, "faiss.index")

    np.save(emb_path, Xn)
    meta.to_csv(meta_path, index=False)

    # Build FAISS index (Inner Product on normalized vectors == cosine sim)
    dim = Xn.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(Xn)
    faiss.write_index(index, idx_path)

    cfg = {
        "model": MODEL,
        "batch_size": BATCH_SIZE,
        "vector_dim": dim,
        "count": int(Xn.shape[0]),
        "normalized": True,
        "similarity": "cosine (via inner product on L2-normalized vectors)",
        "files": {
            "embeddings": emb_path,
            "meta": meta_path,
            "faiss_index": idx_path
        }
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\nDONE")
    print("Saved:", emb_path)
    print("Saved:", meta_path)
    print("Saved:", idx_path)
    print("Saved:", cfg_path)
    print("Rows:", len(df), " Dim:", dim)

if __name__ == "__main__":
    main()