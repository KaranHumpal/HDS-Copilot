import os
import numpy as np
import pandas as pd
import faiss

INDEX_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"
TOPK_LIST = [1, 3, 5, 10, 20, 30, 50]

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

def main():
    meta_path = os.path.join(INDEX_DIR, "meta.csv")
    idx_path  = os.path.join(INDEX_DIR, "faiss.index")

    meta = pd.read_csv(meta_path)
    index = faiss.read_index(idx_path)

    # Load embeddings so we can do "leave-one-out" queries exactly
    X = np.load(os.path.join(INDEX_DIR, "embeddings_l2.npy")).astype(np.float32)

    if "Unit_Price_num" not in meta.columns:
        raise ValueError("meta.csv missing Unit_Price_num")

    y = meta["Unit_Price_num"].astype(float).values
    n = len(meta)

    # Precompute neighbor lists once at max K + 1 (because first hit is itself)
    max_k = max(TOPK_LIST)
    D, I = index.search(X, max_k + 1)  # includes itself at rank 1 typically

    results = []

    for k in TOPK_LIST:
        preds_med = np.zeros(n, dtype=float)
        preds_mean = np.zeros(n, dtype=float)

        for i in range(n):
            neigh = I[i].tolist()

            # drop self if present
            neigh = [j for j in neigh if j != i]

            # take top-k after removal
            neigh_k = neigh[:k]

            prices = y[neigh_k]
            preds_med[i] = float(np.median(prices))
            preds_mean[i] = float(np.mean(prices))

        r = {
            "K": k,
            "MAE_median": mae(y, preds_med),
            "MAPE_median_%": mape(y, preds_med),
            "MAE_mean": mae(y, preds_mean),
            "MAPE_mean_%": mape(y, preds_mean),
        }
        results.append(r)

    out = pd.DataFrame(results).sort_values("K")
    print("\nKNN BASELINE RESULTS (Unit_Price)")
    print(out.to_string(index=False))

    out_path = os.path.join(INDEX_DIR, "step4_knn_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()