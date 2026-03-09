import os
import math
import numpy as np
import pandas as pd
import faiss

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INDEX_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"
TRAIN_CSV = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_TRAIN_READY.csv"

# neighbor K used to make stats (use best-ish small K; we can tune later)
NEIGH_K = 10

# Ridge strength candidates
ALPHAS = [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]

N_SPLITS = 5
SEED = 42

def safe_log(x):
    if x is None or pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 0:
        return np.nan
    return math.log(x)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-9
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)

def build_neighbor_stats(y, neigh_idx):
    """neigh_idx: (n, K) indices"""
    n, k = neigh_idx.shape
    stats = {}
    prices = y[neigh_idx]  # (n, K)
    stats["nb_top1"] = prices[:, 0]
    stats["nb_mean"] = prices.mean(axis=1)
    stats["nb_median"] = np.median(prices, axis=1)
    stats["nb_min"] = prices.min(axis=1)
    stats["nb_max"] = prices.max(axis=1)
    stats["nb_std"] = prices.std(axis=1)
    return pd.DataFrame(stats)

def main():
    df = pd.read_csv(TRAIN_CSV)

    # Labels
    y = df["Unit_Price_num"].astype(float).values
    y_log = np.log(y)

    # Load embeddings + FAISS for neighbor lookup
    Xemb = np.load(os.path.join(INDEX_DIR, "embeddings_l2.npy")).astype(np.float32)
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    # Precompute neighbors for each row (exclude itself)
    D, I = index.search(Xemb, NEIGH_K + 1)
    neigh = []
    for i in range(len(df)):
        lst = [j for j in I[i].tolist() if j != i]
        neigh.append(lst[:NEIGH_K])
    neigh = np.array(neigh, dtype=int)

    # Neighbor price stats
    nb = build_neighbor_stats(y, neigh)

    # Base numeric features (keep NA -> fill later)
    num_cols = []
    for c in ["Overall_X_num", "Overall_Y_num", "Overall_Z_num", "Tol_min_value"]:
        if c in df.columns:
            num_cols.append(c)
    # log qty helps a ton
    df["log_qty"] = df["Qty_num"].apply(safe_log)
    num_cols = ["log_qty"] + num_cols

    # Flags
    flag_cols = [c for c in ["has_gdt", "has_datums", "has_true_position", "has_profile", "has_runout", "has_cmm", "tight_tol_flag"]
                 if c in df.columns]

    # Categoricals
    cat_cols = [c for c in ["Company", "Process_Class", "Units_norm", "Material_norm", "Finish_norm"] if c in df.columns]

    # Assemble feature frame
    X = pd.concat(
        [
            df[num_cols].copy(),
            df[flag_cols].astype(int).copy() if flag_cols else pd.DataFrame(index=df.index),
            nb
        ],
        axis=1
    )

    # Fill numeric NaNs with column median (safe)
    for c in X.columns:
        if X[c].dtype != object:
            med = X[c].median()
            X[c] = X[c].fillna(med)

    # One-hot encode categoricals and append
    if cat_cols:
        X_cat = pd.get_dummies(df[cat_cols].fillna("unknown").astype(str), prefix=cat_cols, drop_first=False)
        X = pd.concat([X, X_cat], axis=1)

    X = X.astype(float).values

    # Cross-validate + pick alpha
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    best_alpha = None
    best_mape = 1e18
    summary = []

    for a in ALPHAS:
        preds_log = np.zeros(len(df), dtype=float)

        for tr, te in kf.split(X):
            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=a, random_state=SEED))
            ])
            model.fit(X[tr], y_log[tr])
            preds_log[te] = model.predict(X[te])

        preds = np.exp(preds_log)
        row = {
            "alpha": a,
            "MAE": mae(y, preds),
            "MAPE_%": mape(y, preds),
        }
        summary.append(row)

        if row["MAPE_%"] < best_mape:
            best_mape = row["MAPE_%"]
            best_alpha = a

    out = pd.DataFrame(summary).sort_values("MAPE_%")
    print("\nHYBRID RIDGE CV RESULTS")
    print(out.to_string(index=False))
    print(f"\nBest alpha: {best_alpha}  (MAPE={best_mape:.2f}%)")

    out_path = os.path.join(INDEX_DIR, "step5_hybrid_ridge_cv.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()