import os
import json
import math
import numpy as np
import pandas as pd
import faiss
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

INDEX_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"
TRAIN_CSV = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_TRAIN_READY.csv"
OUT_DIR   = r"C:\Users\khump\OneDrive\Desktop\quote_model_step6"

NEIGH_K = 10
ALPHA = 1.0
SEED = 42

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_log(x):
    if x is None or pd.isna(x):
        return np.nan
    x = float(x)
    if x <= 0:
        return np.nan
    return math.log(x)

def build_neighbor_stats(y, neigh_idx):
    prices = y[neigh_idx]
    return pd.DataFrame({
        "nb_top1": prices[:, 0],
        "nb_mean": prices.mean(axis=1),
        "nb_median": np.median(prices, axis=1),
        "nb_min": prices.min(axis=1),
        "nb_max": prices.max(axis=1),
        "nb_std": prices.std(axis=1),
    })

def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(TRAIN_CSV)
    y = df["Unit_Price_num"].astype(float).values
    y_log = np.log(y)

    # load embeddings + faiss
    Xemb = np.load(os.path.join(INDEX_DIR, "embeddings_l2.npy")).astype(np.float32)
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    # neighbors exclude self
    D, I = index.search(Xemb, NEIGH_K + 1)
    neigh = []
    for i in range(len(df)):
        lst = [j for j in I[i].tolist() if j != i]
        neigh.append(lst[:NEIGH_K])
    neigh = np.array(neigh, dtype=int)

    nb = build_neighbor_stats(y, neigh)

    # numeric features
    num_cols = []
    for c in ["Overall_X_num", "Overall_Y_num", "Overall_Z_num", "Tol_min_value"]:
        if c in df.columns:
            num_cols.append(c)

    flag_cols = [c for c in ["has_gdt", "has_datums", "has_true_position", "has_profile",
                             "has_runout", "has_cmm", "tight_tol_flag"] if c in df.columns]

    cat_cols = [c for c in ["Company", "Process_Class", "Units_norm", "Material_norm", "Finish_norm"] if c in df.columns]

    Xdf = pd.concat(
        [
            df[num_cols].copy(),
            df[flag_cols].astype(int).copy() if flag_cols else pd.DataFrame(index=df.index),
            nb
        ],
        axis=1
    )

    # fill numeric NaNs with medians
    for c in Xdf.columns:
        med = Xdf[c].median()
        Xdf[c] = Xdf[c].fillna(med)

    # one-hot categoricals
    if cat_cols:
        Xcat = pd.get_dummies(df[cat_cols].fillna("unknown").astype(str), prefix=cat_cols, drop_first=False)
        Xdf = pd.concat([Xdf, Xcat], axis=1)

    feature_columns = list(Xdf.columns)
    X = Xdf.astype(float).values

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=ALPHA, random_state=SEED))
    ])
    model.fit(X, y_log)

    joblib_path = os.path.join(OUT_DIR, "hybrid_ridge.joblib")
    joblib.dump(model, joblib_path)

    cfg = {
        "alpha": ALPHA,
        "neighbor_k": NEIGH_K,
        "feature_columns": feature_columns,
        "num_cols": num_cols,
        "flag_cols": flag_cols,
        "cat_cols": cat_cols,
        "index_dir": INDEX_DIR,
        "train_csv": TRAIN_CSV,
        "note": "Model predicts log(Unit_Price_num); output is exp(pred)."
    }
    cfg_path = os.path.join(OUT_DIR, "model_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Saved model:", joblib_path)
    print("Saved config:", cfg_path)
    print("Features:", len(feature_columns))

if __name__ == "__main__":
    main()