# step7_predict_quote.py
# Python 3.14
# Robust inference:
# - can pick a sample by exact PN, substring PN, or random
# - never hard-crashes; prints candidates instead
# - predicts unit price using Step6 model + Step3 FAISS comps

import os, re, json, math, random
import numpy as np
import pandas as pd
import faiss
import joblib
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
INDEX_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"
MODEL_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_model_step6"
FULL_CLEAN_CSV = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_FULL_CLEAN.csv"

EMBED_MODEL = "text-embedding-3-large"
TOPK = 10

# ----------------------------
# helpers
# ----------------------------
def _norm_pn(x: str) -> str:
    x = "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)
    x = x.strip().upper()
    x = x.replace("–", "-").replace("—", "-")
    x = re.sub(r"\s+", "", x)
    return x

def clean_text(s):
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).replace("\n", " ").replace("\t", " ")
    s = s.replace("±", "+/-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def safe_log(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return np.nan
        x = float(x)
        if x <= 0:
            return np.nan
        return math.log(x)
    except:
        return np.nan

def build_jobcard_text(sample: dict) -> str:
    fields = [
        ("Part", "Part_Number"),
        ("Title", "Title"),
        ("Company", "Company"),
        ("Units", "Units"),
        ("Material", "Material"),
        ("Finish", "Finish"),
        ("Process_Class", "Process_Class"),
        ("Overall_Dims_Text", "Overall_Dims_Text"),
        ("Overall_X", "Overall_X"),
        ("Overall_Y", "Overall_Y"),
        ("Overall_Z", "Overall_Z"),
        ("Tol_General", "Tol_General"),
        ("Tol_Tightest", "Tol_Tightest"),
        ("GD_T", "GD_T"),
        ("Key_Notes", "Key_Notes"),
        ("LLM_Quote_Notes", "LLM_Quote_Notes"),
        ("Process_Reason", "Process_Reason"),
    ]
    parts = []
    for label, key in fields:
        txt = clean_text(sample.get(key))
        if txt:
            parts.append(f"{label}: {txt}")
    return " | ".join(parts)

def neighbor_stats(prices: np.ndarray):
    return {
        "nb_top1": float(prices[0]),
        "nb_mean": float(prices.mean()),
        "nb_median": float(np.median(prices)),
        "nb_min": float(prices.min()),
        "nb_max": float(prices.max()),
        "nb_std": float(prices.std()),
    }

def confidence_heuristic(top1_sim: float, nb_mean: float, nb_std: float) -> float:
    cv = nb_std / max(nb_mean, 1e-9)
    conf = 0.6 * top1_sim + 0.4 * (1.0 - cv)
    return float(max(0.0, min(1.0, conf)))

# ----------------------------
# feature builder (must match Step6 config)
# ----------------------------
def make_feature_vector(sample: dict, cfg: dict, nb_stat: dict) -> pd.DataFrame:
    X = pd.DataFrame([{c: 0.0 for c in cfg["feature_columns"]}])

    # numeric
    for c in cfg["num_cols"]:
        if c in sample:
            try:
                X.at[0, c] = float(sample.get(c))
            except:
                X.at[0, c] = 0.0

    # flags
    for c in cfg["flag_cols"]:
        X.at[0, c] = float(int(bool(sample.get(c, False))))

    # neighbor stats
    for k, v in nb_stat.items():
        if k in X.columns:
            X.at[0, k] = float(v)

    # one-hot cats (only if the exact dummy column exists)
    for cat in cfg["cat_cols"]:
        val = sample.get(cat, "unknown")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = "unknown"
        col = f"{cat}_{str(val)}"
        if col in X.columns:
            X.at[0, col] = 1.0

    return X.fillna(0.0)

# ----------------------------
# sample selection (robust)
# ----------------------------
def load_full_clean_df():
    df = pd.read_csv(FULL_CLEAN_CSV)
    if "Part_Number" not in df.columns:
        raise ValueError("FULL_CLEAN CSV missing Part_Number")
    df["_pn_norm"] = df["Part_Number"].fillna("").astype(str).apply(_norm_pn)
    return df

def pick_sample(df: pd.DataFrame, pn_exact: str | None = None, pn_contains: str | None = None, random_pick: bool = False):
    if pn_exact:
        t = _norm_pn(pn_exact)
        m = df[df["_pn_norm"] == t]
        if len(m) > 0:
            return m.iloc[0].to_dict(), f"exact match: {m.iloc[0]['Part_Number']}"
        print(f"No exact match for: {pn_exact}")

    if pn_contains:
        t = _norm_pn(pn_contains)
        m = df[df["_pn_norm"].str.contains(re.escape(t), na=False)]
        if len(m) > 0:
            print("Top candidates (contains):")
            print(m[["Company", "Part_Number"]].head(15).to_string(index=False))
            return m.iloc[0].to_dict(), f"contains match: {m.iloc[0]['Part_Number']}"
        print(f"No contains match for: {pn_contains}")

    if random_pick:
        i = random.randint(0, len(df) - 1)
        return df.iloc[i].to_dict(), f"random pick: {df.iloc[i]['Part_Number']}"

    # default: first row
    return df.iloc[0].to_dict(), f"default first row: {df.iloc[0]['Part_Number']}"

# ----------------------------
# predict
# ----------------------------
def predict_quote(sample: dict, topk: int = TOPK) -> dict:
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    model_path = os.path.join(MODEL_DIR, "hybrid_ridge.joblib")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = joblib.load(model_path)

    meta = pd.read_csv(os.path.join(INDEX_DIR, "meta.csv"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    jobcard_text = clean_text(sample.get("jobcard_text")) or build_jobcard_text(sample)
    jobcard_text = clean_text(jobcard_text)

    client = OpenAI()
    resp = client.embeddings.create(model=EMBED_MODEL, input=[jobcard_text], encoding_format="float")
    q = np.array([resp.data[0].embedding], dtype=np.float32)
    q = l2_normalize(q)

    scores, idxs = index.search(q, topk)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    comps = meta.iloc[idxs].copy()
    comps["score"] = scores

    picked_pn = str(sample.get("Part_Number", "")).strip()
    if picked_pn:
        comps = comps[comps["Part_Number"].astype(str).str.strip() != picked_pn].copy()
        
    comps = comps.sort_values("score", ascending=False)
    comps = comps.drop_duplicates(subset=["Part_Number"], keep="first").head(TOPK).copy()

    prices = comps["Unit_Price_num"].astype(float).values
    nb_stat = neighbor_stats(prices)

    X = make_feature_vector(sample, cfg, nb_stat)
    pred_log = float(model.predict(X.values)[0])
    pred_base = math.exp(pred_log)

    qty = sample.get("Qty_num", 15)
    if qty is None:
        qty = 15

    def qty_multiplier(q):
        if q <= 3:
            return 1.25
        elif q <= 9:
            return 1.10
        elif q <= 15:
            return 1.00
        elif q <= 25:
            return 0.92
        elif q <= 50:
            return 0.85
        else:
            return 0.78

    mult = qty_multiplier(int(qty))
    pred = pred_base * mult
    conf = confidence_heuristic(float(comps["score"].iloc[0]), nb_stat["nb_mean"], nb_stat["nb_std"])

    return {
        "pred_unit_price": pred,
        "confidence_0_1": conf,
        "top1_similarity": float(comps["score"].iloc[0]),
        "pred_base_unit_price_ref15": pred_base,
        "qty_multiplier": mult,
        "neighbor_stats": nb_stat,
        "picked_part_number": sample.get("Part_Number", None),
        "top_comps": comps[["Company", "Part_Number", "Qty_num", "Unit_Price_num", "score"]].to_dict(orient="records"),
    }

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    df = load_full_clean_df()

    # OPTION 1: set one of these:
    PN_EXACT = None          # e.g. "660-00142"
    PN_CONTAINS = None       # e.g. "0312-03"
    RANDOM_PICK = True       # easiest: True

    sample, how = pick_sample(df, pn_exact=PN_EXACT, pn_contains=PN_CONTAINS, random_pick=RANDOM_PICK)
    print("Picked:", how)

    result = predict_quote(sample, topk=TOPK)
    print(json.dumps(result, indent=2))