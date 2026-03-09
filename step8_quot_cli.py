import os
import re
import json
import math
import numpy as np
import pandas as pd
import faiss
import joblib
from openai import OpenAI
from step9 import generate_quote_package

# ----------------------------
# PATHS
# ----------------------------
INDEX_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3"
MODEL_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_model_step6"

# Models
EMBED_MODEL = "text-embedding-3-large"
VISION_MODEL = "gpt-4o"

TOPK = 10

client = OpenAI()

# ----------------------------
# helpers
# ----------------------------



def qty_multiplier(qty: int) -> float:
    if qty is None:
        return 1.0
    if qty <= 3:
        return 1.25
    elif qty <= 9:
        return 1.10
    elif qty <= 15:
        return 1.00
    elif qty <= 25:
        return 0.92
    elif qty <= 50:
        return 0.85
    else:
        return 0.78
    
def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = s.replace("±", "+/-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_float(x):
    try:
        s = str(x).strip().replace("$", "").replace(",", "")
        if s == "" or s.lower() in {"none", "nan"}:
            return None
        return float(s)
    except:
        return None


def safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in {"none", "nan"}:
            return None
        if re.fullmatch(r"\d+", s):
            return int(s)
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return None
    except:
        return None


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


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


# ----------------------------
# normalization helpers
# ----------------------------
def norm_units(u):
    s = clean_text(u).lower()
    if s in {"mm", "millimeter", "millimeters"}:
        return "mm"
    if s in {"in", "inch", "inches", "\""}:
        return "inch"
    return "unknown"


def norm_material(m):
    s = clean_text(m).lower()
    if not s:
        return "unknown"
    s = re.sub(r"\s+", " ", s)
    s = s.replace("alum.", "aluminum").replace("alum", "aluminum")

    if "6061" in s:
        return "aluminum 6061-t6" if "t6" in s else "aluminum 6061"
    if "7075" in s:
        return "aluminum 7075-t6" if "t6" in s else "aluminum 7075"
    if "5052" in s:
        return "aluminum 5052"
    if "17-4" in s or "17 4" in s:
        return "stainless 17-4"
    if "316" in s:
        return "stainless 316"
    if "304" in s:
        return "stainless 304"
    if "stainless" in s:
        return "stainless"
    if "titanium" in s or "ti-6al-4v" in s or "6al4v" in s:
        return "titanium"
    if "delrin" in s or "acetal" in s:
        return "delrin (acetal)"
    if "peek" in s:
        return "peek"
    if "brass" in s:
        return "brass"
    if "copper" in s:
        return "copper"
    if "steel" in s:
        return "steel"

    return s


def norm_finish(f):
    s = clean_text(f).lower()
    if not s:
        return "unknown"
    s = re.sub(r"\s+", " ", s)

    if "anod" in s:
        if "type ii" in s or "type 2" in s:
            return "anodize type ii"
        if "type iii" in s or "type 3" in s or "hard" in s:
            return "anodize type iii"
        return "anodize"
    if "passiv" in s:
        return "passivate"
    if "chem film" in s or "chromate" in s or "alodine" in s:
        return "chem film"
    if "nickel" in s and "plate" in s:
        return "nickel plate"
    if "zinc" in s and "plate" in s:
        return "zinc plate"
    if "powder" in s and "coat" in s:
        return "powder coat"
    if "paint" in s:
        return "paint"
    if "bead" in s and "blast" in s:
        return "bead blast"

    return s


def extract_tol_min_value(tol_text, units_norm_val):
    if not tol_text:
        return None

    s = str(tol_text).lower().replace("±", "+/-")
    nums = re.findall(r"(?<!\w)(?:\d+\.\d+|\d+|\.\d+)", s)

    vals = []
    for n in nums:
        try:
            v = float(n)
            if v > 0:
                vals.append(v)
        except:
            pass

    if not vals:
        return None

    vmin = min(vals)
    has_mm = "mm" in s
    has_in = ("inch" in s) or (re.search(r'(?<!\w)in(?!\w)', s) is not None) or ('"' in s)

    if units_norm_val == "inch" and has_mm and not has_in:
        return vmin / 25.4
    if units_norm_val == "mm" and has_in and not has_mm:
        return vmin * 25.4

    return vmin


def gdt_flags(text: str) -> dict:
    t = clean_text(text).lower()
    return {
        "has_gdt": any(k in t for k in [
            "true position", "position", "profile", "flatness", "parallelism",
            "perpendicularity", "circularity", "cylindricity", "runout",
            "concentricity", "symmetry", "datum", "gd&t"
        ]),
        "has_datums": ("datum" in t) or (re.search(r"\bdatum\s*[a-z]\b", t) is not None),
        "has_true_position": ("true position" in t) or (re.search(r"\bpos(ition)?\b", t) is not None),
        "has_profile": ("profile" in t),
        "has_runout": ("runout" in t),
        "has_cmm": ("cmm" in t) or ("inspection" in t),
    }


def build_jobcard_text(sample: dict) -> str:
    fields = [
        ("Company", "Company"),
        ("Part", "Part_Number"),
        ("Title", "Title"),
        ("Units", "Units_norm"),
        ("Material", "Material_norm"),
        ("Finish", "Finish_norm"),
        ("Process_Class", "Process_Class"),
        ("Dims", "Overall_Dims_Text"),
        ("Tol_General", "Tol_General"),
        ("Tol_Tightest", "Tol_Tightest"),
        ("GD_T", "GD_T"),
        ("Notes", "Key_Notes"),
        ("Quote_Notes", "LLM_Quote_Notes"),
        ("Process_Reason", "Process_Reason"),
    ]
    parts = []
    for label, key in fields:
        txt = clean_text(sample.get(key, ""))
        if txt:
            parts.append(f"{label}: {txt}")
    return " | ".join(parts)


def finalize_sample(sample: dict) -> dict:
    """
    Canonicalize all incoming samples so manual mode and AI parser mode
    produce the exact same downstream schema.
    """
    s = dict(sample) if sample else {}

    # Required text-ish fields
    defaults = {
        "Company": "UNKNOWN",
        "Part_Number": "",
        "Title": "",
        "Units_norm": "unknown",
        "Material_norm": "unknown",
        "Finish_norm": "unknown",
        "Process_Class": "unknown",
        "Overall_Dims_Text": "",
        "Tol_General": "",
        "Tol_Tightest": "",
        "GD_T": "",
        "Key_Notes": "",
        "LLM_Quote_Notes": "",
        "Process_Reason": "",
    }
    for k, v in defaults.items():
        if k not in s or s[k] is None:
            s[k] = v

    # Clean base text
    for k in [
        "Company", "Part_Number", "Title", "Process_Class", "Overall_Dims_Text",
        "Tol_General", "Tol_Tightest", "GD_T", "Key_Notes",
        "LLM_Quote_Notes", "Process_Reason"
    ]:
        s[k] = clean_text(s.get(k, ""))

    s["Company"] = s["Company"].upper() if s["Company"] else "UNKNOWN"

    # Normalize categorical fields
    s["Units_norm"] = norm_units(s.get("Units_norm"))
    s["Material_norm"] = norm_material(s.get("Material_norm"))
    s["Finish_norm"] = norm_finish(s.get("Finish_norm"))

    # Numeric fields
    s["Qty_num"] = safe_int(s.get("Qty_num"))
    if s["Qty_num"] is None or s["Qty_num"] <= 0:
        s["Qty_num"] = 1

    s["Overall_X_num"] = safe_float(s.get("Overall_X_num"))
    s["Overall_Y_num"] = safe_float(s.get("Overall_Y_num"))
    s["Overall_Z_num"] = safe_float(s.get("Overall_Z_num"))

    # Tolerance-derived features
    tol_source = s.get("Tol_Tightest") or s.get("Tol_General")
    s["Tol_min_value"] = extract_tol_min_value(tol_source, s["Units_norm"])

    tol_combo = " ".join([
        s.get("Tol_General", ""),
        s.get("Tol_Tightest", ""),
        s.get("GD_T", ""),
        s.get("Key_Notes", ""),
    ]).strip()

    flags = gdt_flags(tol_combo)
    s.update(flags)

    tol_min = s.get("Tol_min_value")
    tight_tol = False
    if tol_min is not None:
        if s["Units_norm"] == "inch":
            tight_tol = tol_min <= 0.005
        elif s["Units_norm"] == "mm":
            tight_tol = tol_min <= 0.10
    s["tight_tol_flag"] = tight_tol

    # Build final embedding text
    s["jobcard_text"] = build_jobcard_text(s)

    return s


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


def make_feature_vector(sample: dict, cfg: dict, nb_stat: dict) -> pd.DataFrame:
    X = pd.DataFrame([{c: 0.0 for c in cfg["feature_columns"]}])

    for c in cfg["num_cols"]:
        if c in sample and sample[c] is not None:
            try:
                X.at[0, c] = float(sample[c])
            except:
                X.at[0, c] = 0.0

    for c in cfg["flag_cols"]:
        X.at[0, c] = float(int(bool(sample.get(c, False))))

    for k, v in nb_stat.items():
        if k in X.columns:
            X.at[0, k] = float(v)

    for cat in cfg["cat_cols"]:
        val = sample.get(cat, "unknown")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = "unknown"
        col = f"{cat}_{str(val)}"
        if col in X.columns:
            X.at[0, col] = 1.0

    return X.fillna(0.0)


# ----------------------------
# MODE 1: Manual/local jobcard entry (NO PDF sent)
# ----------------------------
def manual_jobcard():
    print("\nManual mode: enter what you know. Leave blank if unknown.\n")

    raw = {
        "Company": input("Company (): ").strip().upper() or "UNKNOWN",
        "Part_Number": input("Part_Number: ").strip(),
        "Title": input("Title / short description: ").strip(),
        "Units_norm": input("Units (mm/inch): ").strip(),
        "Material_norm": input("Material (free text): ").strip(),
        "Finish_norm": input("Finish (free text): ").strip(),
        "Process_Class": input("Process_Class (mill_only/lathe_only/lathe_and_mill/unknown): ").strip() or "unknown",
        "Overall_Dims_Text": input("Overall dims text (e.g., 4 x 2 x 1): ").strip(),
        "Tol_General": input("Tol_General (paste text): ").strip(),
        "Tol_Tightest": input("Tol_Tightest (paste text): ").strip(),
        "GD_T": input("GD_T (paste text): ").strip(),
        "Qty_num": input("Qty (integer): ").strip(),
        "Key_Notes": input("Notes (ops, inspection, risks): ").strip(),
        "Overall_X_num": input("Overall_X_num (optional numeric): ").strip(),
        "Overall_Y_num": input("Overall_Y_num (optional numeric): ").strip(),
        "Overall_Z_num": input("Overall_Z_num (optional numeric): ").strip(),
        "LLM_Quote_Notes": "",
        "Process_Reason": "",
    }

    return finalize_sample(raw)


# ----------------------------
# MODE 2: OpenAI PDF -> Jobcard extraction (PDF IS SENT)
# ----------------------------
def ai_parser_jobcard():
    print("\nAI Parser mode (OpenAI).")
    print("WARNING: This mode uploads the PDF to OpenAI for extraction.")

    ok = input("Type YES to continue: ").strip()
    if ok != "YES":
        print("Cancelled. Switching to Manual mode.")
        return manual_jobcard()

    pdf_path = input("PDF path (full path to drawing PDF): ").strip().strip('"')
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    company = input("Company (ARGO/NOVA/ZEE/JOBY): ").strip().upper() or "UNKNOWN"
    qty = safe_int(input("Qty (integer): ").strip()) or 1

    with open(pdf_path, "rb") as fobj:
        uploaded = client.files.create(
            file=fobj,
            purpose="user_data"
        )

    prompt = f"""
You are extracting a CNC job card from a manufacturing drawing PDF.

Return ONLY valid JSON. No markdown. No commentary.

Use this exact schema:

{{
  "Company": "{company}",
  "Qty_num": {qty},
  "Part_Number": "",
  "Title": "",
  "Units_norm": "mm" | "inch" | "unknown",
  "Material_norm": "",
  "Finish_norm": "",
  "Process_Class": "mill_only" | "lathe_only" | "lathe_and_mill" | "unknown",
  "Overall_Dims_Text": "",
  "Overall_X_num": null,
  "Overall_Y_num": null,
  "Overall_Z_num": null,
  "Tol_General": "",
  "Tol_Tightest": "",
  "GD_T": "",
  "Key_Notes": "",
  "LLM_Quote_Notes": "",
  "Process_Reason": ""
}}

Rules:
- If unknown, use null for numeric fields and "" for strings.
- Units_norm must be exactly mm, inch, or unknown.
- Material_norm and Finish_norm should be short normalized strings if possible.
- Extract tolerances and GD&T as faithfully as possible from the drawing.
- Do not invent dimensions or notes.
""".strip()

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    text = text.strip()

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Model did not return JSON. Raw output:\n" + text)
        raw = json.loads(m.group(0))

    raw["Company"] = company
    raw["Qty_num"] = qty

    return finalize_sample(raw)


# ----------------------------
# inference core
# ----------------------------
def predict(sample: dict):
    cfg_path = os.path.join(MODEL_DIR, "model_config.json")
    model_path = os.path.join(MODEL_DIR, "hybrid_ridge.joblib")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = joblib.load(model_path)
    meta = pd.read_csv(os.path.join(INDEX_DIR, "meta.csv"))
    index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

    text = clean_text(sample.get("jobcard_text")) or build_jobcard_text(sample)

    emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
        encoding_format="float"
    )
    q = np.array([emb.data[0].embedding], dtype=np.float32)
    q = l2_normalize(q)

    scores, idxs = index.search(q, TOPK)
    idxs = idxs[0].tolist()
    scores = scores[0].tolist()

    comps = meta.iloc[idxs].copy()
    comps["score"] = scores

    comps = (
        comps.sort_values("score", ascending=False)
        .drop_duplicates(subset=["Part_Number"], keep="first")
        .head(TOPK)
        .copy()
    )

    prices = comps["Unit_Price_num"].astype(float).values
    nb_stat = neighbor_stats(prices)
    top1_sim = float(comps["score"].iloc[0])

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

    qty = sample.get("Qty_num", 15)
    mult = qty_multiplier(qty)
    pred = pred_base * mult

    conf = confidence_heuristic(top1_sim, nb_stat["nb_mean"], nb_stat["nb_std"])

    return {
        "pred_unit_price": pred,
        "confidence_0_1": conf,
        "top1_similarity": top1_sim,
        "neighbor_stats": nb_stat,
        "pred_base_unit_price_ref15": pred_base,
        "pred_base_unit_price_ref15": pred_base,
        "qty_multiplier": mult,
        "qty_multiplier": mult,
        "jobcard": sample,
        "top_comps": comps[
            ["Company", "Part_Number", "Qty_num", "Unit_Price_num", "score"]
        ].to_dict(orient="records"),
    }


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    print("\nQUOTE CLI")
    print("1) Manual / Local jobcard entry (NO PDF sent)")
    print("2) OpenAI PDF parser mode (PDF IS SENT)\n")

    mode = input("Choose mode [1/2]: ").strip()

    if mode == "2":
        sample = ai_parser_jobcard()
    else:
        sample = manual_jobcard()

    print("\n--- FINALIZED SAMPLE ---")
    print(json.dumps(sample, indent=2))

    result = predict(sample)

    quote_pkg = generate_quote_package(sample, result)

    print("\n--- LLM QUOTE PACKAGE (JSON) ---")
    print(json.dumps(quote_pkg, indent=2))

    print("\n--- QUOTE RESULT (JSON) ---")
    print(json.dumps(result, indent=2))