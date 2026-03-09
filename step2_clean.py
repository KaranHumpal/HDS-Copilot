# clean_step2_best.py  (Python 3.14)
# هدف: أفضل تنظيف لآخر هدف المشروع (Retrieval + Pricing)
# - لا نعدل أي عمود خام (raw) أبداً
# - نضيف أعمدة *_num / *_norm / *_flag
# - نطلع ملفين:
#   1) FULL_CLEAN.csv  (كل الصفوف، بدون حذف)
#   2) TRAIN_READY.csv (فقط الصفوف اللي فيها Qty + Unit_Price صالحة)

import re
import math
import pandas as pd

# ----------------------------
# PATHS (عدّلهم إذا احتجت)
# ----------------------------
IN_CSV = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_500parts_dataset_ONLY_MATCHED.csv"

OUT_FULL  = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_FULL_CLEAN.csv"
OUT_TRAIN = r"C:\Users\khump\OneDrive\Desktop\ALL_COMPANIES_STEP2_TRAIN_READY.csv"

# ----------------------------
# Helpers: numeric parsing
# ----------------------------
_money_re = re.compile(r"^\$?\s*(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*$")

def to_float(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA
    s = s.replace("$", "").replace(",", "")
    try:
        return float(s)
    except:
        return pd.NA

def to_int(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan", "none"}:
        return pd.NA
    if re.fullmatch(r"\d+", s):
        return int(s)
    # accept "2.0"
    try:
        f = float(s)
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
    except:
        pass
    return pd.NA

def safe_log(x):
    if pd.isna(x):
        return pd.NA
    try:
        x = float(x)
        if x <= 0:
            return pd.NA
        return math.log(x)
    except:
        return pd.NA

# ----------------------------
# Helpers: text normalization
# ----------------------------
def norm_units(u):
    if pd.isna(u):
        return pd.NA
    s = str(u).strip().lower()
    if s in {"mm", "millimeter", "millimeters"}:
        return "mm"
    if s in {"in", "inch", "inches", "\""}:
        return "inch"
    return pd.NA

def clean_text_for_embedding(x):
    """Light cleanup only: keep meaning, remove junk whitespace."""
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("±", "+/-")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# Helpers: material normalization (lightweight)
# ----------------------------
def norm_material(m):
    if pd.isna(m):
        return pd.NA
    s = str(m).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("alum.", "aluminum").replace("alum", "aluminum")
    s = s.replace("–", "-").replace("—", "-")

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
    if "titanium" in s or "ti-" in s or "6al4v" in s or "ti 6al-4v" in s:
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
    if pd.isna(f):
        return pd.NA
    s = str(f).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # small mapping; extend later if needed
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

# ----------------------------
# Helpers: tolerance parsing + GD&T flags
# ----------------------------
_gdt_keywords = [
    "true position", "position", "profile", "flatness", "parallelism", "perpendicularity",
    "circularity", "cylindricity", "runout", "concentricity", "symmetry", "datum", "gd&t",
]

def has_any(s, words):
    s = (s or "").lower()
    return any(w in s for w in words)

def extract_tol_min_value(tol_text, units_norm_val):
    """
    Best-effort: find the smallest positive numeric value in tolerance text.
    Converts mm <-> inch ONLY if the tolerance string explicitly contains units (mm or inch).
    Otherwise assumes it's already in the part's Units.
    """
    if pd.isna(tol_text):
        return pd.NA
    s = str(tol_text).strip().lower()
    if s == "" or s in {"nan", "none"}:
        return pd.NA
    s = s.replace("±", "+/-")
    s = re.sub(r"\s+", " ", s)

    nums = re.findall(r"(?<!\w)(?:\d+\.\d+|\d+|\.\d+)", s)
    if not nums:
        return pd.NA

    vals = []
    for n in nums:
        try:
            v = float(n)
            if v > 0:
                vals.append(v)
        except:
            pass
    if not vals:
        return pd.NA

    vmin = min(vals)
    has_mm = "mm" in s
    has_in = ("inch" in s) or re.search(r'(?<!\w)in(?!\w)', s) is not None or ('"' in s)

    if units_norm_val == "inch" and has_mm and not has_in:
        return vmin / 25.4
    if units_norm_val == "mm" and has_in and not has_mm:
        return vmin * 25.4

    return vmin

def tight_tol_flag(tol_min_value, units_norm_val):
    """
    Simple heuristic:
      inch: tight if <= 0.005
      mm:   tight if <= 0.10
    """
    if pd.isna(tol_min_value) or pd.isna(units_norm_val):
        return False
    try:
        v = float(tol_min_value)
    except:
        return False
    if units_norm_val == "inch":
        return v <= 0.005
    if units_norm_val == "mm":
        return v <= 0.10
    return False

# ----------------------------
# jobcard_text builder
# ----------------------------
def build_jobcard_text(row):
    parts = []

    # Pick the best columns if they exist
    def add(label, col):
        if col in row and not pd.isna(row[col]):
            txt = clean_text_for_embedding(row[col])
            if txt:
                parts.append(f"{label}: {txt}")

    add("Part", "Part_Number")
    add("Title", "Title")
    add("Company", "Company")
    add("Units", "Units")
    add("Material", "Material")
    add("Finish", "Finish")
    add("Process_Class", "Process_Class")

    # dims: include raw text + numeric if available
    add("Overall_Dims_Text", "Overall_Dims_Text")
    for c in ["Overall_X", "Overall_Y", "Overall_Z"]:
        add(c, c)

    # tolerance / gdt
    add("Tol_General", "Tol_General")
    add("Tol_Tightest", "Tol_Tightest")
    add("GD_T", "GD_T")

    # notes (these matter a lot for retrieval)
    add("Key_Notes", "Key_Notes")
    add("LLM_Quote_Notes", "LLM_Quote_Notes")
    add("Process_Reason", "Process_Reason")

    return " | ".join(parts)

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(IN_CSV)

# ----------------------------
# Add safe parsed columns (DO NOT overwrite raw)
# ----------------------------
# Labels
if "Qty" not in df.columns:
    df["Qty"] = pd.NA
if "Unit_Price" not in df.columns:
    df["Unit_Price"] = pd.NA

df["Qty_num"] = df["Qty"].apply(to_int)
df["Unit_Price_num"] = df["Unit_Price"].apply(to_float)
df["log_Unit_Price"] = df["Unit_Price_num"].apply(safe_log)

# Core normals
df["Units_norm"] = df["Units"].apply(norm_units) if "Units" in df.columns else pd.NA

for c in ["Overall_X", "Overall_Y", "Overall_Z"]:
    if c in df.columns:
        df[c + "_num"] = df[c].apply(to_float)
    else:
        df[c + "_num"] = pd.NA

df["Material_norm"] = df["Material"].apply(norm_material) if "Material" in df.columns else pd.NA
df["Finish_norm"] = df["Finish"].apply(norm_finish) if "Finish" in df.columns else pd.NA

# Tolerance parsing + flags (raw stays)
tol_text_combo = (
    (df["Tol_General"].fillna("").astype(str) if "Tol_General" in df.columns else "") +
    " " +
    (df["Tol_Tightest"].fillna("").astype(str) if "Tol_Tightest" in df.columns else "") +
    " " +
    (df["GD_T"].fillna("").astype(str) if "GD_T" in df.columns else "")
)

df["Tol_min_value"] = [
    extract_tol_min_value(t, u)
    for t, u in zip(df["Tol_Tightest"] if "Tol_Tightest" in df.columns else [pd.NA]*len(df),
                    df["Units_norm"] if "Units_norm" in df.columns else [pd.NA]*len(df))
]

df["has_gdt"] = [has_any(t, _gdt_keywords) for t in tol_text_combo]
df["has_datums"] = [("datum" in str(t).lower()) or re.search(r"\bdatum\s*[a-z]\b", str(t).lower()) is not None for t in tol_text_combo]
df["has_true_position"] = [("true position" in str(t).lower()) or re.search(r"\bpos(ition)?\b", str(t).lower()) is not None for t in tol_text_combo]
df["has_profile"] = ["profile" in str(t).lower() for t in tol_text_combo]
df["has_runout"] = ["runout" in str(t).lower() for t in tol_text_combo]
df["has_cmm"] = [("cmm" in str(t).lower()) or ("inspection" in str(t).lower()) for t in tol_text_combo]
df["tight_tol_flag"] = [tight_tol_flag(v, u) for v, u in zip(df["Tol_min_value"], df["Units_norm"])]

# Build jobcard_text (for embeddings/retrieval)
df["jobcard_text"] = df.apply(build_jobcard_text, axis=1)

# ----------------------------
# Save FULL (no row drops)
# ----------------------------
df.to_csv(OUT_FULL, index=False)

# ----------------------------
# TRAIN_READY subset (purposeful filtering)
# ----------------------------
train = df.copy()

# Must have Part_Number to match + train
if "Part_Number" in train.columns:
    train = train[train["Part_Number"].notna() & (train["Part_Number"].astype(str).str.strip() != "")]
else:
    train = train.iloc[0:0]  # empty if no part number column

train = train[
    train["Qty_num"].notna() & (train["Qty_num"] > 0) &
    train["Unit_Price_num"].notna() & (train["Unit_Price_num"] > 0)
].copy()

# Optional mild outlier trim (top 0.5%) ONLY if enough samples
if len(train) >= 60:
    p995 = train["Unit_Price_num"].quantile(0.995)
    train["price_outlier"] = train["Unit_Price_num"] > p995
    train = train[~train["price_outlier"]].copy()
else:
    train["price_outlier"] = False

train.to_csv(OUT_TRAIN, index=False)

# ----------------------------
# Report
# ----------------------------
def pct(n, d):
    return 0 if d == 0 else round(100.0 * n / d, 2)

total = len(df)
qty_filled = int(df["Qty_num"].notna().sum())
price_filled = int(df["Unit_Price_num"].notna().sum())
tol_parsed = int(df["Tol_min_value"].notna().sum())
units_ok = int(df["Units_norm"].notna().sum())

print("INPUT:", IN_CSV)
print("FULL_CLEAN saved:", OUT_FULL)
print("TRAIN_READY saved:", OUT_TRAIN)
print("Rows total:", total)
print("Rows train-ready:", len(train))
print(f"Qty parsed: {qty_filled}/{total} ({pct(qty_filled,total)}%)")
print(f"Unit_Price parsed: {price_filled}/{total} ({pct(price_filled,total)}%)")
print(f"Units normalized: {units_ok}/{total} ({pct(units_ok,total)}%)")
print(f"Tol_min_value parsed: {tol_parsed}/{total} ({pct(tol_parsed,total)}%)")
print("Done.")