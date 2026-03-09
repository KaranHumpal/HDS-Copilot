import os
import json
import base64
import random
import time
from typing import List, Dict, Any, Optional

import fitz  # pip install pymupdf
import pandas as pd  # pip install pandas
from openai import OpenAI

# ============================================================
# 500 RANDOM PARTS (4 companies) WITH READABILITY GATE + RESUME
# + DRAWING_NUMBER + TITLE + REV (from drawing, not filename)
# + TITLE-BLOCK / NOTES CROPS (better accuracy, fewer tokens)
# + SAFE CSV WRITES
# ============================================================

BASE_DIR = r"C:\Users\khump\OneDrive\Desktop"
FOLDERS = {
    "ARGO": os.path.join(BASE_DIR, "ARGO_PDF"),
    "ZEE":  os.path.join(BASE_DIR, "ZEE_PDF"),
    "NOVA": os.path.join(BASE_DIR, "NOVA_PDF"),
    "JOBY": os.path.join(BASE_DIR, "JOBY_PDF"),
}

MODEL = "gpt-4.1"

# DPI tuning:
# - PREFLIGHT_DPI: keep moderate for speed
# - EXTRACT_DPI: 300 is a good baseline; crops improve readability a lot.
PREFLIGHT_DPI = 160
EXTRACT_DPI = 320
MAX_PAGES = 2

TARGET_N = 500
MAX_TRIES = 6000
SLEEP_BETWEEN_CALLS = 0.10

OUT_CSV = os.path.join(BASE_DIR, "ALL_COMPANIES_500parts_dataset.csv")
OUT_JSONL = os.path.join(BASE_DIR, "ALL_COMPANIES_500parts_dataset.jsonl")
CHECKPOINT_JSONL = os.path.join(BASE_DIR, "ALL_COMPANIES_500parts_checkpoint.jsonl")

client = OpenAI()


# -------------------------
# File iteration
# -------------------------
def iter_pdfs(folder: str):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                yield os.path.join(root, f)


def build_pdf_pool() -> List[Dict[str, str]]:
    pool: List[Dict[str, str]] = []
    for company, folder in FOLDERS.items():
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue
        for p in iter_pdfs(folder):
            pool.append({"company": company, "pdf_path": p})
    return pool


# -------------------------
# JSONL utils
# -------------------------
def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_paths(checkpoint_path: str) -> set:
    done = set()
    if not os.path.isfile(checkpoint_path):
        return done
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                p = obj.get("pdf_path")
                if p:
                    done.add(p)
            except Exception:
                pass
    return done


def safe_get(d: Dict[str, Any], path: List[str], default="unknown"):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# -------------------------
# PDF -> images (full pages + smart crops)
# -------------------------
def _pixmap_to_data_url_png(pix: fitz.Pixmap) -> str:
    b = pix.tobytes("png")
    return "data:image/png;base64," + base64.b64encode(b).decode("utf-8")


def render_page_b64(
    pdf_path: str,
    page_index: int,
    dpi: int,
    clip: Optional[fitz.Rect] = None,
) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=clip)
    doc.close()
    return _pixmap_to_data_url_png(pix)


def page_crops_b64(pdf_path: str, page_index: int, dpi: int) -> Dict[str, str]:
    """
    Heuristic crops that usually cover title block + notes on manufacturing drawings.
    These crops often increase accuracy while reducing tokens (smaller image regions).
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    w, h = page.rect.width, page.rect.height
    doc.close()

    # Bottom band (notes/title area)
    bottom_band = fitz.Rect(0.0, h * 0.60, w, h)

    # Bottom-right (common title block location)
    bottom_right = fitz.Rect(w * 0.45, h * 0.55, w, h)

    # Bottom-left (sometimes notes/material callouts)
    bottom_left = fitz.Rect(0.0, h * 0.55, w * 0.55, h)

    return {
        "crop_bottom_band": render_page_b64(pdf_path, page_index, dpi, clip=bottom_band),
        "crop_bottom_right": render_page_b64(pdf_path, page_index, dpi, clip=bottom_right),
        "crop_bottom_left": render_page_b64(pdf_path, page_index, dpi, clip=bottom_left),
    }


def pdf_to_images_b64(pdf_path: str, max_pages: int, dpi: int) -> List[str]:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out: List[str] = []
    for i in range(min(max_pages, doc.page_count)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        out.append(_pixmap_to_data_url_png(pix))
    doc.close()
    return out


# -------------------------
# Preflight (less strict)
# -------------------------
def preflight_readability(company: str, pdf_filename: str, img_b64: str) -> Dict[str, Any]:
    prompt = f"""
You are checking whether a manufacturing drawing is readable enough to extract quoting information.

Return ONLY a JSON object with:
- readability_ok: "yes" or "no"
- reason: short string
- anchors_found: array of strings from this set if visible:
  ["DIMENSIONS ARE IN", "UNITS", "TOLERANCE", "MATERIAL", "FINISH", "NOTES", "REV", "TITLE BLOCK"]
- severity: "low"|"medium"|"high"

Rules:
- If MOST title block / notes text is unreadable (too small/blurry), set readability_ok="no".
- If SOME is blurry but you can still read key fields (material/units/tolerance/dims), set readability_ok="yes" with severity="medium".
- Do not guess.

Company: {company}
PDF file: {pdf_filename}
"""
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": img_b64, "detail": "low"},
    ]
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)


# -------------------------
# Full extract (drawing_number + title + rev from drawing)
# -------------------------
def full_extract(company: str, pdf_path: str, imgs_full_pages_b64: List[str], crops: Dict[str, str]) -> Dict[str, Any]:
    pdf_filename = os.path.basename(pdf_path)
    pdf_stem = os.path.splitext(pdf_filename)[0]

    prompt = f"""
Return ONE JSON object with keys EXACTLY:

company (="{company}")
pdf_filename (="{pdf_filename}")
pdf_stem (="{pdf_stem}")

# Identity (from the drawing/title block; NOT from filename)
drawing_number: string (aka part number / drawing number; from title block; "unknown" if not found)
title: string (part title/description from title block; "unknown" if not found)
revision: string ("unknown" if not found)

# Core
units: "inch"|"mm"|"unknown"
material: string (resolve "SEE NOTES" by reading notes)
material_form: "plate"|"bar"|"tube"|"sheet"|"unknown"
finish: string (NONE if none; else e.g. ANODIZE / CHEM FILM / PASSIVATE / E-COAT / etc)
finish_spec: string (type/class/color/spec if present; else "unknown")
marking: "Yes"|"No"|"Unknown"

# Process class
process_class: "lathe_only"|"mill_only"|"lathe_and_mill"|"unknown"
process_reason: string (short reason based on drawing features)

# Dimensions / tolerances
overall_dims: {{x:number,y:number,z:number,longest:number,dims_text:string}}
tolerances: {{general:string,tightest:string,gd_t:string}}

# Derived robustness counts
gdt_callouts: array of strings (TRUE POSITION, FLATNESS, PARALLELISM, PERPENDICULARITY, PROFILE, RUNOUT, CONCENTRICITY, CYLINDRICITY)
gdt_count: int
datum_count: int
tight_tol_count: int  (count of dims at/under 0.001in OR 0.025mm; if units unknown, best effort)

# Feature counts
feature_counts: {{
  holes:int, threads:int, bores:int, bosses:int, flanges:int, slots:int, chamfers:int, fillets:int, pockets:int
}}
Multiplicity like 4X, 6X, 42X, X42 are multipliers.

# Secondary ops + requirements (booleans)
inspection_fai: "Yes"|"No"|"Unknown"
inspection_cmm: "Yes"|"No"|"Unknown"
inspection_100pct: "Yes"|"No"|"Unknown"
cert_material: "Yes"|"No"|"Unknown"
cert_coc: "Yes"|"No"|"Unknown"
dfars: "Yes"|"No"|"Unknown"
rohs_reach: "Yes"|"No"|"Unknown"
heat_treat: string (e.g. H900, solution+age, "none"/"unknown")
hardness: string (e.g. HRC 45, HB 150, "unknown")

# Small feature minima (best-effort; use 0 if unknown)
min_hole_dia: number
min_thread_size: string (e.g. 4-40, M3, "unknown")
min_wall_thickness: number

# Hardware / notes
hardware_install: string (pems/keenserts/helicoils/dowel pins/bushings/key inserts/press-fit; else "unknown")
key_notes: string
llm_quote_notes: string

confidence: {{overall:number,dims:number,material:number,tolerances:number}} 0..1

Rules:
- Do not invent. If unsure use "unknown"/"Unknown" and 0.
- drawing_number/title/revision MUST come from the drawing title block or notes (use the cropped images),
  NOT from the PDF filename.
- Prefer notes/title block if other fields say "SEE NOTES".
- For process_class:
  - lathe_only if dominated by turned diameters, centerlines, circular profiles, OD/ID steps, threads typical of turning.
  - mill_only if prismatic with pockets/slots/rectangular profiles, planar faces, hole patterns.
  - lathe_and_mill if both turned diameters AND milled flats/slots/pockets/off-axis holes are indicated.
"""
    content = [{"type": "input_text", "text": prompt}]

    # Put crops first (more likely to contain title block/notes)
    for k in ["crop_bottom_right", "crop_bottom_band", "crop_bottom_left"]:
        if k in crops and crops[k]:
            content.append({"type": "input_image", "image_url": crops[k], "detail": "high"})

    # Then full pages for context/features
    for u in imgs_full_pages_b64:
        content.append({"type": "input_image", "image_url": u, "detail": "high"})

    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": content}],
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)


# -------------------------
# Row mapping + columns
# -------------------------
def to_csv_row(pdf_path: str, preflight: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    overall = data.get("overall_dims", {}) if isinstance(data.get("overall_dims"), dict) else {}
    tol = data.get("tolerances", {}) if isinstance(data.get("tolerances"), dict) else {}
    fc = data.get("feature_counts", {}) if isinstance(data.get("feature_counts"), dict) else {}

    return {
        # Identity
        "Company": safe_get(data, ["company"]),
        "Part_Number": safe_get(data, ["drawing_number"]),  # <-- from drawing
        "Title": safe_get(data, ["title"]),
        "Revision": safe_get(data, ["revision"]),
        "PDF_FileName": safe_get(data, ["pdf_filename"]),
        "PDF_Stem": safe_get(data, ["pdf_stem"]),
        "PDF_Path": pdf_path,

        # Pricing placeholders
        "Qty": "",
        "Unit_Price": "",

        # Process
        "Process_Class": safe_get(data, ["process_class"]),
        "Process_Reason": safe_get(data, ["process_reason"]),

        # Units + Material + Finish
        "Units": safe_get(data, ["units"]),
        "Material": safe_get(data, ["material"]),
        "Material_Form": safe_get(data, ["material_form"]),
        "Finish": safe_get(data, ["finish"]),
        "Finish_Spec": safe_get(data, ["finish_spec"]),
        "Marking": safe_get(data, ["marking"]),

        # Size
        "Overall_X": overall.get("x", 0.0),
        "Overall_Y": overall.get("y", 0.0),
        "Overall_Z": overall.get("z", 0.0),
        "Longest_Dim": overall.get("longest", 0.0),
        "Overall_Dims_Text": overall.get("dims_text", "unknown"),

        # Tolerances + GD&T
        "Tol_General": tol.get("general", "unknown"),
        "Tol_Tightest": tol.get("tightest", "unknown"),
        "GD_T": tol.get("gd_t", "unknown"),
        "GD_T_Callouts": ", ".join(data.get("gdt_callouts", [])) if isinstance(data.get("gdt_callouts"), list) else "",
        "GD_T_Count": safe_get(data, ["gdt_count"], 0),
        "Datum_Count": safe_get(data, ["datum_count"], 0),
        "Tight_Tol_Count": safe_get(data, ["tight_tol_count"], 0),

        # Smallest features (best-effort)
        "Min_Hole_Dia": safe_get(data, ["min_hole_dia"], 0.0),
        "Min_Thread_Size": safe_get(data, ["min_thread_size"]),
        "Min_Wall_Thickness": safe_get(data, ["min_wall_thickness"], 0.0),

        # Feature counts
        "Holes": fc.get("holes", 0),
        "Threads": fc.get("threads", 0),
        "Bores": fc.get("bores", 0),
        "Bosses": fc.get("bosses", 0),
        "Flanges": fc.get("flanges", 0),
        "Slots": fc.get("slots", 0),
        "Chamfers": fc.get("chamfers", 0),
        "Fillets": fc.get("fillets", 0),
        "Pockets": fc.get("pockets", 0),

        # Secondary ops / requirements (booleans)
        "Inspection_FAI": safe_get(data, ["inspection_fai"]),
        "Inspection_CMM": safe_get(data, ["inspection_cmm"]),
        "Inspection_100pct": safe_get(data, ["inspection_100pct"]),
        "Cert_Material": safe_get(data, ["cert_material"]),
        "Cert_CoC": safe_get(data, ["cert_coc"]),
        "DFARS": safe_get(data, ["dfars"]),
        "RoHS_REACH": safe_get(data, ["rohs_reach"]),
        "Heat_Treat": safe_get(data, ["heat_treat"]),
        "Hardness": safe_get(data, ["hardness"]),

        # Hardware / notes
        "Hardware_Install": safe_get(data, ["hardware_install"]),
        "Key_Notes": safe_get(data, ["key_notes"]),
        "LLM_Quote_Notes": safe_get(data, ["llm_quote_notes"]),

        # Confidence
        "Conf_Overall": safe_get(data, ["confidence", "overall"], 0.0),
        "Conf_Dims": safe_get(data, ["confidence", "dims"], 0.0),
        "Conf_Material": safe_get(data, ["confidence", "material"], 0.0),
        "Conf_Tolerances": safe_get(data, ["confidence", "tolerances"], 0.0),

        # Preflight
        "Preflight_OK": preflight.get("readability_ok", "unknown"),
        "Preflight_Severity": preflight.get("severity", "unknown"),
        "Preflight_Reason": preflight.get("reason", "unknown"),
        "Preflight_Anchors": ", ".join(preflight.get("anchors_found", [])) if isinstance(preflight.get("anchors_found"), list) else "",
    }


def load_rows_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Rebuild CSV rows from saved raw JSONL (resume without losing progress).
    """
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(jsonl_path):
        return rows
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pdf_path = obj.get("pdf_path", "")
                preflight = obj.get("preflight", {}) or {}
                result = obj.get("result", {}) or {}
                if pdf_path and isinstance(result, dict):
                    rows.append(to_csv_row(pdf_path, preflight, result))
            except Exception:
                continue
    return rows


# -------------------------
# Safe CSV writes
# -------------------------
def safe_write_csv(df: pd.DataFrame, out_csv: str, tries: int = 10) -> bool:
    tmp = out_csv + ".tmp"
    for attempt in range(1, tries + 1):
        try:
            df.to_csv(tmp, index=False)
            os.replace(tmp, out_csv)
            return True
        except PermissionError as e:
            wait = min(10, attempt)
            print(f"[CSV LOCKED] attempt {attempt}/{tries}: {e} -> retry in {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"[CSV WRITE FAIL] {type(e).__name__}: {e}")
            return False
    return False


def main():
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError('OPENAI_API_KEY not set. In PowerShell: $env:OPENAI_API_KEY="sk-..."')

    pool = build_pdf_pool()
    if not pool:
        raise RuntimeError("No PDFs found in the 4 folders.")

    done = load_done_paths(CHECKPOINT_JSONL)
    collected_rows = load_rows_from_jsonl(OUT_JSONL)
    collected = len(collected_rows)

    print(f"Pool size: {len(pool)} PDFs")
    print(f"Already done (checkpoint): {len(done)} PDFs")
    print(f"Already collected (from JSONL): {collected}/{TARGET_N}")

    csv_columns = [
        # Identity + Pricing
        "Company", "Part_Number", "Title", "Revision", "PDF_FileName", "PDF_Stem", "PDF_Path", "Qty", "Unit_Price",

        # Process + units/material/finish
        "Process_Class", "Process_Reason",
        "Units", "Material", "Material_Form", "Finish", "Finish_Spec", "Marking",

        # Size
        "Overall_X", "Overall_Y", "Overall_Z", "Longest_Dim", "Overall_Dims_Text",

        # Tolerance/GD&T
        "Tol_General", "Tol_Tightest", "GD_T", "GD_T_Callouts", "GD_T_Count", "Datum_Count", "Tight_Tol_Count",

        # Min features
        "Min_Hole_Dia", "Min_Thread_Size", "Min_Wall_Thickness",

        # Feature counts
        "Holes", "Threads", "Bores", "Bosses", "Flanges", "Slots", "Chamfers", "Fillets", "Pockets",

        # Secondary ops / requirements
        "Inspection_FAI", "Inspection_CMM", "Inspection_100pct",
        "Cert_Material", "Cert_CoC", "DFARS", "RoHS_REACH",
        "Heat_Treat", "Hardness",

        # Notes
        "Hardware_Install", "Key_Notes", "LLM_Quote_Notes",

        # Confidence
        "Conf_Overall", "Conf_Dims", "Conf_Material", "Conf_Tolerances",

        # Preflight
        "Preflight_OK", "Preflight_Severity", "Preflight_Reason", "Preflight_Anchors",
    ]

    tries = 0
    while collected < TARGET_N and tries < MAX_TRIES:
        tries += 1
        item = random.choice(pool)
        company = item["company"]
        pdf_path = item["pdf_path"]

        if pdf_path in done:
            continue

        pdf_filename = os.path.basename(pdf_path)
        pdf_stem = os.path.splitext(pdf_filename)[0]
        print(f"\n[TRY {tries}/{MAX_TRIES}] collected={collected}/{TARGET_N} :: {company} :: {pdf_stem}")

        # Preflight
        try:
            pre_img = render_page_b64(pdf_path, 0, PREFLIGHT_DPI, clip=None)
            gate = preflight_readability(company, pdf_filename, pre_img)
            print("Gate:", gate)
        except Exception as e:
            print("Preflight failed:", e)
            done.add(pdf_path)
            append_jsonl(CHECKPOINT_JSONL, {"pdf_path": pdf_path, "error": f"preflight_failed: {e}"})
            continue

        if gate.get("readability_ok") != "yes":
            print("Skipping (unreadable).")
            done.add(pdf_path)
            append_jsonl(CHECKPOINT_JSONL, {"pdf_path": pdf_path, "skipped": True, "preflight": gate})
            continue

        # Extract
        try:
            # Full pages
            imgs_full = pdf_to_images_b64(pdf_path, MAX_PAGES, EXTRACT_DPI)
            # Crops from first page (title block / notes)
            crops = page_crops_b64(pdf_path, 0, EXTRACT_DPI)

            t0 = time.time()
            data = full_extract(company, pdf_path, imgs_full, crops)
            dt = time.time() - t0
            print(f"Extract OK in {dt:.1f}s")
        except Exception as e:
            print("Extract failed:", e)
            done.add(pdf_path)
            append_jsonl(CHECKPOINT_JSONL, {"pdf_path": pdf_path, "error": f"extract_failed: {e}", "preflight": gate})
            continue

        # Save raw record
        record = {
            "pdf_path": pdf_path,
            "company": company,
            "pdf_filename": pdf_filename,
            "pdf_stem": pdf_stem,
            "preflight": gate,
            "result": data,
        }
        append_jsonl(OUT_JSONL, record)

        # checkpoint first
        append_jsonl(CHECKPOINT_JSONL, {"pdf_path": pdf_path})
        done.add(pdf_path)

        # Add row
        row = to_csv_row(pdf_path, gate, data)
        collected_rows.append(row)
        collected += 1

        # Write CSV
        df = pd.DataFrame(collected_rows, columns=csv_columns)
        ok = safe_write_csv(df, OUT_CSV)
        if ok:
            print(f"[CSV] saved {collected} rows -> {OUT_CSV}")
        else:
            print(f"[CSV] could not write right now (locked?). Progress still saved to JSONL/checkpoint.")

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("\nDONE")
    print(f"Collected: {collected}/{TARGET_N}")
    print("CSV:", OUT_CSV)
    print("JSONL:", OUT_JSONL)
    print("Checkpoint:", CHECKPOINT_JSONL)


if __name__ == "__main__":
    main()