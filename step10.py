import os
import re
import json
import random
import shutil
import tempfile
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from step8_quot_cli import finalize_sample, predict
from step9 import generate_quote_package

client = OpenAI()

# ============================================================
# STEP 10: BATCH RUNNER + GT TEMPLATE BUILDER
#
# What this version does:
# 1) samples holdout PDFs not in the train/index set
# 2) parses each PDF with OpenAI
# 3) runs Step 8 prediction
# 4) runs Step 9 quote package generation
# 5) runs OpenAI evaluator scoring
# 6) saves normal eval outputs
# 7) ALSO writes a GT template CSV you can manually fill in later
#
# Then you run a separate eval script on the filled GT CSV.
# ============================================================

PDF_FOLDERS = [
    r"C:\Users\khump\OneDrive\Desktop\ARGO_PDF",
    r"C:\Users\khump\OneDrive\Desktop\ZEE_PDF",
    r"C:\Users\khump\OneDrive\Desktop\NOVA_PDF",
    r"C:\Users\khump\OneDrive\Desktop\JOBY_PDF",
]

INDEX_META_CSV = r"C:\Users\khump\OneDrive\Desktop\quote_index_step3\meta.csv"

VISION_MODEL = "gpt-4o"
EVAL_MODEL = "gpt-4.1"

N_SAMPLES = 20
RANDOM_SEED = 7
MAX_FILES_TO_SCAN = None

# Mostly small-qty jobs, some qty=100 jobs
# More realistic quoting quantities
QTY_CHOICES_MAIN = [3, 5, 10, 15, 20, 25, 30]
QTY_CHOICES_RARE = [50, 100]
QTY_RARE_FRACTION = 0.1

OUT_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_eval_outputs"
OUT_CSV = os.path.join(OUT_DIR, "step10_batch_eval.csv")
OUT_JSONL = os.path.join(OUT_DIR, "step10_batch_eval.jsonl")
OUT_SUMMARY_JSON = os.path.join(OUT_DIR, "step10_batch_eval_summary.json")
OUT_TOPLINE_MD = os.path.join(OUT_DIR, "step10_topline_results.md")
OUT_CASES_MD = os.path.join(OUT_DIR, "step10_example_cases.md")
OUT_GT_TEMPLATE_CSV = os.path.join(OUT_DIR, "step10_ground_truth_template.csv")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\t", " ")
    s = s.replace("±", "+/-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_pn(x: Any) -> str:
    s = clean_text(x).upper()
    s = re.sub(r"\s+", "", s)
    return s


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace("$", "").replace(",", "")
        if s == "" or s.lower() in {"nan", "none"}:
            return None
        return float(s)
    except Exception:
        return None


def infer_company_from_path(pdf_path: str) -> str:
    up = pdf_path.upper()
    for tag in ["ARGO", "ZEE", "NOVA", "JOBY"]:
        if f"\\{tag}_PDF\\" in up or f"/{tag}_PDF/" in up:
            return tag
        if f"\\{tag}\\" in up or f"/{tag}/" in up:
            return tag
    return "UNKNOWN"


def iter_pdfs(folder: str):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".pdf"):
                yield os.path.join(root, f)


def load_train_part_numbers(meta_csv: str) -> set:
    df = pd.read_csv(meta_csv)
    if "Part_Number" not in df.columns:
        raise ValueError("meta.csv is missing Part_Number")
    return set(df["Part_Number"].fillna("").astype(str).map(norm_pn).tolist())


def make_temp_pdf_copy(pdf_path: str) -> str:
    """
    Creates a safe temp copy with lowercase .pdf extension.
    Helps avoid API issues with uppercase .PDF filenames.
    """
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    safe_base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "rfq"
    tmp_dir = tempfile.mkdtemp(prefix="quote_eval_")
    tmp_path = os.path.join(tmp_dir, safe_base + ".pdf")
    shutil.copyfile(pdf_path, tmp_path)
    return tmp_path


def make_eval_qty_plan(n_samples: int, rng: random.Random) -> List[int]:
    n_rare = max(1, int(round(n_samples * QTY_RARE_FRACTION)))
    n_rare = min(n_rare, n_samples)
    n_main = n_samples - n_rare

    qtys = [rng.choice(QTY_CHOICES_MAIN) for _ in range(n_main)]
    qtys += [rng.choice(QTY_CHOICES_RARE) for _ in range(n_rare)]
    rng.shuffle(qtys)
    return qtys


def parse_pdf_to_raw_jobcard(pdf_path: str, qty: int) -> Dict[str, Any]:
    company = infer_company_from_path(pdf_path)
    tmp_pdf_path = make_temp_pdf_copy(pdf_path)

    try:
        with open(tmp_pdf_path, "rb") as fobj:
            uploaded = client.files.create(file=fobj, purpose="user_data")

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
- Process_Class should be inferred if the drawing clearly suggests turned/prismatic/mixed operations.
- Extract tolerances and GD&T as faithfully as possible.
- Do not invent dimensions or notes.
- Company is metadata only; do not depend on it for extraction quality.
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

        if not clean_text(raw.get("Part_Number", "")):
            raw["Part_Number"] = os.path.splitext(os.path.basename(pdf_path))[0]
        if not clean_text(raw.get("Company", "")):
            raw["Company"] = company
        if raw.get("Qty_num") in [None, "", 0]:
            raw["Qty_num"] = qty

        return raw

    finally:
        try:
            if os.path.exists(tmp_pdf_path):
                os.remove(tmp_pdf_path)
            tmp_dir = os.path.dirname(tmp_pdf_path)
            if os.path.isdir(tmp_dir):
                os.rmdir(tmp_dir)
        except Exception:
            pass


EVAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_score_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "parser_quality_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "retrieval_quality_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "quote_usefulness_1_to_5": {"type": "integer", "minimum": 1, "maximum": 5},
        "manual_review_should_be_required": {"type": "boolean"},
        "major_issues": {"type": "array", "items": {"type": "string"}},
        "short_rationale": {"type": "string"}
    },
    "required": [
        "overall_score_1_to_5",
        "parser_quality_1_to_5",
        "retrieval_quality_1_to_5",
        "quote_usefulness_1_to_5",
        "manual_review_should_be_required",
        "major_issues",
        "short_rationale"
    ]
}


def evaluate_case_with_openai(sample: Dict[str, Any], result: Dict[str, Any], quote_pkg: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
You are evaluating a CNC quoting assistant output.

Score the case using only the provided data.
Do NOT assume the exact unit price is ground truth correct.
Judge:
- parser completeness and plausibility
- retrieval usefulness
- whether the quote package is helpful and cautious
- whether manual review is appropriate

Scoring guidance:
- 5 = very strong / useful / plausible
- 3 = mixed / acceptable but notable issues
- 1 = poor / unreliable

Return ONLY JSON matching the schema.

FINALIZED_SAMPLE:
{json.dumps(sample, ensure_ascii=False, indent=2)}

ML_RESULT:
{json.dumps(result, ensure_ascii=False, indent=2)}

LLM_QUOTE_PACKAGE:
{json.dumps(quote_pkg, ensure_ascii=False, indent=2)}
""".strip()

    resp = client.responses.create(
        model=EVAL_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "quote_eval",
                "schema": EVAL_SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(resp.output_text)


def collect_holdout_pdfs() -> List[str]:
    train_pns = load_train_part_numbers(INDEX_META_CSV)
    pdfs: List[str] = []

    scanned = 0
    for folder in PDF_FOLDERS:
        if not os.path.isdir(folder):
            print(f"[WARN] Missing folder: {folder}")
            continue

        for pdf_path in iter_pdfs(folder):
            scanned += 1
            if MAX_FILES_TO_SCAN is not None and scanned > MAX_FILES_TO_SCAN:
                break
            pn = norm_pn(os.path.splitext(os.path.basename(pdf_path))[0])
            if pn not in train_pns:
                pdfs.append(pdf_path)

    return pdfs


def parser_presence_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {}

    def present(col: str) -> float:
        if col not in df.columns:
            return 0.0
        vals = df[col].fillna("").astype(str).map(clean_text)
        return float((vals != "").mean())

    def present_num(col: str) -> float:
        if col not in df.columns:
            return 0.0
        vals = pd.to_numeric(df[col], errors="coerce")
        return float(vals.notna().mean())

    return {
        "title_present_rate": present("title"),
        "units_present_rate": present("units_norm"),
        "material_present_rate": present("material_norm"),
        "finish_present_rate": present("finish_norm"),
        "process_class_present_rate": float(
            (df.get("process_class", pd.Series(dtype=str)).fillna("") != "unknown").mean()
        ) if "process_class" in df.columns else 0.0,
        "overall_dims_text_present_rate": present("overall_dims_text"),
        "tol_general_present_rate": present("tol_general"),
        "tol_tightest_present_rate": present("tol_tightest"),
        "gdt_present_rate": present("gd_t"),
        "overall_x_present_rate": present_num("overall_x_num"),
        "overall_y_present_rate": present_num("overall_y_num"),
        "overall_z_present_rate": present_num("overall_z_num"),
    }


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"count": 0}

    df = pd.DataFrame(records)
    summary: Dict[str, Any] = {
        "count": int(len(df)),
        "mean_pred_unit_price": float(df["pred_unit_price"].mean()) if "pred_unit_price" in df.columns else None,
        "median_pred_unit_price": float(df["pred_unit_price"].median()) if "pred_unit_price" in df.columns else None,
        "mean_model_confidence": float(df["model_confidence"].mean()) if "model_confidence" in df.columns else None,
        "median_model_confidence": float(df["model_confidence"].median()) if "model_confidence" in df.columns else None,
        "mean_top1_similarity": float(df["top1_similarity"].mean()) if "top1_similarity" in df.columns else None,
        "mean_eval_overall_score": float(df["eval_overall_score"].mean()) if "eval_overall_score" in df.columns else None,
        "mean_eval_parser_score": float(df["eval_parser_score"].mean()) if "eval_parser_score" in df.columns else None,
        "mean_eval_retrieval_score": float(df["eval_retrieval_score"].mean()) if "eval_retrieval_score" in df.columns else None,
        "mean_eval_quote_usefulness": float(df["eval_quote_usefulness"].mean()) if "eval_quote_usefulness" in df.columns else None,
        "manual_review_llm_rate": float(df["manual_review_llm"].mean()) if "manual_review_llm" in df.columns else None,
        "eval_manual_review_rate": float(df["eval_manual_review"].mean()) if "eval_manual_review" in df.columns else None,
    }

    if "qty_used" in df.columns:
        qty_counts = df["qty_used"].value_counts().sort_index().to_dict()
        summary["qty_counts"] = {str(int(k)): int(v) for k, v in qty_counts.items()}

    summary["parser_presence"] = parser_presence_metrics(df)
    return summary


def write_topline_results(summary: Dict[str, Any], out_md: str):
    lines = [
        "# Step 10 Topline Results",
        "",
        f"- Cases evaluated: {summary.get('count')}",
        f"- Mean predicted unit price: {summary.get('mean_pred_unit_price')}",
        f"- Median predicted unit price: {summary.get('median_pred_unit_price')}",
        f"- Mean model confidence: {summary.get('mean_model_confidence')}",
        f"- Mean top-1 similarity: {summary.get('mean_top1_similarity')}",
        f"- Mean eval overall score: {summary.get('mean_eval_overall_score')}",
        f"- Mean eval parser score: {summary.get('mean_eval_parser_score')}",
        f"- Mean eval retrieval score: {summary.get('mean_eval_retrieval_score')}",
        f"- Mean eval quote usefulness: {summary.get('mean_eval_quote_usefulness')}",
        f"- LLM manual review rate: {summary.get('manual_review_llm_rate')}",
        f"- Evaluator manual review rate: {summary.get('eval_manual_review_rate')}",
        "",
        "## Parser Presence Metrics",
        "",
    ]

    for k, v in summary.get("parser_presence", {}).items():
        lines.append(f"- {k}: {v}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def write_example_cases(df: pd.DataFrame, out_md: str, n_cases: int = 5):
    if len(df) == 0:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("# Example Cases\n\nNo cases available.\n")
        return

    sub = df.copy().sort_values(["eval_overall_score", "model_confidence"], ascending=[False, False]).head(n_cases)

    lines = ["# Example Cases", ""]
    for _, r in sub.iterrows():
        lines += [
            f"## {r.get('part_number', '')}",
            f"- File: {r.get('filename', '')}",
            f"- Company: {r.get('company_inferred', '')}",
            f"- Qty used: {r.get('qty_used', '')}",
            f"- Predicted unit price: {r.get('pred_unit_price', '')}",
            f"- Model confidence: {r.get('model_confidence', '')}",
            f"- Top-1 similarity: {r.get('top1_similarity', '')}",
            f"- Evaluator overall score: {r.get('eval_overall_score', '')}",
            f"- Evaluator rationale: {r.get('eval_short_rationale', '')}",
            "",
        ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def write_gt_template(rows: List[Dict[str, Any]], out_csv: str):
    """
    Creates a blank GT template for manual fill-in later.
    """
    gt_rows = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        gt_rows.append({
            "Part_Number": r.get("part_number", ""),
            "GT_Unit_Price": "",
            "GT_Qty": r.get("qty_used", ""),
            "Notes": "",
            "PDF_Path": r.get("pdf_path", ""),
            "Company": r.get("company_inferred", ""),
            "Pred_Unit_Price": r.get("pred_unit_price", ""),
            "Model_Confidence": r.get("model_confidence", ""),
        })

    df = pd.DataFrame(gt_rows).drop_duplicates(subset=["Part_Number"])
    df.to_csv(out_csv, index=False)


# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUT_DIR)
    rng = random.Random(RANDOM_SEED)

    holdout_pdfs = collect_holdout_pdfs()
    if not holdout_pdfs:
        raise RuntimeError("No holdout PDFs found after excluding train/index part numbers.")

    print(f"Found {len(holdout_pdfs)} holdout PDFs not present in train/index metadata.")

    n = min(N_SAMPLES, len(holdout_pdfs))
    picked = rng.sample(holdout_pdfs, n)
    qty_plan = make_eval_qty_plan(n, rng)

    print(f"Running batch eval on {n} sampled PDFs...")
    print(f"Quantity plan: {qty_plan}\n")

    all_rows: List[Dict[str, Any]] = []

    with open(OUT_JSONL, "w", encoding="utf-8") as jsonl_f:
        for i, (pdf_path, qty_used) in enumerate(zip(picked, qty_plan), start=1):
            print(f"[{i}/{n}] qty={qty_used}  {pdf_path}")

            row: Dict[str, Any] = {
                "pdf_path": pdf_path,
                "filename": os.path.basename(pdf_path),
                "company_inferred": infer_company_from_path(pdf_path),
                "qty_used": qty_used,
                "status": "ok",
            }

            try:
                raw = parse_pdf_to_raw_jobcard(pdf_path, qty=qty_used)
                sample = finalize_sample(raw)
                result = predict(sample)
                quote_pkg = generate_quote_package(sample, result)
                eval_out = evaluate_case_with_openai(sample, result, quote_pkg)

                row.update({
                    "part_number": sample.get("Part_Number", ""),
                    "title": sample.get("Title", ""),
                    "units_norm": sample.get("Units_norm", ""),
                    "material_norm": sample.get("Material_norm", ""),
                    "finish_norm": sample.get("Finish_norm", ""),
                    "process_class": sample.get("Process_Class", ""),
                    "overall_dims_text": sample.get("Overall_Dims_Text", ""),
                    "overall_x_num": sample.get("Overall_X_num", None),
                    "overall_y_num": sample.get("Overall_Y_num", None),
                    "overall_z_num": sample.get("Overall_Z_num", None),
                    "tol_general": sample.get("Tol_General", ""),
                    "tol_tightest": sample.get("Tol_Tightest", ""),
                    "gd_t": sample.get("GD_T", ""),
                    "sample_qty_num": sample.get("Qty_num", None),
                    "tight_tol_flag": sample.get("tight_tol_flag", False),
                    "pred_unit_price": float(result.get("pred_unit_price", np.nan)),
                    "model_confidence": float(result.get("confidence_0_1", np.nan)),
                    "top1_similarity": float(result.get("top1_similarity", np.nan)),
                    "nb_mean": safe_float(result.get("neighbor_stats", {}).get("nb_mean")),
                    "nb_median": safe_float(result.get("neighbor_stats", {}).get("nb_median")),
                    "manual_review_llm": bool(quote_pkg.get("should_manual_review", False)),
                    "eval_overall_score": int(eval_out.get("overall_score_1_to_5", 0)),
                    "eval_parser_score": int(eval_out.get("parser_quality_1_to_5", 0)),
                    "eval_retrieval_score": int(eval_out.get("retrieval_quality_1_to_5", 0)),
                    "eval_quote_usefulness": int(eval_out.get("quote_usefulness_1_to_5", 0)),
                    "eval_manual_review": bool(eval_out.get("manual_review_should_be_required", False)),
                    "eval_short_rationale": eval_out.get("short_rationale", ""),
                    "sample_json": json.dumps(sample, ensure_ascii=False),
                    "result_json": json.dumps(result, ensure_ascii=False),
                    "quote_pkg_json": json.dumps(quote_pkg, ensure_ascii=False),
                    "eval_json": json.dumps(eval_out, ensure_ascii=False),
                })

                jsonl_f.write(json.dumps({
                    "pdf_path": pdf_path,
                    "qty_used": qty_used,
                    "sample": sample,
                    "result": result,
                    "quote_package": quote_pkg,
                    "eval": eval_out,
                }, ensure_ascii=False) + "\n")

            except Exception as e:
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"
                jsonl_f.write(json.dumps({
                    "pdf_path": pdf_path,
                    "qty_used": qty_used,
                    "status": "error",
                    "error": row["error"],
                }, ensure_ascii=False) + "\n")

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    write_gt_template(all_rows, OUT_GT_TEMPLATE_CSV)

    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    summary = summarize_records(ok_rows)

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_topline_results(summary, OUT_TOPLINE_MD)
    if len(df[df["status"] == "ok"]) > 0:
        write_example_cases(df[df["status"] == "ok"].copy(), OUT_CASES_MD)

    print("\nDONE")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved JSONL: {OUT_JSONL}")
    print(f"Saved summary JSON: {OUT_SUMMARY_JSON}")
    print(f"Saved topline Markdown: {OUT_TOPLINE_MD}")
    print(f"Saved example cases Markdown: {OUT_CASES_MD}")
    print(f"Saved GT template CSV: {OUT_GT_TEMPLATE_CSV}")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()