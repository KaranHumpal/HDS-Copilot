import os
import json
from typing import Optional

import numpy as np
import pandas as pd

# ============================================================
# STEP 11: EVAL METRICS FROM FILLED GT CSV
#
# Run Step 10 first. It creates:
#   step10_batch_eval.csv
#   step10_ground_truth_template.csv
#
# Then manually fill GT_Unit_Price in the GT template CSV.
# This script merges the filled GT file with Step 10 outputs and
# computes quantitative price metrics for your report.
# ============================================================

EVAL_DIR = r"C:\Users\khump\OneDrive\Desktop\quote_eval_outputs"
STEP10_CSV = os.path.join(EVAL_DIR, "step10_batch_eval.csv")
GT_FILLED_CSV = os.path.join(EVAL_DIR, "step10_ground_truth_template.csv")

OUT_MERGED_CSV = os.path.join(EVAL_DIR, "step11_eval_with_gt.csv")
OUT_SUMMARY_JSON = os.path.join(EVAL_DIR, "step11_eval_metrics.json")
OUT_RESULTS_MD = os.path.join(EVAL_DIR, "step11_eval_results.md")

QTY_LOW_MIN = 3
QTY_LOW_MAX = 30


def clean_text(s) -> str:
    if s is None:
        return ""
    return str(s).strip()


def norm_pn(x) -> str:
    return "".join(clean_text(x).upper().split())


def mae(y_true, y_pred) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def median_ape(y_true, y_pred) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    ape = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.median(ape))


def within_pct(y_true, y_pred, pct: float) -> Optional[float]:
    if len(y_true) == 0:
        return None
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    ape = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.mean(ape <= pct))


def compute_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"count": 0}

    y_true = df["GT_Unit_Price"].astype(float).values
    y_pred = df["pred_unit_price"].astype(float).values

    return {
        "count": int(len(df)),
        "mae_usd": mae(y_true, y_pred),
        "rmse_usd": rmse(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "median_ape_pct": median_ape(y_true, y_pred),
        "within_10pct": within_pct(y_true, y_pred, 10.0),
        "within_20pct": within_pct(y_true, y_pred, 20.0),
        "within_30pct": within_pct(y_true, y_pred, 30.0),
        "mean_pred_unit_price": float(df["pred_unit_price"].mean()),
        "mean_gt_unit_price": float(df["GT_Unit_Price"].mean()),
        "mean_model_confidence": float(df["model_confidence"].mean()) if "model_confidence" in df.columns else None,
        "mean_top1_similarity": float(df["top1_similarity"].mean()) if "top1_similarity" in df.columns else None,
    }


def write_results_md(summary: dict, out_md: str):
    lines = [
        "# Step 11 GT-Based Evaluation Results",
        "",
        "## Overall",
        "",
    ]
    for k, v in summary.get("overall", {}).items():
        lines.append(f"- {k}: {v}")

    if "qty_3_30" in summary:
        lines += ["", "## Qty 3-30", ""]
        for k, v in summary["qty_3_30"].items():
            lines.append(f"- {k}: {v}")

    if "qty_100" in summary:
        lines += ["", "## Qty 100", ""]
        for k, v in summary["qty_100"].items():
            lines.append(f"- {k}: {v}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def main():
    if not os.path.isfile(STEP10_CSV):
        raise FileNotFoundError(f"Missing Step 10 CSV: {STEP10_CSV}")
    if not os.path.isfile(GT_FILLED_CSV):
        raise FileNotFoundError(f"Missing filled GT CSV: {GT_FILLED_CSV}")

    pred_df = pd.read_csv(STEP10_CSV)
    gt_df = pd.read_csv(GT_FILLED_CSV)

    if "Part_Number" not in gt_df.columns or "GT_Unit_Price" not in gt_df.columns:
        raise ValueError("GT CSV must contain Part_Number and GT_Unit_Price")

    pred_df["pn_norm"] = pred_df["part_number"].map(norm_pn)
    gt_df["pn_norm"] = gt_df["Part_Number"].map(norm_pn)
    gt_df["GT_Unit_Price"] = pd.to_numeric(gt_df["GT_Unit_Price"], errors="coerce")

    gt_valid = gt_df[gt_df["GT_Unit_Price"].notna()].copy()
    merged = pred_df.merge(
        gt_valid[["pn_norm", "GT_Unit_Price"]],
        on="pn_norm",
        how="inner"
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping part numbers found between Step 10 CSV and filled GT CSV.")

    merged["abs_error_usd"] = (merged["pred_unit_price"] - merged["GT_Unit_Price"]).abs()
    merged["ape_pct"] = merged["abs_error_usd"] / merged["GT_Unit_Price"].abs().clip(lower=1e-9) * 100.0

    merged.to_csv(OUT_MERGED_CSV, index=False)

    summary = {
        "overall": compute_metrics(merged),
    }

    low = merged[merged["qty_used"].between(QTY_LOW_MIN, QTY_LOW_MAX)] if "qty_used" in merged.columns else pd.DataFrame()
    high = merged[merged["qty_used"] == 100] if "qty_used" in merged.columns else pd.DataFrame()

    if len(low) > 0:
        summary["qty_3_30"] = compute_metrics(low)
    if len(high) > 0:
        summary["qty_100"] = compute_metrics(high)

    with open(OUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_results_md(summary, OUT_RESULTS_MD)

    print("DONE")
    print(f"Saved merged CSV: {OUT_MERGED_CSV}")
    print(f"Saved metrics JSON: {OUT_SUMMARY_JSON}")
    print(f"Saved results markdown: {OUT_RESULTS_MD}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
