import json
from openai import OpenAI

# ============================================================
# STEP 9: LLM QUOTE PACKAGE LAYER
#
# Purpose:
# - Takes the finalized sample from Step 8
# - Takes the ML prediction result from Step 8
# - Uses an LLM to produce a structured quote package
#
# Expected usage from Step 8:
#   result = predict(sample)
#   quote_pkg = generate_quote_package(sample, result)
# ============================================================

LLM_MODEL = "gpt-4.1"
client = OpenAI()

QUOTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "company": {"type": "string"},
        "part_number": {"type": "string"},
        "predicted_unit_price_usd": {"type": "number"},
        "model_confidence_0_1": {"type": "number"},
        "price_confidence_label": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        },
        "price_basis": {"type": "string"},
        "should_manual_review": {"type": "boolean"},
        "missing_info_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "process_plan": {
            "type": "array",
            "items": {"type": "string"}
        },
        "top_reasons_price_high_or_low": {
            "type": "array",
            "items": {"type": "string"}
        },
        "retrieved_jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "Company": {"type": "string"},
                    "Part_Number": {"type": "string"},
                    "Qty_num": {"type": ["number", "null"]},
                    "Unit_Price_num": {"type": "number"},
                    "score": {"type": "number"}
                },
                "required": ["Company", "Part_Number", "Qty_num", "Unit_Price_num", "score"]
            }
        }
    },
    "required": [
        "company",
        "part_number",
        "predicted_unit_price_usd",
        "model_confidence_0_1",
        "price_confidence_label",
        "price_basis",
        "should_manual_review",
        "missing_info_questions",
        "risk_flags",
        "process_plan",
        "top_reasons_price_high_or_low",
        "retrieved_jobs"
    ]
}


def generate_quote_package(sample: dict, result: dict) -> dict:
    """
    sample: finalized sample dict from Step 8
    result: output of predict(sample) from Step 8
    """

    prompt = f"""
You are a CNC quoting copilot.

You are given:
1) A structured RFQ job card
2) A machine-learning predicted unit price
3) Similar historical jobs retrieved by embeddings
4) Confidence and neighbor statistics

Return ONLY JSON matching the provided schema.

Rules:
- Use the ML predicted unit price as the main anchor.
- Do not invent unsupported technical details.
- price_basis should explain the price using the similar retrieved jobs and the RFQ characteristics.
- should_manual_review should be true if confidence is weak, critical info is missing, or the job seems high-risk.
- missing_info_questions should only ask questions that would materially change price or manufacturability.
- risk_flags should focus on tolerance, finish, inspection, ambiguity, or process risk.
- process_plan should be short, realistic shop-floor steps.
- top_reasons_price_high_or_low should explain the main cost drivers.
- retrieved_jobs must come from the provided retrieved jobs, not invented examples.

RFQ_JOB_CARD:
{json.dumps(sample, ensure_ascii=False, indent=2)}

ML_RESULT:
{json.dumps(result, ensure_ascii=False, indent=2)}
""".strip()

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "quote_package",
                "schema": QUOTE_SCHEMA,
                "strict": True
            }
        }
    )

    return json.loads(resp.output_text)


if __name__ == "__main__":
    # This file is meant to be imported by Step 8.
    # Example:
    # from step9_quote_llm import generate_quote_package
    # quote_pkg = generate_quote_package(sample, result)
    print("Step 9 ready. Import generate_quote_package(sample, result) from this file.")
