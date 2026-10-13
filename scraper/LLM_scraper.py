"""
llm_scraper.py
"""

import argparse
import json
import re
import time
import requests
import pandas as pd

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
GUIDE_PAGE_KEYWORDS = [
    "living with", "talking to", "recovering from", "coping with",
    "caring for", "supporting", "your feelings about",
]
FIELDS = ["symptoms", "causes", "warnings", "recommendations"]

SYSTEM_PROMPT = """You are helping complete a structured public-health dataset scraped from NHS \
Inform (Scotland's patient information site). For the named condition, write ONLY the missing \
field(s) requested, in the same factual, plain-language, patient-facing style as the reference \
example below - similar sentence length, no meta-commentary, no "as an AI" disclaimers, no medical \
disclaimers, just the informational content itself as NHS Inform would phrase it.

Reference example (real NHS Inform entry, for style/tone/length only):
Condition: {ref_condition}
Symptoms: {ref_symptoms}
Causes: {ref_causes}

Respond with ONLY a JSON object whose keys are exactly the requested missing field names, and whose \
values are the generated text for each (plain string, no markdown). Do not include fields that were \
not requested."""

def is_guide_page(name: str) -> bool:
    n = name.lower()
    return any(kw in n for kw in GUIDE_PAGE_KEYWORDS)

def build_reference(df: pd.DataFrame) -> dict:
    full_rows = df.dropna(subset=FIELDS)
    row = full_rows.iloc[0]
    return {
        "ref_condition": row["condition"],
        "ref_symptoms": str(row["symptoms"])[:400],
        "ref_causes": str(row["causes"])[:400],
    }

def call_llm(condition: str, missing_fields: list, reference: dict, api_url: str) -> dict:
    system = SYSTEM_PROMPT.format(**reference)
    user = f"Condition: {condition}\nMissing fields to generate: {', '.join(missing_fields)}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }
    resp = requests.post(api_url, json=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="nhs_conditions.csv")
    ap.add_argument("--output", default="nhs_conditions_final.csv")
    ap.add_argument("--review-output", default="conditions_to_review.csv")
    ap.add_argument("--api-url", required=True, help="Ngrok URL from the Kaggle API server")
    args = ap.parse_args()

    for path_arg, label in [(args.output, "--output"), (args.review_output, "--review-output")]:
        if "/kaggle/input/" in path_arg.replace("\\", "/"):
            print(f"WARNING: {label}='{path_arg}' is under /kaggle/input/, which is READ-ONLY. Point it at /kaggle/working/ instead.")

    df = pd.read_csv(args.input)
    if "data_source" not in df.columns:
        df["data_source"] = "scraped"

    reference = build_reference(df)
    to_fill = df[df[FIELDS].isnull().any(axis=1)]

    review_rows = []
    filled = 0

    for idx, row in to_fill.iterrows():
        name = row["condition"]
        missing = [f for f in FIELDS if pd.isnull(row[f])]

        if is_guide_page(name) or len(missing) == 4:
            review_rows.append({"condition": name, "missing_fields": ", ".join(missing)})
            continue

        try:
            result = call_llm(name, missing, reference, args.api_url)
        except Exception as e:
            print(f"  FAILED for '{name}': {e}")
            review_rows.append({"condition": name, "missing_fields": ", ".join(missing), "error": str(e)})
            continue

        touched = False
        for field in missing:
            if result.get(field):
                df.at[idx, field] = result[field]
                touched = True
        if touched:
            df.at[idx, "data_source"] = "llm_generated" if df.at[idx, "data_source"] == "scraped" else df.at[idx, "data_source"]
            filled += 1
            print(f"[{filled}] Filled {missing} for '{name}' via LLM")

        if filled and filled % 10 == 0:
            try:
                df.to_csv(args.output, index=False, encoding="utf-8-sig")
                print(f"  [checkpoint] saved progress ({filled} filled so far) -> '{args.output}'")
            except Exception as e:
                print(f"  [checkpoint] WARNING: could not save to '{args.output}' ({e}).")

        time.sleep(0.3)

    seen_text = {}
    for idx, row in df[df["data_source"] == "llm_generated"].iterrows():
        for field in FIELDS:
            val = row[field]
            if pd.isnull(val) or len(str(val).strip()) < 40:
                continue
            key = str(val).strip()
            if key in seen_text and seen_text[key] != row["condition"]:
                review_rows.append({
                    "condition": row["condition"],
                    "missing_fields": "",
                    "reason": f"'{field}' text is identical to the one generated for '{seen_text[key]}' - looks generic",
                })
            else:
                seen_text[key] = row["condition"]

    def safe_save(dataframe, path, label):
        try:
            dataframe.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"{label} -> '{path}'")
        except Exception as e:
            fallback = "./" + path.split("/")[-1]
            print(f"WARNING: could not save to '{path}' ({e}). Falling back to '{fallback}'.")
            dataframe.to_csv(fallback, index=False, encoding="utf-8-sig")
            print(f"{label} -> '{fallback}'")

    safe_save(df, args.output, "Done. Final data")
    safe_save(pd.DataFrame(review_rows), args.review_output, "Rows routed to manual review")
    print(f"{filled} rows filled by LLM. {len(review_rows)} rows need manual review.")

if __name__ == "__main__":
    main()