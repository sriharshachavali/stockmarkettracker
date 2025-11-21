import os
import json
import time
from typing import Any, Dict
import pandas as pd
import openai
import snowflake.connector

# Requirements:
# pip install openai pandas snowflake-connector-python

# Config via environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")  # or your custom GPT name
SNOW_USER = os.getenv("SNOWFLAKE_USER")
SNOW_PWD = os.getenv("SNOWFLAKE_PASSWORD")
SNOW_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOW_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOW_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOW_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
SNOW_TABLE = os.getenv("SNOWFLAKE_TABLE", "GPT_ANALYSIS")

if not OPENAI_API_KEY:
    raise SystemExit("Set OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

def summarize_with_gpt(payload: Dict[str, Any], model: str = OPENAI_MODEL, max_tokens: int = 800) -> str:
    """
    Send JSON payload to the custom GPT defined by OPENAI_MODEL and return text response.
    The assumption is that the custom GPT already contains the instruction set, so we only
    forward the payload as the user message.
    """
    if not model:
        raise ValueError("Set OPENAI_MODEL to your custom GPT name or deployment.")
    user_message = json.dumps(payload)
    for attempt in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        except Exception as e:
            time.sleep(1 + attempt * 2)
            last_err = e
    raise last_err

def write_analysis_to_snowflake(df: pd.DataFrame, analyses: Dict[int, str]):
    """Insert rows into Snowflake. df rows correspond by index to analyses dict."""
    conn = snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PWD,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WAREHOUSE,
        database=SNOW_DATABASE,
        schema=SNOW_SCHEMA,
    )
    cs = conn.cursor()
    try:
        # create table if missing (variant to store JSON)
        cs.execute(f"""
            CREATE TABLE IF NOT EXISTS {SNOW_TABLE} (
                id INTEGER AUTOINCREMENT,
                symbol STRING,
                data VARIANT,
                gpt_result VARIANT,
                created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        insert_sql = f"INSERT INTO {SNOW_TABLE} (symbol, data, gpt_result) VALUES (%s, %s, %s)"
        for idx, row in df.reset_index().iterrows():
            symbol = row.get("symbol") or row.get("Symbol") or None
            data_json = json.loads(row.to_json()) if not row.empty else {}
            gpt_text = analyses.get(idx, "{}")
            # attempt to parse GPT JSON; fallback to raw text
            try:
                gpt_json = json.loads(gpt_text)
            except Exception:
                gpt_json = {"text": gpt_text}
            cs.execute(insert_sql, (symbol, json.dumps(data_json), json.dumps(gpt_json)))
        conn.commit()
    finally:
        cs.close()
        conn.close()

def analyze_df_with_gpt(df: pd.DataFrame, limit_rows: int = 50) -> Dict[int, str]:
    """Prepare row-wise payloads, call GPT, return mapping idx->gpt_response."""
    analyses = {}
    for i, (_, row) in enumerate(df.reset_index().iterrows()):
        if i >= limit_rows:
            break
        payload = {"row": json.loads(row.to_json())}
        gpt_out = summarize_with_gpt(payload)
        analyses[i] = gpt_out
        time.sleep(0.2)
    return analyses

if __name__ == "__main__":
    # Example usage: load precomputed DataFrame (replace path as needed)
    df_path = os.getenv("INPUT_CSV", "input/input_for_sec_summary.csv")
    if not os.path.exists(df_path):
        raise SystemExit(f"Provide INPUT_CSV file at {df_path}")
    df = pd.read_csv(df_path)
    analyses = analyze_df_with_gpt(df)
    print(analyses)
    # write_analysis_to_snowflake(df, analyses)
    print("Done: wrote", len(analyses), "analyses to Snowflake.")
