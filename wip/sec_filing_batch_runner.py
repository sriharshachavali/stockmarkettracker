import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List

import openai
import pandas as pd

from wip.sec_filing_details import (
    extract_section,
    get_full_filing_details,
    download_filing_document,
)

DEFAULT_INPUT = "input/input_for_sec_summary.csv"
DEFAULT_OUTPUT = "output/sec_filing_summary.json"
DEFAULT_METADATA_OUTPUT = "output/sec_filing_run_metadata.json"
ENABLE_GPT_SUMMARY = os.getenv("SEC_ENABLE_GPT_SUMMARY", "1").lower() not in {"0", "false", "no"}
OPENAI_MODEL = os.getenv("SEC_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
OPENAI_MAX_TOKENS = int(os.getenv("SEC_SUMMARY_MAX_TOKENS", "600"))
OPENAI_TEMPERATURE = float(os.getenv("SEC_SUMMARY_TEMPERATURE", "0.2"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SECTION_CHAR_LIMIT = int(os.getenv("SEC_SECTION_CHAR_LIMIT", "4000"))
SUMMARY_SECTIONS: Dict[str, List[str]] = {
    "risk_factors": ["ITEM 1A", "RISK FACTORS"],
    "mdna": ["ITEM 7", "MANAGEMENT'S DISCUSSION", "MANAGEMENTS DISCUSSION"],
    "liquidity": ["LIQUIDITY AND CAPITAL RESOURCES", "ITEM 7A", "ITEM 2"],
}

if ENABLE_GPT_SUMMARY:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY must be set to generate GPT-4o summaries.")
    openai.api_key = OPENAI_API_KEY


def read_tickers(input_path: str) -> List[str]:
    """
    Returns the list of tickers from the first column of the CSV. A header row
    titled 'ticker' or 'symbol' is skipped automatically.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path, header=None)
    first_col = df.iloc[:, 0].astype(str).str.strip()
    tickers = [
        value.upper()
        for value in first_col
        if value and value.lower() not in {"ticker", "symbol"}
    ]
    deduped = list(dict.fromkeys(tickers))
    if not deduped:
        raise ValueError(f"No tickers found in {input_path}")
    return deduped


def run_batch(input_path: str, limit: int) -> List[dict]:
    tickers = read_tickers(input_path)
    filings_output = []
    ticker_output = []
    for ticker in tickers:
        run_ts = current_timestamp()
        try:
            filings = get_full_filing_details(ticker, limit=limit)
            if ENABLE_GPT_SUMMARY:
                enrich_filings_with_summaries(filings)
            print(f"[OK] {ticker}: {len(filings['filings'])} filings")
            filings_output.extend(flatten_filings_for_output(filings, run_ts))
            ticker_output.append(
                {
                    "ticker_name": filings.get("ticker"),
                    "company_name": filings.get("companyName"),
                    "last_watermark": run_ts,
                    "ingest_dt_tm": run_ts,
                    "update_dt_tm": run_ts,
                }
            )
        except Exception as exc:
            print(f"[ERR] {ticker}: {exc}")
            ticker_output.append(
                {
                    "ticker_name": ticker,
                    "company_name": None,
                    "last_watermark": None,
                    "ingest_dt_tm": run_ts,
                    "update_dt_tm": run_ts,
                    "error": str(exc),
                }
            )
    return filings_output, ticker_output


def write_results(results: List[dict], output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote {len(results)} entries to {output_path}")


def enrich_filings_with_summaries(filing_data: dict):
    cik = filing_data.get("cik")
    ticker = filing_data.get("ticker")
    for filing in filing_data.get("filings", []):
        summary_block = {}
        primary_doc = filing.get("primaryDocumentDetected")
        accession = filing.get("accessionNoDashes")
        if not primary_doc or not accession:
            summary_block["error"] = "Primary document not identified."
            filing["analysis"] = summary_block
            continue
        try:
            sections = gather_sections(cik, accession, primary_doc)
            summary_block["sections"] = sections
            if sections:
                summary_block["gptSummary"] = summarize_sections_with_gpt(
                    ticker=ticker,
                    form=filing.get("formType"),
                    filing_date=filing.get("filingDate"),
                    sections=sections,
                )
            else:
                summary_block["warning"] = "No sections extracted from filing document."
        except Exception as exc:
            summary_block["error"] = str(exc)
        filing["analysis"] = summary_block


def gather_sections(cik: str, accession: str, primary_doc: str) -> Dict[str, str]:
    html_text = download_filing_document(cik, accession, primary_doc)
    sections = {}
    for section_key, markers in SUMMARY_SECTIONS.items():
        text = extract_section(html_text, markers)
        if text:
            sections[section_key] = trim_text(text)
    return sections


def trim_text(text: str) -> str:
    clean = text.strip()
    if len(clean) <= SECTION_CHAR_LIMIT:
        return clean
    return clean[:SECTION_CHAR_LIMIT] + "... [truncated]"


def summarize_sections_with_gpt(
    ticker: str,
    form: str,
    filing_date: str,
    sections: Dict[str, str],
) -> str:
    section_chunks = [
        f"{key.upper()}:\n{value}"
        for key, value in sections.items()
    ]
    user_content = (
        f"Ticker: {ticker}\nForm Type: {form}\nFiling Date: {filing_date}\n"
        "Provide a concise summary covering:\n"
        "- Key business or strategic updates\n"
        "- Notable risks highlighted\n"
        "- Liquidity or capital considerations\n"
        "- Actionable insights for investors\n\n"
        "Filing sections:\n\n" + "\n\n".join(section_chunks)
    )
    return call_openai_chat(user_content)


def call_openai_chat(user_content: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert financial analyst. Summarize SEC filings into "
                "concise bullet points with insights, risks, and liquidity notes."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    max_attempts = 5
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=OPENAI_MAX_TOKENS,
                temperature=OPENAI_TEMPERATURE,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = 2 ** attempt
            time.sleep(delay)
    raise last_exc or RuntimeError("OpenAI ChatCompletion failed.")


def flatten_filings_for_output(filing_data: dict, timestamp: str) -> List[dict]:
    rows = []
    filings = filing_data.get("filings", [])
    for idx, filing in enumerate(filings, start=1):
        analysis = filing.get("analysis") or {}
        summary_text = analysis.get("gptSummary") or analysis.get("warning") or analysis.get("error")
        rows.append(
            {
                "number": idx,
                "ticker": filing_data.get("ticker"),
                "form_type": filing.get("formType"),
                "filing_date": filing.get("filingDate"),
                "filing_href": filing.get("filingHref"),
                "accession_number": filing.get("accessionNumber"),
                "accession_no_dashes": filing.get("accessionNoDashes"),
                "file_number": filing.get("fileNumber"),
                "film_number": filing.get("filmNumber"),
                "primary_document": filing.get("primaryDocument"),
                "description": filing.get("description"),
                "summary": summary_text,
                "ingest_dt_tm": timestamp,
                "update_dt_tm": timestamp,
            }
        )
    return rows


def current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    input_path = os.getenv("SEC_SUMMARY_INPUT", DEFAULT_INPUT)
    output_path = os.getenv("SEC_SUMMARY_OUTPUT", DEFAULT_OUTPUT)
    metadata_output_path = os.getenv("SEC_SUMMARY_METADATA_OUTPUT", DEFAULT_METADATA_OUTPUT)
    limit = int(os.getenv("SEC_FILINGS_LIMIT", "10"))

    filings_data, ticker_data = run_batch(input_path, limit=limit)

    if output_path:
        write_results(filings_data, output_path)
    else:
        print(json.dumps(filings_data, indent=2))

    if metadata_output_path:
        write_results(ticker_data, metadata_output_path)
