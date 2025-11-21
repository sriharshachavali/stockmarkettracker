import os
import random
import time

import requests
import pandas as pd
from bs4 import BeautifulSoup

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "YourName SEC Research Script (email@example.com)",
)
print (f"Using SEC_USER_AGENT: {SEC_USER_AGENT}")
HEADERS = {"User-Agent": SEC_USER_AGENT}
RETRY_STATUS_CODES = {403, 408, 429, 500, 502, 503, 504}


def get_with_backoff(
    url: str,
    headers=None,
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: int = 30,
):
    """
    Requests helper that retries with exponential backoff for throttle or transient errors.
    """
    headers = headers or HEADERS
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code in RETRY_STATUS_CODES:
                raise requests.HTTPError(
                    f"Retryable status {resp.status_code} for {url}", response=resp
                )
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            last_exc = exc
        except requests.RequestException as exc:
            last_exc = exc

        if attempt == max_attempts - 1:
            break

        delay = initial_delay * (backoff_factor ** attempt) + random.uniform(0, 0.5)
        time.sleep(delay)

    raise last_exc or RuntimeError(f"Request to {url} failed after {max_attempts} attempts.")


# ---------------------------------------------------------
# 1. TICKER → CIK LOOKUP
# ---------------------------------------------------------
def get_cik_from_ticker(ticker: str) -> str:
    url = f"https://efts.sec.gov/api/search?keys={ticker}"
    r = get_with_backoff(url, headers=HEADERS)
    data = r.json()

    if not data.get("hits"):
        raise ValueError(f"No SEC results found for ticker {ticker}")

    cik = str(data["hits"][0]["cik_str"]).zfill(10)
    return cik


# ---------------------------------------------------------
# 2. GET SUBMISSIONS JSON
# ---------------------------------------------------------
def get_submissions(cik: str):
    url = f"https://data.sec.gov/submissions/{cik}.json"
    r = get_with_backoff(url, headers=HEADERS)
    return r.json()


# ---------------------------------------------------------
# 3. EXTRACT + SORT FILINGS
# ---------------------------------------------------------
def extract_recent_filings(sub_data, cik: str, limit: int = 10):
    recent = sub_data["filings"]["recent"]
    filings = []

    n = len(recent["form"])
    for i in range(n):
        acc = recent["accessionNumber"][i]
        acc_no_dash = acc.replace("-", "")
        primary_doc = recent["primaryDocument"][i]

        filing = {
            "formType": recent["form"][i],
            "filingDate": recent["filingDate"][i],
            "filingHref": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{primary_doc}",
            "accessionNumber": acc,
            "accessionNoDashes": acc_no_dash,
            "fileNumber": recent["fileNumber"][i] if "fileNumber" in recent else None,
            "filmNumber": recent["filmNumber"][i] if "filmNumber" in recent else None,
            "primaryDocument": primary_doc,
            "description": recent["description"][i] if "description" in recent else None,
            "size": recent["size"][i] if "size" in recent else None,
        }

        filings.append(filing)

    filings_sorted = sorted(filings, key=lambda x: x["filingDate"], reverse=True)
    return filings_sorted[:limit]


# ---------------------------------------------------------
# 4. GET FILING INDEX (EXHIBITS & DOCUMENTS)
# ---------------------------------------------------------
def get_filing_index(cik: str, accession_no_dashes: str):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/index.json"
    r = get_with_backoff(url, headers=HEADERS)
    return r.json()


# ---------------------------------------------------------
# 5. IDENTIFY LARGEST TEXT/HTML DOC (PRIMARY)
# ---------------------------------------------------------
def get_primary_document(index_json):
    docs = index_json.get("directory", {}).get("item", [])
    if not docs:
        return None

    # Sort by size descending
    docs_sorted = sorted(docs, key=lambda d: int(d.get("size", 0)), reverse=True)

    # Find first .htm or .txt
    for d in docs_sorted:
        name = d["name"].lower()
        if name.endswith(".htm") or name.endswith(".html") or name.endswith(".txt"):
            return d["name"]

    return docs_sorted[0]["name"]


# ---------------------------------------------------------
# 6. DOWNLOAD FILING DOCUMENT
# ---------------------------------------------------------
def download_filing_document(cik, accession, filename):
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filename}"
    r = get_with_backoff(url, headers=HEADERS)
    return r.text


# ---------------------------------------------------------
# 7. EXTRACT SECTIONS (Risk Factors, MD&A, Liquidity, etc.)
# ---------------------------------------------------------
def extract_section(html_text, section_names):
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text("\n")

    section_markers = [s.upper() for s in section_names]

    lines = text.split("\n")
    extracted = []
    capture = False

    for line in lines:
        upper_line = line.strip().upper()

        # Start capture when we hit a section title
        if any(s in upper_line for s in section_markers):
            capture = True
            extracted = []
            continue

        # Stop when next major header appears
        if capture and upper_line.startswith("ITEM "):
            break

        if capture:
            extracted.append(line)

    return "\n".join(extracted).strip()


# ---------------------------------------------------------
# FULL WORKFLOW WRAPPER
# ---------------------------------------------------------
def get_full_filing_details(ticker: str, limit: int = 10):
    cik = get_cik_from_ticker(ticker)
    sub = get_submissions(cik)
    filings = extract_recent_filings(sub, cik, limit)

    # Add documents + exhibits
    for f in filings:
        index_json = get_filing_index(cik, f["accessionNoDashes"])
        f["index"] = index_json

        primary_doc = get_primary_document(index_json)
        f["primaryDocumentDetected"] = primary_doc

        if primary_doc:
            f["primaryHref"] = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accessionNoDashes']}/{primary_doc}"

    return {
        "ticker": ticker,
        "cik": cik,
        "companyName": sub.get("name"),
        "filings": filings
    }


# ---------------------------------------------------------
# RUN EXAMPLE
# ---------------------------------------------------------
if __name__ == "__main__":
    data = get_full_filing_details("UAMY", limit=10)
    print(data)

    # Make a pandas DataFrame
    df = pd.DataFrame(data["filings"])
    print(df)
