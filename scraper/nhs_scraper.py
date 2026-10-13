"""
NHS Inform Advanced Scraper
"""

import argparse
import time
import re
import logging
from urllib.parse import urljoin
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.nhsinform.scot"
INDEX_URL = "https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/"
FIELDS = ["symptoms", "causes", "warnings", "recommendations"]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "keep-alive",
}

_scraper = cloudscraper.create_scraper(browser={"browser": "chrome", "platform": "windows", "desktop": True})
_scraper.headers.update(HEADERS)

S_PATTERNS = ["symptom", "signs of", "signs and", "how do i know", "how you"]
C_PATTERNS = ["cause", "risk factor", "why does", "what causes", "who is affected"]
W_PATTERNS = ["when to get help", "when to see", "urgent advice", "warning sign",
              "complication", "emergency", "get help", "red flag", "seek help", "diagnosis", "diagnosing"]
R_PATTERNS = ["treatment", "self-help", "self-care", "self care", "living with", "managing",
              "prevention", "medicine", "medication", "how to treat", "outlook", "support"]

def fetch_html(url: str):
    for attempt in range(1, 4):
        try:
            resp = _scraper.get(url, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            logger.warning(f"[{attempt}/3] {url} -> HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"[{attempt}/3] Request failed for {url}: {e}")
        if attempt < 3:
            time.sleep(attempt * 3)
    logger.error(f"Failed to fetch after 3 attempts: {url}")
    return None

def clean_text(text: str) -> str:
    if not text:
        return ""
    junk_patterns = [
        r"Thank\s+You\s+Your\s+feedback\s+has\s+been\s+received",
        r"Was\s+this\s+page\s+helpful\??",
        r"Share\s+this\s+page",
        r"Last\s+updated:?\s*\d{1,2}\s+\w+\s+\d{4}",
        r"Source:?\s*NHS\s+inform",
    ]
    for pattern in junk_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()

def remove_repeated_runs(text: str, min_run: int = 10) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    words = text.split()
    n = len(words)
    out = []
    ngram_index = {}
    i = 0
    while i < n:
        match_start, match_len = None, 0
        if i + min_run <= n:
            key = tuple(words[i:i + min_run])
            for s in ngram_index.get(key, []):
                L = min_run
                while i + L < n and s + L < len(out) and words[i + L] == out[s + L]:
                    L += 1
                if L > match_len:
                    match_len, match_start = L, s
        if match_start is not None:
            i += match_len
        else:
            out.append(words[i])
            pos = len(out) - 1
            if pos - min_run + 1 >= 0:
                new_key = tuple(out[pos - min_run + 1: pos + 1])
                ngram_index.setdefault(new_key, []).append(pos - min_run + 1)
            i += 1
    return " ".join(out)

def clean_field(text: str, passes: int = 3) -> str:
    prev = str(text)
    for _ in range(passes):
        current = remove_repeated_runs(prev)
        if current == prev:
            break
        prev = current
    return prev

def get_index() -> dict:
    soup = fetch_html(INDEX_URL)
    mapping = {}
    if not soup:
        return mapping
    for link in soup.find_all("a", href=True):
        href = link["href"]
        name = link.get_text(strip=True)
        if not name or len(name) < 2 or href.startswith("#"):
            continue
        full_url = urljoin(BASE_URL, href)
        if "/illnesses-and-conditions/" not in full_url:
            continue
        path_segments = [s for s in full_url.replace(BASE_URL, "").strip("/").split("/") if s]
        if len(path_segments) < 3 or path_segments[-1] == "a-to-z":
            continue
        mapping[name] = full_url.rstrip("/") + "/"
    logger.info(f"Index crawl found {len(mapping)} name->url mappings.")
    return mapping

def _find_child_condition_links(soup: BeautifulSoup, page_url: str) -> list:
    prefix = page_url.rstrip("/")
    children, seen = [], set()
    main = soup.find("main") or soup
    for link in main.find_all("a", href=True):
        full = urljoin(BASE_URL, link["href"]).rstrip("/")
        if full.startswith(prefix + "/") and full not in seen and full != prefix:
            seen.add(full)
            children.append(full)
    return children[:6]

def _collect_section_content(heading_tag) -> str:
    heading_level = int(heading_tag.name[1])
    parts = []
    for sibling in heading_tag.find_next_siblings():
        if sibling.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if int(sibling.name[1]) <= heading_level:
                break
        if sibling.name in ("p", "ul", "ol", "dl", "blockquote", "table"):
            text = clean_text(sibling.get_text())
            if text:
                parts.append(text)
        elif sibling.name == "div":
            for child in sibling.find_all(["p", "ul", "ol", "dl", "blockquote"], recursive=True):
                text = clean_text(child.get_text())
                if text:
                    parts.append(text)
    return "\n".join(parts)

def _extract_care_cards(soup: BeautifulSoup) -> str:
    parts = []
    for selector in ["[class*='care-card']", "[class*='callout']", "[class*='urgent']",
                      "[class*='warning']", "[class*='alert']", "[class*='important']"]:
        for el in soup.select(selector):
            text = clean_text(el.get_text())
            if text and len(text) > 20:
                parts.append(text)
    return "\n".join(parts)

def parse_condition_page(soup: BeautifulSoup, name: str, url: str, allow_hub_follow=True) -> dict:
    for selector in ["[class*='feedback']", "[class*='rating']", "[class*='share']",
                      "[id*='feedback']", "nav", "header", "footer",
                      "[class*='breadcrumb']", "[class*='cookie']", "[class*='banner']"]:
        for el in soup.select(selector):
            el.decompose()

    h1 = soup.find("h1")
    title = clean_text(h1.get_text()) if h1 else name

    overview_parts = []
    if h1:
        for sibling in h1.find_next_siblings():
            if sibling.name and sibling.name.startswith("h"):
                break
            if sibling.name in ("p", "ul", "ol"):
                text = clean_text(sibling.get_text())
                if text:
                    overview_parts.append(text)
    overview = "\n".join(overview_parts)

    symptoms, causes, warnings, recommendations = "", "", "", ""
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = heading.get_text().lower().strip()
        section_content = _collect_section_content(heading)
        if not section_content:
            continue
        if any(p in heading_text for p in S_PATTERNS):
            symptoms = (symptoms + "\n" + section_content).strip() if symptoms else section_content
        elif any(p in heading_text for p in C_PATTERNS):
            causes = (causes + "\n" + section_content).strip() if causes else section_content
        elif any(p in heading_text for p in W_PATTERNS):
            warnings = (warnings + "\n" + section_content).strip() if warnings else section_content
        elif any(p in heading_text for p in R_PATTERNS):
            recommendations = (recommendations + "\n" + section_content).strip() if recommendations else section_content

    if not symptoms and overview:
        symptoms = overview

    care_card_text = _extract_care_cards(soup)
    if care_card_text and care_card_text not in warnings:
        warnings = (warnings + "\n" + care_card_text).strip() if warnings else care_card_text

    is_hub_page = False
    body_word_count = len(soup.get_text().split())
    if allow_hub_follow and not any([symptoms, causes, warnings, recommendations]) and body_word_count < 400:
        children = _find_child_condition_links(soup, url)
        if children:
            is_hub_page = True
            merged = {"symptoms": symptoms, "causes": causes, "warnings": warnings, "recommendations": recommendations}
            for child_url in children:
                child_soup = fetch_html(child_url)
                if not child_soup:
                    continue
                cd = parse_condition_page(child_soup, name, child_url, allow_hub_follow=False)
                for field in FIELDS:
                    if cd[field]:
                        merged[field] = (merged[field] + "\n[" + cd["condition"] + "] " + cd[field]).strip() if merged[field] else "[" + cd["condition"] + "] " + cd[field]
            symptoms, causes, warnings, recommendations = merged["symptoms"], merged["causes"], merged["warnings"], merged["recommendations"]

    symptoms, causes, warnings, recommendations = (
        clean_field(symptoms) if symptoms else "",
        clean_field(causes) if causes else "",
        clean_field(warnings) if warnings else "",
        clean_field(recommendations) if recommendations else "",
    )

    return {
        "condition": title,
        "symptoms": symptoms or None,
        "causes": causes or None,
        "warnings": warnings or None,
        "recommendations": recommendations or None,
        "url": url,
        "is_hub_page": is_hub_page,
    }

def run_full_scrape(output_path: str):
    name_to_url = get_index()
    if not name_to_url:
        logger.error("Failed to load the A-Z index - aborting.")
        return

    records = []
    total = len(name_to_url)
    for i, (name, url) in enumerate(name_to_url.items(), 1):
        logger.info(f"[{i}/{total}] Scraping: {name} ...")
        soup = fetch_html(url)
        if soup:
            records.append(parse_condition_page(soup, name, url))
        else:
            logger.error(f"  Failed to scrape {name}")

        if i % 10 == 0:
            pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"  [Progress] saved {len(records)}/{total} to '{output_path}'")
        time.sleep(1.5)

    pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Done. Saved {len(records)} records to '{output_path}'.")

def run_fill_missing(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    for col in FIELDS:
        if col not in df.columns:
            df[col] = None
    if "url" not in df.columns:
        df["url"] = None
    if "is_hub_page" not in df.columns:
        df["is_hub_page"] = False

    to_fix = df[df[FIELDS].isnull().any(axis=1)]
    logger.info(f"{len(to_fix)} / {len(df)} rows have at least one missing field. Re-crawling index for URLs...")
    name_to_url = get_index()
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

    for i, (idx, row) in enumerate(to_fix.iterrows(), 1):
        name = row["condition"]
        url = name_to_url.get(name)
        if not url:
            for cand_name, cand_url in name_to_url.items():
                if norm(cand_name) == norm(name):
                    url = cand_url
                    break
        if not url:
            logger.warning(f"[{i}/{len(to_fix)}] No URL match found for '{name}', skipping.")
            continue

        logger.info(f"[{i}/{len(to_fix)}] Re-scraping: {name} ({url})")
        soup = fetch_html(url)
        if not soup:
            continue
        data = parse_condition_page(soup, name, url)

        for field in FIELDS:
            if pd.isnull(df.at[idx, field]) and data[field]:
                df.at[idx, field] = data[field]
        df.at[idx, "url"] = url
        df.at[idx, "is_hub_page"] = data["is_hub_page"]

        if i % 10 == 0:
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info(f"  [Progress] saved {i}/{len(to_fix)} to '{output_path}'")
        time.sleep(1.5)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    still_missing = df[df[FIELDS].isnull().any(axis=1)]
    logger.info(f"Done. Saved to '{output_path}'. {len(still_missing)} rows still missing a field.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "fill"], default="fill")
    ap.add_argument("--input", default="nhs_conditions.csv")
    ap.add_argument("--output", default="nhs_conditions.csv")
    args = ap.parse_args()

    if args.mode == "full":
        run_full_scrape(args.output)
    else:
        run_fill_missing(args.input, args.output)

if __name__ == "__main__":
    main()