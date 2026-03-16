import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.a-gc.com"
SOLD_URL = f"{BASE_URL}/vehicles/sold"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}

OUTPUT_PATH = Path("data") / "avant_garde_listings_raw.csv"


@dataclass
class Listing:
    listing_url: str
    title: str
    year: Optional[int]
    make: Optional[str]
    model: Optional[str]
    miles: Optional[int]
    vin: Optional[str]
    stock_number: Optional[str]
    description: str


def fetch_html(url: str) -> Optional[str]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        print(f"[WARN] {url} returned {resp.status_code}")
        return None
    return resp.text


def parse_sold_page(html: str) -> List[str]:
    """
    Extract vehicle detail URLs from a sold-inventory page.
    """
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        text = (a.get_text(strip=True) or "")
        href = a["href"]
        if text.startswith("Sold") and "/vehicles/" in href:
            # Normalize to absolute URL
            if href.startswith("/"):
                links.append(BASE_URL + href)
            elif href.startswith("http"):
                links.append(href)
    return sorted(set(links))


def _extract_spec_value(soup: BeautifulSoup, label: str) -> Optional[str]:
    """
    Best-effort extraction of a value that follows a given label in the Specs area.
    """
    # Look for 'Specs' header, then search within its next sibling container.
    specs_header = soup.find(lambda tag: tag.name in ("h2", "h3") and "Specs" in tag.get_text())
    if specs_header:
        container = specs_header.find_next_sibling()
    else:
        container = soup

    # Try structured dt/dd or th/td layouts first.
    for lbl in container.find_all(["dt", "th"]):
        if label.lower() in lbl.get_text(strip=True).lower():
            val = lbl.find_next_sibling(["dd", "td"])
            if val:
                return val.get_text(strip=True)

    # Fallback: simple text search.
    text = container.get_text(" ", strip=True)
    lower = text.lower()
    idx = lower.find(label.lower())
    if idx != -1:
        snippet = text[idx : idx + 80]
        parts = snippet.split(":")
        if len(parts) > 1:
            return parts[1].split()[0]
    return None


def parse_listing(url: str) -> Optional[Listing]:
    html = fetch_html(url)
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else ""

    year_str = _extract_spec_value(soup, "Year")
    make = _extract_spec_value(soup, "Make")
    model = _extract_spec_value(soup, "Model")
    miles_str = _extract_spec_value(soup, "Miles")
    vin = _extract_spec_value(soup, "Vin") or _extract_spec_value(soup, "VIN")
    stock = _extract_spec_value(soup, "Stock")

    def to_int(s: Optional[str]) -> Optional[int]:
        if not s:
            return None
        # Strip commas and non-digits
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None

    year = to_int(year_str)
    miles = to_int(miles_str)

    # Description: text under a 'Description' header.
    desc = ""
    desc_header = soup.find(lambda tag: tag.name in ("h2", "h3") and "Description" in tag.get_text())
    if desc_header:
        desc_container = desc_header.find_next_sibling()
        if desc_container:
            desc = desc_container.get_text(" ", strip=True)

    return Listing(
        listing_url=url,
        title=title,
        year=year,
        make=make,
        model=model,
        miles=miles,
        vin=vin,
        stock_number=stock,
        description=desc,
    )


def scrape_sold_inventory(max_pages: int = 42) -> List[Listing]:
    """
    Scrape a subset of sold inventory pages for testing.

    Increase max_pages as needed once the flow is validated.
    """
    listings: List[Listing] = []
    seen_urls = set()

    for page in range(1, max_pages + 1):
        url = SOLD_URL if page == 1 else f"{SOLD_URL}?page={page}"
        print(f"[INFO] Fetching sold page {page}: {url}")
        html = fetch_html(url)
        if not html:
            break

        detail_urls = parse_sold_page(html)
        print(f"[INFO] Found {len(detail_urls)} detail URLs on page {page}")

        for detail_url in detail_urls:
            if detail_url in seen_urls:
                continue
            seen_urls.add(detail_url)

            print(f"  [INFO] Scraping listing: {detail_url}")
            listing = parse_listing(detail_url)
            if listing:
                listings.append(listing)
            time.sleep(0.3)  # Be polite to the host without being too slow

        time.sleep(0.5)

    return listings


def main() -> None:
    listings = scrape_sold_inventory(max_pages=3)
    print(f"[INFO] Total listings scraped: {len(listings)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(l) for l in listings])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved raw listings to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

