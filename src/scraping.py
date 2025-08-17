# scraping.py

import requests
import time
import random
import sys
import argparse
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from pathlib import Path

# -----------------------
# Config
# -----------------------
BASE_PATTERN = "https://www.zameen.com/Homes/Lahore-1-{}.html"
UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)   # ensure data/ exists

session = requests.Session()
session.headers.update({
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/"
})

# -----------------------
# Helpers
# -----------------------
def get(url, retries=3, timeout=30):
    """GET with retry + rotating UA."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            session.headers["User-Agent"] = random.choice(UA_POOL)
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp
            else:
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_exc = e
        time.sleep(0.8 * attempt + random.uniform(0, 0.6))
    raise last_exc if last_exc else RuntimeError("Unknown request error")

def parse_card(card):
    """Extract fields from a single <article> card."""
    title, link, price, location, area, beds, baths = (None,) * 7

    a_title = card.find("a", title=True, href=True)
    if a_title:
        title = a_title.get("title")
        href = a_title.get("href")
        link = "https://www.zameen.com" + href if href and href.startswith("/") else href

    currency_tag = card.find("span", {"aria-label": "Currency"})
    price_tag = card.find("span", {"aria-label": "Price"})
    if currency_tag and price_tag:
        price = f"{currency_tag.get_text(strip=True)} {price_tag.get_text(strip=True)}"

    loc_tag = card.find("div", {"aria-label": "Location"})
    location = loc_tag.get_text(strip=True) if loc_tag else None

    area_wrap = card.find("span", {"aria-label": "Area"})
    if area_wrap:
        inner = area_wrap.find("span")
        area = inner.get_text(strip=True) if inner else area_wrap.get_text(strip=True)

    beds_tag = card.find("span", {"aria-label": "Beds"})
    baths_tag = card.find("span", {"aria-label": "Baths"})
    beds = beds_tag.get_text(strip=True) if beds_tag else None
    baths = baths_tag.get_text(strip=True) if baths_tag else None

    return {
        "Title": title,
        "Price": price,
        "Location": location,
        "Area": area,
        "Beds": beds,
        "Baths": baths,
        "Link": link
    }

def scrape_page(page_num, base_pattern=BASE_PATTERN):
    url = base_pattern.format(page_num)
    resp = get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.find_all("article")
    return [parse_card(card) for card in cards if any(parse_card(card).values())]

def write_checkpoint(df, tag):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = DATA_DIR / f"zameen_lahore_checkpoint_p{tag}_{ts}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"💾 Checkpoint written: {fname}")

# -----------------------
# Main
# -----------------------
def main(max_pages, stop_after, delay_min, delay_max, retries, checkpoint_every, output_csv):
    all_rows = []
    empty_streak = 0

    for p in range(1, max_pages + 1):
        try:
            rows = scrape_page(p)
        except Exception as e:
            print(f"[Page {p}] Error: {e}", file=sys.stderr)
            rows = []

        if not rows:
            empty_streak += 1
            print(f"[Page {p}] 0 listings (empty_streak={empty_streak})")
        else:
            empty_streak = 0
            all_rows.extend(rows)
            print(f"[Page {p}] +{len(rows)} listings (total={len(all_rows)})")

        if p % checkpoint_every == 0 and all_rows:
            df_ckpt = pd.DataFrame(all_rows).drop_duplicates()
            df_ckpt = df_ckpt.dropna()   # remove rows with missing values
            write_checkpoint(df_ckpt, tag=p)

        if empty_streak >= stop_after:
            print(f"Stopping early after {empty_streak} consecutive empty pages.")
            break

        time.sleep(random.uniform(delay_min, delay_max))

    # Final save (cleaned)
    df = pd.DataFrame(all_rows).drop_duplicates()
    df_clean = df.dropna()   # drop rows with any empty/missing values
    final_path = DATA_DIR / output_csv
    df_clean.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"✅ Done. Saved {len(df_clean)} cleaned rows to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape property listings from zameen.com")
    parser.add_argument("--max-pages", type=int, default=1000, help="Maximum number of pages to scrape")
    parser.add_argument("--stop-after", type=int, default=5, help="Stop after N consecutive empty pages")
    parser.add_argument("--delay-min", type=float, default=1.2, help="Minimum delay between requests")
    parser.add_argument("--delay-max", type=float, default=2.8, help="Maximum delay between requests")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per request")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Write CSV checkpoint every N pages")
    parser.add_argument("--output-csv", type=str, default="zameen_lahore_listings_clean.csv", help="Final cleaned CSV filename")

    args = parser.parse_args()

    main(
        max_pages=args.max_pages,
        stop_after=args.stop_after,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        retries=args.retries,
        checkpoint_every=args.checkpoint_every,
        output_csv=args.output_csv,
    )
