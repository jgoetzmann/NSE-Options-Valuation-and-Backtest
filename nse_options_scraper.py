#!/usr/bin/env python3
"""
NSE Options Chain Scraper
=========================

This script scrapes NSE (National Stock Exchange of India) options data and provides:
- Fetches option chain JSON for an index (default: NIFTY) or equity
- Writes raw response to raw_option_chain_<symbol>_<timestamp>.json
- Normalizes to a tidy per-leg (CE/PE) list and writes option_chain_slim_<symbol>_<timestamp>.json
- All outputs are saved to outputs/json/ folder with timestamped filenames

Notes:
- Use reasonable rate limits; respect NSE terms.
- If you see 401/403, rotate UA or slow down; NSE may block aggressive scraping.
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Literal, Optional

import requests


Kind = Literal["indices", "equities"]


class NSEOptionChainScraper:
    BASE = "https://www.nseindia.com"
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

    def __init__(self, user_agent: Optional[str] = None, timeout: int = 20):
        self.s = requests.Session()
        self.timeout = timeout
        self.s.headers.update({
            "User-Agent": user_agent or self.UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        })

    def _bootstrap(self) -> None:
        """Prime cookies (important for NSE API)."""
        # Hit a couple of pages that usually set cookies used by /api/*
        for path in ("", "/option-chain", "/api/marketStatus"):
            url = f"{self.BASE}{'/' if path and not path.startswith('/') else ''}{path}"
            try:
                r = self.s.get(url, timeout=self.timeout)
                r.raise_for_status()
                # brief pause to be polite and mimic a browser
                time.sleep(0.4)
            except requests.RequestException:
                # Keep going; some endpoints might intermittently fail
                pass

    def fetch_option_chain(self, symbol: str, kind: Kind = "indices", max_retries: int = 5) -> Dict[str, Any]:
        """Fetch raw option chain JSON for symbol."""
        self._bootstrap()

        if kind == "indices":
            url = f"{self.BASE}/api/option-chain-indices"
        elif kind == "equities":
            url = f"{self.BASE}/api/option-chain-equities"
        else:
            raise ValueError("kind must be 'indices' or 'equities'")

        params = {"symbol": symbol.upper()}
        # Set Referer to a plausible page
        self.s.headers.update({"Referer": f"{self.BASE}/option-chain"})

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                # 200 OK required; 401/403 can happen if cookies/headers not accepted
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_err = e
                # Small exponential backoff
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"Failed to fetch option chain for {symbol} after {max_retries} attempts: {last_err}")

    @staticmethod
    def _get(d: Dict[str, Any], *keys: str) -> Any:
        """Safe getter with fallbacks for inconsistent field names."""
        for k in keys:
            if k in d:
                return d[k]
        return None

    def normalize(self, raw: Dict[str, Any], symbol: str, kind: Kind) -> List[Dict[str, Any]]:
        """
        Flatten raw['records']['data'] where each item can have CE/PE legs.
        Produces one row per leg (CE or PE) per strike & expiry.
        """
        rows: List[Dict[str, Any]] = []
        records = raw.get("records", {})
        data = records.get("data", []) or []
        underlying_value = records.get("underlyingValue")

        for item in data:
            strike = item.get("strikePrice")
            expiry = item.get("expiryDate")

            for leg_type in ("CE", "PE"):
                if leg_type in item and isinstance(item[leg_type], dict):
                    leg = item[leg_type]

                    row = {
                        "symbol": symbol.upper(),
                        "instrumentType": "INDEX" if kind == "indices" else "EQUITY",
                        "expiry": expiry,
                        "strike": strike,
                        "optionType": leg_type,
                        "lastPrice": self._get(leg, "lastPrice"),
                        "change": self._get(leg, "change"),
                        "percentChange": self._get(leg, "pChange", "percentChange"),
                        "openInterest": self._get(leg, "openInterest"),
                        "changeInOI": self._get(leg, "changeinOpenInterest", "changeInOI"),
                        "impliedVolatility": self._get(leg, "impliedVolatility", "IV"),
                        "totalTradedVolume": self._get(leg, "totalTradedVolume", "totalTradedQty"),
                        "totalTradedValue": self._get(leg, "totalTradedValue"),
                        "bidQty": self._get(leg, "bidQty"),
                        "bidPrice": self._get(leg, "bidprice", "bidPrice"),
                        "askQty": self._get(leg, "askQty"),
                        "askPrice": self._get(leg, "askPrice", "askprice"),
                        "underlyingValue": underlying_value,
                        "timestamp": raw.get("records", {}).get("timestamp") or raw.get("records", {}).get("timeStamp"),
                        "fetchedAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    }
                    rows.append(row)

        return rows


def main():
    ap = argparse.ArgumentParser(description="Scrape NSE option chain JSON and produce raw + normalized JSON files.")
    ap.add_argument("--symbol", "-s", default="NIFTY", help="Symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)")
    ap.add_argument("--kind", "-k", choices=["indices", "equities"], default="indices",
                    help="Use 'indices' for NIFTY/BANKNIFTY/FINNIFTY etc., 'equities' for stocks")
    ap.add_argument("--outdir", "-o", default="outputs/json", help="Output directory (default: outputs/json)")
    ap.add_argument("--user-agent", "-u", default=None, help="Override User-Agent if needed")
    args = ap.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    scraper = NSEOptionChainScraper(user_agent=args.user_agent)
    raw = scraper.fetch_option_chain(args.symbol, kind=args.kind)

    raw_path = f"{args.outdir.rstrip('/')}/raw_option_chain_{args.symbol.upper()}_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"Wrote raw JSON -> {raw_path}")

    slim = scraper.normalize(raw, symbol=args.symbol, kind=args.kind)
    slim_path = f"{args.outdir.rstrip('/')}/option_chain_slim_{args.symbol.upper()}_{timestamp}.json"
    with open(slim_path, "w", encoding="utf-8") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f"Wrote normalized JSON -> {slim_path}")


if __name__ == "__main__":
    main()
