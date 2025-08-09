# Simple Black-Scholes Options Analysis

A small toolkit for options analysis that combines Black–Scholes pricing/Greeks, yfinance-based US options lookups, and NSE (India) option-chain scraping and processing.

### What you can do
- Manual US option lookup (yfinance): Greeks and pricing
- Manual US option lookup with valuation vs market
- Scrape NSE option-chain data for an index and a single expiration date
- Process scraped CSVs to compute Greeks, theoretical prices, and valuation
- Explore large NSE datasets from previously saved raw JSON files

---

## Requirements
- Python 3.9+
- pip packages:
  - pandas
  - numpy
  - scipy
  - requests
  - yfinance

Install packages:
```bash
pip install pandas numpy scipy requests yfinance
```

---

## Project structure
```
.
├─ LICENSE
├─ README.md
├─ utils.py                         # Black-Scholes pricing, Greeks, valuation utilities
├─ option_info_manual.py            # Manual US option lookup (Greeks)
├─ option_info_manual_with_valuation.py  # Manual US option lookup + valuation
├─ nse_scraper.py                   # NSE: fetch option-chain and process ALL options for a chosen date
├─ valuation_from_csv.py            # Compute Greeks/theoretical price/valuation from CSV
├─ testers/                         # Test and utility scripts (development)
│  ├─ analyze_nse_data.py
│  ├─ test_full_nse_processing.py
│  ├─ test_single_date_processing.py
│  ├─ test_nse_full_data.py
│  ├─ test_nse_scraper.py
│  └─ test_nse_scraper_enhanced.py
├─ outputs/                         # Output folders (created)
│  ├─ csv/
│  └─ json/
└─ previous_work/
```
Note: Some CSV/JSON test artifacts may currently reside in the project root. The `outputs/csv` and `outputs/json` folders exist and are intended for organizing outputs going forward.

---

## Core module: `utils.py`
The shared financial utilities used across scripts.

- `black_scholes_price(S, K, T, r, sigma, option_type)`
  - Returns the Black–Scholes price for a European call/put
- `black_scholes_greeks(S, K, T, r, sigma, option_type)`
  - Returns a dict with delta, gamma, theta, vega, rho
- `option_valuation(theoretical_price, market_price, S=None, K=None, T=None, sigma=None, bid=None, ask=None, option_type='call', tolerance=0.05)`
  - Enhanced valuation that adjusts tolerance using moneyness, time to expiry, IV, and liquidity (bid/ask spread)
  - Returns `(rating, pct_diff, confidence)`
- `simple_option_valuation(theoretical_price, market_price, tolerance=0.05)`
  - Legacy simple valuation by percent difference

---

## Scripts and how to run them

### 1) Manual US option info (Greeks only)
Interactive; fetches a single option from yfinance and prints Greeks.
```bash
python option_info_manual.py
```
You will be prompted for: ticker, expiration (YYYY-MM-DD), strike, and option type (call/put).

### 2) Manual US option info with valuation
Interactive; prints Greeks, theoretical price, valuation rating, and confidence.
```bash
python option_info_manual_with_valuation.py
```
Same inputs as above. Uses `utils.option_valuation` with additional context (S, K, T, sigma, bid/ask, option type).

### 3) NSE scraper (single date, ALL options)
Fetches NSE option-chain JSON for a chosen index and processes ALL options for a single expiration date.
```bash
python nse_scraper.py
```
- Choose a ticker (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)
- The script shows available expiries; enter one exactly as listed (DD-MMM-YYYY)
- Saves a raw JSON snapshot and a processed CSV (currently into project root)

Notes:
- Includes retry logic, randomized delays, and graceful handling of rate limits.
- If you encounter 401/429 responses, rerun later or use the test scripts with existing JSON files.

### 4) Process valuation from a CSV
Takes a CSV (e.g., from the NSE scraper), computes Greeks and theoretical prices, and applies valuation.
```bash
python valuation_from_csv.py
```
Follow the prompt to provide the CSV path. Outputs top undervalued options in the console and saves a full results CSV.

---

## Test and utility scripts (in `testers/`)
These are provided to validate functionality non-interactively and to work with already saved datasets.

- `test_single_date_processing.py`
  - Reads an existing raw JSON file and processes ALL options for a user-chosen expiration date
- `test_full_nse_processing.py`
  - Processes a large dataset from an existing raw JSON file; helpful for stress testing
- `analyze_nse_data.py`
  - Prints summary stats of a raw NSE JSON file (expiries, strikes, counts)
- `test_nse_scraper.py`, `test_nse_scraper_enhanced.py`, `test_nse_full_data.py`
  - Earlier test harnesses; superseded by the single-date processors above

Run, for example:
```bash
python testers\test_single_date_processing.py
```

---

## Outputs and organization
- Raw JSON snapshots (NSE) and processed CSVs are currently saved in the project root.
- Output folders exist for organizing artifacts:
  - `outputs/json/` for JSON
  - `outputs/csv/` for CSV
- You may move existing artifacts into these folders. Future updates can direct scripts to write there by default.

---

## Known issues and notes
- NSE endpoint can return 401/429 due to bot detection/rate limits. The scraper includes retries and delays but may still fail intermittently.
- Enter expiration dates exactly as displayed (e.g., `07-Aug-2025`).
- When time-to-expiry T is very small, some Greeks can approach zero or become unstable; NaNs may appear near expiry.
- yfinance expirations are dynamic; test scripts attempt to select valid expiries automatically.

---

## Roadmap / TODO
- Direct all JSON/CSV outputs to `outputs/json` and `outputs/csv` respectively by default
- Implement `option_info_auto_scan.py` for scanning a range of expirations/strikes via yfinance
- Add CLI arguments to reduce interactive prompts and enable batch processing
- Harden NSE fetching further (cookie persistence, rotating headers, or official data sources)
- More unit tests for valuation logic and greeks

---

## License
MIT (see LICENSE)
