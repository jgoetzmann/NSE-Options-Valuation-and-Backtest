# NSE Options Valuation and Backtest Project
## Professional-Grade Options Analysis Platform

A comprehensive toolkit for options analysis that combines Black–Scholes pricing/Greeks, yfinance-based US options lookups, and NSE (India) option-chain scraping and processing. This project provides enterprise-level options analysis capabilities with real-time data processing, advanced valuation algorithms, and comprehensive testing frameworks.

### 🚀 What You Can Do
- **Manual US Options Analysis**: Interactive Greeks calculation and pricing for individual US options
- **Advanced US Options Valuation**: Sophisticated valuation analysis with confidence scoring and market condition assessment
- **NSE Data Scraping**: Real-time option-chain data collection from National Stock Exchange of India
- **Comprehensive Data Processing**: Process large NSE datasets with professional-grade analysis
- **Advanced Valuation Engine**: Multi-factor valuation considering moneyness, time decay, volatility, and liquidity
- **Professional Testing Suite**: Comprehensive testing frameworks for development and quality assurance

### 🏗️ Project Architecture
The project is built with a modular architecture:
- **Core Engine** (`utils.py`): Black-Scholes calculations, Greeks, and advanced valuation
- **Data Collection** (`nse_options_scraper.py`): NSE data scraping with robust error handling
- **Analysis Tools**: Multiple analysis scripts for different use cases
- **Testing Framework**: Comprehensive test suites for validation and quality assurance
- **Output Management**: Organized CSV and JSON output with timestamping

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
├─ .gitignore
├─ README.md
├─ utils.py                         # Black-Scholes pricing, Greeks, valuation utilities
├─ option_info_manual.py            # Manual US option lookup (Greeks)
├─ option_info_manual_Valuation_processor.py  # Manual US option lookup + valuation
├─ nse_options_scraper.py           # NSE: fetch option-chain and process ALL options for a chosen date
├─ nse_options_valuation_processor.py  # Complete NSE options analysis pipeline with valuation
├─ outputs/                         # Output folders (created)
│  ├─ csv/                         # CSV outputs with timestamped filenames
│  ├─ json/                        # JSON outputs with timestamped filenames
│  └─ results_from_nse_valuations/ # Summary reports from NSE analysis
└─ previous_work/                   # Legacy code and previous iterations
```

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
python option_info_manual_Valuation_processor.py
```
Same inputs as above. Uses `utils.option_valuation` with additional context (S, K, T, sigma, bid/ask, option type).

### 3) NSE scraper (single date, ALL options)
Fetches NSE option-chain JSON for a chosen index and processes ALL options for a single expiration date.
```bash
python nse_options_scraper.py
```
- Choose a ticker (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)
- The script shows available expiries; enter one exactly as listed (DD-MMM-YYYY)
- Saves a raw JSON snapshot and a processed CSV (currently into project root)

Notes:
- Includes retry logic, randomized delays, and graceful handling of rate limits.
- If you encounter 401/429 responses, rerun later or use the test scripts with existing JSON files.

### 4) Complete NSE Options Analysis Pipeline
The flagship feature - automatically scrapes NSE data, runs comprehensive valuation analysis, and generates detailed reports.
```bash
python nse_options_valuation_processor.py
```

**What it does:**
1. **Calls NSE Options Scraper** → Gets fresh data from NSE
2. **Loads Slim JSON** → Processes the normalized options data
3. **Runs Valuation Analysis** → Uses `utils.py` for Black-Scholes pricing and advanced valuation
4. **Saves Detailed Results** → CSV in `outputs/csv/` with timestamped filenames
5. **Creates Summary Report** → Top over/under valued options in `outputs/results_from_nse_valuations/`
6. **Provides Dual Models** → Both enhanced and simple valuation approaches

**Output Files:**
- **CSV**: `outputs/csv/nse_options_valuation_{SYMBOL}_{TIMESTAMP}.csv`
  - Contains all options with valuation results from both models
  - Includes theoretical prices, mispricing percentages, confidence scores
- **Summary TXT**: `outputs/results_from_nse_valuations/nse_valuation_summary_{SYMBOL}_{TIMESTAMP}.txt`
  - Top 10 undervalued options with both valuation models
  - Top 10 overvalued options with both valuation models
  - Comprehensive methodology explanation

**Key Features:**
- **Smart Data Handling**: Automatically handles missing implied volatility and market prices
- **Option Type Conversion**: Converts CE/PE to call/put for Black-Scholes calculations
- **Quality Filtering**: Focuses on liquid options with reasonable mispricing
- **Confidence Boosting**: Multiple factors contribute to higher confidence scores
- **Dual Model Comparison**: Both simple and sophisticated analysis options

---

## Outputs and organization
All outputs are automatically organized into timestamped folders:
- `outputs/json/` - Raw and processed JSON files from NSE scraper
- `outputs/csv/` - Detailed CSV analysis results
- `outputs/results_from_nse_valuations/` - Summary reports and analysis

---

## Known issues and notes
- NSE endpoint can return 401/429 due to bot detection/rate limits. The scraper includes retries and delays but may still fail intermittently.
- Enter expiration dates exactly as displayed (e.g., `07-Aug-2025`).
- When time-to-expiry T is very small, some Greeks can approach zero or become unstable; NaNs may appear near expiry.
- yfinance expirations are dynamic; test scripts attempt to select valid expiries automatically.

---

## Roadmap / TODO
- Add more symbols and indices support for NSE analysis
- Implement portfolio-level analysis and backtesting capabilities
- Add more sophisticated volatility surface modeling
- Enhance confidence scoring with additional market factors
- Add web interface for easier data visualization

---

## License
MIT (see LICENSE)
