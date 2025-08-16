import json
import pandas as pd
from datetime import datetime
import sys
import os
import glob

OUTPUT_JSON_DIR = os.path.join('outputs', 'json')
OUTPUT_CSV_DIR = os.path.join('outputs', 'csv')

KNOWN_TICKERS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def find_latest_json() -> str:
    candidates = sorted(glob.glob(os.path.join(OUTPUT_JSON_DIR, "*.json")), key=os.path.getmtime, reverse=True)
    if not candidates:
        # Fallback: look in project root
        candidates = sorted(glob.glob("*.json"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else ""

def infer_symbol_from_filename(path: str, default: str = "NIFTY") -> str:
    upper = os.path.basename(path).upper()
    for sym in KNOWN_TICKERS:
        if sym in upper:
            return sym
    return default

def process_all_option_data_from_json(json_filename, symbol, expiration, max_options=None):
    print(f"Loading data from {json_filename}...")
    with open(json_filename, 'r') as f:
        data = json.load(f)

    spot_price = data["records"]["underlyingValue"]
    all_expiries = data["records"]["expiryDates"]

    if expiration not in all_expiries:
        print(f"Expiration {expiration} not found for {symbol}")
        return pd.DataFrame()

    expiry_date = datetime.strptime(expiration, "%d-%b-%Y")
    T = (expiry_date - datetime.now()).days / 365

    print(f"Processing {symbol} expiration: {expiration} (T = {T:.3f} years)")

    rows = []
    total_strikes = len(data["records"]["data"])
    print(f"Total strikes available: {total_strikes}")

    processed_count = 0
    valid_options = 0

    for i, item in enumerate(data["records"]["data"]):
        if i % 100 == 0:
            print(f"Processing strike {i+1}/{total_strikes}... (valid options: {valid_options})")

        if max_options and valid_options >= max_options:
            print(f"Reached limit of {max_options} options, stopping...")
            break

        strike = item["strikePrice"]
        for side, opt_key in [("CE", "call"), ("PE", "put")]:
            if side not in item:
                continue
            opt = item[side]
            if opt.get("lastPrice") is None or opt.get("lastPrice") == 0:
                continue
            try:
                ltp = float(opt["lastPrice"])
                iv = float(opt.get("impliedVolatility") or 0) / 100
                volume = int(opt.get("totalTradedVolume") or 0)
                oi = int(opt.get("openInterest") or 0)
                bid = float(opt.get("bidprice") or 0)
                ask = float(opt.get("askPrice") or 0)
                change = float(opt.get("change") or 0)
                if bid == 0 and ask == 0:
                    continue
                rows.append({
                    "Ticker": symbol,
                    "Expiration": expiration,
                    "Strike": strike,
                    "Type": opt_key,
                    "LTP": round(ltp, 2),
                    "IV": round(iv * 100, 2),
                    "Volume": volume,
                    "OI": oi,
                    "Bid": round(bid, 2),
                    "Ask": round(ask, 2),
                    "Change": round(change, 2),
                    "Spot": round(spot_price, 2),
                    "T_years": round(T, 4)
                })
                valid_options += 1
            except (ValueError, TypeError) as e:
                print(f"Error processing option data: {e}")
                continue
        processed_count += 1

    print(f"Processed {processed_count} strikes, found {valid_options} valid options")
    return pd.DataFrame(rows)

def main():
    print("FULL NSE PROCESSING TEST")
    print("Processing ALL available options from existing JSON file")

    json_filename = find_latest_json()
    if not json_filename:
        print("No JSON files found in outputs/json or project root.")
        sys.exit(1)

    symbol = infer_symbol_from_filename(json_filename)

    # Read available expiries and pick the first by default
    with open(json_filename, 'r') as f:
        data = json.load(f)
    all_expiries = data["records"]["expiryDates"]
    if not all_expiries:
        print("No expiries available in JSON")
        sys.exit(1)
    expiration = all_expiries[0]
    print(f"Using expiration: {expiration}")

    limit_input = input("Enter max options to process (or press Enter for ALL): ").strip()
    max_options = int(limit_input) if limit_input else None
    if max_options:
        print(f"Will process maximum {max_options} options")
    else:
        print("Will process ALL available options")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = process_all_option_data_from_json(json_filename, symbol, expiration, max_options)

    if not df.empty:
        ensure_dir(OUTPUT_CSV_DIR)
        csv_filename = f"full_nse_processed_{symbol}_{timestamp}.csv"
        csv_path = os.path.join(OUTPUT_CSV_DIR, csv_filename)
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*60}")
        print("FULL PROCESSING RESULTS:")
        print(f"{'='*60}")
        print(f"Source JSON: {json_filename}")
        print(f"Output CSV: {csv_path}")
        print(f"Total options processed: {len(df)}")
        print(f"Total strikes processed: {len(df['Strike'].unique())}")

        print(f"\nBy Option Type:")
        calls = df[df['Type'] == 'call']
        puts = df[df['Type'] == 'put']
        print(f"Calls: {len(calls)}")
        print(f"Puts: {len(puts)}")

        print(f"\nStrike Analysis:")
        print(f"Strike range: {df['Strike'].min():,} - {df['Strike'].max():,}")
        print(f"Number of unique strikes: {len(df['Strike'].unique())}")

        print(f"\nVolatility Analysis:")
        print(f"IV range: {df['IV'].min():.1f}% - {df['IV'].max():.1f}%")
        print(f"Average IV: {df['IV'].mean():.1f}%")

        print(f"\nLiquidity Analysis:")
        print(f"Options with volume > 0: {len(df[df['Volume'] > 0])}")
        print(f"Options with OI > 0: {len(df[df['OI'] > 0])}")
        print(f"Average volume: {df['Volume'].mean():.0f}")
        print(f"Average OI: {df['OI'].mean():.0f}")

        print(f"\nMoneyness Analysis:")
        atm_strike = df['Spot'].iloc[0]
        itm_calls = len(df[(df['Type'] == 'call') & (df['Strike'] < atm_strike)])
        otm_calls = len(df[(df['Type'] == 'call') & (df['Strike'] > atm_strike)])
        itm_puts = len(df[(df['Type'] == 'put') & (df['Strike'] > atm_strike)])
        otm_puts = len(df[(df['Type'] == 'put') & (df['Strike'] < atm_strike)])
        atm_options = len(df[df['Strike'] == atm_strike])
        print(f"ATM options (strike = {atm_strike:,.0f}): {atm_options}")
        print(f"ITM calls: {itm_calls}")
        print(f"OTM calls: {otm_calls}")
        print(f"ITM puts: {itm_puts}")
        print(f"OTM puts: {otm_puts}")

        json_size = os.path.getsize(json_filename) / (1024*1024)
        csv_size = os.path.getsize(csv_path) / 1024
        print(f"\nFile sizes:")
        print(f"JSON: {json_size:.1f} MB")
        print(f"CSV: {csv_size:.1f} KB")
    else:
        print("No valid options found")

if __name__ == "__main__":
    main() 