import requests
import json
import time
import pandas as pd
from datetime import datetime
import sys

def get_nse_chain(symbol="NIFTY"):
    """
    Download option-chain JSON from nseindia.com (no API key required).
    Uses spoofed headers to mimic browser requests.
    """
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/option-chain",
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()
    try:
        # First call to home page sets cookies that the JSON endpoint wants
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        time.sleep(0.5)  # polite pause
        resp = session.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching NSE data for {symbol}: {e}")
        return None

def save_json_with_timestamp(data, symbol, timestamp):
    """Save raw JSON data with timestamp"""
    json_filename = f"full_nse_raw_{symbol}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
    return json_filename

def process_all_option_data(data, symbol, expiration):
    """
    Process ALL option data for a specific symbol and expiration.
    No limits - processes every available option.
    """
    if not data:
        return pd.DataFrame()
    
    # Extract spot price and validate expiration
    spot_price = data["records"]["underlyingValue"]
    all_expiries = data["records"]["expiryDates"]
    
    if expiration not in all_expiries:
        print(f"Expiration {expiration} not found for {symbol}")
        return pd.DataFrame()
    
    # Parse expiration date
    expiry_date = datetime.strptime(expiration, "%d-%b-%Y")
    T = (expiry_date - datetime.now()).days / 365
    
    print(f"Processing {symbol} expiration: {expiration} (T = {T:.3f} years)")
    
    # Process ALL option data (no limits)
    rows = []
    total_strikes = len(data["records"]["data"])
    print(f"Total strikes available: {total_strikes}")
    
    for i, item in enumerate(data["records"]["data"]):
        if i % 50 == 0:  # Progress indicator every 50 strikes
            print(f"Processing strike {i+1}/{total_strikes}...")
            
        strike = item["strikePrice"]
        
        # Process both calls and puts
        for side, opt_key in [("CE", "call"), ("PE", "put")]:
            if side not in item:
                continue
                
            opt = item[side]
            
            # Skip illiquid rows
            if opt.get("lastPrice") is None or opt.get("lastPrice") == 0:
                continue
            
            # Extract required data
            ltp = float(opt["lastPrice"])
            iv = float(opt.get("impliedVolatility") or 0) / 100  # Convert % to decimal
            volume = int(opt.get("totalTradedVolume") or 0)
            oi = int(opt.get("openInterest") or 0)
            bid = float(opt.get("bidprice") or 0)
            ask = float(opt.get("askPrice") or 0)
            change = float(opt.get("change") or 0)
            
            # Skip if no bid/ask data
            if bid == 0 and ask == 0:
                continue
            
            rows.append({
                "Ticker": symbol,
                "Expiration": expiration,
                "Strike": strike,
                "Type": opt_key,
                "LTP": round(ltp, 2),
                "IV": round(iv * 100, 2),  # Convert back to percentage
                "Volume": volume,
                "OI": oi,
                "Bid": round(bid, 2),
                "Ask": round(ask, 2),
                "Change": round(change, 2),
                "Spot": round(spot_price, 2),
                "T_years": round(T, 4)
            })
    
    return pd.DataFrame(rows)

def main():
    """Test with full NSE data - no limits"""
    print("FULL NSE DATA TEST")
    print("Processing ALL available options (no limits)")
    
    symbol = "NIFTY"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nFetching ALL data for {symbol}...")
    
    # Fetch data from NSE
    data = get_nse_chain(symbol)
    if not data:
        print(f"Failed to fetch data for {symbol}")
        return
    
    # Save raw JSON with timestamp
    json_filename = save_json_with_timestamp(data, symbol, timestamp)
    print(f"Raw JSON saved: {json_filename}")
    
    # Extract spot price and available expiries
    spot_price = data["records"]["underlyingValue"]
    all_expiries = data["records"]["expiryDates"]
    
    print(f"Spot Price: {spot_price}")
    print(f"Available expiries: {len(all_expiries)}")
    for i, exp in enumerate(all_expiries[:5]):  # Show first 5
        print(f"  {i+1}. {exp}")
    if len(all_expiries) > 5:
        print(f"  ... and {len(all_expiries) - 5} more")
    
    # Use first available expiration for testing
    expiration = all_expiries[0]
    print(f"\nUsing expiration: {expiration}")
    
    # Process ALL option data
    df = process_all_option_data(data, symbol, expiration)
    
    if not df.empty:
        # Save CSV
        csv_filename = f"full_nse_options_{symbol}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"\n{'='*60}")
        print("FULL RESULTS:")
        print(f"{'='*60}")
        print(f"Raw JSON saved: {json_filename}")
        print(f"Full CSV saved: {csv_filename}")
        print(f"Total options processed: {len(df)}")
        print(f"Total strikes processed: {len(df['Strike'].unique())}")
        
        # Show summary by type
        print(f"\nBy Option Type:")
        calls = df[df['Type'] == 'call']
        puts = df[df['Type'] == 'put']
        print(f"Calls: {len(calls)}")
        print(f"Puts: {len(puts)}")
        
        # Show strike range
        print(f"\nStrike Analysis:")
        print(f"Strike range: {df['Strike'].min()} - {df['Strike'].max()}")
        print(f"Number of unique strikes: {len(df['Strike'].unique())}")
        
        # Show IV analysis
        print(f"\nVolatility Analysis:")
        print(f"IV range: {df['IV'].min():.1f}% - {df['IV'].max():.1f}%")
        print(f"Average IV: {df['IV'].mean():.1f}%")
        
        # Show liquidity analysis
        print(f"\nLiquidity Analysis:")
        print(f"Options with volume > 0: {len(df[df['Volume'] > 0])}")
        print(f"Options with OI > 0: {len(df[df['OI'] > 0])}")
        print(f"Average volume: {df['Volume'].mean():.0f}")
        print(f"Average OI: {df['OI'].mean():.0f}")
        
        # Show first few rows
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))
        
        # Show file sizes
        import os
        json_size = os.path.getsize(json_filename) / (1024*1024)  # MB
        csv_size = os.path.getsize(csv_filename) / 1024  # KB
        print(f"\nFile sizes:")
        print(f"JSON: {json_size:.1f} MB")
        print(f"CSV: {csv_size:.1f} KB")
        
    else:
        print("No valid options found")

if __name__ == "__main__":
    main() 