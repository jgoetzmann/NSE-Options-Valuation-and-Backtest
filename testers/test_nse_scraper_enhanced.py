import requests
import json
import time
import pandas as pd
from datetime import datetime
import sys
import os

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
    json_filename = f"test_nse_raw_{symbol}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)
    return json_filename

def process_option_data(data, symbol, expiration, save_raw=True):
    """
    Process option data for a specific symbol and expiration.
    Returns DataFrame with processed options.
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
    
    # Process option data (limit to first 5 strikes for testing)
    rows = []
    strike_count = 0
    max_strikes = 5
    
    for item in data["records"]["data"]:
        if strike_count >= max_strikes:
            break
            
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
        
        strike_count += 1
    
    return pd.DataFrame(rows)

def test_specific_ticker():
    """Test with a specific ticker (NIFTY)"""
    print("="*60)
    print("TEST 1: SPECIFIC TICKER (NIFTY)")
    print("="*60)
    
    symbol = "NIFTY"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Testing {symbol}...")
    
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
    print(f"Available expiries: {all_expiries}")
    
    # Use first available expiration for testing
    expiration = all_expiries[0]
    print(f"Using expiration: {expiration}")
    
    # Process option data
    df = process_option_data(data, symbol, expiration)
    
    if not df.empty:
        # Save CSV
        csv_filename = f"test_nse_specific_{symbol}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"\nResults for {symbol}:")
        print(f"CSV saved: {csv_filename}")
        print(f"Options processed: {len(df)}")
        print(f"Calls: {len(df[df['Type'] == 'call'])}")
        print(f"Puts: {len(df[df['Type'] == 'put'])}")
        print(f"Strike range: {df['Strike'].min()} - {df['Strike'].max()}")
        
        # Show first few rows
        print(f"\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))
    else:
        print(f"No valid options found for {symbol}")

def test_all_tickers():
    """Test with all available tickers"""
    print("\n" + "="*60)
    print("TEST 2: ALL TICKERS")
    print("="*60)
    
    # Available NSE indices (limit to first 2 for testing)
    available_tickers = ["NIFTY", "BANKNIFTY"]  # Limit for testing
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Testing all tickers: {', '.join(available_tickers)}")
    
    # Process each ticker
    all_results = []
    json_files = []
    
    for symbol in available_tickers:
        print(f"\nFetching data for {symbol}...")
        
        # Fetch data from NSE
        data = get_nse_chain(symbol)
        if not data:
            print(f"Failed to fetch data for {symbol}, skipping...")
            continue
        
        # Save raw JSON with timestamp
        json_filename = save_json_with_timestamp(data, symbol, timestamp)
        json_files.append(json_filename)
        print(f"Raw JSON saved: {json_filename}")
        
        # Extract spot price and available expiries
        spot_price = data["records"]["underlyingValue"]
        all_expiries = data["records"]["expiryDates"]
        
        print(f"Spot Price: {spot_price}")
        print(f"Available expiries: {all_expiries}")
        
        # Use first available expiration for testing
        expiration = all_expiries[0]
        print(f"Using expiration: {expiration}")
        
        # Process option data
        df = process_option_data(data, symbol, expiration)
        
        if not df.empty:
            all_results.append(df)
            print(f"Processed {len(df)} options for {symbol}")
        else:
            print(f"No valid options found for {symbol}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined CSV
        csv_filename = f"test_nse_all_tickers_{timestamp}.csv"
        combined_df.to_csv(csv_filename, index=False)
        
        print(f"\n{'='*60}")
        print("COMBINED RESULTS:")
        print(f"{'='*60}")
        print(f"Raw JSON files saved: {len(json_files)}")
        for json_file in json_files:
            print(f"  {json_file}")
        
        print(f"\nCombined CSV saved: {csv_filename}")
        print(f"Total options processed: {len(combined_df)}")
        
        # Show summary by ticker
        print(f"\nBy Ticker:")
        for symbol in combined_df['Ticker'].unique():
            ticker_data = combined_df[combined_df['Ticker'] == symbol]
            calls = len(ticker_data[ticker_data['Type'] == 'call'])
            puts = len(ticker_data[ticker_data['Type'] == 'put'])
            print(f"  {symbol}: {calls} calls, {puts} puts")
        
        # Show overall statistics
        print(f"\nOverall Statistics:")
        print(f"Strike range: {combined_df['Strike'].min()} - {combined_df['Strike'].max()}")
        print(f"IV range: {combined_df['IV'].min():.1f}% - {combined_df['IV'].max():.1f}%")
        print(f"Total calls: {len(combined_df[combined_df['Type'] == 'call'])}")
        print(f"Total puts: {len(combined_df[combined_df['Type'] == 'put'])}")
        
        # Show first few rows
        print(f"\nFirst 5 rows of combined data:")
        print(combined_df.head().to_string(index=False))
        
    else:
        print("No valid option data found for any ticker")

def main():
    """Run both tests"""
    print("ENHANCED NSE SCRAPER TEST")
    print("Testing both specific ticker and all tickers functionality")
    
    # Test 1: Specific ticker
    test_specific_ticker()
    
    # Test 2: All tickers
    test_all_tickers()
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 