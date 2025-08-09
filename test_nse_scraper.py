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
        print(f"Error fetching NSE data: {e}")
        return None

def main():
    # Test values
    symbol = "NIFTY"
    
    print(f"Testing NSE scraper for {symbol}...")
    
    # Fetch data from NSE
    data = get_nse_chain(symbol)
    if not data:
        print("Failed to fetch data from NSE")
        sys.exit(1)
    
    # Extract spot price and available expiries
    spot_price = data["records"]["underlyingValue"]
    all_expiries = data["records"]["expiryDates"]
    
    print(f"Spot Price: {spot_price}")
    print(f"Available expiries: {all_expiries}")
    
    # Use first available expiration for testing
    expiration = all_expiries[0]
    print(f"Using expiration: {expiration}")
    
    # Parse expiration date
    expiry_date = datetime.strptime(expiration, "%d-%b-%Y")
    T = (expiry_date - datetime.now()).days / 365
    
    print(f"Processing expiration: {expiration} (T = {T:.3f} years)")
    
    # Process option data (limit to first 10 strikes for testing)
    rows = []
    strike_count = 0
    max_strikes = 10
    
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
    
    # Create DataFrame and save to CSV
    if rows:
        df = pd.DataFrame(rows)
        filename = f"test_nse_options_{symbol}_{expiration.replace('-', '_')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\nTest data saved to {filename}")
        print(f"Total options processed: {len(rows)}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Show summary
        print(f"\nSummary:")
        print(f"Calls: {len(df[df['Type'] == 'call'])}")
        print(f"Puts: {len(df[df['Type'] == 'put'])}")
        print(f"Strike range: {df['Strike'].min()} - {df['Strike'].max()}")
        print(f"IV range: {df['IV'].min():.1f}% - {df['IV'].max():.1f}%")
        
        # Show first few rows
        print(f"\nFirst 5 rows:")
        print(df.head().to_string(index=False))
        
    else:
        print("No valid option data found")

if __name__ == "__main__":
    main() 