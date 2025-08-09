import json
import pandas as pd
from datetime import datetime
import sys

def process_all_options_for_date_from_json(json_filename, symbol, target_expiration):
    """
    Process ALL options for a specific expiration date from existing JSON file.
    No API calls - just processes the data we already have.
    """
    print(f"Loading data from {json_filename}...")
    
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    # Extract spot price and available expiries
    spot_price = data["records"]["underlyingValue"]
    all_expiries = data["records"]["expiryDates"]
    
    print(f"Spot Price: {spot_price}")
    print(f"Available expiries: {len(all_expiries)}")
    print(f"Target expiration: {target_expiration}")
    
    # Validate target expiration
    if target_expiration not in all_expiries:
        print(f"Target expiration {target_expiration} not found!")
        print("Available expiries:")
        for i, exp in enumerate(all_expiries):
            print(f"  {i+1:2d}. {exp}")
        return pd.DataFrame()
    
    # Parse expiration date
    expiry_date = datetime.strptime(target_expiration, "%d-%b-%Y")
    T = (expiry_date - datetime.now()).days / 365
    
    print(f"\nProcessing {symbol} expiration: {target_expiration}")
    print(f"Time to expiry: {T:.3f} years ({T*365:.1f} days)")
    
    # Process ALL option data for this expiration
    rows = []
    total_strikes = len(data["records"]["data"])
    print(f"Total strikes available: {total_strikes}")
    
    processed_count = 0
    valid_options = 0
    
    for i, item in enumerate(data["records"]["data"]):
        # Progress indicator every 100 strikes
        if i % 100 == 0:
            print(f"Processing strike {i+1}/{total_strikes}... (valid options: {valid_options})")
            
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
            try:
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
                    "Expiration": target_expiration,
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
                
                valid_options += 1
                
            except (ValueError, TypeError) as e:
                print(f"Error processing option data: {e}")
                continue
        
        processed_count += 1
    
    print(f"Processed {processed_count} strikes, found {valid_options} valid options")
    return pd.DataFrame(rows)

def main():
    """Test processing ALL options for a single date from existing JSON file"""
    print("NSE SINGLE DATE PROCESSING TEST")
    print("Processing ALL options for a specific date from existing JSON file")
    
    # Use existing JSON file
    json_filename = "test_nse_raw_NIFTY_20250806_192134.json"
    symbol = "NIFTY"
    
    # Load data to show available expiries
    with open(json_filename, 'r') as f:
        data = json.load(f)
    
    all_expiries = data["records"]["expiryDates"]
    
    print(f"\nAvailable expiries for {symbol}:")
    for i, exp in enumerate(all_expiries):
        print(f"  {i+1:2d}. {exp}")
    
    # Get target expiration from user
    target_expiration = input(f"\nEnter expiration date (DD-MMM-YYYY format): ").strip()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process ALL options for the target date
    df = process_all_options_for_date_from_json(json_filename, symbol, target_expiration)
    
    if not df.empty:
        # Save CSV
        csv_filename = f"nse_single_date_{symbol}_{target_expiration.replace('-', '_')}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        
        print(f"\n{'='*60}")
        print("SINGLE DATE PROCESSING RESULTS:")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Expiration: {target_expiration}")
        print(f"Source JSON: {json_filename}")
        print(f"Output CSV: {csv_filename}")
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
        print(f"Strike range: {df['Strike'].min():,} - {df['Strike'].max():,}")
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
        
        # Show moneyness analysis
        print(f"\nMoneyness Analysis:")
        atm_strike = df['Spot'].iloc[0]  # Current spot price
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
        
        # Show data quality metrics
        print(f"\nData Quality:")
        print(f"Options with valid IV: {len(df[df['IV'] > 0])}")
        print(f"Options with valid bid/ask: {len(df[(df['Bid'] > 0) & (df['Ask'] > 0)])}")
        print(f"Options with volume: {len(df[df['Volume'] > 0])}")
        print(f"Options with OI: {len(df[df['OI'] > 0])}")
        
        # Show first few rows
        print(f"\nFirst 10 rows:")
        print(df.head(10).to_string(index=False))
        
        # File sizes
        import os
        json_size = os.path.getsize(json_filename) / (1024*1024)  # MB
        csv_size = os.path.getsize(csv_filename) / 1024  # KB
        print(f"\nFile sizes:")
        print(f"JSON: {json_size:.1f} MB")
        print(f"CSV: {csv_size:.1f} KB")
        
    else:
        print("No valid options found for the specified date")

if __name__ == "__main__":
    main() 