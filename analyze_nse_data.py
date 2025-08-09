import json
import pandas as pd

def analyze_nse_json(filename):
    """Analyze the NSE JSON file to show full data scope"""
    print(f"Analyzing {filename}...")
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Basic info
    total_strikes = len(data["records"]["data"])
    total_expiries = len(data["records"]["expiryDates"])
    spot_price = data["records"]["underlyingValue"]
    
    print(f"\n{'='*60}")
    print("NSE DATA ANALYSIS")
    print(f"{'='*60}")
    print(f"Spot Price: {spot_price}")
    print(f"Total strikes available: {total_strikes}")
    print(f"Total expiries available: {total_expiries}")
    print(f"Estimated total options: {total_strikes * total_expiries * 2:,} (strikes × expiries × 2 for calls/puts)")
    
    # Show expiries
    print(f"\nAvailable Expiries:")
    for i, exp in enumerate(data["records"]["expiryDates"]):
        print(f"  {i+1:2d}. {exp}")
    
    # Analyze strikes
    strikes = [item["strikePrice"] for item in data["records"]["data"]]
    print(f"\nStrike Analysis:")
    print(f"Strike range: {min(strikes):,} - {max(strikes):,}")
    print(f"Number of unique strikes: {len(set(strikes))}")
    
    # Count options by type for first expiration
    first_expiry = data["records"]["expiryDates"][0]
    print(f"\nOptions for {first_expiry}:")
    
    calls_count = 0
    puts_count = 0
    
    for item in data["records"]["data"]:
        if "CE" in item:
            calls_count += 1
        if "PE" in item:
            puts_count += 1
    
    print(f"Calls available: {calls_count}")
    print(f"Puts available: {puts_count}")
    print(f"Total options for this expiry: {calls_count + puts_count}")
    
    # Show sample data structure
    print(f"\nSample data structure:")
    sample_item = data["records"]["data"][0]
    print(f"Keys in data item: {list(sample_item.keys())}")
    
    if "CE" in sample_item:
        print(f"Call option keys: {list(sample_item['CE'].keys())}")
    if "PE" in sample_item:
        print(f"Put option keys: {list(sample_item['PE'].keys())}")
    
    # File size
    import os
    file_size = os.path.getsize(filename) / (1024*1024)  # MB
    print(f"\nFile size: {file_size:.1f} MB")
    
    return data

if __name__ == "__main__":
    # Analyze the existing JSON file
    analyze_nse_json("test_nse_raw_NIFTY_20250806_192134.json") 