#!/usr/bin/env python3
"""
NSE Options Data Scraper and Processor
=====================================

This module provides comprehensive functionality for scraping and processing NSE (National Stock Exchange of India) 
options data. It's designed to fetch real-time option chain data from NSE's API and process it into structured formats.

Key Features:
- Fetches option chain data for major NSE indices (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)
- Implements robust retry logic with randomized delays to handle rate limiting
- Processes options with configurable limits to avoid overwhelming data
- Saves both raw JSON data and processed CSV files with timestamps
- Handles authentication, session management, and error recovery gracefully
- NEW: best_nse_options_for_date() function for finding best options by valuation

Main Functions:
- get_nse_chain(): Downloads option-chain JSON from nseindia.com with retry logic
- process_all_options_for_date(): Processes options for a specific expiration date with limits
- best_nse_options_for_date(): Finds best options for a date using valuation analysis
- save_json_with_timestamp(): Saves raw JSON data with timestamp
- save_csv_with_timestamp(): Saves processed CSV data with timestamp

Output:
- Raw JSON files saved to outputs/json/ directory
- Processed CSV files saved to outputs/csv/ directory
- Best options CSV files saved to outputs/best_nse_options_for_date/ directory
- CSV includes calculated fields like time to expiry, moneyness, and option characteristics

Usage:
    python nse_scraper.py
    # Follow prompts to select index and expiration date
    
    # Or use the new function directly:
    from nse_scraper import best_nse_options_for_date
    best_nse_options_for_date("NIFTY", "15-Dec-2024")

Dependencies:
- requests: For HTTP requests to NSE API
- pandas: For data processing and CSV operations
- json: For JSON data handling
- datetime: For timestamp generation
- os: For directory operations
- random: For randomized delays to avoid rate limiting
- time: For delay implementation
- utils: For Black-Scholes calculations and valuation

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import random

# =============================================================================
# CONFIGURATION VARIABLES - EASILY MODIFIABLE
# =============================================================================

# Data processing limits - adjust these as needed
MAX_STRIKES_TO_PROCESS = 10        # Maximum number of strike prices to process
MAX_OPTIONS_PER_STRIKE = 10        # Maximum options (calls + puts) per strike
MAX_TOTAL_OPTIONS = 100            # Maximum total options to process

# Alternative: Set to None to process ALL data (use with caution)
# MAX_STRIKES_TO_PROCESS = None    # Process all strikes
# MAX_OPTIONS_PER_STRIKE = None    # Process all options per strike
# MAX_TOTAL_OPTIONS = None         # Process all options

# Data quality filters
MIN_VOLUME_THRESHOLD = 0           # Minimum volume to consider option liquid
MIN_OI_THRESHOLD = 0               # Minimum open interest to consider option liquid
MIN_IV_THRESHOLD = 0.01            # Minimum implied volatility (1%) to consider valid
MIN_BID_ASK_SPREAD = 0             # Minimum bid-ask spread to consider option liquid

# Rate limiting and retry settings
MAX_RETRIES = 3                    # Maximum API retry attempts
DELAY_RANGE = (1, 3)              # Random delay range between requests (seconds)
TIMEOUT_SECONDS = 15               # Request timeout in seconds

# =============================================================================

# Import valuation functions
try:
    from utils import black_scholes_price, black_scholes_greeks, option_valuation
except ImportError:
    print("Warning: utils module not found. Valuation features will be disabled.")
    black_scholes_price = None
    black_scholes_greeks = None
    option_valuation = None

OUTPUT_JSON_DIR = os.path.join('outputs', 'json')
OUTPUT_CSV_DIR = os.path.join('outputs', 'csv')
OUTPUT_BEST_OPTIONS_DIR = os.path.join('outputs', 'best_nse_options_for_date')

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def update_configuration(max_strikes=None, max_options_per_strike=None, max_total_options=None, 
                        min_iv=None, min_volume=None, min_oi=None, min_spread=None):
    """
    Update configuration variables dynamically.
    Call this function to change limits without editing the file.
    
    Examples:
        update_configuration(max_strikes=5, max_total_options=50)  # Process only 5 strikes, max 50 options
        update_configuration(max_strikes=None, max_total_options=None)  # Process all data (unlimited)
    """
    global MAX_STRIKES_TO_PROCESS, MAX_OPTIONS_PER_STRIKE, MAX_TOTAL_OPTIONS
    global MIN_IV_THRESHOLD, MIN_VOLUME_THRESHOLD, MIN_OI_THRESHOLD, MIN_BID_ASK_SPREAD
    
    if max_strikes is not None:
        MAX_STRIKES_TO_PROCESS = max_strikes
    if max_options_per_strike is not None:
        MAX_OPTIONS_PER_STRIKE = max_options_per_strike
    if max_total_options is not None:
        MAX_TOTAL_OPTIONS = max_total_options
    if min_iv is not None:
        MIN_IV_THRESHOLD = min_iv
    if min_volume is not None:
        MIN_VOLUME_THRESHOLD = min_volume
    if min_oi is not None:
        MIN_OI_THRESHOLD = min_oi
    if min_spread is not None:
        MIN_BID_ASK_SPREAD = min_spread
    
    print("Configuration updated:")
    print(f"  Max strikes: {MAX_STRIKES_TO_PROCESS or 'Unlimited'}")
    print(f"  Max options per strike: {MAX_OPTIONS_PER_STRIKE or 'Unlimited'}")
    print(f"  Max total options: {MAX_TOTAL_OPTIONS or 'Unlimited'}")
    print(f"  Min IV: {MIN_IV_THRESHOLD*100:.1f}%")
    print(f"  Min volume: {MIN_VOLUME_THRESHOLD}")
    print(f"  Min OI: {MIN_OI_THRESHOLD}")
    print(f"  Min spread: {MIN_BID_ASK_SPREAD*100:.1f}%")

def get_nse_chain(symbol="NIFTY", max_retries=None, delay_range=None):
    """
    Download option-chain JSON from nseindia.com with retry logic.
    Uses multiple fallback approaches to handle NSE's changing authentication.
    """
    # Use configuration values if not provided
    if max_retries is None:
        max_retries = MAX_RETRIES
    if delay_range is None:
        delay_range = DELAY_RANGE
        
    # Multiple endpoint attempts
    endpoints = [
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}",
        f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}&expiryDate=",
        f"https://www.nseindia.com/get-quotes/derivatives?symbol={symbol}",
    ]
    
    # Multiple header configurations
    header_configs = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept-Language": "en-US,en;q=0.9",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": "https://www.nseindia.com/option-chain",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        },
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Referer": "https://www.nseindia.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
    ]

    session = requests.Session()
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} for {symbol}...")
            
            # Try different endpoints and header combinations
            for endpoint_idx, url in enumerate(endpoints):
                for header_idx, headers in enumerate(header_configs):
                    print(f"  Trying endpoint {endpoint_idx+1}/{len(endpoints)}, headers {header_idx+1}/{len(header_configs)}...")
                    
                    try:
                        # First call to home page sets cookies
                        session.get("https://www.nseindia.com", headers=headers, timeout=10)
                        
                        # Polite pause
                        delay = random.uniform(0.5, 1.0)
                        time.sleep(delay)
                        
                        # Make the API call
                        resp = session.get(url, headers=headers, timeout=10)
                        
                        if resp.status_code == 200:
                            print(f"Successfully fetched data for {symbol} with endpoint {endpoint_idx+1}, headers {header_idx+1}")
                            return resp.json()
                        elif resp.status_code == 401:
                            print(f"  Unauthorized (401) - trying next combination...")
                            continue
                        elif resp.status_code == 429:
                            print(f"  Rate limited (429) - waiting...")
                            time.sleep(random.uniform(5, 10))
                            continue
                        else:
                            print(f"  HTTP {resp.status_code} - trying next combination...")
                            continue
                            
                    except Exception as e:
                        print(f"  Error with this combination: {e}")
                        continue
            
            # If all combinations failed, wait before next attempt
            if attempt < max_retries - 1:
                print(f"All combinations failed, waiting before retry...")
                time.sleep(random.uniform(2, 5))
                    
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))
    
    print(f"Failed to fetch data for {symbol} after trying all combinations")
    return None

def save_json_with_timestamp(data, symbol, timestamp):
    """Save raw JSON data with timestamp under outputs/json"""
    ensure_dir(OUTPUT_JSON_DIR)
    json_filename = f"nse_raw_{symbol}_{timestamp}.json"
    json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    return json_path

def process_all_options_for_date(data, symbol, target_expiration):
    """
    Process options for a specific expiration date with configurable limits.
    Applies data quality filters and respects processing limits.
    """
    if not data:
        return pd.DataFrame()
    
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
    
    # Parse expiration date - handle both DD-MMM-YYYY and DD-MMM-YY formats
    try:
        # Try DD-MMM-YYYY format first
        expiry_date = datetime.strptime(target_expiration, "%d-%b-%Y")
    except ValueError:
        try:
            # Try DD-MMM-YY format
            expiry_date = datetime.strptime(target_expiration, "%d-%b-%y")
        except ValueError:
            print(f"Error: Cannot parse expiration date '{target_expiration}'")
            print("Expected format: DD-MMM-YYYY (e.g., 15-Dec-2025)")
            return pd.DataFrame()
    
    # Calculate time to expiry
    current_time = datetime.now()
    T = (expiry_date - current_time).days / 365
    
    # Validate time to expiry
    if T <= 0:
        print(f"Warning: Expiration date {target_expiration} is in the past or today!")
        print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Expiration time: {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time to expiry: {T:.3f} years ({T*365:.1f} days)")
        
        # Ask user if they want to continue
        response = input("Do you want to continue processing? (y/n): ").strip().lower()
        if response != 'y':
            return pd.DataFrame()
    
    print(f"\nProcessing {symbol} expiration: {target_expiration}")
    print(f"Time to expiry: {T:.3f} years ({T*365:.1f} days)")
    
    # Apply processing limits
    print(f"\nProcessing limits:")
    print(f"  Max strikes: {MAX_STRIKES_TO_PROCESS or 'Unlimited'}")
    print(f"  Max options per strike: {MAX_OPTIONS_PER_STRIKE or 'Unlimited'}")
    print(f"  Max total options: {MAX_TOTAL_OPTIONS or 'Unlimited'}")
    
    # Process option data with limits
    rows = []
    total_strikes = len(data["records"]["data"])
    print(f"Total strikes available: {total_strikes}")
    
    processed_count = 0
    valid_options = 0
    strikes_processed = 0
    
    for i, item in enumerate(data["records"]["data"]):
        # Check strike limit
        if MAX_STRIKES_TO_PROCESS and strikes_processed >= MAX_STRIKES_TO_PROCESS:
            print(f"Reached strike limit ({MAX_STRIKES_TO_PROCESS})")
            break
            
        # Check total options limit
        if MAX_TOTAL_OPTIONS and valid_options >= MAX_TOTAL_OPTIONS:
            print(f"Reached total options limit ({MAX_TOTAL_OPTIONS})")
            break
            
        strike = item["strikePrice"]
        strikes_processed += 1
        
        # Progress indicator every 10 strikes
        if strikes_processed % 10 == 0:
            print(f"Processing strike {strikes_processed}/{min(total_strikes, MAX_STRIKES_TO_PROCESS or total_strikes)}... (valid options: {valid_options})")
            
        # Process both calls and puts
        options_this_strike = 0
        for side, opt_key in [("CE", "call"), ("PE", "put")]:
            if side not in item:
                continue
                
            # Check options per strike limit
            if MAX_OPTIONS_PER_STRIKE and options_this_strike >= MAX_OPTIONS_PER_STRIKE:
                break
                
            opt = item[side]
            
            # Skip illiquid rows based on quality filters
            if opt.get("lastPrice") is None or opt.get("lastPrice") == 0:
                continue
                
            # Apply volume and OI filters
            volume = int(opt.get("totalTradedVolume") or 0)
            oi = int(opt.get("openInterest") or 0)
            if volume < MIN_VOLUME_THRESHOLD or oi < MIN_OI_THRESHOLD:
                continue
            
            # Extract required data
            try:
                ltp = float(opt["lastPrice"])
                iv = float(opt.get("impliedVolatility") or 0) / 100  # Convert % to decimal
                bid = float(opt.get("bidprice") or 0)
                ask = float(opt.get("askPrice") or 0)
                change = float(opt.get("change") or 0)
                
                # Skip if no bid/ask data
                if bid == 0 and ask == 0:
                    continue
                
                # Apply IV filter
                if iv < MIN_IV_THRESHOLD:
                    continue
                
                # Apply bid-ask spread filter
                if ask > 0 and bid > 0:
                    spread = (ask - bid) / ((ask + bid) / 2)  # Percentage spread
                    if spread < MIN_BID_ASK_SPREAD:
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
                options_this_strike += 1
                
            except (ValueError, TypeError) as e:
                print(f"Error processing option data: {e}")
                continue
        
        processed_count += 1
    
    print(f"Processed {strikes_processed} strikes, found {valid_options} valid options")
    print(f"Data quality filters applied:")
    print(f"  Min volume: {MIN_VOLUME_THRESHOLD}")
    print(f"  Min OI: {MIN_OI_THRESHOLD}")
    print(f"  Min IV: {MIN_IV_THRESHOLD*100:.1f}%")
    print(f"  Min bid-ask spread: {MIN_BID_ASK_SPREAD*100:.1f}%")
    
    return pd.DataFrame(rows)

def best_nse_options_for_date(symbol="NIFTY", target_expiration=None, risk_free_rate=0.063):
    """
    Find the best NSE options for a specific expiration date using valuation analysis.
    
    This function:
    1. Scrapes NSE options data for the specified symbol and expiration date (with configurable limits)
    2. Calculates theoretical prices using Black-Scholes model
    3. Performs comprehensive valuation analysis
    4. Sorts options by valuation rating (best undervalued first)
    5. Saves results to outputs/best_nse_options_for_date/ directory
    
    Parameters:
    -----------
    symbol : str
        NSE index symbol (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, SENSEX)
        Default: "NIFTY"
    
    target_expiration : str
        Expiration date in DD-MMM-YYYY format (e.g., "15-Dec-2024")
        If None, defaults to tomorrow's date
        Default: None
    
    risk_free_rate : float
        Risk-free interest rate for Black-Scholes calculations
        Default: 0.063 (6.3% - Indian market standard)
    
    Returns:
    --------
    tuple : (success, output_path, options_count)
        success: bool - Whether the operation was successful
        output_path: str - Path to the output CSV file
        options_count: int - Number of options processed and analyzed
    """
    
    # Check if valuation functions are available
    if not all([black_scholes_price, black_scholes_greeks, option_valuation]):
        print("Error: Valuation functions not available. Please ensure utils.py is accessible.")
        return False, None, 0
    
    # Set default expiration to tomorrow if none provided
    if target_expiration is None:
        tomorrow = datetime.now() + timedelta(days=1)
        target_expiration = tomorrow.strftime("%d-%b-%Y")
        print(f"No expiration date provided. Using tomorrow: {target_expiration}")
    
    print(f"Finding best {symbol} options for expiration: {target_expiration}")
    print("=" * 60)
    
    # Step 1: Fetch NSE data
    print("Step 1: Fetching NSE options data...")
    data = get_nse_chain(symbol)
    if not data:
        print(f"Failed to fetch data for {symbol}")
        return False, None, 0
    
    # Step 2: Process all options for the target date
    print("Step 2: Processing all options for target expiration...")
    df = process_all_options_for_date(data, symbol, target_expiration)
    
    if df.empty:
        print(f"No options found for {symbol} expiration {target_expiration}")
        return False, None, 0
    
    print(f"Found {len(df)} options to analyze")
    
    # Step 3: Calculate theoretical prices and valuation for each option
    print("Step 3: Calculating theoretical prices and valuation...")
    results = []
    processed = 0
    
    for idx, row in df.iterrows():
        try:
            # Extract required data
            S = row['Spot']  # Current spot price
            K = row['Strike']  # Strike price
            T = row['T_years']  # Time to expiry in years
            sigma = row['IV'] / 100  # Implied volatility (convert % to decimal)
            market_price = row['LTP']  # Last traded price
            option_type = row['Type']  # call or put
            bid = row['Bid']  # Bid price
            ask = row['Ask']  # Ask price
            
            # Skip if invalid data
            if T <= 0 or sigma <= 0 or market_price <= 0:
                continue
            
            # Calculate theoretical price using Black-Scholes
            theoretical_price = black_scholes_price(S, K, T, risk_free_rate, sigma, option_type)
            
            # Calculate Greeks
            greeks = black_scholes_greeks(S, K, T, risk_free_rate, sigma, option_type)
            
            # Perform comprehensive valuation
            rating, pct_diff, confidence = option_valuation(
                theoretical_price, market_price, S, K, T, sigma, bid, ask, option_type
            )
            
            # Create result row with all calculated values
            result_row = row.copy()
            result_row['Theoretical_Price'] = round(theoretical_price, 4)
            result_row['Valuation_Rating'] = rating
            result_row['Pct_Difference'] = round(pct_diff * 100, 2)  # Convert to percentage
            result_row['Confidence'] = round(confidence, 3)
            result_row['Delta'] = round(greeks['delta'], 4)
            result_row['Gamma'] = round(greeks['gamma'], 6)
            result_row['Theta'] = round(greeks['theta'], 4)
            result_row['Vega'] = round(greeks['vega'], 4)
            result_row['Rho'] = round(greeks['rho'], 4)
            
            # Add moneyness information
            if option_type == 'call':
                moneyness = 'ITM' if K < S else 'OTM' if K > S else 'ATM'
            else:  # put
                moneyness = 'ITM' if K > S else 'OTM' if K < S else 'ATM'
            result_row['Moneyness'] = moneyness
            
            results.append(result_row)
            processed += 1
            
            # Progress indicator
            if processed % 100 == 0:
                print(f"Processed {processed}/{len(df)} options...")
                
        except Exception as e:
            print(f"Error processing option {idx}: {e}")
            continue
    
    if not results:
        print("No valid options could be analyzed")
        return False, None, 0
    
    print(f"Successfully analyzed {processed} options")
    
    # Step 4: Create results DataFrame and sort by valuation
    print("Step 4: Sorting options by valuation rating...")
    results_df = pd.DataFrame(results)
    
    # Sort by percentage difference (best undervalued first)
    # Also consider confidence level for better ranking
    results_df = results_df.sort_values(['Pct_Difference', 'Confidence'], ascending=[False, False])
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    ensure_dir(OUTPUT_BEST_OPTIONS_DIR)
    
    # Create filename with date in header
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"best_{symbol}_options_{target_expiration.replace('-', '_')}_{timestamp}.csv"
    output_path = os.path.join(OUTPUT_BEST_OPTIONS_DIR, filename)
    
    # Add date information to the CSV header
    with open(output_path, 'w') as f:
        f.write(f"# Best {symbol} Options for Expiration: {target_expiration}\n")
        f.write(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Risk-Free Rate: {risk_free_rate:.1%}\n")
        f.write(f"# Total Options Analyzed: {len(results_df)}\n")
        f.write(f"# Spot Price: {results_df['Spot'].iloc[0]}\n")
        f.write(f"# Time to Expiry: {results_df['T_years'].iloc[0]:.4f} years\n")
        f.write("#" + "="*80 + "\n")
        f.write("\n")
    
    # Append the DataFrame to the file
    results_df.to_csv(output_path, mode='a', index=False)
    
    # Step 6: Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Symbol: {symbol}")
    print(f"Expiration: {target_expiration}")
    print(f"Output file: {output_path}")
    print(f"Total options analyzed: {len(results_df)}")
    print(f"Risk-free rate used: {risk_free_rate:.1%}")
    
    # Show top 10 best options
    print(f"\nTop 10 Best Options (by valuation):")
    print("-" * 40)
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"{idx+1:2d}. {row['Type'].upper()} {row['Strike']:>8,.0f} - "
              f"{row['Valuation_Rating']:20s} "
              f"(+{row['Pct_Difference']:6.1f}%, conf: {row['Confidence']:.2f})")
    
    # Show valuation distribution
    print(f"\nValuation Distribution:")
    print("-" * 40)
    rating_counts = results_df['Valuation_Rating'].value_counts()
    for rating, count in rating_counts.items():
        print(f"{rating:25s}: {count:4d} options")
    
    # Show moneyness distribution
    print(f"\nMoneyness Distribution:")
    print("-" * 40)
    moneyness_counts = results_df['Moneyness'].value_counts()
    for moneyness, count in moneyness_counts.items():
        print(f"{moneyness:8s}: {count:4d} options")
    
    # File size
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"\nOutput file size: {file_size:.1f} KB")
    
    return True, output_path, len(results_df)

def main():
    # Available NSE indices
    available_tickers = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]
    
    print("NSE OPTION SCRAPER - SINGLE DATE")
    print("="*50)
    print("Available NSE indices:", ", ".join(available_tickers))
    
    # Show current configuration
    print(f"\nCurrent Configuration:")
    print(f"  Max strikes to process: {MAX_STRIKES_TO_PROCESS or 'Unlimited'}")
    print(f"  Max options per strike: {MAX_OPTIONS_PER_STRIKE or 'Unlimited'}")
    print(f"  Max total options: {MAX_TOTAL_OPTIONS or 'Unlimited'}")
    print(f"  Min IV threshold: {MIN_IV_THRESHOLD*100:.1f}%")
    print(f"  Min volume threshold: {MIN_VOLUME_THRESHOLD}")
    print(f"  Min OI threshold: {MIN_OI_THRESHOLD}")
    print(f"  Min bid-ask spread: {MIN_BID_ASK_SPREAD*100:.1f}%")
    print(f"  Max retries: {MAX_RETRIES}")
    print(f"  Request timeout: {TIMEOUT_SECONDS}s")
    
    # User input
    symbol = input("\nEnter ticker symbol: ").strip().upper()
    if symbol not in available_tickers:
        print(f"Invalid ticker. Available: {', '.join(available_tickers)}")
        sys.exit(1)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nFetching data for {symbol}...")
    
    # Fetch data from NSE with retry logic
    data = get_nse_chain(symbol)
    if not data:
        print(f"Failed to fetch data for {symbol}")
        sys.exit(1)
    
    # Save raw JSON with timestamp to outputs/json
    json_path = save_json_with_timestamp(data, symbol, timestamp)
    print(f"Raw JSON saved: {json_path}")
    
    # Show available expiries
    all_expiries = data["records"]["expiryDates"]
    print(f"\nAvailable expiries for {symbol}:")
    for i, exp in enumerate(all_expiries):
        print(f"  {i+1:2d}. {exp}")
    
    # Get target expiration from user
    target_expiration = input(f"\nEnter expiration date (DD-MMM-YYYY format): ").strip()
    
    # Process ALL options for the target date
    df = process_all_options_for_date(data, symbol, target_expiration)
    
    if not df.empty:
        # Save CSV under outputs/csv
        ensure_dir(OUTPUT_CSV_DIR)
        csv_filename = f"nse_options_{symbol}_{target_expiration.replace('-', '_')}_{timestamp}.csv"
        csv_path = os.path.join(OUTPUT_CSV_DIR, csv_filename)
        df.to_csv(csv_path, index=False)
        
        print(f"\n{'='*60}")
        print("PROCESSING RESULTS:")
        print(f"{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Expiration: {target_expiration}")
        print(f"Raw JSON: {json_path}")
        print(f"Output CSV: {csv_path}")
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
        json_size = os.path.getsize(json_path) / (1024*1024)  # MB
        csv_size = os.path.getsize(csv_path) / 1024  # KB
        print(f"\nFile sizes:")
        print(f"JSON: {json_size:.1f} MB")
        print(f"CSV: {csv_size:.1f} KB")
        
    else:
        print("No valid options found for the specified date")

if __name__ == "__main__":
    main() 