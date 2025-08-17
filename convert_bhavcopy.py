#!/usr/bin/env python3
"""
Convert BhavCopy format to expected reconstruction format
"""

import pandas as pd
import os

def convert_bhavcopy_format(input_file, output_file, symbol="NIFTY"):
    """
    Convert BhavCopy CSV format to expected reconstruction format
    """
    print(f"Converting {input_file} for symbol {symbol}...")
    
    # Read the BhavCopy file
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} total rows")
    
    # Define instrument types for different symbols
    if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
        # Index derivatives
        instrument_types = ['IDO', 'IDF']  # Index Derivative Options/Futures
    else:
        # Stock options
        instrument_types = ['STO']  # Stock Options
    
    # Filter for options and the target symbol
    options_df = df[(df['FinInstrmTp'].isin(instrument_types)) & (df['TckrSymb'] == symbol)]
    print(f"Found {len(options_df)} contracts for {symbol} (types: {instrument_types})")
    
    if len(options_df) == 0:
        print(f"No contracts found for {symbol}. Available symbols: {sorted(df['TckrSymb'].unique())}")
        return None
    
    # For index derivatives, we want options (CE/PE), not futures
    if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
        options_df = options_df[options_df['FinInstrmTp'] == 'IDO']  # Only options, not futures
        print(f"Filtered to {len(options_df)} option contracts (excluding futures)")
    
    # Convert to expected format
    converted_df = pd.DataFrame()
    
    # Map columns
    converted_df['SYMBOL'] = options_df['TckrSymb']
    
    # Convert date format from YYYY-MM-DD to DD-MM-YYYY
    converted_df['EXPIRY_DT'] = pd.to_datetime(options_df['XpryDt']).dt.strftime('%d-%m-%Y')
    
    converted_df['STRIKE_PR'] = options_df['StrkPric']
    converted_df['OPTION_TYP'] = options_df['OptnTp']
    converted_df['SETTLE_PR'] = options_df['SttlmPric']
    converted_df['OPEN_INT'] = options_df['OpnIntrst']
    converted_df['CHG_IN_OI'] = options_df['ChngInOpnIntrst']
    converted_df['VOLUME'] = options_df['TtlTradgVol']
    
    # Convert trade date format from YYYY-MM-DD to DD-MM-YYYY
    converted_df['TRADE_DATE'] = pd.to_datetime(options_df['TradDt']).dt.strftime('%d-%m-%Y')
    
    # Add additional useful columns
    converted_df['UNDERLYING_PRICE'] = options_df['UndrlygPric']
    converted_df['HIGH_PRICE'] = options_df['HghPric']
    converted_df['LOW_PRICE'] = options_df['LwPric']
    converted_df['OPEN_PRICE'] = options_df['OpnPric']
    converted_df['CLOSE_PRICE'] = options_df['ClsPric']
    converted_df['LAST_PRICE'] = options_df['LastPric']
    
    # Clean up data
    converted_df = converted_df.dropna(subset=['OPTION_TYP'])  # Remove rows without option type
    converted_df = converted_df[converted_df['OPTION_TYP'].isin(['CE', 'PE'])]  # Only CE/PE
    
    print(f"Converted {len(converted_df)} valid option contracts")
    
    # Save converted file
    converted_df.to_csv(output_file, index=False)
    print(f"Saved converted data to {output_file}")
    
    return converted_df

def main():
    # Input and output files
    input_file = "reconstructed/raw/BhavCopy_NSE_FO_0_0_0_20250814_F_0000.csv"
    
    # Convert for NIFTY
    output_file = "reconstructed/raw/converted_bhavcopy_NIFTY_20250814.csv"
    converted_df = convert_bhavcopy_format(input_file, output_file, "NIFTY")
    
    if converted_df is not None:
        print("\nSample converted data:")
        print(converted_df[['SYMBOL', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP', 'SETTLE_PR', 'OPEN_INT']].head())
        
        print(f"\nTotal converted contracts: {len(converted_df)}")
        print(f"Expiry dates: {sorted(converted_df['EXPIRY_DT'].unique())}")
        print(f"Strike range: {converted_df['STRIKE_PR'].min()} - {converted_df['STRIKE_PR'].max()}")
        
        print(f"\n✅ Conversion complete! File saved to: {output_file}")
        print("You can now run Mode C using this converted file.")
        
        # Also try BANKNIFTY and FINNIFTY
        print("\n" + "="*50)
        for symbol in ['BANKNIFTY', 'FINNIFTY']:
            output_file = f"reconstructed/raw/converted_bhavcopy_{symbol}_20250814.csv"
            convert_bhavcopy_format(input_file, output_file, symbol)

if __name__ == "__main__":
    main()
