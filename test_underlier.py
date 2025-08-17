#!/usr/bin/env python3
"""
Test underlier data fetching
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

def test_underlier_fetch():
    # Test the same logic as the reconstruction module
    symbol = "NIFTY"
    underlier_symbol = "^NSEI"  # NIFTY index
    
    # Test date
    test_date = datetime(2025, 8, 14).date()
    start_date = test_date - timedelta(days=5)
    end_date = test_date + timedelta(days=5)
    
    print(f"Fetching {underlier_symbol} data from {start_date} to {end_date}")
    
    try:
        underlier_data = yf.download(
            underlier_symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        print(f"Downloaded data shape: {underlier_data.shape}")
        print(f"Columns: {underlier_data.columns.tolist()}")
        print(f"Index: {underlier_data.index}")
        
        if not underlier_data.empty:
            print("\nSample data:")
            print(underlier_data.head())
            
            # Test the close price extraction
            underlier_closes = underlier_data['Close']
            print(f"\nClose prices shape: {underlier_closes.shape}")
            print(f"Close prices type: {type(underlier_closes)}")
            print(f"Sample close prices:")
            print(underlier_closes.head())
            
            # Test mapping
            test_df = pd.DataFrame({
                'date_t': [test_date],
                'expiry_date': [test_date + timedelta(days=30)]
            })
            
            print(f"\nTest DataFrame:")
            print(test_df)
            
            # Try mapping
            test_df['S_t'] = test_df['date_t'].map(underlier_closes)
            test_df['S_T'] = test_df['expiry_date'].map(underlier_closes)
            
            print(f"\nAfter mapping:")
            print(test_df)
            
        else:
            print("No data downloaded")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_underlier_fetch()
