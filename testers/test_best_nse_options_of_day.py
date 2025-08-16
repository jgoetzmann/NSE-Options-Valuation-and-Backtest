#!/usr/bin/env python3
"""
Test Suite for best_nse_options_for_date Function
================================================

This test file comprehensively tests the best_nse_options_for_date function from nse_scraper.py.
It verifies the function's ability to:
- Scrape ALL NSE options data (not just first 1000)
- Calculate theoretical prices using Black-Scholes
- Perform valuation analysis
- Sort options by valuation rating
- Handle various input parameters and edge cases
- Generate proper output files with headers

Test Scenarios:
1. Default behavior (no date specified - uses tomorrow)
2. Specific expiration date
3. Different NSE symbols
4. Error handling for invalid inputs
5. Output file validation
6. Data completeness verification

Usage:
    python testers/test_best_nse_options_of_day.py

Dependencies:
- nse_scraper: The module containing the function to test
- pandas: For data validation
- os: For file operations
- datetime: For date handling

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path to import nse_scraper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from nse_scraper import best_nse_options_for_date
    print("✓ Successfully imported best_nse_options_for_date function")
except ImportError as e:
    print(f"✗ Failed to import best_nse_options_for_date: {e}")
    sys.exit(1)

def test_default_date():
    """Test the function with no date specified (should default to tomorrow)"""
    print("\n" + "="*60)
    print("TEST 1: Default Date (Tomorrow)")
    print("="*60)
    
    try:
        success, output_path, options_count = best_nse_options_for_date("NIFTY")
        
        if success:
            print(f"✓ Test PASSED: Function executed successfully")
            print(f"  - Output file: {output_path}")
            print(f"  - Options analyzed: {options_count}")
            
            # Verify output file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024  # KB
                print(f"  - File size: {file_size:.1f} KB")
                
                # Check if file has proper header
                with open(output_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(10)]
                
                header_found = any("Best NIFTY Options for Expiration:" in line for line in first_lines)
                if header_found:
                    print("  - ✓ Header information properly written")
                else:
                    print("  - ✗ Header information missing or incorrect")
                
                # Verify CSV data
                try:
                    df = pd.read_csv(output_path, comment='#')
                    print(f"  - ✓ CSV data readable: {len(df)} rows")
                    
                    # Check required columns
                    required_cols = ['Ticker', 'Expiration', 'Strike', 'Type', 'LTP', 'IV', 
                                   'Theoretical_Price', 'Valuation_Rating', 'Pct_Difference', 'Confidence']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if not missing_cols:
                        print("  - ✓ All required columns present")
                    else:
                        print(f"  - ✗ Missing columns: {missing_cols}")
                    
                    # Check data quality
                    if len(df) > 0:
                        print(f"  - ✓ Data contains {len(df)} options")
                        print(f"  - ✓ Valuation ratings: {df['Valuation_Rating'].nunique()} unique values")
                        print(f"  - ✓ Strike range: {df['Strike'].min():,.0f} - {df['Strike'].max():,.0f}")
                    else:
                        print("  - ✗ No data rows found")
                        
                except Exception as e:
                    print(f"  - ✗ Error reading CSV data: {e}")
            else:
                print("  - ✗ Output file not created")
                
        else:
            print(f"✗ Test FAILED: Function returned success=False")
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

def test_specific_date():
    """Test the function with a specific expiration date"""
    print("\n" + "="*60)
    print("TEST 2: Specific Expiration Date")
    print("="*60)
    
    # Get tomorrow's date in the required format
    tomorrow = datetime.now() + timedelta(days=1)
    specific_date = tomorrow.strftime("%d-%b-%Y")
    
    print(f"Testing with specific date: {specific_date}")
    
    try:
        success, output_path, options_count = best_nse_options_for_date("NIFTY", specific_date)
        
        if success:
            print(f"✓ Test PASSED: Function executed successfully with specific date")
            print(f"  - Output file: {output_path}")
            print(f"  - Options analyzed: {options_count}")
            
            # Verify the date in the filename
            if specific_date.replace('-', '_') in output_path:
                print("  - ✓ Date properly included in filename")
            else:
                print("  - ✗ Date not properly included in filename")
                
        else:
            print(f"✗ Test FAILED: Function returned success=False")
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

def test_different_symbol():
    """Test the function with a different NSE symbol"""
    print("\n" + "="*60)
    print("TEST 3: Different NSE Symbol (BANKNIFTY)")
    print("="*60)
    
    try:
        success, output_path, options_count = best_nse_options_for_date("BANKNIFTY")
        
        if success:
            print(f"✓ Test PASSED: Function executed successfully with BANKNIFTY")
            print(f"  - Output file: {output_path}")
            print(f"  - Options analyzed: {options_count}")
            
            # Verify the symbol in the filename
            if "BANKNIFTY" in output_path:
                print("  - ✓ Symbol properly included in filename")
            else:
                print("  - ✗ Symbol not properly included in filename")
                
        else:
            print(f"✗ Test FAILED: Function returned success=False")
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "="*60)
    print("TEST 4: Error Handling")
    print("="*60)
    
    # Test with invalid symbol
    print("Testing with invalid symbol...")
    try:
        success, output_path, options_count = best_nse_options_for_date("INVALID_SYMBOL")
        
        if not success:
            print("  - ✓ Correctly handled invalid symbol")
        else:
            print("  - ✗ Should have failed with invalid symbol")
            
    except Exception as e:
        print(f"  - ✓ Exception caught for invalid symbol: {e}")
    
    # Test with invalid date format
    print("Testing with invalid date format...")
    try:
        success, output_path, options_count = best_nse_options_for_date("NIFTY", "invalid-date")
        
        if not success:
            print("  - ✓ Correctly handled invalid date format")
        else:
            print("  - ✗ Should have failed with invalid date format")
            
    except Exception as e:
        print(f"  - ✓ Exception caught for invalid date format: {e}")

def test_data_completeness():
    """Test that ALL options are processed, not just first 1000"""
    print("\n" + "="*60)
    print("TEST 5: Data Completeness (ALL Options)")
    print("="*60)
    
    try:
        success, output_path, options_count = best_nse_options_for_date("NIFTY")
        
        if success and output_path and os.path.exists(output_path):
            # Read the output file
            df = pd.read_csv(output_path, comment='#')
            
            print(f"Total options processed: {options_count}")
            print(f"Options in output file: {len(df)}")
            
            # Check if we have a substantial number of options (should be > 1000 for NIFTY)
            if options_count > 1000:
                print("  - ✓ Processing ALL options (not limited to 1000)")
            else:
                print("  - ⚠ Warning: Only processed {options_count} options - may be limited")
            
            # Check for variety in strikes
            unique_strikes = df['Strike'].nunique()
            print(f"Unique strikes: {unique_strikes}")
            
            if unique_strikes > 50:  # NIFTY typically has 100+ strikes
                print("  - ✓ Good variety of strike prices")
            else:
                print("  - ⚠ Limited variety of strike prices")
            
            # Check for both calls and puts
            call_count = len(df[df['Type'] == 'call'])
            put_count = len(df[df['Type'] == 'put'])
            print(f"Call options: {call_count}")
            print(f"Put options: {put_count}")
            
            if call_count > 0 and put_count > 0:
                print("  - ✓ Both call and put options present")
            else:
                print("  - ✗ Missing call or put options")
                
        else:
            print("✗ Cannot test data completeness - function failed")
            
    except Exception as e:
        print(f"✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()

def test_output_directory():
    """Test that output directory is created properly"""
    print("\n" + "="*60)
    print("TEST 6: Output Directory Creation")
    print("="*60)
    
    output_dir = os.path.join('outputs', 'best_nse_options_for_date')
    
    if os.path.exists(output_dir):
        print(f"✓ Output directory exists: {output_dir}")
        
        # List files in directory
        files = os.listdir(output_dir)
        print(f"  - Files in directory: {len(files)}")
        
        if files:
            print("  - Recent files:")
            for file in sorted(files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)[:3]:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"    * {file} ({file_size:.1f} KB, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("  - Directory is empty")
    else:
        print(f"✗ Output directory does not exist: {output_dir}")

def main():
    """Run all tests"""
    print("BEST NSE OPTIONS FOR DATE - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    
    # Run all tests
    test_default_date()
    test_specific_date()
    test_different_symbol()
    test_error_handling()
    test_data_completeness()
    test_output_directory()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print(f"Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main() 