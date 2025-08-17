#!/usr/bin/env python3
"""
Mode C Runner: True EOD Backtest via Reconstructed Past Chains
=============================================================
Simple script to run Mode C and show where to find results.
"""

import os
import sys
import glob
from datetime import datetime

def main():
    print("📊 MODE C: True EOD Backtest via Reconstructed Past Chains")
    print("=" * 70)
    
    # Check if required modules exist
    if not os.path.exists("backtests/run_true_backtest.py"):
        print("❌ True backtest module not found!")
        print("   Make sure you're in the project root directory")
        return
    
    # Check for config
    config_path = "configs/backtest_true.yml"
    if not os.path.exists(config_path):
        print(f"❌ True backtest config file not found: {config_path}")
        print("   Please create the config file first")
        return
    
    print("📋 Backtest Configuration found!")
    
    # Check if we have historical data
    print("\n🔍 Checking for historical data...")
    
    # Look for reconstructed data
    reconstructed_parquet = "reconstructed/parquet/daily_chain_norm.parquet"
    reconstructed_json = glob.glob("reconstructed/json/chain_NIFTY_*.json")
    
    if os.path.exists(reconstructed_parquet):
        print(f"✅ Found reconstructed data: {reconstructed_parquet}")
        data_source = "parquet"
    elif reconstructed_json:
        print(f"✅ Found {len(reconstructed_json)} reconstructed JSON files")
        data_source = "json"
    else:
        print("⚠️  No reconstructed historical data found!")
        print("\n🔧 You need to reconstruct historical chains first:")
        print("   1. Get EOD derivatives data (bhavcopy files)")
        print("   2. Run: python data_pipeline/reconstruct_chain_from_eod.py")
        print("   3. Or use sample data if available")
        
        # Ask if user wants to proceed anyway
        response = input("\n❓ Do you want to proceed anyway? (y/n): ").lower()
        if response != 'y':
            print("   Exiting. Please reconstruct data first.")
            return
        data_source = "none"
    
    # Check config for date span
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        date_span = config.get('date_span', {})
        start_date = date_span.get('start', 'N/A')
        end_date = date_span.get('end', 'N/A')
        
        print(f"\n📅 Backtest Period: {start_date} to {end_date}")
        
    except Exception as e:
        print(f"⚠️  Could not read config dates: {e}")
        start_date = "N/A"
        end_date = "N/A"
    
    print("\n🔄 Running true EOD backtest...")
    print("   This may take 15-60 minutes depending on data size...")
    
    try:
        # Import and run the backtest
        sys.path.append("backtests")
        from run_true_backtest import TrueBacktestRunner
        
        runner = TrueBacktestRunner(config_path)
        
        # Run backtest over the configured date span
        if start_date != "N/A" and end_date != "N/A":
            results = runner.run_backtest(start_date, end_date)
        else:
            # Use default dates if config doesn't specify
            results = runner.run_backtest("2022-01-01", "2024-12-31")
        
        print("\n✅ MODE C COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Show where to find results
        print("\n📊 RESULTS LOCATIONS:")
        print("   📁 Detailed Backtest CSV:")
        csv_files = glob.glob("outputs/csv/true_backtest_*.csv")
        for csv_file in csv_files:
            print(f"      {csv_file}")
        
        print("\n   📄 Performance Summary:")
        summary_files = glob.glob("outputs/results_from_nse_valuations/true_backtest_summary_*.txt")
        for summary_file in summary_files:
            print(f"      {summary_file}")
        
        print("\n   📈 Performance Plots:")
        plot_files = glob.glob("outputs/plots/true_backtest_*.png")
        for plot_file in plot_files:
            print(f"      {plot_file}")
        
        print("\n🎯 KEY METRICS:")
        if 'performance_summary' in results:
            summary = results['performance_summary']
            print(f"   Total PnL: ₹{summary.get('total_pnl', 'N/A')}")
            print(f"   Win Rate: {summary.get('win_rate', 'N/A'):.1%}")
            print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"   Max Drawdown: {summary.get('max_drawdown', 'N/A'):.1%}")
            print(f"   Total Trades: {summary.get('total_trades', 'N/A')}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Check the CSV file for detailed trade-by-trade results")
        print("   2. Review the summary TXT file for key performance metrics")
        print("   3. Examine plots for visual analysis")
        print("   4. Compare with Mode A (synthetic) results")
        print("   5. Use results to validate trading strategies")
        
        print("\n🔍 DATA QUALITY:")
        if 'data_quality' in results:
            quality = results['data_quality']
            print(f"   Total Contracts: {quality.get('total_contracts', 'N/A')}")
            print(f"   Valid Contracts: {quality.get('valid_contracts', 'N/A')}")
            print(f"   Data Coverage: {quality.get('data_coverage', 'N/A'):.1%}")
        
    except Exception as e:
        print(f"\n❌ ERROR running Mode C: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you have historical data in reconstructed/")
        print("   3. Verify the backtest config file is valid")
        print("   4. Ensure the date range in config matches available data")
        
        print("\n📚 HELP:")
        print("   - Check README_NEW_SYSTEM.md for detailed setup instructions")
        print("   - Verify data reconstruction completed successfully")
        print("   - Check config file parameters and date ranges")

if __name__ == "__main__":
    main()
