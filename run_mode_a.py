#!/usr/bin/env python3
"""
Mode A Runner: Synthetic Backtest on Current Snapshot
====================================================
Simple script to run Mode A and show where to find results.
"""

import os
import sys
import glob
from datetime import datetime

def main():
    print("🚀 MODE A: Synthetic Backtest on Current Snapshot")
    print("=" * 60)
    
    # Find the most recent snapshot
    snapshot_files = glob.glob("outputs/json/option_chain_slim_NIFTY_*.json")
    if not snapshot_files:
        print("❌ No snapshot files found!")
        print("   Run: python nse_options_scraper.py")
        return
    
    # Get the most recent snapshot
    latest_snapshot = max(snapshot_files, key=os.path.getctime)
    print(f"📁 Using snapshot: {os.path.basename(latest_snapshot)}")
    
    # Check if backtest module exists
    if not os.path.exists("backtests/run_synthetic_on_snapshot.py"):
        print("❌ Backtest module not found!")
        print("   Make sure you're in the project root directory")
        return
    
    print("\n🔄 Running synthetic backtest...")
    print("   This may take a few minutes...")
    
    try:
        # Import and run the backtest
        sys.path.append("backtests")
        from run_synthetic_on_snapshot import SyntheticBacktestRunner
        
        # Use default config
        config_path = "configs/backtest_synth.yml"
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found: {config_path}")
            print("   Using default settings...")
            config_path = None
        
        runner = SyntheticBacktestRunner(config_path)
        results = runner.run_synthetic_backtest(latest_snapshot)
        
        print("\n✅ MODE A COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Show where to find results
        print("\n📊 RESULTS LOCATIONS:")
        print("   📁 Detailed Results CSV:")
        csv_files = glob.glob("outputs/csv/synthetic_backtest_*.csv")
        for csv_file in csv_files:
            print(f"      {csv_file}")
        
        print("\n   📄 Performance Summary:")
        summary_files = glob.glob("outputs/results_from_nse_valuations/synthetic_summary_*.txt")
        for summary_file in summary_files:
            print(f"      {summary_file}")
        
        print("\n   📈 Performance Plots:")
        plot_files = glob.glob("outputs/plots/synthetic_*.png")
        for plot_file in plot_files:
            print(f"      {plot_file}")
        
        print("\n🎯 KEY METRICS:")
        if 'performance_summary' in results:
            summary = results['performance_summary']
            print(f"   Total PnL: ₹{summary.get('total_pnl', 'N/A')}")
            print(f"   Win Rate: {summary.get('win_rate', 'N/A'):.1%}")
            print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 'N/A'):.2f}")
        
        print("\n💡 NEXT STEPS:")
        print("   1. Check the CSV file for detailed trade-by-trade results")
        print("   2. Review the summary TXT file for key performance metrics")
        print("   3. Examine plots for visual analysis")
        print("   4. Try different horizons or filters in configs/backtest_synth.yml")
        
        # Generate enhanced results automatically
        print("\n📊 GENERATING ENHANCED RESULTS...")
        try:
            from generate_mode_results import EnhancedModeResultsGenerator
            generator = EnhancedModeResultsGenerator()
            mode_a_report = generator.generate_mode_a_results()
            print(f"✅ Enhanced results generated: {os.path.basename(mode_a_report)}")
        except Exception as e:
            print(f"⚠️  Could not generate enhanced results: {e}")
        
    except Exception as e:
        print(f"\n❌ ERROR running Mode A: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   2. Check that you're in the project root directory")
        print("   3. Verify the snapshot file exists and is valid")

if __name__ == "__main__":
    main()
