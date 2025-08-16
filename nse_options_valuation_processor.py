#!/usr/bin/env python3
"""
NSE Options Valuation Processor
==============================

This script processes NSE options data through valuation analysis:
1. Calls nse_options_scraper to get fresh data
2. Loads the slim JSON output
3. Runs valuation analysis for each option using utils.py
4. Saves detailed results to CSV
5. Creates summary report with top over/under valued options
6. Saves summary to TXT file

Dependencies:
- nse_options_scraper.py
- utils.py
- os, json, csv, datetime, subprocess
"""

import os
import json
import csv
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
import math

# Import our utilities
from utils import black_scholes_price, option_valuation, simple_option_valuation


class NSEOptionsValuationProcessor:
    def __init__(self, symbol: str = "NIFTY", kind: str = "indices"):
        self.symbol = symbol.upper()
        self.kind = kind
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        os.makedirs("outputs/csv", exist_ok=True)
        os.makedirs("outputs/results_from_nse_valuations", exist_ok=True)
        
        # Risk-free rate (can be made configurable)
        self.risk_free_rate = 0.06  # 6% annual rate
        
    def call_nse_scraper(self) -> str:
        """Call nse_options_scraper and return the slim JSON file path."""
        print(f"🔄 Calling NSE options scraper for {self.symbol}...")
        
        try:
            # Run the scraper
            result = subprocess.run([
                sys.executable, "nse_options_scraper.py",
                "--symbol", self.symbol,
                "--kind", self.kind
            ], capture_output=True, text=True, check=True)
            
            # Extract the slim JSON file path from output
            for line in result.stdout.split('\n'):
                if "option_chain_slim" in line and ".json" in line:
                    # Extract filename from the output
                    filename = line.split('->')[-1].strip()
                    return filename
            
            raise RuntimeError("Could not find slim JSON file path in scraper output")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error calling NSE scraper: {e}")
            print(f"Stderr: {e.stderr}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
    
    def load_slim_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load the slim JSON file."""
        print(f"📖 Loading slim JSON from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Loaded {len(data)} option records")
            return data
            
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            raise
    
    def calculate_time_to_expiry(self, expiry_str: str) -> float:
        """Calculate time to expiry in years from expiry date string."""
        try:
            # Parse expiry date (format: "29-Dec-2026")
            expiry_date = datetime.strptime(expiry_str, "%d-%b-%Y")
            current_date = datetime.now()
            
            # Calculate difference in days and convert to years
            days_to_expiry = (expiry_date - current_date).days
            years_to_expiry = max(days_to_expiry / 365.25, 0.001)  # Minimum 0.001 years
            
            return years_to_expiry
            
        except Exception as e:
            print(f"⚠️ Warning: Could not parse expiry date '{expiry_str}': {e}")
            return 0.25  # Default to 3 months
    
    def run_valuation(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run valuation analysis for a single option."""
        try:
            # Extract required data
            S = option_data.get('underlyingValue')  # Spot price
            K = option_data.get('strike')  # Strike price
            T = self.calculate_time_to_expiry(option_data.get('expiry', ''))
            sigma = option_data.get('impliedVolatility', 0) / 100  # Convert to decimal
            market_price = option_data.get('lastPrice', 0)
            option_type = option_data.get('optionType', '').lower()
            # Convert CE/PE to call/put
            if option_type == 'ce':
                option_type = 'call'
            elif option_type == 'pe':
                option_type = 'put'
            bid = option_data.get('bidPrice')
            ask = option_data.get('askPrice')
            
            # Skip if missing critical data
            if not all([S, K, T, option_type]):
                return {
                    'valuation_status': 'missing_data',
                    'theoretical_price': None,
                    'rating': 'N/A',
                    'mispricing_pct': None,
                    'confidence': None,
                    'error': 'Missing critical data for valuation'
                }
            
            # Handle missing market price - try to use mid-price from bid/ask
            if market_price <= 0:
                if bid and ask and bid > 0 and ask > 0:
                    market_price = (bid + ask) / 2
                    option_data['lastPrice'] = market_price  # Update the option data
                elif bid and bid > 0:
                    market_price = bid
                    option_data['lastPrice'] = market_price
                elif ask and ask > 0:
                    market_price = ask
                    option_data['lastPrice'] = market_price
                else:
                    return {
                        'valuation_status': 'missing_data',
                        'theoretical_price': None,
                        'rating': 'N/A',
                        'mispricing_pct': None,
                        'confidence': None,
                        'error': 'No market price data available'
                    }
            
            # Skip if implied volatility is 0 or invalid
            if sigma <= 0:
                # Try to use a reasonable default volatility based on moneyness
                if S and K:
                    moneyness = abs(math.log(S / K))
                    # Use higher volatility for OTM options, lower for ITM
                    if option_type == 'call':
                        if K > S:  # OTM call
                            sigma = 0.25  # 25% default
                        else:  # ITM call
                            sigma = 0.20  # 20% default
                    else:  # put
                        if K < S:  # OTM put
                            sigma = 0.25  # 25% default
                        else:  # ITM put
                            sigma = 0.20  # 20% default
                else:
                    sigma = 0.22  # 22% default
                
                # Update the sigma value for the option
                option_data['impliedVolatility'] = sigma * 100  # Convert back to percentage
            
            # Calculate theoretical price using Black-Scholes
            theoretical_price = black_scholes_price(S, K, T, self.risk_free_rate, sigma, option_type)
            
            # Run both valuation models
            enhanced_rating, enhanced_mispricing_pct, enhanced_confidence = option_valuation(
                theoretical_price, market_price, S, K, T, sigma, bid, ask, option_type
            )
            
            simple_rating, simple_mispricing_pct = simple_option_valuation(
                theoretical_price, market_price
            )
            
            return {
                'valuation_status': 'success',
                'theoretical_price': round(theoretical_price, 4),
                'enhanced_rating': enhanced_rating,
                'enhanced_mispricing_pct': round(enhanced_mispricing_pct * 100, 2),  # Convert to percentage
                'enhanced_confidence': round(enhanced_confidence, 3),
                'simple_rating': simple_rating,
                'simple_mispricing_pct': round(simple_mispricing_pct * 100, 2),  # Convert to percentage
                'error': None
            }
            
        except Exception as e:
            return {
                'valuation_status': 'error',
                'theoretical_price': None,
                'rating': 'N/A',
                'mispricing_pct': None,
                'confidence': None,
                'error': str(e)
            }
    
    def process_all_options(self, options_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all options through valuation."""
        print(f"🔍 Running valuation analysis for {len(options_data)} options...")
        
        processed_options = []
        total_options = len(options_data)
        
        for i, option in enumerate(options_data, 1):
            if i % 1000 == 0:
                print(f"   Processed {i}/{total_options} options...")
            
            # Run valuation
            valuation_result = self.run_valuation(option)
            
            # Combine original data with valuation results
            processed_option = {**option, **valuation_result}
            processed_options.append(processed_option)
        
        print(f"✅ Completed valuation analysis for {len(processed_options)} options")
        return processed_options
    
    def save_to_csv(self, processed_options: List[Dict[str, Any]]) -> str:
        """Save processed options to CSV file."""
        csv_filename = f"outputs/csv/nse_options_valuation_{self.symbol}_{self.timestamp}.csv"
        
        print(f"💾 Saving results to CSV: {csv_filename}")
        
        # Define CSV columns
        fieldnames = [
            'symbol', 'expiry', 'strike', 'optionType', 'lastPrice', 'underlyingValue',
            'impliedVolatility', 'openInterest', 'totalTradedVolume', 'bidPrice', 'askPrice',
            'theoretical_price', 'enhanced_rating', 'enhanced_mispricing_pct', 'enhanced_confidence',
            'simple_rating', 'simple_mispricing_pct', 'valuation_status', 'error'
        ]
        
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for option in processed_options:
                    # Extract only the fields we want in CSV
                    row = {field: option.get(field) for field in fieldnames}
                    writer.writerow(row)
            
            print(f"✅ CSV saved successfully: {csv_filename}")
            return csv_filename
            
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")
            raise
    
    def create_summary_report(self, processed_options: List[Dict[str, Any]]) -> str:
        """Create summary report with top over/under valued options."""
        txt_filename = f"outputs/results_from_nse_valuations/nse_valuation_summary_{self.symbol}_{self.timestamp}.txt"
        
        print(f"📝 Creating summary report: {txt_filename}")
        
        try:
            # Filter successful valuations
            successful_options = [
                opt for opt in processed_options 
                if opt.get('valuation_status') == 'success' and opt.get('enhanced_mispricing_pct') is not None
            ]
            
            # Filter out extreme mispricing and focus on more liquid options
            reasonable_options = [
                opt for opt in successful_options 
                if (abs(opt.get('enhanced_mispricing_pct', 0)) < 500 and  # Filter out >500% mispricing
                    opt.get('openInterest', 0) > 0 and  # Has open interest
                    opt.get('totalTradedVolume', 0) > 0)  # Has some trading volume
            ]
            
            # Sort by mispricing percentage (most undervalued first, then most overvalued)
            undervalued = [opt for opt in reasonable_options if opt.get('enhanced_mispricing_pct', 0) > 0]
            overvalued = [opt for opt in reasonable_options if opt.get('enhanced_mispricing_pct', 0) < 0]
            
            # Sort by absolute mispricing percentage
            undervalued.sort(key=lambda x: abs(x.get('enhanced_mispricing_pct', 0)), reverse=True)
            overvalued.sort(key=lambda x: abs(x.get('enhanced_mispricing_pct', 0)), reverse=True)
            
            # Take top 10 from each category
            top_undervalued = undervalued[:10]
            top_overvalued = overvalued[:10]
            
            # Generate summary text
            summary_lines = [
                f"NSE Options Valuation Summary - {self.symbol}",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total options analyzed: {len(processed_options)}",
                f"Successful valuations: {len(successful_options)}",
                f"Failed valuations: {len(processed_options) - len(successful_options)}",
                "",
                "=" * 80,
                "",
                f"TOP UNDERVALUED OPTIONS (Top {len(top_undervalued)}):",
                "-" * 50
            ]
            
            for i, opt in enumerate(top_undervalued, 1):
                summary_lines.append(
                    f"{i:2d}. {opt['optionType']} {opt['strike']} {opt['expiry']} | "
                    f"Market: ₹{opt['lastPrice']} | Fair: ₹{opt['theoretical_price']} | "
                    f"Enhanced: {opt['enhanced_mispricing_pct']:+.2f}% (Conf: {opt['enhanced_confidence']:.1%}) | "
                    f"Simple: {opt['simple_mispricing_pct']:+.2f}%"
                )
            
            summary_lines.extend([
                "",
                f"TOP OVERVALUED OPTIONS (Top {len(top_overvalued)}):",
                "-" * 50
            ])
            
            for i, opt in enumerate(top_overvalued, 1):
                summary_lines.append(
                    f"{i:2d}. {opt['optionType']} {opt['strike']} {opt['expiry']} | "
                    f"Market: ₹{opt['lastPrice']} | Fair: ₹{opt['theoretical_price']} | "
                    f"Enhanced: {opt['enhanced_mispricing_pct']:+.2f}% (Conf: {opt['enhanced_confidence']:.1%}) | "
                    f"Simple: {opt['simple_mispricing_pct']:+.2f}%"
                )
            
            summary_lines.extend([
                "",
                "=" * 80,
                "",
                "VALUATION METHODOLOGY:",
                "",
                "1. THEORETICAL PRICING (Black-Scholes Model):",
                "   - Calculates fair value using Black-Scholes formula for European options",
                "   - Inputs: Spot price, Strike price, Time to expiry, Risk-free rate (6%), Implied volatility",
                "   - Handles both calls and puts with proper mathematical implementation",
                "",
                "2. ENHANCED VALUATION MODEL (Advanced):",
                "   - Multi-factor scoring system beyond simple percentage difference",
                "   - Moneyness Factor: Higher tolerance for OTM options (more uncertainty)",
                "   - Time Decay Factor: Higher tolerance for near-expiry options",
                "   - Volatility Factor: Higher tolerance in high-volatility environments",
                "   - Liquidity Factor: Higher tolerance for wide bid-ask spreads",
                "   - Confidence Score: Indicates reliability based on data quality factors",
                "   - Rating Categories: strongly undervalued → strongly overvalued",
                "",
                "3. SIMPLE VALUATION MODEL (Basic):",
                "   - Traditional percentage-based approach",
                "   - Compares theoretical vs. market price directly",
                "   - Uses 5% tolerance threshold for over/under valuation",
                "   - Rating Categories: undervalued, fairly priced, overvalued",
                "",
                "4. CONFIDENCE SCORING (Enhanced Model Only):",
                "   - Base confidence from sigmoid function around tolerance levels",
                "   - Data quality boost for tight bid-ask spreads (< 5% = +15%, < 10% = +10%)",
                "   - Volume boost for actively traded options (> 100 contracts = +10%)",
                "   - Moneyness boost for ATM options (closer to spot = higher confidence)",
                "   - Time boost for longer-dated options (less time decay noise)",
                "",
                "5. DATA QUALITY FILTERS:",
                "   - Focuses on liquid options with open interest and trading volume",
                "   - Filters out extreme mispricing (>500%) likely due to data errors",
                "   - Uses bid-ask mid-price when last price is unavailable",
                "   - Applies reasonable default volatility when implied vol is missing",
                "",
                "Note: This analysis is for educational purposes only.",
                "Always conduct your own research before making investment decisions.",
                "The enhanced model provides more nuanced analysis but requires quality market data."
            ])
            
            # Write summary to file
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            print(f"✅ Summary report saved: {txt_filename}")
            return txt_filename
            
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            raise
    
    def run_full_analysis(self) -> Tuple[str, str]:
        """Run the complete analysis pipeline."""
        try:
            # Step 1: Call NSE scraper
            slim_json_path = self.call_nse_scraper()
            
            # Step 2: Load slim JSON
            options_data = self.load_slim_json(slim_json_path)
            
            # Step 3: Process all options through valuation
            processed_options = self.process_all_options(options_data)
            
            # Step 4: Save to CSV
            csv_path = self.save_to_csv(processed_options)
            
            # Step 5: Create summary report
            summary_path = self.create_summary_report(processed_options)
            
            return csv_path, summary_path
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main function to run the NSE options valuation processor."""
    print("🚀 NSE Options Valuation Processor")
    print("=" * 50)
    
    # You can modify these parameters as needed
    symbol = "NIFTY"  # or "BANKNIFTY", "RELIANCE", etc.
    kind = "indices"  # or "equities"
    
    try:
        processor = NSEOptionsValuationProcessor(symbol=symbol, kind=kind)
        csv_path, summary_path = processor.run_full_analysis()
        
        print("\n" + "=" * 50)
        print("✅ Done! Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   Summary: {summary_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 