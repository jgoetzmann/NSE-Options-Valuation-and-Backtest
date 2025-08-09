import pandas as pd
import numpy as np
from utils import black_scholes_greeks, black_scholes_price, option_valuation
from datetime import datetime
import sys

def calculate_greeks_and_valuation(row, r=0.063):
    """
    Calculate Greeks and valuation for a single option row.
    Uses Indian risk-free rate of 6.3% by default.
    """
    try:
        # Extract data from row
        S = row['Spot']
        K = row['Strike']
        T = row['T_years']
        sigma = row['IV'] / 100  # Convert percentage to decimal
        market_price = row['LTP']
        option_type = row['Type']
        bid = row['Bid']
        ask = row['Ask']
        
        # Skip if invalid data
        if T <= 0 or sigma <= 0 or market_price <= 0:
            return None, None, None, None
        
        # Calculate Greeks
        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
        
        # Calculate theoretical price
        theoretical_price = black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Perform valuation
        rating, pct_diff, confidence = option_valuation(
            theoretical_price, market_price, S, K, T, sigma, bid, ask, option_type
        )
        
        return greeks, theoretical_price, rating, pct_diff, confidence
        
    except Exception as e:
        print(f"Error processing row: {e}")
        return None, None, None, None, None

def main():
    # Get CSV filename from user
    csv_file = input("Enter CSV filename from nse_scraper.py: ").strip()
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} options from {csv_file}")
        
        # Validate required columns
        required_cols = ['Ticker', 'Expiration', 'Strike', 'Type', 'LTP', 'IV', 'Bid', 'Ask', 'Spot', 'T_years']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            sys.exit(1)
        
        # Process each option
        results = []
        processed = 0
        
        print("\nProcessing options...")
        for idx, row in df.iterrows():
            greeks, theo_price, rating, pct_diff, confidence = calculate_greeks_and_valuation(row)
            
            if greeks is not None:
                # Add results to row
                result_row = row.copy()
                result_row['Theoretical_Price'] = theo_price
                result_row['Valuation_Rating'] = rating
                result_row['Pct_Difference'] = pct_diff
                result_row['Confidence'] = confidence
                result_row['Delta'] = greeks['delta']
                result_row['Gamma'] = greeks['gamma']
                result_row['Theta'] = greeks['theta']
                result_row['Vega'] = greeks['vega']
                result_row['Rho'] = greeks['rho']
                
                results.append(result_row)
                processed += 1
                
                # Progress indicator
                if processed % 50 == 0:
                    print(f"Processed {processed} options...")
        
        if not results:
            print("No valid options found for analysis")
            sys.exit(1)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by percentage difference (best undervalued first)
        results_df = results_df.sort_values('Pct_Difference', ascending=False)
        
        # Save full results
        output_file = f"valuation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        
        # Display top 10 undervalued options
        print(f"\n{'='*80}")
        print("TOP 10 UNDERVALUED OPTIONS:")
        print(f"{'='*80}")
        
        top_10 = results_df.head(10)
        for idx, row in top_10.iterrows():
            print(f"\n{row['Ticker']} {row['Type'].upper()} {row['Strike']} expiring {row['Expiration']}")
            print(f"  Market: ${row['LTP']:.2f} | Theoretical: ${row['Theoretical_Price']:.2f}")
            print(f"  Valuation: {row['Valuation_Rating'].upper()} ({row['Pct_Difference']*100:+.1f}%)")
            print(f"  Confidence: {row['Confidence']:.1%} | IV: {row['IV']:.1f}%")
            print(f"  Greeks: Δ={row['Delta']:.3f}, Γ={row['Gamma']:.4f}, Θ={row['Theta']:.1f}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS:")
        print(f"{'='*80}")
        
        undervalued = results_df[results_df['Pct_Difference'] > 0.05]  # >5% undervalued
        overvalued = results_df[results_df['Pct_Difference'] < -0.05]  # >5% overvalued
        fairly_priced = results_df[(results_df['Pct_Difference'] >= -0.05) & (results_df['Pct_Difference'] <= 0.05)]
        
        print(f"Total options analyzed: {len(results_df)}")
        print(f"Undervalued (>5%): {len(undervalued)} ({len(undervalued)/len(results_df)*100:.1f}%)")
        print(f"Overvalued (>5%): {len(overvalued)} ({len(overvalued)/len(results_df)*100:.1f}%)")
        print(f"Fairly priced (±5%): {len(fairly_priced)} ({len(fairly_priced)/len(results_df)*100:.1f}%)")
        
        if len(undervalued) > 0:
            print(f"\nBest undervalued: {undervalued.iloc[0]['Pct_Difference']*100:+.1f}%")
        if len(overvalued) > 0:
            print(f"Most overvalued: {overvalued.iloc[-1]['Pct_Difference']*100:+.1f}%")
        
        print(f"\nFull results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"File {csv_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 