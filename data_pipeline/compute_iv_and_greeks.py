#!/usr/bin/env python3
"""
Compute IV and Greeks Module
============================

This module computes implied volatility from option premiums and calculates
all Greeks using the Black-Scholes model from utils.py.

Key Functions:
- invert_implied_volatility(): Find IV that matches market premium
- compute_greeks(): Calculate all Greeks for a contract
- process_options_dataframe(): Apply IV inversion and Greeks to entire DataFrame
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import black_scholes_price, black_scholes_greeks
from data_pipeline.schemas import validate_dataframe_schema


class IVAndGreeksComputer:
    """
    Computes implied volatility and Greeks for options data.
    
    This class handles the inversion of Black-Scholes to find implied volatility
    from market premiums and then calculates all Greeks using the estimated IV.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
        
        # Convergence parameters for IV inversion
        self.max_iterations = 100
        self.tolerance = 1e-6
        
    def invert_implied_volatility(self, S: float, K: float, T: float, 
                                 market_price: float, option_type: str) -> Tuple[float, bool]:
        """
        Invert Black-Scholes to find implied volatility.
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiry in years
        market_price : float
            Observed market price
        option_type : str
            Option type ('CE' or 'PE')
            
        Returns:
        --------
        Tuple[float, bool]
            Estimated IV and convergence success flag
        """
        # Convert CE/PE to call/put for Black-Scholes
        bs_option_type = 'call' if option_type == 'CE' else 'put'
        
        # Handle edge cases
        if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
            return 0.20, False  # Return default IV
        
        # Define objective function for minimization
        def objective(sigma):
            try:
                theo_price = black_scholes_price(S, K, T, self.risk_free_rate, sigma, bs_option_type)
                return abs(theo_price - market_price)
            except:
                return float('inf')
        
        # Try different starting points for IV
        starting_points = [0.20, 0.30, 0.40, 0.50, 0.60]
        best_result = None
        best_error = float('inf')
        
        for start_iv in starting_points:
            try:
                result = minimize_scalar(
                    objective,
                    bounds=(0.01, 2.0),  # IV between 1% and 200%
                    method='bounded',
                    options={'maxiter': self.max_iterations}
                )
                
                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
                    
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            estimated_iv = best_result.x
            # Validate the result
            if 0.01 <= estimated_iv <= 2.0:
                return estimated_iv, True
        
        # Fallback: use a reasonable default based on moneyness
        moneyness = abs(np.log(S / K))
        if moneyness < 0.1:  # Near ATM
            default_iv = 0.25
        elif moneyness < 0.3:  # Slightly OTM/ITM
            default_iv = 0.30
        else:  # Far OTM/ITM
            default_iv = 0.40
        
        return default_iv, False
    
    def compute_greeks(self, S: float, K: float, T: float, sigma: float, 
                      option_type: str) -> Dict[str, float]:
        """
        Compute all Greeks for a given contract.
        
        Parameters:
        -----------
        S : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to expiry in years
        sigma : float
            Implied volatility
        option_type : str
            Option type ('CE' or 'PE')
            
        Returns:
        --------
        Dict[str, float]
            Dictionary with all Greeks
        """
        # Convert CE/PE to call/put for Black-Scholes
        bs_option_type = 'call' if option_type == 'CE' else 'put'
        
        try:
            greeks = black_scholes_greeks(S, K, T, self.risk_free_rate, sigma, bs_option_type)
            return greeks
        except Exception as e:
            # Return default values if calculation fails
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0
            }
    
    def process_single_contract(self, row: pd.Series) -> Dict[str, Any]:
        """
        Process a single contract row to compute IV and Greeks.
        
        Parameters:
        -----------
        row : pd.Series
            Single row from options DataFrame
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with computed IV and Greeks
        """
        try:
            # Extract contract parameters
            S = row['S_t']
            K = row['strike']
            T = (row['expiry_date'] - row['date_t']).days / 365.25
            market_price = row['premium_t']
            option_type = row['option_type']
            
            # Skip if missing critical data
            if pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(market_price):
                return {
                    'iv_est_t': None,
                    'delta': None,
                    'gamma': None,
                    'theta': None,
                    'vega': None,
                    'rho': None,
                    'iv_converged': False
                }
            
            # Ensure positive values
            if S <= 0 or K <= 0 or T <= 0 or market_price <= 0:
                return {
                    'iv_est_t': 0.20,
                    'delta': 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0,
                    'iv_converged': False
                }
            
            # Invert for implied volatility
            estimated_iv, converged = self.invert_implied_volatility(
                S, K, T, market_price, option_type
            )
            
            # Compute Greeks using estimated IV
            greeks = self.compute_greeks(S, K, T, estimated_iv, option_type)
            
            return {
                'iv_est_t': estimated_iv,
                'delta': greeks['delta'],
                'gamma': greeks['gamma'],
                'theta': greeks['theta'],
                'vega': greeks['vega'],
                'rho': greeks['rho'],
                'iv_converged': converged
            }
            
        except Exception as e:
            # Return default values on error
            return {
                'iv_est_t': 0.20,
                'delta': 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0,
                'rho': 0.0,
                'iv_converged': False
            }
    
    def process_options_dataframe(self, df: pd.DataFrame, 
                                 batch_size: int = 1000) -> pd.DataFrame:
        """
        Process entire DataFrame to compute IV and Greeks.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Options DataFrame with required columns
        batch_size : int
            Number of rows to process in each batch
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with IV and Greeks computed
        """
        if df.empty:
            return df
        
        # Validate required columns
        required_cols = ['S_t', 'strike', 'expiry_date', 'date_t', 'premium_t', 'option_type']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date columns if needed
        if not pd.api.types.is_datetime64_any_dtype(df['expiry_date']):
            df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        if not pd.api.types.is_datetime64_any_dtype(df['date_t']):
            df['date_t'] = pd.to_datetime(df['date_t'])
        
        # Process in batches to show progress
        total_rows = len(df)
        results = []
        
        print(f"🔄 Computing IV and Greeks for {total_rows} contracts...")
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            print(f"   Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} "
                  f"({i+1}-{batch_end}/{total_rows})")
            
            batch_results = []
            for _, row in batch_df.iterrows():
                result = self.process_single_contract(row)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add computed columns to original DataFrame
        df = df.copy()
        for col in ['iv_est_t', 'delta', 'gamma', 'theta', 'vega', 'rho']:
            if col in results_df.columns:
                df[col] = results_df[col]
        
        # Add convergence flag
        if 'iv_converged' in results_df.columns:
            df['iv_converged'] = results_df['iv_converged']
        
        print(f"✅ Completed IV and Greeks computation")
        print(f"   IV convergence rate: {results_df['iv_converged'].mean():.1%}")
        
        return df
    
    def validate_computed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the computed IV and Greeks data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with computed IV and Greeks
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'statistics': {}
        }
        
        # Check IV ranges
        if 'iv_est_t' in df.columns:
            iv_stats = df['iv_est_t'].describe()
            validation_result['statistics']['iv_est_t'] = iv_stats.to_dict()
            
            # Check for unreasonable IV values
            if iv_stats['max'] > 2.0:  # 200% annualized
                validation_result['warnings'].append(
                    f"Found extremely high IV values (max: {iv_stats['max']:.1%})"
                )
            
            if iv_stats['min'] < 0.01:  # 1% annualized
                validation_result['warnings'].append(
                    f"Found extremely low IV values (min: {iv_stats['min']:.1%})"
                )
        
        # Check Greeks ranges
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for col in greek_cols:
            if col in df.columns:
                greek_stats = df[col].describe()
                validation_result['statistics'][col] = greek_stats.to_dict()
                
                # Check for unreasonable Greek values
                if col == 'delta' and (greek_stats['min'] < -1.1 or greek_stats['max'] > 1.1):
                    validation_result['warnings'].append(
                        f"Delta values outside expected range [-1, 1]: [{greek_stats['min']:.3f}, {greek_stats['max']:.3f}]"
                    )
                
                if col == 'gamma' and greek_stats['max'] > 1.0:
                    validation_result['warnings'].append(
                        f"Found extremely high Gamma values (max: {greek_stats['max']:.3f})"
                    )
        
        # Check convergence rate
        if 'iv_converged' in df.columns:
            convergence_rate = df['iv_converged'].mean()
            validation_result['statistics']['iv_convergence_rate'] = convergence_rate
            
            if convergence_rate < 0.8:
                validation_result['warnings'].append(
                    f"Low IV convergence rate: {convergence_rate:.1%}"
                )
        
        return validation_result


def compute_iv_and_greeks_from_file(input_file: str, output_dir: str = "outputs/csv",
                                   risk_free_rate: float = 0.06) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to compute IV and Greeks from a CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to CSV file with options data
    output_dir : str
        Output directory for enhanced CSV
    risk_free_rate : float
        Risk-free rate for calculations
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        Enhanced DataFrame and path to saved CSV
    """
    # Load options data
    df = pd.read_csv(input_file)
    
    # Convert date columns
    df['date_t'] = pd.to_datetime(df['date_t'])
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    
    # Compute IV and Greeks
    computer = IVAndGreeksComputer(risk_free_rate)
    enhanced_df = computer.process_options_dataframe(df)
    
    # Validate
    validation_result = computer.validate_computed_data(enhanced_df)
    for warning in validation_result['warnings']:
        print(f"  Warning: {warning}")
    
    # Save enhanced data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"options_with_iv_greeks_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    enhanced_df.to_csv(filepath, index=False)
    
    return enhanced_df, filepath


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute IV and Greeks for options data")
    parser.add_argument("input_file", help="Path to CSV file with options data")
    parser.add_argument("--output-dir", default="outputs/csv", help="Output directory")
    parser.add_argument("--risk-free-rate", type=float, default=0.06, 
                       help="Risk-free rate (default: 0.06)")
    parser.add_argument("--batch-size", type=int, default=1000, 
                       help="Batch size for processing (default: 1000)")
    
    args = parser.parse_args()
    
    try:
        enhanced_df, csv_path = compute_iv_and_greeks_from_file(
            args.input_file, args.output_dir, args.risk_free_rate
        )
        
        print(f"✅ Successfully computed IV and Greeks")
        print(f"   Input: {args.input_file}")
        print(f"   Output: {csv_path}")
        print(f"   Contracts: {len(enhanced_df)}")
        
        # Show summary statistics
        if 'iv_est_t' in enhanced_df.columns:
            print(f"\n📊 IV Summary:")
            print(f"   Mean IV: {enhanced_df['iv_est_t'].mean():.1%}")
            print(f"   Min IV: {enhanced_df['iv_est_t'].min():.1%}")
            print(f"   Max IV: {enhanced_df['iv_est_t'].max():.1%}")
        
        if 'iv_converged' in enhanced_df.columns:
            convergence_rate = enhanced_df['iv_converged'].mean()
            print(f"   IV Convergence: {convergence_rate:.1%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
