#!/usr/bin/env python3
"""
Enhanced Feature Computation Module
==================================

This module computes enhanced features for options analysis including:
- Enhanced mispricing percentages using the advanced valuation engine
- Confidence scores based on multiple factors
- Market vs theoretical price comparisons
- Advanced ranking features

It integrates with the enhanced valuation engine from utils.py to provide
professional-grade options analysis capabilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import black_scholes_price, option_valuation
from data_pipeline.schemas import add_derived_features


class EnhancedFeatureComputer:
    """
    Computes enhanced features for options analysis using the advanced valuation engine.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
        
    def compute_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute enhanced features for options DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Options DataFrame with basic features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with enhanced features added
        """
        enhanced_df = df.copy()
        
        print("🧮 Computing enhanced features...")
        
        # Compute theoretical prices and enhanced valuation
        enhanced_df = self._compute_theoretical_prices(enhanced_df)
        enhanced_df = self._compute_enhanced_valuation(enhanced_df)
        enhanced_df = self._compute_ranking_features(enhanced_df)
        
        print(f"✅ Enhanced features computed for {len(enhanced_df)} contracts")
        return enhanced_df
    
    def _compute_theoretical_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute theoretical Black-Scholes prices for all options.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with theoretical prices added
        """
        print("   Computing theoretical prices...")
        
        # Add theoretical price column
        df['theoretical_price'] = np.nan
        
        # Process in batches for efficiency
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            print(f"     Processing batch {(i//batch_size)+1}/{total_batches} ({i+1}-{batch_end})")
            
            for idx in batch_df.index:
                try:
                    # Extract parameters
                    S = df.loc[idx, 'S_t']
                    K = df.loc[idx, 'strike']
                    T = df.loc[idx, 'ttm_days'] / 365.25  # Convert to years
                    sigma = df.loc[idx, 'iv_est_t']  # IV is already in decimal form
                    option_type_raw = df.loc[idx, 'option_type']
                    
                    # Convert option type to expected format
                    if option_type_raw.upper() == 'CE':
                        option_type = 'call'
                    elif option_type_raw.upper() == 'PE':
                        option_type = 'put'
                    else:
                        continue  # Skip invalid option types
                    
                    # Validate parameters
                    if (pd.isna(S) or pd.isna(K) or pd.isna(T) or pd.isna(sigma) or
                        S <= 0 or K <= 0 or T <= 0 or sigma <= 0):
                        continue
                    
                    # Compute theoretical price
                    try:
                        theo_price = black_scholes_price(S, K, T, self.risk_free_rate, sigma, option_type)
                        df.loc[idx, 'theoretical_price'] = theo_price
                    except Exception as e:
                        continue
                    
                except Exception as e:
                    # Skip contracts with calculation errors
                    continue
        
        # Calculate percentage difference
        df['pct_diff'] = np.where(
            df['premium_t'] > 0,
            (df['theoretical_price'] - df['premium_t']) / df['premium_t'],
            np.nan
        )
        
        return df
    
    def _compute_enhanced_valuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute enhanced valuation using the advanced valuation engine.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with theoretical prices
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with enhanced valuation features
        """
        print("   Computing enhanced valuation...")
        
        # Add enhanced valuation columns
        df['enhanced_rating'] = ''
        df['enhanced_mispricing_pct'] = np.nan
        df['enhanced_confidence'] = np.nan
        
        # Process in batches
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            print(f"     Processing batch {(i//batch_size)+1}/{total_batches} ({i+1}-{batch_end})")
            
            for idx in batch_df.index:
                try:
                    # Extract parameters for enhanced valuation
                    theo_price = df.loc[idx, 'theoretical_price']
                    market_price = df.loc[idx, 'premium_t']
                    
                    if pd.isna(theo_price) or pd.isna(market_price) or market_price <= 0:
                        continue
                    
                    # Get additional parameters for enhanced valuation
                    S = df.loc[idx, 'S_t']
                    K = df.loc[idx, 'strike']
                    T = df.loc[idx, 'ttm_days'] / 365.25
                    sigma = df.loc[idx, 'iv_est_t']
                    bid = df.loc[idx, 'bidPrice'] if 'bidPrice' in df.columns else None
                    ask = df.loc[idx, 'askPrice'] if 'askPrice' in df.columns else None
                    total_volume = df.loc[idx, 'totalTradedVolume'] if 'totalTradedVolume' in df.columns else None
                    
                    # Compute enhanced valuation
                    rating, pct_diff, confidence = option_valuation(
                        theoretical_price=theo_price,
                        market_price=market_price,
                        S=S, K=K, T=T, sigma=sigma,
                        bid=bid, ask=ask
                    )
                    
                    # Store results
                    df.loc[idx, 'enhanced_rating'] = rating
                    df.loc[idx, 'enhanced_mispricing_pct'] = pct_diff
                    df.loc[idx, 'enhanced_confidence'] = confidence
                    
                except Exception as e:
                    # Skip contracts with calculation errors
                    continue
        
        return df
    
    def _compute_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional features for ranking and portfolio construction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with enhanced valuation
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with ranking features added
        """
        print("   Computing ranking features...")
        
        # Moneyness-based features
        df['moneyness_ratio'] = np.where(
            df['S_t'] > 0,
            abs(df['S_t'] - df['strike']) / df['S_t'],
            np.nan
        )
        
        # Liquidity features
        if 'bidPrice' in df.columns and 'askPrice' in df.columns:
            df['spread_absolute'] = df['askPrice'] - df['bidPrice']
            df['spread_pct'] = np.where(
                df['bidPrice'] > 0,
                df['spread_absolute'] / df['bidPrice'],
                np.nan
            )
        
        # Volume-based features
        if 'totalTradedVolume' in df.columns:
            df['volume_rank'] = df['totalTradedVolume'].rank(pct=True)
        
        # Open Interest features
        if 'openInterest' in df.columns:
            df['oi_rank'] = df['openInterest'].rank(pct=True)
        
        # Time decay features
        df['time_decay_factor'] = np.exp(-df['ttm_days'] / 30)  # Exponential decay
        
        # Volatility features
        if 'iv_est_t' in df.columns:
            df['iv_rank'] = df['iv_est_t'].rank(pct=True)
            df['iv_skew'] = df['iv_est_t'] - df['iv_est_t'].median()
        
        # Alternative ranking scores
        df['simple_mispricing_score'] = np.abs(df['pct_diff'])
        df['liquidity_score'] = np.where(
            'spread_pct' in df.columns,
            1 / (1 + df['spread_pct']),  # Higher score for tighter spreads
            0.5  # Default score
        )
        
        # Combined ranking score (enhanced mispricing × confidence)
        # Fallback to simple mispricing if enhanced features not available
        df['enhanced_ranking_score'] = np.where(
            (df['enhanced_mispricing_pct'].notna()) & (df['enhanced_confidence'].notna()),
            df['enhanced_mispricing_pct'] * df['enhanced_confidence'],
            df['simple_mispricing_score']  # Fallback to simple mispricing
        )
        
        return df
    
    def validate_enhanced_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the enhanced features for quality and completeness.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with enhanced features
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'missing_features': [],
            'feature_stats': {},
            'warnings': []
        }
        
        # Required enhanced features
        required_features = [
            'theoretical_price', 'enhanced_rating', 'enhanced_mispricing_pct', 
            'enhanced_confidence', 'enhanced_ranking_score'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                validation_result['missing_features'].append(feature)
                validation_result['is_valid'] = False
        
        # Check feature statistics
        for feature in required_features:
            if feature in df.columns:
                validation_result['feature_stats'][feature] = {
                    'count': df[feature].count(),
                    'null_count': df[feature].isnull().sum(),
                    'min': df[feature].min() if df[feature].dtype in ['float64', 'int64'] else None,
                    'max': df[feature].max() if df[feature].dtype in ['float64', 'int64'] else None,
                    'mean': df[feature].mean() if df[feature].dtype in ['float64', 'int64'] else None
                }
        
        # Check for reasonable values
        if 'enhanced_confidence' in df.columns:
            confidence_range = df['enhanced_confidence'].dropna()
            if len(confidence_range) > 0:
                if confidence_range.min() < 0 or confidence_range.max() > 1:
                    validation_result['warnings'].append("Confidence scores outside [0,1] range")
        
        if 'enhanced_mispricing_pct' in df.columns:
            mispricing_range = df['enhanced_mispricing_pct'].dropna()
            if len(mispricing_range) > 0:
                if abs(mispricing_range.max()) > 2.0:  # 200% mispricing
                    validation_result['warnings'].append("Found extremely high mispricing values")
        
        return validation_result


def compute_enhanced_features_from_file(input_file: str, output_dir: str = "outputs/csv") -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to compute enhanced features from a CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to CSV file with options data
    output_dir : str
        Output directory for enhanced CSV
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        Enhanced DataFrame and path to saved CSV
    """
    # Load options data
    options_df = pd.read_csv(input_file)
    options_df['date_t'] = pd.to_datetime(options_df['date_t'])
    options_df['expiry_date'] = pd.to_datetime(options_df['expiry_date'])
    
    # Compute enhanced features
    computer = EnhancedFeatureComputer()
    enhanced_df = computer.compute_enhanced_features(options_df)
    
    # Validate
    validation_result = computer.validate_enhanced_features(enhanced_df)
    if not validation_result['is_valid']:
        print("Warning: Enhanced feature validation issues:")
        for feature in validation_result['missing_features']:
            print(f"  Missing: {feature}")
    
    for warning in validation_result['warnings']:
        print(f"  Warning: {warning}")
    
    # Save enhanced data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"options_with_enhanced_features_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    enhanced_df.to_csv(filepath, index=False)
    
    return enhanced_df, filepath


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute enhanced features for options data")
    parser.add_argument("input_file", help="Path to CSV file with options data")
    parser.add_argument("--output-dir", default="outputs/csv", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        enhanced_df, csv_path = compute_enhanced_features_from_file(
            args.input_file, args.output_dir
        )
        
        print(f"✅ Successfully computed enhanced features")
        print(f"   Input: {args.input_file}")
        print(f"   Output: {csv_path}")
        print(f"   Contracts: {len(enhanced_df)}")
        
        # Show feature statistics
        print(f"\n📊 Enhanced Feature Statistics:")
        if 'enhanced_ranking_score' in enhanced_df.columns:
            valid_scores = enhanced_df['enhanced_ranking_score'].dropna()
            if len(valid_scores) > 0:
                print(f"   Ranking Scores: {len(valid_scores)} valid, range [{valid_scores.min():.4f}, {valid_scores.max():.4f}]")
        
        if 'enhanced_confidence' in enhanced_df.columns:
            valid_conf = enhanced_df['enhanced_confidence'].dropna()
            if len(valid_conf) > 0:
                print(f"   Confidence: {len(valid_conf)} valid, avg {valid_conf.mean():.3f}")
        
        if 'enhanced_mispricing_pct' in enhanced_df.columns:
            valid_misp = enhanced_df['enhanced_mispricing_pct'].dropna()
            if len(valid_misp) > 0:
                print(f"   Mispricing: {len(valid_misp)} valid, avg {valid_misp.mean():.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
