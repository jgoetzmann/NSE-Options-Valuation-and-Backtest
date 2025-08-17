#!/usr/bin/env python3
"""
Attach Underlier Features Module
================================

This module attaches underlier-derived features to the normalized options data.
It fetches historical data from yfinance and computes returns, realized volatility,
and other market regime indicators.

Key Functions:
- fetch_underlier_history(): Get historical underlier data from yfinance
- compute_returns(): Calculate various return measures
- compute_realized_volatility(): Calculate realized volatility over different windows
- attach_features(): Main function to attach all features to options DataFrame
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.schemas import validate_dataframe_schema


class UnderlierFeatureAttacher:
    """
    Attaches underlier-derived features to options data.
    
    This class handles fetching historical underlier data and computing
    features like returns, realized volatility, and market regime indicators.
    """
    
    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol.upper()
        
        # Map NSE symbols to yfinance symbols
        self.symbol_mapping = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'FINNIFTY': '^NSEBANK',  # Use BANKNIFTY as proxy for now
            'MIDCPNIFTY': '^NSEI',   # Use NIFTY as proxy for now
            'SENSEX': '^BSESN'
        }
        
        self.yf_symbol = self.symbol_mapping.get(self.symbol, self.symbol)
        
    def fetch_underlier_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical underlier data from yfinance.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame
            Historical data with OHLCV columns
        """
        try:
            print(f"📊 Fetching {self.yf_symbol} data from {start_date} to {end_date}...")
            
            ticker = yf.Ticker(self.yf_symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise ValueError(f"No data found for {self.yf_symbol}")
            
            print(f"✅ Fetched {len(hist)} days of data")
            return hist
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {self.yf_symbol} data: {e}")
    
    def compute_returns(self, hist_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various return measures from historical data.
        
        Parameters:
        -----------
        hist_data : pd.DataFrame
            Historical OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            Data with return columns added
        """
        df = hist_data.copy()
        
        # Daily returns
        df['ret_1d'] = df['Close'].pct_change()
        
        # 5-day returns (rolling)
        df['ret_5d'] = df['Close'].pct_change(periods=5)
        
        # 10-day returns
        df['ret_10d'] = df['Close'].pct_change(periods=10)
        
        # 20-day returns
        df['ret_20d'] = df['Close'].pct_change(periods=20)
        
        return df
    
    def compute_realized_volatility(self, hist_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute realized volatility over different windows.
        
        Parameters:
        -----------
        hist_data : pd.DataFrame
            Historical data with returns computed
            
        Returns:
        --------
        pd.DataFrame
            Data with realized volatility columns added
        """
        df = hist_data.copy()
        
        # 10-day realized volatility (annualized)
        df['rv_10d'] = df['ret_1d'].rolling(window=10).std() * np.sqrt(252)
        
        # 20-day realized volatility (annualized)
        df['rv_20d'] = df['ret_1d'].rolling(window=20).std() * np.sqrt(252)
        
        # 5-day realized volatility (annualized)
        df['rv_5d'] = df['ret_1d'].rolling(window=5).std() * np.sqrt(252)
        
        return df
    
    def compute_market_regime_indicators(self, hist_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market regime indicators.
        
        Parameters:
        -----------
        hist_data : pd.DataFrame
            Historical data with returns and volatility
            
        Returns:
        --------
        pd.DataFrame
            Data with regime indicators added
        """
        df = hist_data.copy()
        
        # Volatility regime (percentile rank)
        df['vol_regime_20d'] = df['rv_20d'].rolling(window=60).rank(pct=True)
        
        # Trend regime (20-day moving average vs current price)
        df['ma_20d'] = df['Close'].rolling(window=20).mean()
        df['trend_regime'] = (df['Close'] > df['ma_20d']).astype(int)
        
        # Momentum regime (5-day vs 20-day returns)
        df['momentum_regime'] = np.where(
            df['ret_5d'] > df['ret_20d'], 1, 0
        )
        
        return df
    
    def attach_features(self, options_df: pd.DataFrame, 
                       lookback_days: int = 60) -> pd.DataFrame:
        """
        Attach underlier features to options DataFrame.
        
        Parameters:
        -----------
        options_df : pd.DataFrame
            Options DataFrame with date_t column
        lookback_days : int
            Number of days to look back for feature computation
            
        Returns:
        --------
        pd.DataFrame
            Options DataFrame with underlier features attached
        """
        if 'date_t' not in options_df.columns:
            raise ValueError("Options DataFrame must have 'date_t' column")
        
        # Get date range for underlier data
        min_date = options_df['date_t'].min()
        max_date = options_df['date_t'].max()
        
        # Add buffer for lookback
        start_date = (min_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        end_date = max_date.strftime('%Y-%m-%d')
        
        # Fetch underlier data
        hist_data = self.fetch_underlier_history(start_date, end_date)
        
        # Compute features
        hist_data = self.compute_returns(hist_data)
        hist_data = self.compute_realized_volatility(hist_data)
        hist_data = self.compute_market_regime_indicators(hist_data)
        
        # Prepare for merge
        hist_data = hist_data.reset_index()
        hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.date
        
        # Use the actual snapshot date for all options (since they're from the same snapshot)
        snapshot_date = options_df['date_t'].iloc[0].date()
        options_df['date_t_date'] = snapshot_date
        
        # Merge features
        result_df = pd.merge(
            options_df,
            hist_data[['Date', 'Close', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
                      'rv_5d', 'rv_10d', 'rv_20d', 'vol_regime_20d', 'trend_regime', 'momentum_regime']],
            left_on='date_t_date',
            right_on='Date',
            how='left'
        )
        
        # Clean up merge columns
        result_df = result_df.drop(['date_t_date', 'Date'], axis=1)
        
        # Rename Close to S_t if not already present
        if 'S_t' not in result_df.columns:
            result_df['S_t'] = result_df['Close']
            result_df = result_df.drop('Close', axis=1)
        
        # Fill missing values
        result_df = self._fill_missing_features(result_df)
        
        return result_df
    
    def _fill_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing feature values with reasonable defaults.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with potentially missing features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing values filled
        """
        # Fill returns with 0 (no change)
        return_cols = ['ret_1d', 'ret_5d', 'ret_10d', 'ret_20d']
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill volatility with historical average or reasonable default
        vol_cols = ['rv_5d', 'rv_10d', 'rv_20d']
        for col in vol_cols:
            if col in df.columns:
                # Use median of non-null values, or default to 20% annualized
                median_vol = df[col].median()
                if pd.isna(median_vol):
                    median_vol = 0.20
                df[col] = df[col].fillna(median_vol)
        
        # Fill regime indicators with neutral values
        if 'vol_regime_20d' in df.columns:
            df['vol_regime_20d'] = df['vol_regime_20d'].fillna(0.5)
        
        if 'trend_regime' in df.columns:
            df['trend_regime'] = df['trend_regime'].fillna(0)
        
        if 'momentum_regime' in df.columns:
            df['momentum_regime'] = df['momentum_regime'].fillna(0)
        
        return df
    
    def validate_attached_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that features were attached correctly.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features attached
            
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
        
        # Check for required features
        required_features = ['S_t', 'ret_1d', 'ret_5d', 'rv_10d', 'rv_20d']
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
        
        # Check for unreasonable values
        if 'S_t' in df.columns:
            if df['S_t'].min() <= 0:
                validation_result['warnings'].append("Found non-positive underlier prices")
        
        if 'rv_10d' in df.columns:
            if df['rv_10d'].max() > 2.0:  # 200% annualized volatility
                validation_result['warnings'].append("Found extremely high realized volatility values")
        
        return validation_result


def attach_underlier_features_from_file(options_file: str, symbol: str = "NIFTY",
                                       output_dir: str = "outputs/csv") -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to attach features from a CSV file.
    
    Parameters:
    -----------
    options_file : str
        Path to CSV file with options data
    symbol : str
        Underlying symbol
    output_dir : str
        Output directory for enhanced CSV
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        Enhanced DataFrame and path to saved CSV
    """
    # Load options data
    options_df = pd.read_csv(options_file)
    options_df['date_t'] = pd.to_datetime(options_df['date_t'])
    options_df['expiry_date'] = pd.to_datetime(options_df['expiry_date'])
    
    # Attach features
    attacher = UnderlierFeatureAttacher(symbol)
    enhanced_df = attacher.attach_features(options_df)
    
    # Validate
    validation_result = attacher.validate_attached_features(enhanced_df)
    if not validation_result['is_valid']:
        print("Warning: Feature attachment validation issues:")
        for feature in validation_result['missing_features']:
            print(f"  Missing: {feature}")
    
    for warning in validation_result['warnings']:
        print(f"  Warning: {warning}")
    
    # Save enhanced data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"options_with_features_{symbol}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    enhanced_df.to_csv(filepath, index=False)
    
    return enhanced_df, filepath


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Attach underlier features to options data")
    parser.add_argument("options_file", help="Path to CSV file with options data")
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--output-dir", default="outputs/csv", help="Output directory")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback days for features")
    
    args = parser.parse_args()
    
    try:
        enhanced_df, csv_path = attach_underlier_features_from_file(
            args.options_file, args.symbol, args.output_dir
        )
        
        print(f"✅ Successfully attached underlier features")
        print(f"   Input: {args.options_file}")
        print(f"   Output: {csv_path}")
        print(f"   Contracts: {len(enhanced_df)}")
        print(f"   Symbol: {args.symbol}")
        
        # Show feature summary
        feature_cols = ['S_t', 'ret_1d', 'ret_5d', 'rv_10d', 'rv_20d']
        print(f"\n📊 Feature Summary:")
        for col in feature_cols:
            if col in enhanced_df.columns:
                non_null = enhanced_df[col].count()
                total = len(enhanced_df)
                print(f"   {col}: {non_null}/{total} ({non_null/total*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
