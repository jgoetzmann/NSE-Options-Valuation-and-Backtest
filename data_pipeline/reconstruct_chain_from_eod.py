#!/usr/bin/env python3
"""
Reconstruct Historical Option Chains from EOD Data for Mode C (EOD-True)
======================================================================

This module reconstructs per-day option chains using official historical derivatives data
(bhavcopy files) and yfinance underlier data. It's the foundation for Mode C backtesting.

Key Functions:
- load_eod_derivatives_data: Load and parse bhavcopy-style CSV files
- map_to_internal_schema: Convert external format to internal schema
- attach_underlier_data: Join with yfinance underlier closes
- reconstruct_daily_chain: Main reconstruction pipeline
- validate_reconstructed_data: Data quality checks
- persist_reconstructed_chain: Save to JSON and parquet formats

Usage:
    python reconstruct_chain_from_eod.py --date 2024-01-15 --symbol NIFTY
    python reconstruct_chain_from_eod.py --date-range 2024-01-01 2024-12-31 --symbol NIFTY
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.schemas import (
    NORMALIZED_TABLE_SCHEMA, 
    PANDAS_DTYPES,
    validate_dataframe_schema
)
from data_pipeline.compute_iv_and_greeks import IVAndGreeksComputer
from data_pipeline.make_labels import LabelGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EODChainReconstructor:
    """
    Reconstructs historical option chains from EOD derivatives data.
    
    This class handles the complete pipeline from loading raw EOD data to
    producing normalized, labeled option chains suitable for backtesting.
    """
    
    def __init__(self, symbol: str = "NIFTY", risk_free_rate: float = 0.06):
        """
        Initialize the reconstructor.
        
        Args:
            symbol: Underlier symbol (NIFTY, BANKNIFTY, etc.)
            risk_free_rate: Risk-free rate for option pricing
        """
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.iv_computer = IVAndGreeksComputer(risk_free_rate)
        self.label_generator = LabelGenerator(risk_free_rate)
        
        # Define underlier mapping for yfinance
        self.underlier_symbols = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "^NSEBANK"  # Approximate mapping
        }
        
        # Define expected CSV columns for bhavcopy format
        self.expected_columns = [
            'SYMBOL', 'EXPIRY_DT', 'STRIKE_PR', 'OPTION_TYP', 'SETTLE_PR',
            'OPEN_INT', 'CHG_IN_OI', 'VOLUME', 'TRADE_DATE'
        ]
        
        # Output directories
        self.output_dirs = {
            'json': 'reconstructed/json',
            'parquet': 'reconstructed/parquet'
        }
        
        # Ensure output directories exist
        for dir_path in self.output_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_eod_derivatives_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and parse EOD derivatives data from CSV file.
        
        Args:
            file_path: Path to bhavcopy CSV file
            
        Returns:
            DataFrame with derivatives data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EOD derivatives file not found: {file_path}")
        
        try:
            # Try different CSV formats
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            missing_cols = set(self.expected_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns in {file_path}: {missing_cols}")
                logger.info(f"Available columns: {list(df.columns)}")
                
                # Try to map common variations
                column_mapping = {
                    'TRADING_DATE': 'TRADE_DATE',
                    'STRIKE': 'STRIKE_PR',
                    'OPTION_TYPE': 'OPTION_TYP',
                    'SETTLE_PRICE': 'SETTLE_PR',
                    'OPEN_INTEREST': 'OPEN_INT',
                    'CHANGE_IN_OI': 'CHG_IN_OI',
                    'TOTAL_TRADED_VOLUME': 'VOLUME'
                }
                
                df = df.rename(columns=column_mapping)
                
                # Check again after mapping
                missing_cols = set(self.expected_columns) - set(df.columns)
                if missing_cols:
                    raise ValueError(f"Critical columns missing after mapping: {missing_cols}")
            
            # Filter for the target symbol
            df = df[df['SYMBOL'] == self.symbol].copy()
            
            if df.empty:
                raise ValueError(f"No data found for symbol {self.symbol} in {file_path}")
            
            logger.info(f"Loaded {len(df)} contracts for {self.symbol} from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading EOD data from {file_path}: {e}")
            raise
    
    def map_to_internal_schema(self, df: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """
        Map external EOD format to internal normalized schema.
        
        Args:
            df: Raw EOD derivatives DataFrame
            trade_date: Trading date in YYYY-MM-DD format
            
        Returns:
            DataFrame with internal schema
        """
        # Create mapping
        mapped_df = pd.DataFrame()
        
        # Map basic contract fields
        mapped_df['symbol'] = df['SYMBOL']
        mapped_df['expiry_date'] = pd.to_datetime(df['EXPIRY_DT'], format='%d-%m-%Y')
        mapped_df['strike'] = df['STRIKE_PR'].astype(float)
        mapped_df['option_type'] = df['OPTION_TYP'].str.upper()
        mapped_df['premium_t'] = df['SETTLE_PR'].astype(float)
        mapped_df['openInterest'] = df['OPEN_INT'].astype(int)
        mapped_df['changeinOpenInterest'] = df['CHG_IN_OI'].astype(int)
        mapped_df['totalTradedVolume'] = df['VOLUME'].astype(int)
        
        # Add date fields
        mapped_df['date_t'] = pd.to_datetime(trade_date)
        mapped_df['ttm_days'] = (mapped_df['expiry_date'] - mapped_df['date_t']).dt.days
        
        # Add synthetic flag (0 for true EOD data)
        mapped_df['synthetic_flag'] = 0
        
        # Add valid horizon flags
        mapped_df['valid_horizon'] = 1  # Will be updated based on actual horizons
        
        # Filter out invalid contracts
        valid_mask = (
            (mapped_df['premium_t'] > 0) &
            (mapped_df['ttm_days'] > 0) &
            (mapped_df['expiry_date'] >= mapped_df['date_t'])
        )
        
        mapped_df = mapped_df[valid_mask].copy()
        
        # Reset index
        mapped_df = mapped_df.reset_index(drop=True)
        
        logger.info(f"Mapped {len(mapped_df)} valid contracts to internal schema")
        return mapped_df
    
    def attach_underlier_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach underlier price data from yfinance.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with underlier data attached
        """
        if df.empty:
            return df
        
        # Get unique dates
        unique_dates = df['date_t'].dt.date.unique()
        logger.info(f"Fetching underlier data for {len(unique_dates)} unique dates")
        
        # Get underlier symbol
        underlier_symbol = self.underlier_symbols.get(self.symbol, "^NSEI")
        
        # Fetch underlier data for the date range
        start_date = min(unique_dates) - timedelta(days=5)  # Buffer for timezone
        end_date = max(unique_dates) + timedelta(days=5)
        
        try:
            underlier_data = yf.download(
                underlier_symbol,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if underlier_data.empty:
                logger.warning(f"No underlier data found for {underlier_symbol}")
                return df
            
            # Extract close prices and ensure it's a Series
            underlier_closes = underlier_data['Close']
            if isinstance(underlier_closes, pd.DataFrame):
                underlier_closes = underlier_closes.iloc[:, 0]  # Take first column if DataFrame
            elif isinstance(underlier_closes, pd.Series):
                pass  # Already correct format
            else:
                logger.warning(f"Unexpected underlier_closes type: {type(underlier_closes)}")
                return df
            
            # Convert dates to datetime for proper mapping
            df['date_t_dt'] = pd.to_datetime(df['date_t'])
            df['expiry_date_dt'] = pd.to_datetime(df['expiry_date'])
            
            # Attach S_t (underlier price at trade date)
            df['S_t'] = df['date_t_dt'].map(underlier_closes)
            
            # Attach S_T (underlier price at expiry) for expired contracts
            df['S_T'] = df['expiry_date_dt'].map(underlier_closes)
            
            # Compute returns and volatility features
            df = self._compute_underlier_features(df, underlier_closes)
            
            # Filter out contracts with missing underlier data
            valid_mask = df['S_t'].notna()
            df = df[valid_mask].copy()
            
            logger.info(f"Attached underlier data to {len(df)} contracts")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching underlier data: {e}")
            return df
    
    def _compute_underlier_features(self, df: pd.DataFrame, underlier_closes: pd.Series) -> pd.DataFrame:
        """
        Compute underlier-derived features.
        
        Args:
            df: DataFrame with contract data
            underlier_closes: Series of underlier close prices
            
        Returns:
            DataFrame with additional features
        """
        # Compute returns
        df['ret_1d'] = underlier_closes.pct_change(1).reindex(df['date_t']).values
        df['ret_5d'] = underlier_closes.pct_change(5).reindex(df['date_t']).values
        
        # Compute realized volatility (10d and 20d)
        for window in [10, 20]:
            vol_col = f'rv_{window}d'
            df[vol_col] = (
                underlier_closes.pct_change()
                .rolling(window=window)
                .std()
                .multiply((252 ** 0.5))  # Annualize
                .reindex(df['date_t'])
                .values
            )
        
        return df
    
    def compute_iv_and_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute implied volatility and Greeks for all contracts.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with IV and Greeks computed
        """
        if df.empty:
            return df
        
        logger.info("Computing IV and Greeks for all contracts...")
        
        # Process in batches for memory efficiency
        df = self.iv_computer.process_options_dataframe(df, batch_size=1000)
        
        # Add cost basis points (placeholder - should come from config)
        df['cost_bps'] = 60.0  # Default 60 bps round-turn
        
        logger.info(f"Computed IV and Greeks for {len(df)} contracts")
        return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate payoff, PnL, ROI, and POP labels for expired contracts.
        
        Args:
            df: DataFrame with contract data
            
        Returns:
            DataFrame with labels generated
        """
        if df.empty:
            return df
        
        logger.info("Generating labels for expired contracts...")
        
        # Generate labels using the label generator
        df = self.label_generator.generate_labels(df, cost_bps=df['cost_bps'].iloc[0])
        
        # Count labeled contracts
        labeled_count = df['POP_label'].notna().sum()
        logger.info(f"Generated labels for {labeled_count} expired contracts")
        
        return df
    
    def validate_reconstructed_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate reconstructed data quality and integrity.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_contracts': len(df),
            'valid_contracts': 0,
            'expired_contracts': 0,
            'pending_contracts': 0,
            'errors': [],
            'warnings': []
        }
        
        if df.empty:
            validation_results['errors'].append("No contracts found")
            return validation_results
        
        # Check for required columns
        required_cols = ['symbol', 'expiry_date', 'strike', 'option_type', 'date_t']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_results['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        try:
            validate_dataframe_schema(df, strict=False)
            validation_results['valid_contracts'] = len(df)
        except Exception as e:
            validation_results['errors'].append(f"Schema validation failed: {e}")
        
        # Check for expired vs pending contracts
        if 'S_T' in df.columns:
            expired_mask = df['S_T'].notna()
            validation_results['expired_contracts'] = expired_mask.sum()
            validation_results['pending_contracts'] = (~expired_mask).sum()
        
        # Check for duplicate contracts
        if 'symbol' in df.columns and 'expiry_date' in df.columns and 'strike' in df.columns and 'option_type' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'expiry_date', 'strike', 'option_type', 'date_t']).sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Found {duplicates} duplicate contracts")
        
        # Check for invalid dates
        if 'date_t' in df.columns and 'expiry_date' in df.columns:
            invalid_dates = (df['date_t'] >= df['expiry_date']).sum()
            if invalid_dates > 0:
                validation_results['errors'].append(f"Found {invalid_dates} contracts with invalid dates")
        
        return validation_results
    
    def persist_reconstructed_chain(self, df: pd.DataFrame, trade_date: str) -> Dict[str, str]:
        """
        Save reconstructed chain to JSON and parquet formats.
        
        Args:
            df: DataFrame to save
            trade_date: Trading date in YYYY-MM-DD format
            
        Returns:
            Dictionary with file paths
        """
        if df.empty:
            logger.warning("No data to persist")
            return {}
        
        # Format date for filename
        date_str = trade_date.replace('-', '')
        
        # Save as JSON (mirror of live schema)
        json_filename = f"chain_{self.symbol}_{date_str}.json"
        json_path = os.path.join(self.output_dirs['json'], json_filename)
        
        # Convert to JSON format similar to live scraper
        json_data = self._convert_to_live_schema(df)
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Save as parquet (normalized table)
        parquet_filename = f"chain_{self.symbol}_{date_str}.parquet"
        parquet_path = os.path.join(self.output_dirs['parquet'], parquet_filename)
        
        df.to_parquet(parquet_path, index=False)
        
        # Append to daily chain norm parquet
        daily_norm_path = os.path.join(self.output_dirs['parquet'], 'daily_chain_norm.parquet')
        
        if os.path.exists(daily_norm_path):
            # Load existing and append
            existing_df = pd.read_parquet(daily_norm_path)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(
                subset=['symbol', 'expiry_date', 'strike', 'option_type', 'date_t'],
                keep='last'
            )
        else:
            combined_df = df
        
        combined_df.to_parquet(daily_norm_path, index=False)
        
        logger.info(f"Persisted chain to {json_path} and {parquet_path}")
        logger.info(f"Updated daily chain norm with {len(df)} contracts")
        
        return {
            'json': json_path,
            'parquet': parquet_path,
            'daily_norm': daily_norm_path
        }
    
    def _convert_to_live_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Convert normalized DataFrame to live scraper JSON schema.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            Dictionary in live schema format
        """
        # Group by expiry date
        result = {}
        
        for expiry_date, group in df.groupby('expiry_date'):
            expiry_str = expiry_date.strftime('%Y-%m-%d')
            
            # Separate CE and PE contracts
            ce_contracts = group[group['option_type'] == 'CE']
            pe_contracts = group[group['option_type'] == 'PE']
            
            result[expiry_str] = {
                'CE': ce_contracts.to_dict('records') if not ce_contracts.empty else [],
                'PE': pe_contracts.to_dict('records') if not pe_contracts.empty else []
            }
        
        return result
    
    def reconstruct_daily_chain(self, eod_file_path: str, trade_date: str) -> Dict[str, Any]:
        """
        Complete pipeline to reconstruct a daily option chain.
        
        Args:
            eod_file_path: Path to EOD derivatives CSV file
            trade_date: Trading date in YYYY-MM-DD format
            
        Returns:
            Dictionary with reconstruction results
        """
        logger.info(f"Starting reconstruction for {trade_date} from {eod_file_path}")
        
        try:
            # Step 1: Load EOD data
            raw_df = self.load_eod_derivatives_data(eod_file_path)
            
            # Step 2: Map to internal schema
            mapped_df = self.map_to_internal_schema(raw_df, trade_date)
            
            # Step 3: Attach underlier data
            enriched_df = self.attach_underlier_data(mapped_df)
            
            # Step 4: Compute IV and Greeks
            computed_df = self.compute_iv_and_greeks(enriched_df)
            
            # Step 5: Generate labels
            labeled_df = self.generate_labels(computed_df)
            
            # Step 6: Validate data
            validation_results = self.validate_reconstructed_data(labeled_df)
            
            # Step 7: Persist results
            file_paths = self.persist_reconstructed_chain(labeled_df, trade_date)
            
            # Compile results
            results = {
                'trade_date': trade_date,
                'symbol': self.symbol,
                'validation': validation_results,
                'file_paths': file_paths,
                'contract_count': len(labeled_df),
                'status': 'success'
            }
            
            logger.info(f"Reconstruction completed successfully for {trade_date}")
            return results
            
        except Exception as e:
            logger.error(f"Reconstruction failed for {trade_date}: {e}")
            return {
                'trade_date': trade_date,
                'symbol': self.symbol,
                'status': 'failed',
                'error': str(e)
            }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Reconstruct historical option chains from EOD data'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='NIFTY',
        help='Underlier symbol (default: NIFTY)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Single date to reconstruct (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--date-range',
        nargs=2,
        metavar=('START_DATE', 'END_DATE'),
        help='Date range to reconstruct (YYYY-MM-DD YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--eod-file',
        type=str,
        help='Path to EOD derivatives CSV file'
    )
    
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.06,
        help='Risk-free rate for option pricing (default: 0.06)'
    )
    
    args = parser.parse_args()
    
    if not args.date and not args.date_range:
        parser.error("Must specify either --date or --date-range")
    
    if args.date_range and len(args.date_range) != 2:
        parser.error("--date-range requires exactly two dates")
    
    # Initialize reconstructor
    reconstructor = EODChainReconstructor(
        symbol=args.symbol,
        risk_free_rate=args.risk_free_rate
    )
    
    if args.date:
        # Single date reconstruction
        if not args.eod_file:
            parser.error("--eod-file required for single date reconstruction")
        
        results = reconstructor.reconstruct_daily_chain(args.eod_file, args.date)
        print(json.dumps(results, indent=2, default=str))
        
    elif args.date_range:
        # Date range reconstruction
        start_date = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        
        # Generate date range
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Filter for trading days (weekdays)
        trading_dates = [d for d in date_range if d.weekday() < 5]
        
        print(f"Reconstructing chains for {len(trading_dates)} trading days...")
        
        all_results = []
        for date in tqdm(trading_dates):
            date_str = date.strftime('%Y-%m-%d')
            
            # For now, we'll need EOD files for each date
            # In practice, this would be integrated with a data source
            print(f"Note: EOD file needed for {date_str}")
            
            # Placeholder for actual reconstruction
            # results = reconstructor.reconstruct_daily_chain(eod_file, date_str)
            # all_results.append(results)
        
        print(f"Date range reconstruction completed for {len(trading_dates)} dates")


if __name__ == "__main__":
    main()
