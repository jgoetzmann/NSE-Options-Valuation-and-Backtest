#!/usr/bin/env python3
"""
Snapshot Normalization Module for Mode A (SB-CS)
===============================================

This module normalizes live JSON snapshots from nse_options_scraper.py into tabular format
for synthetic backtesting. It parses CE/PE blocks into one row per contract and extracts
all necessary fields for analysis.

Key Functions:
- normalize_snapshot_json(): Convert JSON snapshot to normalized DataFrame
- extract_contract_data(): Extract individual contract data from CE/PE blocks
- validate_normalized_data(): Ensure data quality and completeness
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.schemas import (
    NORMALIZED_TABLE_SCHEMA, 
    PANDAS_DTYPES, 
    validate_dataframe_schema,
    add_derived_features
)


class SnapshotNormalizer:
    """
    Normalizes NSE options snapshot JSON into tabular format.
    
    This class handles the conversion from the nested JSON structure returned by
    nse_options_scraper.py into a flat DataFrame suitable for analysis.
    """
    
    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol.upper()
        self.timestamp = datetime.now()
        
    def load_snapshot_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load snapshot JSON file.
        
        Parameters:
        -----------
        file_path : str
            Path to the JSON snapshot file
            
        Returns:
        --------
        Dict[str, Any]
            Loaded JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load snapshot JSON: {e}")
    
    def extract_contract_data(self, ce_data: Dict[str, Any], pe_data: Dict[str, Any], 
                            expiry: str, underlying_value: float) -> List[Dict[str, Any]]:
        """
        Extract contract data from CE and PE blocks.
        
        Parameters:
        -----------
        ce_data : Dict[str, Any]
            Call options data block
        pe_data : Dict[str, Any]
            Put options data block
        expiry : str
            Expiry date string
        underlying_value : float
            Current underlying asset price
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of contract dictionaries
        """
        contracts = []
        
        # Process CE (Call) options
        if ce_data and 'data' in ce_data:
            for contract in ce_data['data']:
                contract_dict = self._extract_single_contract(
                    contract, expiry, underlying_value, 'CE'
                )
                if contract_dict:
                    contracts.append(contract_dict)
        
        # Process PE (Put) options
        if pe_data and 'data' in pe_data:
            for contract in pe_data['data']:
                contract_dict = self._extract_single_contract(
                    contract, expiry, underlying_value, 'PE'
                )
                if contract_dict:
                    contracts.append(contract_dict)
        
        return contracts
    
    def _extract_single_contract(self, contract: Dict[str, Any], expiry: str, 
                                underlying_value: float, option_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract data for a single contract.
        
        Parameters:
        -----------
        contract : Dict[str, Any]
            Individual contract data
        expiry : str
            Expiry date string
        underlying_value : float
            Current underlying asset price
        option_type : str
            Option type (CE or PE)
            
        Returns:
        --------
        Optional[Dict[str, Any]]
            Extracted contract data or None if invalid
        """
        try:
            # Skip contracts with missing critical data
            if not all(key in contract for key in ['strikePrice', 'lastPrice']):
                return None
            
            # Extract basic contract information
            contract_dict = {
                'symbol': self.symbol,
                'expiry_date': expiry,
                'strike': float(contract.get('strikePrice', 0)),
                'option_type': option_type,
                'premium_t': float(contract.get('lastPrice', 0)),
                'openInterest': int(contract.get('openInterest', 0)),
                'changeinOpenInterest': int(contract.get('changeinOpenInterest', 0)),
                'totalTradedVolume': int(contract.get('totalTradedVolume', 0)),
                'bidPrice': float(contract.get('bidprice', 0)) if contract.get('bidprice') else None,
                'askPrice': float(contract.get('askPrice', 0)) if contract.get('askPrice') else None,
                'lastPrice': float(contract.get('lastPrice', 0)),
                'impliedVolatility': float(contract.get('impliedVolatility', 0)) if contract.get('impliedVolatility') else None,
                'S_t': underlying_value,
                'date_t': self.timestamp,
                'synthetic_flag': 1,  # Mode A is synthetic
                'valid_horizon': 1    # Will be updated based on horizon validation
            }
            
            # Skip contracts with invalid data
            if (contract_dict['strike'] <= 0 or 
                contract_dict['premium_t'] <= 0 or
                contract_dict['openInterest'] <= 0):
                return None
            
            return contract_dict
            
        except (ValueError, TypeError) as e:
            # Skip contracts with data conversion errors
            return None
    
    def normalize_snapshot_json(self, json_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Normalize the complete snapshot JSON into a DataFrame.
        
        Parameters:
        -----------
        json_data : Dict[str, Any]
            Raw JSON data from NSE scraper
            
        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame with one row per contract
        """
        all_contracts = []
        
        # Extract records from the JSON structure
        records = json_data.get('records', {})
        
        # Get underlying value
        underlying_value = float(records.get('underlyingValue', 0))
        if underlying_value <= 0:
            raise ValueError("Invalid underlying value in snapshot")
        
        # Process each expiry date
        expiry_dates = records.get('expiryDates', [])
        
        for expiry in expiry_dates:
            # Get CE and PE data for this expiry
            ce_data = records.get('data', {}).get(expiry, {}).get('CE', {})
            pe_data = records.get('data', {}).get(expiry, {}).get('PE', {})
            
            # Extract contracts for this expiry
            contracts = self.extract_contract_data(ce_data, pe_data, expiry, underlying_value)
            all_contracts.extend(contracts)
        
        # Convert to DataFrame
        if not all_contracts:
            raise ValueError("No valid contracts found in snapshot")
        
        df = pd.DataFrame(all_contracts)
        
        # Convert date columns
        df['date_t'] = pd.to_datetime(df['date_t'])
        df['expiry_date'] = pd.to_datetime(df['expiry_date'], format='%d-%b-%Y')
        
        # Set dtypes according to schema
        for col, dtype in PANDAS_DTYPES.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    # Handle nullable columns
                    if 'Int' in dtype:
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add derived features
        df = add_derived_features(df)
        
        return df
    
    def validate_normalized_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the normalized data for quality and completeness.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Normalized DataFrame to validate
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        validation_result = validate_dataframe_schema(df, strict=False)
        
        # Additional validation specific to snapshots
        if len(df) == 0:
            validation_result['warnings'].append("No contracts found in snapshot")
        
        # Check for reasonable data ranges
        if 'strike' in df.columns:
            min_strike = df['strike'].min()
            max_strike = df['strike'].max()
            if min_strike <= 0 or max_strike > 100000:
                validation_result['warnings'].append(
                    f"Strike prices seem unreasonable: min={min_strike}, max={max_strike}"
                )
        
        if 'premium_t' in df.columns:
            negative_premium = df[df['premium_t'] <= 0]
            if len(negative_premium) > 0:
                validation_result['warnings'].append(
                    f"Found {len(negative_premium)} contracts with non-positive premium"
                )
        
        # Check option type distribution
        if 'option_type' in df.columns:
            option_counts = df['option_type'].value_counts()
            if len(option_counts) != 2:
                validation_result['warnings'].append(
                    f"Expected 2 option types, found: {option_counts.index.tolist()}"
                )
        
        return validation_result
    
    def save_normalized_snapshot(self, df: pd.DataFrame, output_dir: str = "outputs/csv") -> str:
        """
        Save normalized snapshot to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Normalized DataFrame to save
        output_dir : str
            Output directory for CSV file
            
        Returns:
        --------
        str
            Path to saved CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_norm_{self.symbol}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        return filepath


def normalize_snapshot_from_file(file_path: str, symbol: str = "NIFTY") -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to normalize a snapshot from a JSON file.
    
    Parameters:
    -----------
    file_path : str
        Path to the JSON snapshot file
    symbol : str
        Underlying symbol
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        Normalized DataFrame and path to saved CSV
    """
    normalizer = SnapshotNormalizer(symbol)
    
    # Load and normalize
    json_data = normalizer.load_snapshot_json(file_path)
    df = normalizer.normalize_snapshot_json(json_data)
    
    # Validate
    validation_result = normalizer.validate_normalized_data(df)
    if not validation_result['is_valid']:
        print("Warning: Data validation issues found:")
        for error in validation_result['errors']:
            print(f"  Error: {error}")
        for warning in validation_result['warnings']:
            print(f"  Warning: {warning}")
    
    # Save
    csv_path = normalizer.save_normalized_snapshot(df)
    
    return df, csv_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize NSE options snapshot JSON")
    parser.add_argument("json_file", help="Path to JSON snapshot file")
    parser.add_argument("--symbol", default="NIFTY", help="Underlying symbol")
    parser.add_argument("--output-dir", default="outputs/csv", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        df, csv_path = normalize_snapshot_from_file(args.json_file, args.symbol)
        
        print(f"✅ Successfully normalized snapshot")
        print(f"   Input: {args.json_file}")
        print(f"   Output: {csv_path}")
        print(f"   Contracts: {len(df)}")
        print(f"   Symbol: {args.symbol}")
        
        # Show summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"   CE contracts: {len(df[df['option_type'] == 'CE'])}")
        print(f"   PE contracts: {len(df[df['option_type'] == 'PE'])}")
        print(f"   Expiry dates: {df['expiry_date'].nunique()}")
        print(f"   Strike range: {df['strike'].min():.0f} - {df['strike'].max():.0f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
