#!/usr/bin/env python3
"""
Central Schema Definitions for NSE Options Backtest Project
==========================================================

This module defines the standardized data structures, column names, and data types
used across all three execution modes (SB-CS, ML-Live, EOD-True).

Key Concepts:
- Primary Key (PK): (symbol, expiry_date, strike, option_type, date_t)
- As-of date (date_t): Trading date when features and entry premium are observed
- Expiry date (expiry_T): Contract settlement date for payoff computation
- Horizon (d): Days before expiry for synthetic backtest runs
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

# =============================================================================
# COLUMN DEFINITIONS
# =============================================================================

# Core contract identification
CONTRACT_COLUMNS = {
    'date_t': 'date',                    # Trading date (as-of date)
    'expiry_date': 'date',               # Contract expiry date
    'horizon_d': 'Int64',                # Days to expiry (nullable for synthetic runs)
    'symbol': 'string',                  # Underlying symbol (NIFTY, BANKNIFTY, etc.)
    'option_type': 'string',             # CE/PE
    'strike': 'float64',                 # Strike price
    'synthetic_flag': 'Int8',            # 1 for synthetic runs, 0 for true EOD
    'valid_horizon': 'Int8'              # 1 if horizon is valid for this contract
}

# Market data columns
MARKET_DATA_COLUMNS = {
    'premium_t': 'float64',              # Entry price (EOD settle in Mode C, snapshot in Mode A)
    'openInterest': 'Int64',             # Open interest
    'changeinOpenInterest': 'Int64',     # Change in open interest (nullable)
    'totalTradedVolume': 'Int64',        # Total traded volume
    'bidPrice': 'float64',               # Bid price (nullable)
    'askPrice': 'float64',               # Ask price (nullable)
    'lastPrice': 'float64',              # Last traded price
    'impliedVolatility': 'float64'       # Implied volatility (percentage)
}

# Underlier data columns
UNDERLIER_COLUMNS = {
    'S_t': 'float64',                    # Underlier price at date_t
    'S_T': 'float64',                    # Underlier price at expiry (nullable until expiry)
    'ret_1d': 'float64',                 # 1-day return (nullable)
    'ret_5d': 'float64',                 # 5-day return (nullable)
    'rv_10d': 'float64',                 # 10-day realized volatility (nullable)
    'rv_20d': 'float64'                  # 20-day realized volatility (nullable)
}

# Computed features columns
COMPUTED_FEATURES_COLUMNS = {
    'iv_est_t': 'float64',               # Estimated IV (from premium inversion)
    'delta': 'float64',                  # Delta Greek
    'gamma': 'float64',                  # Gamma Greek
    'theta': 'float64',                  # Theta Greek
    'vega': 'float64',                   # Vega Greek
    'rho': 'float64',                    # Rho Greek
    'cost_bps': 'float64'                # Transaction costs in basis points
}

# Label columns (computed at expiry)
LABEL_COLUMNS = {
    'payoff_T': 'float64',               # Option payoff at expiry (nullable until expiry)
    'PnL': 'float64',                    # Profit/Loss (nullable until expiry)
    'ROI': 'float64',                    # Return on Investment (nullable until expiry)
    'POP_label': 'Int8'                  # Profit or Loss label: 1 if PnL >= 0, else 0
}

# =============================================================================
# COMPLETE SCHEMA
# =============================================================================

# Combine all column definitions
NORMALIZED_TABLE_SCHEMA = {
    **CONTRACT_COLUMNS,
    **MARKET_DATA_COLUMNS,
    **UNDERLIER_COLUMNS,
    **COMPUTED_FEATURES_COLUMNS,
    **LABEL_COLUMNS
}

# =============================================================================
# DATA TYPE MAPPINGS
# =============================================================================

# Pandas dtype mappings for the schema
PANDAS_DTYPES = {
    'date_t': 'datetime64[ns]',
    'expiry_date': 'datetime64[ns]',
    'horizon_d': 'Int64',
    'symbol': 'string',
    'option_type': 'string',
    'strike': 'float64',
    'premium_t': 'float64',
    'openInterest': 'Int64',
    'changeinOpenInterest': 'Int64',
    'totalTradedVolume': 'Int64',
    'bidPrice': 'float64',
    'askPrice': 'float64',
    'lastPrice': 'float64',
    'impliedVolatility': 'float64',
    'S_t': 'float64',
    'S_T': 'float64',
    'ret_1d': 'float64',
    'ret_5d': 'float64',
    'rv_10d': 'float64',
    'rv_20d': 'float64',
    'iv_est_t': 'float64',
    'delta': 'float64',
    'gamma': 'float64',
    'theta': 'float64',
    'vega': 'float64',
    'rho': 'float64',
    'cost_bps': 'float64',
    'payoff_T': 'float64',
    'PnL': 'float64',
    'ROI': 'float64',
    'POP_label': 'Int8',
    'synthetic_flag': 'Int8',
    'valid_horizon': 'Int8'
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_dataframe_schema(df: pd.DataFrame, strict: bool = True) -> Dict[str, Any]:
    """
    Validate that a DataFrame conforms to the expected schema.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    strict : bool
        If True, raise errors for missing columns. If False, return warnings.
    
    Returns:
    --------
    Dict[str, Any]
        Validation results with status and details
    """
    validation_result = {
        'is_valid': True,
        'missing_columns': [],
        'type_mismatches': [],
        'warnings': []
    }
    
    # Check for required columns
    required_columns = [
        'date_t', 'expiry_date', 'symbol', 'option_type', 'strike',
        'premium_t', 'openInterest', 'totalTradedVolume'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            validation_result['missing_columns'].append(col)
            if strict:
                validation_result['is_valid'] = False
    
    # Check data types for present columns
    for col, expected_dtype in PANDAS_DTYPES.items():
        if col in df.columns:
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                validation_result['type_mismatches'].append({
                    'column': col,
                    'expected': expected_dtype,
                    'actual': actual_dtype
                })
                if strict:
                    validation_result['is_valid'] = False
    
    # Check for data quality issues
    if 'date_t' in df.columns and 'expiry_date' in df.columns:
        invalid_expiry = df[df['date_t'] >= df['expiry_date']]
        if len(invalid_expiry) > 0:
            validation_result['warnings'].append(
                f"Found {len(invalid_expiry)} rows where date_t >= expiry_date"
            )
    
    if 'premium_t' in df.columns:
        negative_premium = df[df['premium_t'] <= 0]
        if len(negative_premium) > 0:
            validation_result['warnings'].append(
                f"Found {len(negative_premium)} rows with non-positive premium_t"
            )
    
    return validation_result

def get_feature_columns(include_interactions: bool = False) -> List[str]:
    """
    Get the list of feature columns for ML training.
    
    Parameters:
    -----------
    include_interactions : bool
        Whether to include interaction features
        
    Returns:
    --------
    List[str]
        List of feature column names
    """
    base_features = [
        'moneyness', 'ttm_days', 'iv_est_t', 'delta', 'gamma', 'theta', 'vega', 'rho',
        'openInterest', 'totalTradedVolume', 'premium_t', 'ret_5d', 'rv_10d'
    ]
    
    if include_interactions:
        interaction_features = [
            'moneyness*ttm_days', 'delta*theta', 'spread_pct*oi_rank'
        ]
        return base_features + interaction_features
    
    return base_features

def get_label_columns() -> List[str]:
    """Get the list of label columns for ML training."""
    return ['POP_label', 'PnL', 'ROI']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_empty_normalized_table() -> pd.DataFrame:
    """
    Create an empty DataFrame with the correct schema.
    
    Returns:
    --------
    pd.DataFrame
        Empty DataFrame with correct column names and dtypes
    """
    df = pd.DataFrame(columns=list(NORMALIZED_TABLE_SCHEMA.keys()))
    
    # Set dtypes
    for col, dtype in PANDAS_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    return df

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the normalized table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with basic columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # Moneyness (S_t / strike)
    if 'S_t' in df.columns and 'strike' in df.columns:
        df['moneyness'] = df['S_t'] / df['strike']
    
    # Time to maturity in days
    if 'date_t' in df.columns and 'expiry_date' in df.columns:
        df['ttm_days'] = (df['expiry_date'] - df['date_t']).dt.days
    
    # Spread percentage
    if 'bidPrice' in df.columns and 'askPrice' in df.columns:
        df['spread_pct'] = (df['askPrice'] - df['bidPrice']) / ((df['askPrice'] + df['bidPrice']) / 2)
    
    # OI rank (percentile within each date)
    if 'openInterest' in df.columns and 'date_t' in df.columns:
        df['oi_rank'] = df.groupby('date_t')['openInterest'].rank(pct=True)
    
    return df

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""
    symbol: str
    synthetic_flag: bool = False
    date_span: Dict[str, str] = None  # start, end
    horizons: List[int] = None
    filters: Dict[str, Any] = None
    costs: Dict[str, Any] = None
    portfolio: Dict[str, Any] = None
    reconstruction: Dict[str, Any] = None
    features: Dict[str, Any] = None
    risk: Dict[str, Any] = None
    output: Dict[str, Any] = None
    validation: Dict[str, Any] = None
    metrics: Dict[str, Any] = None
    quality: Dict[str, Any] = None
    ml_prep: Dict[str, Any] = None
    methodology: Dict[str, Any] = None

def validate_backtest_config(config: BacktestConfig) -> Dict[str, Any]:
    """
    Validate backtest configuration.
    
    Parameters:
    -----------
    config : BacktestConfig
        Configuration to validate
        
    Returns:
    --------
    Dict[str, Any]
        Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate symbol
    valid_symbols = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SENSEX']
    if config.symbol not in valid_symbols:
        validation_result['errors'].append(f"Invalid symbol: {config.symbol}")
        validation_result['is_valid'] = False
    
    # Validate horizons
    if not config.horizons or not all(isinstance(h, int) and h > 0 for h in config.horizons):
        validation_result['errors'].append("Horizons must be positive integers")
        validation_result['is_valid'] = False
    
    # Validate date span
    try:
        if config.date_span:
            start_date = pd.to_datetime(config.date_span['start'])
            end_date = pd.to_datetime(config.date_span['end'])
            if start_date > end_date:
                validation_result['errors'].append("Start date must be before or equal to end date")
                validation_result['is_valid'] = False
    except (KeyError, ValueError) as e:
        validation_result['errors'].append(f"Invalid date format: {e}")
        validation_result['is_valid'] = False
    
    return validation_result
