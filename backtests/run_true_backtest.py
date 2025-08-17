#!/usr/bin/env python3
"""
True EOD Backtest Runner for Mode C (EOD-True)
==============================================

This module runs historically accurate backtests using reconstructed option chains
from Mode C. It loads historical data, applies filters, constructs portfolios,
and computes performance metrics.

Key Functions:
- load_reconstructed_data: Load historical option chains
- apply_filters: Apply trading filters and constraints
- construct_portfolio: Rank and select positions
- compute_performance: Calculate KPIs and metrics
- generate_reports: Create comprehensive backtest reports

Usage:
    python run_true_backtest.py --config configs/backtest_true.yml
    python run_true_backtest.py --config configs/backtest_true.yml --start-date 2024-01-01 --end-date 2024-12-31
"""

import os
import sys
import yaml
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
    NORMALIZED_TABLE_SCHEMA, 
    PANDAS_DTYPES,
    validate_dataframe_schema,
    BacktestConfig,
    validate_backtest_config
)
from data_pipeline.attach_underlier_features import UnderlierFeatureAttacher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrueBacktestRunner:
    """
    Runs true EOD backtests using reconstructed historical option chains.
    
    This class implements the complete backtesting pipeline for Mode C,
    including data loading, filtering, portfolio construction, and
    performance evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the backtest runner.
        
        Args:
            config_path: Path to backtest configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate configuration
        validation_results = validate_backtest_config(self.config)
        if validation_results.get('errors'):
            raise ValueError(f"Configuration validation failed: {validation_results['errors']}")
        
        # Initialize components
        self.feature_attacher = UnderlierFeatureAttacher(self.config.symbol)
        
        # Output directories
        self.output_dir = "outputs"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TrueBacktestRunner for {self.config.symbol}")
    
    def _load_config(self) -> BacktestConfig:
        """Load and parse backtest configuration."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to BacktestConfig dataclass
        return BacktestConfig(**config_dict)
    
    def load_reconstructed_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load reconstructed option chains for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical option data
        """
        logger.info(f"Loading reconstructed data from {start_date} to {end_date}")
        
        # Try to load from daily chain norm parquet first
        daily_norm_path = "reconstructed/parquet/daily_chain_norm.parquet"
        
        if os.path.exists(daily_norm_path):
            try:
                # Load and filter by date range
                df = pd.read_parquet(daily_norm_path)
                
                # Convert date columns to datetime if needed
                if 'date_t' in df.columns:
                    df['date_t'] = pd.to_datetime(df['date_t'])
                
                # Filter by date range
                mask = (
                    (df['date_t'] >= start_date) &
                    (df['date_t'] <= end_date) &
                    (df['symbol'] == self.config.symbol)
                )
                
                filtered_df = df[mask].copy()
                
                if not filtered_df.empty:
                    logger.info(f"Loaded {len(filtered_df)} contracts from daily chain norm")
                    return filtered_df
                else:
                    logger.warning("No data found in date range from daily chain norm")
            
            except Exception as e:
                logger.warning(f"Error loading from daily chain norm: {e}")
        
        # Fallback: try to load individual parquet files
        logger.info("Attempting to load individual parquet files...")
        
        parquet_dir = "reconstructed/parquet"
        if not os.path.exists(parquet_dir):
            raise FileNotFoundError(f"Reconstructed data directory not found: {parquet_dir}")
        
        all_data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Generate date range
        current_dt = start_dt
        while current_dt <= end_dt:
            if current_dt.weekday() < 5:  # Skip weekends
                date_str = current_dt.strftime('%Y%m%d')
                parquet_file = f"chain_{self.config.symbol}_{date_str}.parquet"
                parquet_path = os.path.join(parquet_dir, parquet_file)
                
                if os.path.exists(parquet_path):
                    try:
                        daily_df = pd.read_parquet(parquet_path)
                        all_data.append(daily_df)
                        logger.debug(f"Loaded {len(daily_df)} contracts for {current_dt.date()}")
                    except Exception as e:
                        logger.warning(f"Error loading {parquet_path}: {e}")
            
            current_dt += timedelta(days=1)
        
        if not all_data:
            raise FileNotFoundError(f"No reconstructed data found for {self.config.symbol} in date range")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total contracts from individual files")
        
        return combined_df
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trading filters and constraints.
        
        Args:
            df: DataFrame with option data
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        logger.info("Applying trading filters...")
        initial_count = len(df)
        
        # Apply basic filters
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Minimum open interest
        if hasattr(self.config.filters, 'min_oi') and self.config.filters.min_oi:
            mask &= (df['openInterest'] >= self.config.filters.min_oi)
        
        # Minimum premium
        if hasattr(self.config.filters, 'min_premium') and self.config.filters.min_premium:
            mask &= (df['premium_t'] >= self.config.filters.min_premium)
        
        # Maximum spread percentage
        if hasattr(self.config.filters, 'max_spread_pct') and self.config.filters.max_spread_pct:
            # Calculate spread percentage if bid/ask available
            if 'bidPrice' in df.columns and 'askPrice' in df.columns:
                spread_pct = (df['askPrice'] - df['bidPrice']) / df['premium_t']
                mask &= (spread_pct <= self.config.filters.max_spread_pct)
        
        # Time to maturity bounds
        if hasattr(self.config.filters, 'ttm_bounds'):
            ttm_bounds = self.config.filters.ttm_bounds
            if hasattr(ttm_bounds, 'min') and ttm_bounds.min:
                mask &= (df['ttm_days'] >= ttm_bounds.min)
            if hasattr(ttm_bounds, 'max') and ttm_bounds.max:
                mask &= (df['ttm_days'] <= ttm_bounds.max)
        
        # Minimum volume
        if hasattr(self.config.filters, 'min_volume') and self.config.filters.min_volume:
            mask &= (df['totalTradedVolume'] >= self.config.filters.min_volume)
        
        # Maximum implied volatility
        if hasattr(self.config.filters, 'max_iv') and self.config.filters.max_iv:
            if 'iv_est_t' in df.columns:
                mask &= (df['iv_est_t'] <= self.config.filters.max_iv)
        
        # Apply mask
        filtered_df = df[mask].copy()
        
        logger.info(f"Applied filters: {initial_count} -> {len(filtered_df)} contracts")
        return filtered_df
    
    def compute_ranking_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute ranking score for portfolio construction.
        
        Args:
            df: DataFrame with option data
            
        Returns:
            Series with ranking scores
        """
        if df.empty:
            return pd.Series(dtype=float)
        
        rank_score = self.config.portfolio.rank_score
        
        if rank_score == "pct_diff_times_confidence":
            # Use percentage difference from theoretical value times confidence
            if 'pct_diff' in df.columns and 'confidence' in df.columns:
                return df['pct_diff'] * df['confidence']
            else:
                logger.warning("pct_diff or confidence not available, using simple ranking")
                return pd.Series(1.0, index=df.index)
        
        elif rank_score == "iv_rank":
            # Rank by implied volatility (lower IV = higher rank for selling)
            if 'iv_est_t' in df.columns:
                return -df['iv_est_t']  # Negative for descending sort
        
        elif rank_score == "theta_rank":
            # Rank by theta (higher theta = higher rank for selling)
            if 'theta' in df.columns:
                return df['theta']
        
        elif rank_score == "gamma_rank":
            # Rank by gamma (lower gamma = higher rank for selling)
            if 'gamma' in df.columns:
                return -df['gamma']  # Negative for descending sort
        
        else:
            logger.warning(f"Unknown ranking score: {rank_score}, using random ranking")
            return pd.Series(np.random.random(len(df)), index=df.index)
    
    def construct_portfolio(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        Construct portfolio for a specific date.
        
        Args:
            df: DataFrame with option data for the date
            date: Trading date
            
        Returns:
            DataFrame with selected positions
        """
        if df.empty:
            return df
        
        logger.info(f"Constructing portfolio for {date}")
        
        # Compute ranking scores
        df['ranking_score'] = self.compute_ranking_score(df)
        
        # Sort by ranking score (descending)
        df = df.sort_values('ranking_score', ascending=False).reset_index(drop=True)
        
        # Apply position limits
        max_positions = getattr(self.config.portfolio, 'daily_max_positions', 20)
        if len(df) > max_positions:
            df = df.head(max_positions).copy()
            logger.info(f"Limited portfolio to {max_positions} positions")
        
        # Add portfolio metadata
        df['portfolio_date'] = date
        df['position_rank'] = range(1, len(df) + 1)
        
        # Apply position sizing if configured
        if hasattr(self.config.portfolio, 'position_sizing'):
            sizing_config = self.config.portfolio.position_sizing
            if sizing_config == "equal_weight":
                df['position_weight'] = 1.0 / len(df)
            elif sizing_config == "rank_weight":
                # Weight inversely proportional to rank
                df['position_weight'] = 1.0 / df['position_rank']
                df['position_weight'] = df['position_weight'] / df['position_weight'].sum()
            else:
                df['position_weight'] = 1.0 / len(df)
        else:
            df['position_weight'] = 1.0 / len(df)
        
        logger.info(f"Constructed portfolio with {len(df)} positions")
        return df
    
    def compute_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute performance metrics for the portfolio.
        
        Args:
            portfolio_df: DataFrame with portfolio positions
            
        Returns:
            Dictionary with performance metrics
        """
        if portfolio_df.empty:
            return {}
        
        metrics = {}
        
        # Basic portfolio stats
        metrics['total_positions'] = len(portfolio_df)
        metrics['total_premium'] = portfolio_df['premium_t'].sum()
        
        # Performance metrics (only for expired contracts)
        expired_mask = portfolio_df['POP_label'].notna()
        if expired_mask.sum() > 0:
            expired_df = portfolio_df[expired_mask]
            
            # PnL metrics
            metrics['total_pnl'] = expired_df['PnL'].sum()
            metrics['avg_pnl'] = expired_df['PnL'].mean()
            metrics['pnl_std'] = expired_df['PnL'].std()
            
            # ROI metrics
            metrics['total_roi'] = expired_df['ROI'].sum()
            metrics['avg_roi'] = expired_df['ROI'].mean()
            metrics['roi_std'] = expired_df['ROI'].std()
            
            # Hit rate
            metrics['hit_rate'] = expired_df['POP_label'].mean()
            metrics['total_hits'] = expired_df['POP_label'].sum()
            metrics['total_misses'] = len(expired_df) - metrics['total_hits']
            
            # Win/Loss analysis
            wins = expired_df[expired_df['POP_label'] == 1]
            losses = expired_df[expired_df['POP_label'] == 0]
            
            if len(wins) > 0:
                metrics['avg_win'] = wins['PnL'].mean()
                metrics['max_win'] = wins['PnL'].max()
            else:
                metrics['avg_win'] = 0.0
                metrics['max_win'] = 0.0
            
            if len(losses) > 0:
                metrics['avg_loss'] = losses['PnL'].mean()
                metrics['max_loss'] = losses['PnL'].min()
            else:
                metrics['avg_loss'] = 0.0
                metrics['max_loss'] = 0.0
            
            # Risk metrics
            if metrics['pnl_std'] > 0:
                metrics['sharpe_ratio'] = metrics['avg_pnl'] / metrics['pnl_std'] * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Sortino ratio (using downside deviation)
            downside_returns = expired_df[expired_df['PnL'] < 0]['PnL']
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if downside_std > 0:
                    metrics['sortino_ratio'] = metrics['avg_pnl'] / downside_std * np.sqrt(252)
                else:
                    metrics['sortino_ratio'] = 0.0
            else:
                metrics['sortino_ratio'] = float('inf')
        
        else:
            # No expired contracts
            metrics['total_pnl'] = 0.0
            metrics['avg_pnl'] = 0.0
            metrics['hit_rate'] = 0.0
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
        
        # Portfolio composition
        if 'option_type' in portfolio_df.columns:
            ce_count = (portfolio_df['option_type'] == 'CE').sum()
            pe_count = (portfolio_df['option_type'] == 'PE').sum()
            metrics['ce_positions'] = ce_count
            metrics['pe_positions'] = pe_count
        
        if 'ttm_days' in portfolio_df.columns:
            metrics['avg_ttm'] = portfolio_df['ttm_days'].mean()
            metrics['min_ttm'] = portfolio_df['ttm_days'].min()
            metrics['max_ttm'] = portfolio_df['ttm_days'].max()
        
        return metrics
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run the complete backtest.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting true EOD backtest from {start_date} to {end_date}")
        
        try:
            # Step 1: Load reconstructed data
            df = self.load_reconstructed_data(start_date, end_date)
            
            # Step 2: Apply filters
            filtered_df = self.apply_filters(df)
            
            if filtered_df.empty:
                logger.warning("No data remaining after filtering")
                return {
                    'status': 'no_data',
                    'start_date': start_date,
                    'end_date': end_date,
                    'symbol': self.config.symbol
                }
            
            # Step 3: Group by date and construct portfolios
            daily_portfolios = {}
            daily_metrics = {}
            
            unique_dates = sorted(filtered_df['date_t'].dt.date.unique())
            logger.info(f"Processing {len(unique_dates)} trading days")
            
            for date in tqdm(unique_dates, desc="Processing dates"):
                date_str = date.strftime('%Y-%m-%d')
                date_mask = filtered_df['date_t'].dt.date == date
                daily_df = filtered_df[date_mask].copy()
                
                if not daily_df.empty:
                    # Construct portfolio for the date
                    portfolio = self.construct_portfolio(daily_df, date_str)
                    daily_portfolios[date_str] = portfolio
                    
                    # Compute performance metrics
                    metrics = self.compute_performance_metrics(portfolio)
                    daily_metrics[date_str] = metrics
            
            # Step 4: Aggregate results
            all_portfolios = pd.concat(daily_portfolios.values(), ignore_index=True)
            
            # Step 5: Compute overall metrics
            overall_metrics = self._compute_overall_metrics(daily_metrics, all_portfolios)
            
            # Step 6: Generate reports
            report_paths = self._generate_reports(
                all_portfolios, daily_metrics, overall_metrics, start_date, end_date
            )
            
            # Compile results
            results = {
                'status': 'success',
                'start_date': start_date,
                'end_date': end_date,
                'symbol': self.config.symbol,
                'total_trading_days': len(unique_dates),
                'total_contracts': len(all_portfolios),
                'daily_metrics': daily_metrics,
                'overall_metrics': overall_metrics,
                'report_paths': report_paths
            }
            
            logger.info("True EOD backtest completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                'status': 'failed',
                'start_date': start_date,
                'end_date': end_date,
                'symbol': self.config.symbol,
                'error': str(e)
            }
    
    def _compute_overall_metrics(self, daily_metrics: Dict, all_portfolios: pd.DataFrame) -> Dict[str, Any]:
        """Compute overall performance metrics across all dates."""
        if not daily_metrics:
            return {}
        
        overall = {}
        
        # Aggregate daily metrics
        total_pnl = sum(m.get('total_pnl', 0) for m in daily_metrics.values())
        total_positions = sum(m.get('total_positions', 0) for m in daily_metrics.values())
        
        overall['total_pnl'] = total_pnl
        overall['total_positions'] = total_positions
        
        # Average daily metrics
        valid_days = [m for m in daily_metrics.values() if m.get('total_positions', 0) > 0]
        if valid_days:
            overall['avg_daily_pnl'] = np.mean([m.get('total_pnl', 0) for m in valid_days])
            overall['avg_daily_positions'] = np.mean([m.get('total_positions', 0) for m in valid_days])
            overall['avg_hit_rate'] = np.mean([m.get('hit_rate', 0) for m in valid_days])
            overall['avg_sharpe'] = np.mean([m.get('sharpe_ratio', 0) for m in valid_days])
        
        # Portfolio composition
        if 'option_type' in all_portfolios.columns:
            overall['total_ce'] = (all_portfolios['option_type'] == 'CE').sum()
            overall['total_pe'] = (all_portfolios['option_type'] == 'PE').sum()
        
        return overall
    
    def _generate_reports(self, all_portfolios: pd.DataFrame, daily_metrics: Dict, 
                         overall_metrics: Dict, start_date: str, end_date: str) -> Dict[str, str]:
        """Generate backtest reports and save to files."""
        report_paths = {}
        
        # Save portfolio data
        portfolio_filename = f"true_backtest_{self.config.symbol}_{start_date}_{end_date}.csv"
        portfolio_path = os.path.join(self.output_dir, portfolio_filename)
        all_portfolios.to_csv(portfolio_path, index=False)
        report_paths['portfolio_csv'] = portfolio_path
        
        # Save daily metrics
        metrics_filename = f"true_backtest_metrics_{self.config.symbol}_{start_date}_{end_date}.json"
        metrics_path = os.path.join(self.output_dir, metrics_filename)
        with open(metrics_path, 'w') as f:
            json.dump(daily_metrics, f, indent=2, default=str)
        report_paths['daily_metrics_json'] = metrics_path
        
        # Generate summary report
        summary_filename = f"true_backtest_summary_{self.config.symbol}_{start_date}_{end_date}.txt"
        summary_path = os.path.join(self.output_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(f"True EOD Backtest Summary for {self.config.symbol}\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Performance:\n")
            f.write(f"Total PnL: {overall_metrics.get('total_pnl', 0):.2f}\n")
            f.write(f"Total Positions: {overall_metrics.get('total_positions', 0)}\n")
            f.write(f"Average Daily PnL: {overall_metrics.get('avg_daily_pnl', 0):.2f}\n")
            f.write(f"Average Hit Rate: {overall_metrics.get('avg_hit_rate', 0):.2%}\n")
            f.write(f"Average Sharpe Ratio: {overall_metrics.get('avg_sharpe', 0):.2f}\n\n")
            
            f.write("Portfolio Composition:\n")
            f.write(f"CE Positions: {overall_metrics.get('total_ce', 0)}\n")
            f.write(f"PE Positions: {overall_metrics.get('total_pe', 0)}\n\n")
            
            f.write("Configuration:\n")
            f.write(f"Symbol: {self.config.symbol}\n")
            f.write(f"Filters: {self.config.filters}\n")
            f.write(f"Portfolio: {self.config.portfolio}\n")
            f.write(f"Costs: {self.config.costs}\n")
        
        report_paths['summary_txt'] = summary_path
        
        logger.info(f"Generated reports: {list(report_paths.values())}")
        return report_paths


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Run true EOD backtest using reconstructed option chains'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to backtest configuration YAML file'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize backtest runner
    try:
        runner = TrueBacktestRunner(args.config)
    except Exception as e:
        print(f"Error initializing backtest runner: {e}")
        sys.exit(1)
    
    # Get date range from config if not provided
    if not args.start_date:
        if hasattr(runner.config, 'date_span') and hasattr(runner.config.date_span, 'start'):
            args.start_date = runner.config.date_span.start
        else:
            print("Error: Start date not provided and not found in config")
            sys.exit(1)
    
    if not args.end_date:
        if hasattr(runner.config, 'date_span') and hasattr(runner.config.date_span, 'end'):
            args.end_date = runner.config.date_span.end
        else:
            print("Error: End date not provided and not found in config")
            sys.exit(1)
    
    # Run backtest
    print(f"Running true EOD backtest for {runner.config.symbol}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print("=" * 50)
    
    results = runner.run_backtest(args.start_date, args.end_date)
    
    if results['status'] == 'success':
        print("\nBacktest completed successfully!")
        print(f"Total trading days: {results['total_trading_days']}")
        print(f"Total contracts: {results['total_contracts']}")
        print(f"Total PnL: {results['overall_metrics'].get('total_pnl', 0):.2f}")
        print(f"Average hit rate: {results['overall_metrics'].get('avg_hit_rate', 0):.2%}")
        
        print("\nReports generated:")
        for report_type, path in results['report_paths'].items():
            print(f"  {report_type}: {path}")
    else:
        print(f"\nBacktest failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
