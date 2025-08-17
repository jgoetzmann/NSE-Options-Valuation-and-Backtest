#!/usr/bin/env python3
"""
Synthetic Backtest Runner for Mode A (SB-CS)
============================================

This module runs synthetic backtests on current snapshots to validate end-to-end
plumbing and explore feature importance. It simulates option trading scenarios
using current market data and historical underlier movements.

Key Functions:
- run_synthetic_backtest(): Main synthetic backtest function
- simulate_horizon_scenarios(): Simulate different entry horizons
- rank_and_select_positions(): Rank and select trading positions
- compute_performance_metrics(): Calculate backtest performance metrics
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.normalize_snapshot import normalize_snapshot_from_file
from data_pipeline.attach_underlier_features import attach_underlier_features_from_file
from data_pipeline.compute_iv_and_greeks import compute_iv_and_greeks_from_file
from data_pipeline.make_labels import generate_labels_from_file
from data_pipeline.schemas import validate_dataframe_schema, add_derived_features


class SyntheticBacktestRunner:
    """
    Runs synthetic backtests on current snapshots.
    
    This class implements Mode A (SB-CS) which validates the end-to-end pipeline
    and explores feature importance without requiring historical data reconstruction.
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.symbol = self.config['symbol']
        self.horizons = self.config['horizons']
        self.filters = self.config['filters']
        self.costs = self.config['costs']
        self.portfolio = self.config['portfolio']
        
        # Initialize results storage
        self.backtest_results = []
        self.performance_metrics = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    def run_synthetic_backtest(self, snapshot_file: str) -> Dict[str, Any]:
        """
        Run complete synthetic backtest pipeline.
        
        Parameters:
        -----------
        snapshot_file : str
            Path to JSON snapshot file
            
        Returns:
        --------
        Dict[str, Any]
            Complete backtest results
        """
        print(f"🚀 Starting Synthetic Backtest for {self.symbol}")
        print(f"   Config: {os.path.basename(snapshot_file)}")
        print(f"   Horizons: {self.horizons}")
        print("=" * 60)
        
        try:
            # Step 1: Normalize snapshot
            print("📊 Step 1: Normalizing snapshot...")
            normalized_df, norm_csv = self._normalize_snapshot(snapshot_file)
            
            # Step 2: Attach underlier features
            print("📈 Step 2: Attaching underlier features...")
            enhanced_df, enhanced_csv = self._attach_features(norm_csv)
            
            # Step 3: Compute IV and Greeks
            print("🧮 Step 3: Computing IV and Greeks...")
            iv_greeks_df, iv_csv = self._compute_iv_greeks(enhanced_csv)
            
            # Step 4: Generate synthetic labels
            print("🏷️  Step 4: Generating synthetic labels...")
            labeled_df, labeled_csv = self._generate_labels(iv_csv)
            
            # Step 5: Run horizon simulations
            print("⏰ Step 5: Running horizon simulations...")
            horizon_results = self._simulate_horizon_scenarios(labeled_df)
            
            # Step 6: Rank and select positions
            print("📋 Step 6: Ranking and selecting positions...")
            portfolio_results = self._rank_and_select_positions(horizon_results)
            
            # Step 7: Compute performance metrics
            print("📊 Step 7: Computing performance metrics...")
            performance_metrics = self._compute_performance_metrics(portfolio_results)
            
            # Step 8: Generate reports
            print("📝 Step 8: Generating reports...")
            reports = self._generate_reports(portfolio_results, performance_metrics)
            
            # Compile results
            results = {
                'backtest_type': 'synthetic',
                'symbol': self.symbol,
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'data_files': {
                    'normalized': norm_csv,
                    'enhanced': enhanced_csv,
                    'iv_greeks': iv_csv,
                    'labeled': labeled_csv
                },
                'horizon_results': horizon_results,
                'portfolio_results': portfolio_results,
                'performance_metrics': performance_metrics,
                'reports': reports
            }
            
            print("✅ Synthetic backtest completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Synthetic backtest failed: {e}")
            raise
    
    def _normalize_snapshot(self, snapshot_file: str) -> Tuple[pd.DataFrame, str]:
        """Normalize snapshot JSON to tabular format."""
        df, csv_path = normalize_snapshot_from_file(snapshot_file, self.symbol)
        return df, csv_path
    
    def _attach_features(self, input_csv: str) -> Tuple[pd.DataFrame, str]:
        """Attach underlier features to normalized data."""
        df, csv_path = attach_underlier_features_from_file(input_csv, self.symbol)
        return df, csv_path
    
    def _compute_iv_greeks(self, input_csv: str) -> Tuple[pd.DataFrame, str]:
        """Compute IV and Greeks for options data."""
        df, csv_path = compute_iv_and_greeks_from_file(input_csv)
        return df, csv_path
    
    def _generate_labels(self, input_csv: str) -> Tuple[pd.DataFrame, str]:
        """Generate synthetic labels for options data."""
        cost_bps = self.costs.get('round_turn_bps', 60)
        df, csv_path = generate_labels_from_file(input_csv, cost_bps=cost_bps)
        return df, csv_path
    
    def _simulate_horizon_scenarios(self, labeled_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Simulate different entry horizon scenarios.
        
        Parameters:
        -----------
        labeled_df : pd.DataFrame
            DataFrame with labeled options data
            
        Returns:
        --------
        Dict[int, pd.DataFrame]
            Dictionary mapping horizons to scenario DataFrames
        """
        horizon_results = {}
        
        for horizon in self.horizons:
            print(f"   Simulating {horizon}-day horizon...")
            
            # Create horizon-specific scenario
            horizon_df = self._create_horizon_scenario(labeled_df, horizon)
            
            # Apply filters
            filtered_df = self._apply_filters(horizon_df)
            
            # Add synthetic flag
            filtered_df['synthetic_flag'] = 1
            filtered_df['horizon_d'] = horizon
            
            horizon_results[horizon] = filtered_df
        
        return horizon_results
    
    def _create_horizon_scenario(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Create scenario for specific entry horizon.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Base options DataFrame
        horizon : int
            Days before expiry for entry
            
        Returns:
        --------
        pd.DataFrame
            Horizon-specific scenario DataFrame
        """
        scenario_df = df.copy()
        
        # Filter contracts with sufficient time to expiry
        min_ttm = horizon + 1  # Need at least horizon + 1 days
        scenario_df = scenario_df[scenario_df['ttm_days'] >= min_ttm]
        
        # Add horizon-specific features
        scenario_df['entry_date'] = scenario_df['expiry_date'] - pd.Timedelta(days=horizon)
        scenario_df['horizon_ttm'] = horizon / 365.25  # Convert to years
        
        # Validate horizon feasibility
        scenario_df['valid_horizon'] = 1
        
        # For weekly options, some horizons may not be valid
        if horizon > 7:
            # Check if this is a weekly expiry
            weekly_mask = scenario_df['ttm_days'] <= 7
            scenario_df.loc[weekly_mask, 'valid_horizon'] = 0
        
        return scenario_df
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality and trading filters.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Apply filters from config
        if 'min_oi' in self.filters:
            filtered_df = filtered_df[filtered_df['openInterest'] >= self.filters['min_oi']]
        
        if 'min_premium' in self.filters:
            filtered_df = filtered_df[filtered_df['premium_t'] >= self.filters['min_premium']]
        
        if 'min_volume' in self.filters:
            filtered_df = filtered_df[filtered_df['totalTradedVolume'] >= self.filters['min_volume']]
        
        if 'max_iv' in self.filters:
            filtered_df = filtered_df[filtered_df['iv_est_t'] <= self.filters['max_iv']]
        
        # Apply TTM bounds
        if 'ttm_bounds' in self.filters:
            ttm_bounds = self.filters['ttm_bounds']
            if 'min' in ttm_bounds:
                filtered_df = filtered_df[filtered_df['ttm_days'] >= ttm_bounds['min']]
            if 'max' in ttm_bounds:
                filtered_df = filtered_df[filtered_df['ttm_days'] <= ttm_bounds['max']]
        
        # Apply spread filter if bid/ask data available
        if 'max_spread_pct' in self.filters and 'spread_pct' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['spread_pct'] <= self.filters['max_spread_pct']]
        
        # Only keep valid horizon contracts
        filtered_df = filtered_df[filtered_df['valid_horizon'] == 1]
        
        print(f"     Applied filters: {len(df)} → {len(filtered_df)} contracts")
        return filtered_df
    
    def _rank_and_select_positions(self, horizon_results: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
        """
        Rank and select trading positions for each horizon.
        
        Parameters:
        -----------
        horizon_results : Dict[int, pd.DataFrame]
            Dictionary of horizon-specific results
            
        Returns:
        --------
        Dict[str, Any]
            Portfolio selection results
        """
        portfolio_results = {}
        
        for horizon, df in horizon_results.items():
            print(f"   Ranking positions for {horizon}-day horizon...")
            
            if df.empty:
                portfolio_results[horizon] = {'positions': [], 'summary': {}}
                continue
            
            # Calculate ranking scores
            ranked_df = self._calculate_ranking_scores(df)
            
            # Select top positions
            max_positions = self.portfolio.get('daily_max_positions', 20)
            selected_positions = ranked_df.head(max_positions)
            
            # Calculate position weights
            selected_positions = self._calculate_position_weights(selected_positions)
            
            # Compile results
            portfolio_results[horizon] = {
                'positions': selected_positions.to_dict('records'),
                'summary': {
                    'total_contracts': len(df),
                    'selected_contracts': len(selected_positions),
                    'avg_score': selected_positions['ranking_score'].mean(),
                    'score_std': selected_positions['ranking_score'].std()
                }
            }
        
        return portfolio_results
    
    def _calculate_ranking_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ranking scores for position selection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with ranking scores
        """
        scored_df = df.copy()
        
        # Get ranking formula from config
        rank_formula = self.portfolio.get('rank_score', 'pct_diff_times_confidence')
        
        if rank_formula == 'pct_diff_times_confidence':
            # Use mispricing percentage × confidence
            if 'enhanced_mispricing_pct' in scored_df.columns and 'enhanced_confidence' in scored_df.columns:
                scored_df['ranking_score'] = (
                    scored_df['enhanced_mispricing_pct'] * scored_df['enhanced_confidence']
                )
            else:
                # Fallback to simple mispricing
                scored_df['ranking_score'] = scored_df.get('enhanced_mispricing_pct', 0)
        
        elif rank_formula == 'iv_skew':
            # Use IV skew as ranking score
            if 'iv_est_t' in scored_df.columns:
                atm_iv = scored_df['iv_est_t'].median()
                scored_df['ranking_score'] = (scored_df['iv_est_t'] - atm_iv) / atm_iv
        
        else:
            # Default to random ranking
            scored_df['ranking_score'] = np.random.random(len(scored_df))
        
        # Sort by ranking score (descending)
        scored_df = scored_df.sort_values('ranking_score', ascending=False)
        
        return scored_df
    
    def _calculate_position_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate position weights for portfolio construction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with selected positions
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with position weights
        """
        weighted_df = df.copy()
        
        position_sizing = self.portfolio.get('position_sizing', 'equal_weight')
        max_weight = self.portfolio.get('max_portfolio_weight', 0.05)
        
        if position_sizing == 'equal_weight':
            # Equal weight allocation
            n_positions = len(weighted_df)
            if n_positions > 0:
                weight = min(1.0 / n_positions, max_weight)
                weighted_df['position_weight'] = weight
        
        elif position_sizing == 'confidence_weight':
            # Weight by confidence score
            if 'enhanced_confidence' in weighted_df.columns:
                confidence_scores = weighted_df['enhanced_confidence']
                total_confidence = confidence_scores.sum()
                if total_confidence > 0:
                    weights = confidence_scores / total_confidence
                    weights = weights.clip(upper=max_weight)
                    weighted_df['position_weight'] = weights
        
        else:
            # Default to equal weight
            n_positions = len(weighted_df)
            if n_positions > 0:
                weight = min(1.0 / n_positions, max_weight)
                weighted_df['position_weight'] = weight
        
        return weighted_df
    
    def _compute_performance_metrics(self, portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute performance metrics for the synthetic backtest.
        
        Parameters:
        -----------
        portfolio_results : Dict[str, Any]
            Portfolio selection results
            
        Returns:
        --------
        Dict[str, Any]
            Performance metrics
        """
        metrics = {}
        
        for horizon, results in portfolio_results.items():
            positions = results['positions']
            if not positions:
                continue
            
            # Convert to DataFrame for analysis
            pos_df = pd.DataFrame(positions)
            
            # Calculate basic metrics
            horizon_metrics = {
                'total_positions': len(positions),
                'avg_ranking_score': pos_df['ranking_score'].mean(),
                'score_std': pos_df['ranking_score'].std(),
                'avg_confidence': pos_df.get('enhanced_confidence', pd.Series([0])).mean(),
                'avg_mispricing': pos_df.get('enhanced_mispricing_pct', pd.Series([0])).mean(),
                'position_concentration': pos_df['position_weight'].std() if 'position_weight' in pos_df.columns else 0
            }
            
            # Add option type distribution
            if 'option_type' in pos_df.columns:
                option_dist = pos_df['option_type'].value_counts()
                horizon_metrics['option_distribution'] = option_dist.to_dict()
            
            # Add moneyness distribution
            if 'moneyness' in pos_df.columns:
                horizon_metrics['avg_moneyness'] = pos_df['moneyness'].mean()
                horizon_metrics['moneyness_std'] = pos_df['moneyness'].std()
            
            metrics[f'horizon_{horizon}'] = horizon_metrics
        
        # Overall portfolio metrics
        all_positions = []
        for horizon, results in portfolio_results.items():
            all_positions.extend(results['positions'])
        
        if all_positions:
            all_pos_df = pd.DataFrame(all_positions)
            metrics['overall'] = {
                'total_positions': len(all_positions),
                'avg_ranking_score': all_pos_df['ranking_score'].mean(),
                'avg_confidence': all_pos_df.get('enhanced_confidence', pd.Series([0])).mean(),
                'avg_mispricing': all_pos_df.get('enhanced_mispricing_pct', pd.Series([0])).mean()
            }
        
        return metrics
    
    def _generate_reports(self, portfolio_results: Dict[str, Any], 
                         performance_metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate backtest reports.
        
        Parameters:
        -----------
        portfolio_results : Dict[str, Any]
            Portfolio selection results
        performance_metrics : Dict[str, Any]
            Performance metrics
            
        Returns:
        --------
        Dict[str, str]
            Dictionary of report file paths
        """
        reports = {}
        
        # Generate summary report
        summary_report = self._generate_summary_report(portfolio_results, performance_metrics)
        reports['summary'] = summary_report
        
        # Generate detailed CSV reports
        for horizon, results in portfolio_results.items():
            if results['positions']:
                csv_path = self._save_horizon_csv(horizon, results['positions'])
                reports[f'horizon_{horizon}_csv'] = csv_path
        
        return reports
    
    def _generate_summary_report(self, portfolio_results: Dict[str, Any], 
                                performance_metrics: Dict[str, Any]) -> str:
        """Generate summary text report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"outputs/results_from_nse_valuations/synthetic_summary_{self.symbol}_{timestamp}.txt"
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"Synthetic Backtest Summary - {self.symbol}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backtest Type: Mode A (SB-CS)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  Symbol: {self.symbol}\n")
            f.write(f"  Horizons: {self.horizons}\n")
            f.write(f"  Max Positions: {self.portfolio.get('daily_max_positions', 20)}\n")
            f.write(f"  Ranking Formula: {self.portfolio.get('rank_score', 'N/A')}\n\n")
            
            f.write("PORTFOLIO RESULTS:\n")
            for horizon, results in portfolio_results.items():
                f.write(f"  {horizon}-Day Horizon:\n")
                f.write(f"    Total Contracts: {results['summary'].get('total_contracts', 0)}\n")
                f.write(f"    Selected Positions: {results['summary'].get('selected_contracts', 0)}\n")
                f.write(f"    Average Score: {results['summary'].get('avg_score', 0):.4f}\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            if 'overall' in performance_metrics:
                overall = performance_metrics['overall']
                f.write(f"  Overall Average Score: {overall.get('avg_ranking_score', 0):.4f}\n")
                f.write(f"  Overall Average Confidence: {overall.get('avg_confidence', 0):.4f}\n")
                f.write(f"  Overall Average Mispricing: {overall.get('avg_mispricing', 0):.2f}%\n\n")
            
            f.write("IMPORTANT NOTES:\n")
            f.write("- This is a SYNTHETIC backtest for pipeline validation only\n")
            f.write("- Results are NOT indicative of actual trading performance\n")
            f.write("- Use for feature exploration and system validation\n")
            f.write("- Always conduct thorough testing before live trading\n")
        
        return report_path
    
    def _save_horizon_csv(self, horizon: int, positions: List[Dict[str, Any]]) -> str:
        """Save horizon-specific positions to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"outputs/csv/synthetic_backtest_{self.symbol}_horizon{horizon}_{timestamp}.csv"
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        df = pd.DataFrame(positions)
        df.to_csv(csv_path, index=False)
        
        return csv_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run synthetic backtest on NSE options snapshot")
    parser.add_argument("snapshot_file", help="Path to JSON snapshot file")
    parser.add_argument("--config", default="configs/backtest_synth.yml", 
                       help="Path to configuration file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        # Run synthetic backtest
        runner = SyntheticBacktestRunner(args.config)
        results = runner.run_synthetic_backtest(args.snapshot_file)
        
        print(f"\n✅ Synthetic backtest completed successfully!")
        print(f"   Results saved to: {args.output_dir}")
        print(f"   Summary report: {results['reports']['summary']}")
        
        # Print key metrics
        print(f"\n📊 Key Metrics:")
        for horizon in results['portfolio_results']:
            summary = results['portfolio_results'][horizon]['summary']
            print(f"   {horizon}-day horizon: {summary['selected_contracts']} positions selected")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
