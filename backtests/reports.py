#!/usr/bin/env python3
"""
Common Reports Module for NSE Options Backtest Project
=====================================================

This module provides common reporting functionality across all three execution modes:
- Mode A (SB-CS): Synthetic Backtest on Current Snapshot
- Mode B (ML-Live): Train-and-Score ML on Historical Reconstructions
- Mode C (EOD-True): True EOD Backtest via Reconstructed Past Chains

Key Functions:
- generate_performance_summary: Common KPIs and metrics
- create_performance_plots: Visualization of results
- generate_risk_analysis: Risk metrics and analysis
- create_portfolio_summary: Portfolio composition and statistics
- export_reports: Save reports in various formats

Usage:
    from reports import generate_performance_summary, create_performance_plots
    summary = generate_performance_summary(backtest_results)
    plots = create_performance_plots(backtest_results)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

# Try to import visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available. Plotting disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestReporter:
    """
    Generates comprehensive reports for backtest results.
    
    This class provides standardized reporting across all execution modes,
    ensuring consistency in metrics, visualizations, and output formats.
    """
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Define color schemes for different option types
        self.colors = {
            'CE': '#2E86AB',  # Blue for calls
            'PE': '#A23B72',  # Purple for puts
            'total': '#F18F01',  # Orange for totals
            'pnl': '#C73E1D',  # Red for PnL
            'roi': '#3A1772'   # Dark purple for ROI
        }
    
    def generate_performance_summary(self, results: Dict[str, Any], 
                                   mode: str = "unknown") -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Args:
            results: Backtest results dictionary
            mode: Execution mode (SB-CS, ML-Live, EOD-True)
            
        Returns:
            Dictionary with performance summary
        """
        summary = {
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'basic_metrics': {},
            'risk_metrics': {},
            'portfolio_metrics': {},
            'cost_analysis': {},
            'data_quality': {}
        }
        
        try:
            # Basic performance metrics
            if 'overall_metrics' in results:
                overall = results['overall_metrics']
                summary['basic_metrics'] = {
                    'total_pnl': overall.get('total_pnl', 0.0),
                    'total_positions': overall.get('total_positions', 0),
                    'avg_daily_pnl': overall.get('avg_daily_pnl', 0.0),
                    'avg_hit_rate': overall.get('avg_hit_rate', 0.0),
                    'total_trading_days': results.get('total_trading_days', 0)
                }
            
            # Risk metrics
            if 'daily_metrics' in results:
                daily_metrics = results['daily_metrics']
                summary['risk_metrics'] = self._compute_risk_metrics(daily_metrics)
            
            # Portfolio composition
            if 'portfolio_csv' in results.get('report_paths', {}):
                portfolio_path = results['report_paths']['portfolio_csv']
                if os.path.exists(portfolio_path):
                    portfolio_df = pd.read_csv(portfolio_path)
                    summary['portfolio_metrics'] = self._analyze_portfolio_composition(portfolio_df)
            
            # Cost analysis
            summary['cost_analysis'] = self._analyze_costs(results)
            
            # Data quality metrics
            summary['data_quality'] = self._assess_data_quality(results)
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def _compute_risk_metrics(self, daily_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compute risk metrics from daily performance data."""
        if not daily_metrics:
            return {}
        
        risk_metrics = {}
        
        try:
            # Extract daily PnL values
            daily_pnls = []
            daily_returns = []
            
            for date, metrics in daily_metrics.items():
                if 'total_pnl' in metrics and metrics['total_pnl'] is not None:
                    daily_pnls.append(metrics['total_pnl'])
                    
                    # Calculate daily return (assuming equal position sizing)
                    if 'total_premium' in metrics and metrics['total_premium'] > 0:
                        daily_return = metrics['total_pnl'] / metrics['total_premium']
                        daily_returns.append(daily_return)
            
            if daily_pnls:
                # Basic statistics
                risk_metrics['total_pnl'] = sum(daily_pnls)
                risk_metrics['avg_daily_pnl'] = np.mean(daily_pnls)
                risk_metrics['pnl_std'] = np.std(daily_pnls)
                risk_metrics['min_daily_pnl'] = min(daily_pnls)
                risk_metrics['max_daily_pnl'] = max(daily_pnls)
                
                # Risk-adjusted metrics
                if risk_metrics['pnl_std'] > 0:
                    risk_metrics['sharpe_ratio'] = (
                        risk_metrics['avg_daily_pnl'] / risk_metrics['pnl_std'] * np.sqrt(252)
                    )
                else:
                    risk_metrics['sharpe_ratio'] = 0.0
                
                # Drawdown analysis
                cumulative_pnl = np.cumsum(daily_pnls)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = cumulative_pnl - running_max
                risk_metrics['max_drawdown'] = abs(min(drawdown)) if len(drawdown) > 0 else 0.0
                
                # Win/Loss analysis
                winning_days = sum(1 for pnl in daily_pnls if pnl > 0)
                losing_days = sum(1 for pnl in daily_pnls if pnl < 0)
                risk_metrics['winning_days'] = winning_days
                risk_metrics['losing_days'] = losing_days
                risk_metrics['win_rate'] = winning_days / len(daily_pnls) if daily_pnls else 0.0
            
            if daily_returns:
                # Return-based metrics
                risk_metrics['avg_daily_return'] = np.mean(daily_returns)
                risk_metrics['return_std'] = np.std(daily_returns)
                risk_metrics['volatility'] = risk_metrics['return_std'] * np.sqrt(252)
                
                # Sortino ratio (downside deviation)
                negative_returns = [r for r in daily_returns if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        risk_metrics['sortino_ratio'] = (
                            risk_metrics['avg_daily_return'] / downside_std * np.sqrt(252)
                        )
                    else:
                        risk_metrics['sortino_ratio'] = 0.0
                else:
                    risk_metrics['sortino_ratio'] = float('inf')
        
        except Exception as e:
            logger.error(f"Error computing risk metrics: {e}")
            risk_metrics['error'] = str(e)
        
        return risk_metrics
    
    def _analyze_portfolio_composition(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio composition and characteristics."""
        if portfolio_df.empty:
            return {}
        
        composition = {}
        
        try:
            # Option type distribution
            if 'option_type' in portfolio_df.columns:
                type_counts = portfolio_df['option_type'].value_counts()
                composition['ce_count'] = type_counts.get('CE', 0)
                composition['pe_count'] = type_counts.get('PE', 0)
                composition['total_contracts'] = len(portfolio_df)
            
            # Time to maturity analysis
            if 'ttm_days' in portfolio_df.columns:
                composition['avg_ttm'] = portfolio_df['ttm_days'].mean()
                composition['min_ttm'] = portfolio_df['ttm_days'].min()
                composition['max_ttm'] = portfolio_df['ttm_days'].max()
                composition['ttm_std'] = portfolio_df['ttm_days'].std()
            
            # Strike price analysis
            if 'strike' in portfolio_df.columns:
                composition['avg_strike'] = portfolio_df['strike'].mean()
                composition['min_strike'] = portfolio_df['strike'].min()
                composition['max_strike'] = portfolio_df['strike'].max()
                composition['strike_std'] = portfolio_df['strike'].std()
            
            # Moneyness analysis
            if 'moneyness' in portfolio_df.columns:
                composition['avg_moneyness'] = portfolio_df['moneyness'].mean()
                composition['itm_count'] = (portfolio_df['moneyness'] > 1.0).sum()
                composition['otm_count'] = (portfolio_df['moneyness'] < 1.0).sum()
                composition['atm_count'] = (abs(portfolio_df['moneyness'] - 1.0) <= 0.02).sum()
            
            # Greeks analysis
            greek_columns = ['delta', 'gamma', 'theta', 'vega', 'rho']
            for greek in greek_columns:
                if greek in portfolio_df.columns:
                    composition[f'avg_{greek}'] = portfolio_df[greek].mean()
                    composition[f'{greek}_std'] = portfolio_df[greek].std()
            
            # Premium analysis
            if 'premium_t' in portfolio_df.columns:
                composition['total_premium'] = portfolio_df['premium_t'].sum()
                composition['avg_premium'] = portfolio_df['premium_t'].mean()
                composition['premium_std'] = portfolio_df['premium_t'].std()
            
            # Open interest and volume analysis
            if 'openInterest' in portfolio_df.columns:
                composition['total_oi'] = portfolio_df['openInterest'].sum()
                composition['avg_oi'] = portfolio_df['openInterest'].mean()
            
            if 'totalTradedVolume' in portfolio_df.columns:
                composition['total_volume'] = portfolio_df['totalTradedVolume'].sum()
                composition['avg_volume'] = portfolio_df['totalTradedVolume'].mean()
        
        except Exception as e:
            logger.error(f"Error analyzing portfolio composition: {e}")
            composition['error'] = str(e)
        
        return composition
    
    def _analyze_costs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze transaction costs and their impact."""
        cost_analysis = {}
        
        try:
            # Extract cost configuration
            if 'config' in results:
                config = results['config']
                if hasattr(config, 'costs'):
                    costs = config.costs
                    cost_analysis['round_turn_bps'] = getattr(costs, 'round_turn_bps', 0)
                    cost_analysis['slippage_mode'] = getattr(costs, 'slippage_mode', 'unknown')
            
            # Cost impact analysis (if available)
            if 'portfolio_csv' in results.get('report_paths', {}):
                portfolio_path = results['report_paths']['portfolio_csv']
                if os.path.exists(portfolio_path):
                    portfolio_df = pd.read_csv(portfolio_path)
                    if 'cost_bps' in portfolio_df.columns:
                        cost_analysis['total_cost_bps'] = portfolio_df['cost_bps'].sum()
                        cost_analysis['avg_cost_bps'] = portfolio_df['cost_bps'].mean()
            
            # Stress test scenarios
            base_costs = cost_analysis.get('round_turn_bps', 60)
            cost_analysis['stress_scenarios'] = {
                'low_costs': base_costs * 0.75,
                'base_costs': base_costs,
                'high_costs': base_costs * 1.25,
                'very_high_costs': base_costs * 1.5
            }
        
        except Exception as e:
            logger.error(f"Error analyzing costs: {e}")
            cost_analysis['error'] = str(e)
        
        return cost_analysis
    
    def _assess_data_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality and completeness."""
        quality_metrics = {}
        
        try:
            # Data completeness
            if 'total_contracts' in results:
                quality_metrics['total_contracts'] = results['total_contracts']
            
            if 'total_trading_days' in results:
                quality_metrics['total_trading_days'] = results['total_trading_days']
            
            # Missing data analysis
            if 'portfolio_csv' in results.get('report_paths', {}):
                portfolio_path = results['report_paths']['portfolio_csv']
                if os.path.exists(portfolio_path):
                    portfolio_df = pd.read_csv(portfolio_path)
                    
                    # Check for missing values in key columns
                    key_columns = ['premium_t', 'strike', 'expiry_date', 'option_type']
                    missing_data = {}
                    for col in key_columns:
                        if col in portfolio_df.columns:
                            missing_count = portfolio_df[col].isna().sum()
                            missing_data[col] = {
                                'missing_count': missing_count,
                                'missing_pct': missing_count / len(portfolio_df) * 100
                            }
                    
                    quality_metrics['missing_data'] = missing_data
                    
                    # Data type consistency
                    quality_metrics['data_types'] = portfolio_df.dtypes.to_dict()
            
            # Validation results
            if 'validation' in results:
                quality_metrics['validation'] = results['validation']
        
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            quality_metrics['error'] = str(e)
        
        return quality_metrics
    
    def create_performance_plots(self, results: Dict[str, Any], 
                                save_plots: bool = True) -> Dict[str, str]:
        """
        Create performance visualization plots.
        
        Args:
            results: Backtest results dictionary
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with plot file paths
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available. Skipping plot generation.")
            return {}
        
        plot_paths = {}
        
        try:
            # Create output directory for plots
            plots_dir = os.path.join(self.output_dir, "plots")
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            # 1. Daily PnL Chart
            if 'daily_metrics' in results:
                plot_paths['daily_pnl'] = self._create_daily_pnl_chart(
                    results['daily_metrics'], plots_dir
                )
            
            # 2. Cumulative PnL Chart
            if 'daily_metrics' in results:
                plot_paths['cumulative_pnl'] = self._create_cumulative_pnl_chart(
                    results['daily_metrics'], plots_dir
                )
            
            # 3. Portfolio Composition Chart
            if 'portfolio_csv' in results.get('report_paths', {}):
                portfolio_path = results['report_paths']['portfolio_csv']
                if os.path.exists(portfolio_path):
                    portfolio_df = pd.read_csv(portfolio_path)
                    plot_paths['portfolio_composition'] = self._create_portfolio_composition_chart(
                        portfolio_df, plots_dir
                    )
            
            # 4. Risk-Return Scatter Plot
            if 'daily_metrics' in results:
                plot_paths['risk_return'] = self._create_risk_return_chart(
                    results['daily_metrics'], plots_dir
                )
            
            # 5. Drawdown Chart
            if 'daily_metrics' in results:
                plot_paths['drawdown'] = self._create_drawdown_chart(
                    results['daily_metrics'], plots_dir
                )
            
            logger.info(f"Generated {len(plot_paths)} performance plots")
            
        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")
        
        return plot_paths
    
    def _create_daily_pnl_chart(self, daily_metrics: Dict[str, Any], 
                                plots_dir: str) -> str:
        """Create daily PnL bar chart."""
        try:
            dates = []
            pnls = []
            
            for date, metrics in daily_metrics.items():
                if 'total_pnl' in metrics and metrics['total_pnl'] is not None:
                    dates.append(date)
                    pnls.append(metrics['total_pnl'])
            
            if not dates:
                return ""
            
            # Convert dates to datetime
            date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            
            plt.figure(figsize=(12, 6))
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
            plt.bar(range(len(dates)), pnls, color=colors, alpha=0.7)
            
            plt.title('Daily PnL Performance', fontsize=14, fontweight='bold')
            plt.xlabel('Trading Days')
            plt.ylabel('PnL')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(range(len(dates)), [d.strftime('%m-%d') for d in date_objects], 
                       rotation=45)
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "daily_pnl.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating daily PnL chart: {e}")
            return ""
    
    def _create_cumulative_pnl_chart(self, daily_metrics: Dict[str, Any], 
                                     plots_dir: str) -> str:
        """Create cumulative PnL line chart."""
        try:
            dates = []
            pnls = []
            
            for date, metrics in daily_metrics.items():
                if 'total_pnl' in metrics and metrics['total_pnl'] is not None:
                    dates.append(date)
                    pnls.append(metrics['total_pnl'])
            
            if not dates:
                return ""
            
            # Convert dates to datetime
            date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            
            # Calculate cumulative PnL
            cumulative_pnl = np.cumsum(pnls)
            
            plt.figure(figsize=(12, 6))
            plt.plot(date_objects, cumulative_pnl, linewidth=2, color=self.colors['pnl'])
            
            plt.title('Cumulative PnL Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Cumulative PnL')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "cumulative_pnl.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating cumulative PnL chart: {e}")
            return ""
    
    def _create_portfolio_composition_chart(self, portfolio_df: pd.DataFrame, 
                                           plots_dir: str) -> str:
        """Create portfolio composition pie chart."""
        try:
            if 'option_type' not in portfolio_df.columns:
                return ""
            
            # Count option types
            type_counts = portfolio_df['option_type'].value_counts()
            
            if len(type_counts) == 0:
                return ""
            
            plt.figure(figsize=(8, 8))
            colors = [self.colors.get(opt_type, '#666666') for opt_type in type_counts.index]
            
            plt.pie(type_counts.values, labels=type_counts.index, colors=colors, 
                    autopct='%1.1f%%', startangle=90)
            plt.title('Portfolio Composition by Option Type', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "portfolio_composition.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating portfolio composition chart: {e}")
            return ""
    
    def _create_risk_return_chart(self, daily_metrics: Dict[str, Any], 
                                  plots_dir: str) -> str:
        """Create risk-return scatter plot."""
        try:
            returns = []
            risks = []
            
            for date, metrics in daily_metrics.items():
                if 'total_pnl' in metrics and 'total_premium' in metrics:
                    if metrics['total_premium'] > 0:
                        daily_return = metrics['total_pnl'] / metrics['total_premium']
                        returns.append(daily_return)
                        risks.append(abs(daily_return))  # Use absolute return as risk proxy
            
            if len(returns) < 2:
                return ""
            
            plt.figure(figsize=(10, 8))
            plt.scatter(risks, returns, alpha=0.6, color=self.colors['roi'])
            
            plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
            plt.xlabel('Risk (Absolute Return)')
            plt.ylabel('Return')
            plt.grid(True, alpha=0.3)
            
            # Add zero lines
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "risk_return.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating risk-return chart: {e}")
            return ""
    
    def _create_drawdown_chart(self, daily_metrics: Dict[str, Any], 
                               plots_dir: str) -> str:
        """Create drawdown chart."""
        try:
            dates = []
            pnls = []
            
            for date, metrics in daily_metrics.items():
                if 'total_pnl' in metrics and metrics['total_pnl'] is not None:
                    dates.append(date)
                    pnls.append(metrics['total_pnl'])
            
            if not dates:
                return ""
            
            # Convert dates to datetime
            date_objects = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            
            # Calculate drawdown
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / running_max * 100  # Percentage
            
            plt.figure(figsize=(12, 6))
            plt.fill_between(date_objects, drawdown, 0, alpha=0.3, color='red')
            plt.plot(date_objects, drawdown, linewidth=2, color='red')
            
            plt.title('Portfolio Drawdown', fontsize=14, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            
            # Add zero line
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(plots_dir, "drawdown.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return ""
    
    def export_reports(self, results: Dict[str, Any], mode: str = "unknown",
                       output_format: str = "all") -> Dict[str, str]:
        """
        Export comprehensive reports in various formats.
        
        Args:
            results: Backtest results dictionary
            mode: Execution mode
            output_format: Output format ('json', 'txt', 'csv', 'all')
            
        Returns:
            Dictionary with exported file paths
        """
        export_paths = {}
        
        try:
            # Generate performance summary
            summary = self.generate_performance_summary(results, mode)
            
            # Export summary in different formats
            if output_format in ['json', 'all']:
                json_path = os.path.join(self.output_dir, f"{mode}_summary.json")
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                export_paths['summary_json'] = json_path
            
            if output_format in ['txt', 'all']:
                txt_path = os.path.join(self.output_dir, f"{mode}_summary.txt")
                with open(txt_path, 'w') as f:
                    f.write(f"{mode} Backtest Summary\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Basic metrics
                    f.write("Basic Performance Metrics:\n")
                    basic = summary.get('basic_metrics', {})
                    for key, value in basic.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.2f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # Risk metrics
                    f.write("Risk Metrics:\n")
                    risk = summary.get('risk_metrics', {})
                    for key, value in risk.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                    
                    # Portfolio metrics
                    f.write("Portfolio Composition:\n")
                    portfolio = summary.get('portfolio_metrics', {})
                    for key, value in portfolio.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.2f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                
                export_paths['summary_txt'] = txt_path
            
            # Generate plots if requested
            if output_format in ['plots', 'all']:
                plot_paths = self.create_performance_plots(results, save_plots=True)
                export_paths.update(plot_paths)
            
            logger.info(f"Exported reports in {output_format} format")
            
        except Exception as e:
            logger.error(f"Error exporting reports: {e}")
            export_paths['error'] = str(e)
        
        return export_paths


def generate_performance_summary(results: Dict[str, Any], mode: str = "unknown") -> Dict[str, Any]:
    """Convenience function to generate performance summary."""
    reporter = BacktestReporter()
    return reporter.generate_performance_summary(results, mode)


def create_performance_plots(results: Dict[str, Any], save_plots: bool = True) -> Dict[str, str]:
    """Convenience function to create performance plots."""
    reporter = BacktestReporter()
    return reporter.create_performance_plots(results, save_plots)


def export_reports(results: Dict[str, Any], mode: str = "unknown", 
                   output_format: str = "all") -> Dict[str, str]:
    """Convenience function to export reports."""
    reporter = BacktestReporter()
    return reporter.export_reports(results, mode, output_format)


if __name__ == "__main__":
    # Example usage
    print("BacktestReporter module loaded successfully.")
    print("Use the convenience functions or instantiate BacktestReporter class directly.")
