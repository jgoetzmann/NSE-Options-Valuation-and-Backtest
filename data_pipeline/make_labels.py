#!/usr/bin/env python3
"""
Make Labels Module
==================

This module generates payoff, PnL, ROI, and POP (Profit or Loss) labels for options data.
It handles both expired contracts (with realized outcomes) and non-expired contracts
(labeled as pending until maturity).

Key Functions:
- compute_payoff(): Calculate option payoff at expiry
- compute_pnl_and_roi(): Calculate profit/loss and return on investment
- generate_labels(): Main function to generate all labels for a DataFrame
- validate_labels(): Ensure label quality and consistency
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.schemas import validate_dataframe_schema


class LabelGenerator:
    """
    Generates payoff, PnL, ROI, and POP labels for options data.
    
    This class handles the computation of option outcomes including payoff calculations,
    profit/loss analysis, and binary profit indicators for machine learning.
    """
    
    def __init__(self, risk_free_rate: float = 0.06):
        self.risk_free_rate = risk_free_rate
        
    def compute_payoff(self, S_T: float, K: float, option_type: str) -> float:
        """
        Compute option payoff at expiry.
        
        Parameters:
        -----------
        S_T : float
            Underlier price at expiry
        K : float
            Strike price
        option_type : str
            Option type ('CE' or 'PE')
            
        Returns:
        --------
        float
            Option payoff at expiry
        """
        if option_type == 'CE':  # Call option
            payoff = max(S_T - K, 0)
        elif option_type == 'PE':  # Put option
            payoff = max(K - S_T, 0)
        else:
            raise ValueError(f"Invalid option type: {option_type}")
        
        return payoff
    
    def compute_pnl_and_roi(self, payoff: float, premium_t: float, 
                           cost_bps: float = 0.0) -> Tuple[float, float]:
        """
        Compute profit/loss and return on investment.
        
        Parameters:
        -----------
        payoff : float
            Option payoff at expiry
        premium_t : float
            Entry premium (cost to enter position)
        cost_bps : float
            Transaction costs in basis points
            
        Returns:
        --------
        Tuple[float, float]
            (PnL, ROI) tuple
        """
        # Calculate transaction costs
        if cost_bps > 0:
            transaction_cost = premium_t * (cost_bps / 10000)  # Convert bps to decimal
        else:
            transaction_cost = 0.0
        
        # Total cost including transaction costs
        total_cost = premium_t + transaction_cost
        
        # PnL = Payoff - Total Cost
        pnl = payoff - total_cost
        
        # ROI = PnL / Total Cost
        if total_cost > 0:
            roi = pnl / total_cost
        else:
            roi = 0.0
        
        return pnl, roi
    
    def generate_pop_label(self, pnl: float, threshold: float = 0.0) -> int:
        """
        Generate POP (Profit or Loss) label.
        
        Parameters:
        -----------
        pnl : float
            Profit/loss value
        threshold : float
            Threshold for positive label (default: 0.0)
            
        Returns:
        --------
        int
            1 if PnL >= threshold, 0 otherwise
        """
        return 1 if pnl >= threshold else 0
    
    def process_single_contract(self, row: pd.Series, 
                               cost_bps: float = 0.0) -> Dict[str, Any]:
        """
        Process a single contract to generate labels.
        
        Parameters:
        -----------
        row : pd.Series
            Single row from options DataFrame
        cost_bps : float
            Transaction costs in basis points
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary with computed labels
        """
        try:
            # Extract contract parameters
            S_T = row.get('S_T')
            K = row.get('strike')
            option_type = row.get('option_type')
            premium_t = row.get('premium_t')
            
            # Check if contract has expired (S_T is available)
            if pd.isna(S_T) or S_T <= 0:
                # Contract not yet expired
                return {
                    'payoff_T': None,
                    'PnL': None,
                    'ROI': None,
                    'POP_label': None,
                    'label_status': 'pending'
                }
            
            # Contract has expired, compute outcomes
            if pd.isna(K) or pd.isna(option_type) or pd.isna(premium_t):
                return {
                    'payoff_T': None,
                    'PnL': None,
                    'ROI': None,
                    'POP_label': None,
                    'label_status': 'missing_data'
                }
            
            # Compute payoff
            payoff = self.compute_payoff(S_T, K, option_type)
            
            # Compute PnL and ROI
            pnl, roi = self.compute_pnl_and_roi(payoff, premium_t, cost_bps)
            
            # Generate POP label
            pop_label = self.generate_pop_label(pnl)
            
            return {
                'payoff_T': payoff,
                'PnL': pnl,
                'ROI': roi,
                'POP_label': pop_label,
                'label_status': 'completed'
            }
            
        except Exception as e:
            # Return error status on failure
            return {
                'payoff_T': None,
                'PnL': None,
                'ROI': None,
                'POP_label': None,
                'label_status': f'error: {str(e)}'
            }
    
    def generate_labels(self, df: pd.DataFrame, cost_bps: float = 0.0,
                       batch_size: int = 1000) -> pd.DataFrame:
        """
        Generate labels for entire DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Options DataFrame with required columns
        cost_bps : float
            Transaction costs in basis points
        batch_size : int
            Number of rows to process in each batch
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with labels generated
        """
        if df.empty:
            return df
        
        # Validate required columns
        required_cols = ['strike', 'option_type', 'premium_t']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if S_T column exists (for expiry data)
        has_expiry_data = 'S_T' in df.columns
        
        if not has_expiry_data:
            print("⚠️  Warning: No S_T column found. All contracts will be labeled as pending.")
            print("   To generate realized labels, attach expiry data using attach_expiry_data().")
        
        # Process in batches to show progress
        total_rows = len(df)
        results = []
        
        print(f"🔄 Generating labels for {total_rows} contracts...")
        
        for i in range(0, total_rows, batch_size):
            batch_end = min(i + batch_size, total_rows)
            batch_df = df.iloc[i:batch_end]
            
            print(f"   Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size} "
                  f"({i+1}-{batch_end}/{total_rows})")
            
            batch_results = []
            for _, row in batch_df.iterrows():
                result = self.process_single_contract(row, cost_bps)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add computed columns to original DataFrame
        df = df.copy()
        for col in ['payoff_T', 'PnL', 'ROI', 'POP_label']:
            if col in results_df.columns:
                df[col] = results_df[col]
        
        # Add label status
        if 'label_status' in results_df.columns:
            df['label_status'] = results_df['label_status']
        
        # Generate summary statistics
        self._print_label_summary(df, results_df)
        
        return df
    
    def _print_label_summary(self, df: pd.DataFrame, results_df: pd.DataFrame):
        """Print summary of label generation results."""
        print(f"✅ Completed label generation")
        
        # Count by status
        if 'label_status' in results_df.columns:
            status_counts = results_df['label_status'].value_counts()
            print(f"   Label Status Summary:")
            for status, count in status_counts.items():
                print(f"     {status}: {count}")
        
        # Count completed labels
        completed_mask = results_df['label_status'] == 'completed'
        completed_count = completed_mask.sum()
        total_count = len(results_df)
        
        if completed_count > 0:
            print(f"   Completed Labels: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%)")
            
            # PnL statistics for completed labels
            completed_pnl = results_df.loc[completed_mask, 'PnL']
            if not completed_pnl.empty:
                print(f"   PnL Statistics:")
                print(f"     Mean: ₹{completed_pnl.mean():.2f}")
                print(f"     Median: ₹{completed_pnl.median():.2f}")
                print(f"     Min: ₹{completed_pnl.min():.2f}")
                print(f"     Max: ₹{completed_pnl.max():.2f}")
            
            # ROI statistics for completed labels
            completed_roi = results_df.loc[completed_mask, 'ROI']
            if not completed_roi.empty:
                print(f"   ROI Statistics:")
                print(f"     Mean: {completed_roi.mean():.1%}")
                print(f"     Median: {completed_roi.median():.1%}")
                print(f"     Min: {completed_roi.min():.1%}")
                print(f"     Max: {completed_roi.max():.1%}")
            
            # POP label distribution
            completed_pop = results_df.loc[completed_mask, 'POP_label']
            if not completed_pop.empty:
                pop_counts = completed_pop.value_counts()
                print(f"   POP Label Distribution:")
                for label, count in pop_counts.items():
                    pct = count / len(completed_pop) * 100
                    print(f"     {label}: {count} ({pct:.1f}%)")
    
    def attach_expiry_data(self, df: pd.DataFrame, expiry_data: pd.DataFrame,
                           expiry_date_col: str = 'expiry_date') -> pd.DataFrame:
        """
        Attach expiry data to options DataFrame for label generation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Options DataFrame
        expiry_data : pd.DataFrame
            DataFrame with expiry prices (must have date and price columns)
        expiry_date_col : str
            Column name for expiry date in options DataFrame
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with S_T column added
        """
        if df.empty or expiry_data.empty:
            return df
        
        # Ensure expiry_data has required columns
        if 'date' not in expiry_data.columns or 'price' not in expiry_data.columns:
            raise ValueError("expiry_data must have 'date' and 'price' columns")
        
        # Convert date columns
        expiry_data['date'] = pd.to_datetime(expiry_data['date'])
        df[expiry_date_col] = pd.to_datetime(df[expiry_date_col])
        
        # Merge expiry prices
        result_df = pd.merge(
            df,
            expiry_data[['date', 'price']],
            left_on=expiry_date_col,
            right_on='date',
            how='left'
        )
        
        # Rename price column to S_T
        result_df = result_df.rename(columns={'price': 'S_T'})
        result_df = result_df.drop('date', axis=1)
        
        print(f"✅ Attached expiry data for {len(result_df)} contracts")
        print(f"   Contracts with expiry prices: {result_df['S_T'].notna().sum()}")
        
        return result_df
    
    def validate_labels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the generated labels for quality and consistency.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with labels generated
            
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
        
        # Check for required label columns
        label_cols = ['payoff_T', 'PnL', 'ROI', 'POP_label']
        missing_labels = [col for col in label_cols if col not in df.columns]
        if missing_labels:
            validation_result['warnings'].append(f"Missing label columns: {missing_labels}")
        
        # Check label statistics for completed contracts
        if 'label_status' in df.columns:
            completed_mask = df['label_status'] == 'completed'
            completed_count = completed_mask.sum()
            total_count = len(df)
            
            validation_result['statistics']['label_completion'] = {
                'total_contracts': total_count,
                'completed_labels': completed_count,
                'completion_rate': completed_count / total_count if total_count > 0 else 0
            }
            
            if completed_count > 0:
                # Validate PnL consistency
                if 'PnL' in df.columns and 'payoff_T' in df.columns and 'premium_t' in df.columns:
                    completed_df = df[completed_mask]
                    
                    # Check if PnL = payoff_T - premium_t (approximately, accounting for costs)
                    expected_pnl = completed_df['payoff_T'] - completed_df['premium_t']
                    actual_pnl = completed_df['PnL']
                    
                    # Allow for small differences due to transaction costs
                    pnl_diff = abs(expected_pnl - actual_pnl)
                    if pnl_diff.max() > 1.0:  # Allow ₹1 difference
                        validation_result['warnings'].append(
                            f"Large PnL discrepancies found (max diff: ₹{pnl_diff.max():.2f})"
                        )
                
                # Check POP label consistency
                if 'POP_label' in df.columns and 'PnL' in df.columns:
                    completed_df = df[completed_mask]
                    
                    # POP_label should be 1 when PnL >= 0
                    expected_pop = (completed_df['PnL'] >= 0).astype(int)
                    actual_pop = completed_df['POP_label']
                    
                    mismatch_count = (expected_pop != actual_pop).sum()
                    if mismatch_count > 0:
                        validation_result['warnings'].append(
                            f"Found {mismatch_count} POP label mismatches"
                        )
        
        return validation_result


def generate_labels_from_file(input_file: str, output_dir: str = "outputs/csv",
                             cost_bps: float = 60.0) -> Tuple[pd.DataFrame, str]:
    """
    Convenience function to generate labels from a CSV file.
    
    Parameters:
    -----------
    input_file : str
        Path to CSV file with options data
    output_dir : str
        Output directory for labeled CSV
    cost_bps : float
        Transaction costs in basis points
        
    Returns:
    --------
    Tuple[pd.DataFrame, str]
        Labeled DataFrame and path to saved CSV
    """
    # Load options data
    df = pd.read_csv(input_file)
    
    # Convert date columns
    date_cols = ['date_t', 'expiry_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Generate labels
    generator = LabelGenerator()
    labeled_df = generator.generate_labels(df, cost_bps)
    
    # Validate
    validation_result = generator.validate_labels(labeled_df)
    for warning in validation_result['warnings']:
        print(f"  Warning: {warning}")
    
    # Save labeled data
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"options_with_labels_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    labeled_df.to_csv(filepath, index=False)
    
    return labeled_df, filepath


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate labels for options data")
    parser.add_argument("input_file", help="Path to CSV file with options data")
    parser.add_argument("--output-dir", default="outputs/csv", help="Output directory")
    parser.add_argument("--cost-bps", type=float, default=60.0, 
                       help="Transaction costs in basis points (default: 60)")
    parser.add_argument("--batch-size", type=int, default=1000, 
                       help="Batch size for processing (default: 1000)")
    
    args = parser.parse_args()
    
    try:
        labeled_df, csv_path = generate_labels_from_file(
            args.input_file, args.output_dir, args.cost_bps
        )
        
        print(f"✅ Successfully generated labels")
        print(f"   Input: {args.input_file}")
        print(f"   Output: {csv_path}")
        print(f"   Contracts: {len(labeled_df)}")
        print(f"   Cost: {args.cost_bps} bps")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
