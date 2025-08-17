#!/usr/bin/env python3
"""
ML Model Scoring Module for Mode B (ML-Live)
===========================================

This module scores current option snapshots using trained ML models from Mode B.
It loads trained models, normalizes snapshots, computes features, and generates
predictions for both classification (POP_label) and regression (PnL/ROI) tasks.

Key Functions:
- load_trained_models: Load models, scaler, and feature list
- normalize_snapshot: Convert snapshot to normalized format
- engineer_features: Create features matching training data
- score_models: Generate predictions from trained models
- rank_opportunities: Rank options by model scores
- generate_scoring_report: Create comprehensive scoring report

Usage:
    python score_snapshot.py --snapshot outputs/json/chain_NIFTY_YYYYMMDD_HHMM.json --model-dir models/model_store
    python score_snapshot.py --snapshot outputs/json/chain_NIFTY_YYYYMMDD_HHMM.json --model-timestamp 20241201_143022
"""

import os
import sys
import json
import pickle
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.normalize_snapshot import SnapshotNormalizer
from data_pipeline.attach_underlier_features import UnderlierFeatureAttacher
from data_pipeline.compute_iv_and_greeks import IVAndGreeksComputer
from data_pipeline.schemas import (
    NORMALIZED_TABLE_SCHEMA, 
    PANDAS_DTYPES,
    validate_dataframe_schema
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SnapshotScorer:
    """
    Scores current option snapshots using trained ML models.
    
    This class implements the scoring pipeline for Mode B, including
    model loading, snapshot normalization, feature engineering, and
    prediction generation.
    """
    
    def __init__(self, model_dir: str = "models/model_store"):
        """
        Initialize the snapshot scorer.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        
        # Initialize components
        self.normalizer = SnapshotNormalizer()
        self.feature_attacher = UnderlierFeatureAttacher()
        self.iv_computer = IVAndGreeksComputer()
        
        # Model artifacts
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Output directory
        self.output_dir = "outputs"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SnapshotScorer with model directory: {model_dir}")
    
    def load_trained_models(self, model_timestamp: Optional[str] = None) -> Dict[str, str]:
        """
        Load trained models, scaler, and feature list.
        
        Args:
            model_timestamp: Specific timestamp to load models from
            
        Returns:
            Dictionary with loaded model artifacts
        """
        logger.info("Loading trained models...")
        
        loaded_artifacts = {}
        
        try:
            # Find model files
            if model_timestamp:
                # Load specific timestamp
                model_files = self._find_model_files_by_timestamp(model_timestamp)
            else:
                # Load latest models
                model_files = self._find_latest_model_files()
            
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            # Load models
            for model_type, model_path in model_files.items():
                if model_type.endswith('_model'):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Extract model type from filename
                    actual_type = model_type.replace('_model', '')
                    self.models[actual_type] = model
                    loaded_artifacts[actual_type] = model_path
                    logger.info(f"Loaded {actual_type} model from {model_path}")
            
            # Load scaler
            if 'scaler' in model_files:
                with open(model_files['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
                loaded_artifacts['scaler'] = model_files['scaler']
                logger.info(f"Loaded scaler from {model_files['scaler']}")
            
            # Load feature list
            if 'feature_list' in model_files:
                with open(model_files['feature_list'], 'r') as f:
                    self.feature_columns = json.load(f)
                loaded_artifacts['feature_list'] = model_files['feature_list']
                logger.info(f"Loaded feature list with {len(self.feature_columns)} features")
            
            # Load training manifest
            if 'manifest' in model_files:
                with open(model_files['manifest'], 'r') as f:
                    self.model_metadata = json.load(f)
                loaded_artifacts['manifest'] = model_files['manifest']
                logger.info("Loaded training manifest")
            
            # Validate loaded artifacts
            if not self.models:
                raise ValueError("No models loaded")
            if self.scaler is None:
                raise ValueError("Scaler not loaded")
            if not self.feature_columns:
                raise ValueError("Feature list not loaded")
            
            logger.info(f"Successfully loaded {len(loaded_artifacts)} model artifacts")
            return loaded_artifacts
            
        except Exception as e:
            logger.error(f"Error loading trained models: {e}")
            raise
    
    def _find_model_files_by_timestamp(self, timestamp: str) -> Dict[str, str]:
        """Find model files by specific timestamp."""
        model_files = {}
        
        # Look for files with the timestamp
        for file_path in Path(self.model_dir).glob(f"*{timestamp}*"):
            if file_path.suffix == '.pkl':
                if 'model' in file_path.name and ('classifier' in file_path.name or 'regressor' in file_path.name):
                    model_type = 'classifier_model' if 'classifier' in file_path.name else 'regressor_model'
                    model_files[model_type] = str(file_path)
                elif 'scaler' in file_path.name:
                    model_files['scaler'] = str(file_path)
            elif file_path.suffix == '.json':
                if 'feature_list' in file_path.name:
                    model_files['feature_list'] = str(file_path)
                elif 'training_manifest' in file_path.name:
                    model_files['manifest'] = str(file_path)
        
        return model_files
    
    def _find_latest_model_files(self) -> Dict[str, str]:
        """Find latest model files by timestamp."""
        model_files = {}
        
        # Get all files and sort by timestamp
        all_files = []
        for file_path in Path(self.model_dir).glob("*"):
            if file_path.is_file():
                # Extract timestamp from filename
                timestamp_match = None
                for part in file_path.stem.split('_'):
                    if len(part) == 8 and part.isdigit():  # YYYYMMDD format
                        timestamp_match = part
                        break
                
                if timestamp_match:
                    all_files.append((file_path, timestamp_match))
        
        if not all_files:
            return {}
        
        # Sort by timestamp and get latest
        all_files.sort(key=lambda x: x[1], reverse=True)
        latest_timestamp = all_files[0][1]
        
        # Find files with latest timestamp
        return self._find_model_files_by_timestamp(latest_timestamp)
    
    def normalize_snapshot(self, snapshot_file: str) -> pd.DataFrame:
        """
        Normalize snapshot JSON to tabular format.
        
        Args:
            snapshot_file: Path to snapshot JSON file
            
        Returns:
            Normalized DataFrame
        """
        logger.info(f"Normalizing snapshot: {snapshot_file}")
        
        # Load and normalize snapshot
        normalized_df = self.normalizer.normalize_snapshot_json(snapshot_file)
        
        # Validate schema
        validation_results = self.normalizer.validate_normalized_data(normalized_df)
        if validation_results.get('errors'):
            logger.warning(f"Schema validation warnings: {validation_results['errors']}")
        
        logger.info(f"Normalized snapshot: {len(normalized_df)} contracts")
        return normalized_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features matching the training data.
        
        Args:
            df: Normalized snapshot DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
        
        logger.info("Engineering features for scoring...")
        
        # Start with base features
        feature_df = df.copy()
        
        # 1. Attach underlier features
        feature_df = self.feature_attacher.attach_features(feature_df, lookback_days=60)
        
        # 2. Compute IV and Greeks
        feature_df = self.iv_computer.process_options_dataframe(feature_df)
        
        # 3. Add derived features (matching training data)
        feature_df = self._add_derived_features(feature_df)
        
        # 4. Add interaction features (matching training data)
        feature_df = self._add_interaction_features(feature_df)
        
        # 5. Add regime features (matching training data)
        feature_df = self._add_regime_features(feature_df)
        
        # 6. Add time features (matching training data)
        feature_df = self._add_time_features(feature_df)
        
        # 7. Encode categorical features (matching training data)
        feature_df = self._encode_categorical_features(feature_df)
        
        logger.info(f"Feature engineering completed: {len(feature_df)} contracts")
        return feature_df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features matching training data."""
        # Spread percentage
        if 'bidPrice' in df.columns and 'askPrice' in df.columns:
            df['spread_pct'] = (df['askPrice'] - df['bidPrice']) / df['premium_t']
        
        # Open interest rank
        if 'openInterest' in df.columns:
            df['oi_rank'] = df.groupby(['expiry_date'])['openInterest'].rank(pct=True)
        
        # Volume rank
        if 'totalTradedVolume' in df.columns:
            df['volume_rank'] = df.groupby(['expiry_date'])['totalTradedVolume'].rank(pct=True)
        
        # IV rank
        if 'iv_est_t' in df.columns:
            df['iv_rank'] = df.groupby(['expiry_date'])['iv_est_t'].rank(pct=True)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features matching training data."""
        # Common interactions
        if 'moneyness' in df.columns and 'ttm_days' in df.columns:
            df['interaction_moneyness_ttm_days'] = df['moneyness'] * df['ttm_days']
        
        if 'delta' in df.columns and 'theta' in df.columns:
            df['interaction_delta_theta'] = df['delta'] * df['theta']
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime features matching training data."""
        # Volatility regime
        if 'rv_20d' in df.columns:
            df['volatility_regime'] = pd.qcut(
                df['rv_20d'], 
                q=5, 
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            )
        
        # Market regime
        if 'ret_5d' in df.columns:
            df['market_regime'] = pd.cut(
                df['ret_5d'],
                bins=[-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf],
                labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
            )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time features matching training data."""
        if 'date_t' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date_t']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date_t']).dt.month
            df['quarter'] = pd.to_datetime(df['date_t']).dt.quarter
        
        if 'expiry_date' in df.columns:
            df['days_to_expiry'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['date_t'])).dt.days
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features matching training data."""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in df.columns and df[col].nunique() <= 20:
                # Create encoded column name
                encoded_col = f'{col}_encoded'
                if encoded_col in self.feature_columns:
                    # Use the same encoding logic as training
                    df[encoded_col] = df[col].astype(str).map(
                        {val: idx for idx, val in enumerate(df[col].unique())}
                    )
        
        return df
    
    def prepare_features_for_scoring(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model scoring.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Feature matrix for scoring
        """
        if df.empty:
            return np.array([])
        
        # Select only the features used during training
        available_features = [col for col in self.feature_columns if col in df.columns]
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features for scoring: {missing_features}")
        
        if not available_features:
            raise ValueError("No features available for scoring")
        
        # Create feature matrix
        X = df[available_features].values
        
        # Handle missing values
        if np.isnan(X).any():
            logger.warning("Found missing values in features, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        logger.info(f"Prepared features: {X_scaled.shape}")
        return X_scaled
    
    def score_models(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score the snapshot using trained models.
        
        Args:
            feature_df: DataFrame with engineered features
            
        Returns:
            DataFrame with model predictions
        """
        if feature_df.empty:
            return feature_df
        
        logger.info("Scoring snapshot with trained models...")
        
        # Prepare features
        X = self.prepare_features_for_scoring(feature_df)
        
        # Score with each model
        scored_df = feature_df.copy()
        
        # Classifier predictions
        if 'classifier' in self.models:
            classifier = self.models['classifier']
            try:
                y_pred_proba = classifier.predict_proba(X)[:, 1]
                scored_df['model_pop_probability'] = y_pred_proba
                scored_df['model_pop_prediction'] = (y_pred_proba > 0.5).astype(int)
                logger.info("Generated classifier predictions")
            except Exception as e:
                logger.error(f"Error with classifier prediction: {e}")
        
        # Regressor predictions
        if 'regressor' in self.models:
            regressor = self.models['regressor']
            try:
                y_pred = regressor.predict(X)
                scored_df['model_pnl_prediction'] = y_pred
                logger.info("Generated regressor predictions")
            except Exception as e:
                logger.error(f"Error with regressor prediction: {e}")
        
        # Combined score (if both models available)
        if 'classifier' in self.models and 'regressor' in self.models:
            try:
                # Use scoring formula from config or default
                scoring_formula = self.model_metadata.get('config', {}).get('scoring', {}).get('score_formula', 'model_pop * model_ev')
                
                if scoring_formula == 'model_pop * model_ev':
                    # Default: probability * expected value
                    scored_df['model_score'] = (
                        scored_df['model_pop_probability'] * 
                        scored_df['model_pnl_prediction']
                    )
                else:
                    # Custom scoring formula
                    scored_df['model_score'] = self._evaluate_custom_formula(
                        scored_df, scoring_formula
                    )
                
                logger.info("Generated combined model score")
            except Exception as e:
                logger.error(f"Error computing combined score: {e}")
        
        logger.info(f"Scoring completed for {len(scored_df)} contracts")
        return scored_df
    
    def _evaluate_custom_formula(self, df: pd.DataFrame, formula: str) -> pd.Series:
        """Evaluate custom scoring formula."""
        try:
            # Simple formula evaluation (can be extended)
            if formula == 'model_pop_probability':
                return df['model_pop_probability']
            elif formula == 'model_pnl_prediction':
                return df['model_pnl_prediction']
            else:
                # Default to simple multiplication
                return df['model_pop_probability'] * df['model_pnl_prediction']
        except Exception as e:
            logger.error(f"Error evaluating custom formula: {e}")
            return pd.Series(0.0, index=df.index)
    
    def rank_opportunities(self, scored_df: pd.DataFrame, 
                          max_positions: int = 20) -> pd.DataFrame:
        """
        Rank opportunities by model scores.
        
        Args:
            scored_df: DataFrame with model scores
            max_positions: Maximum number of positions to select
            
        Returns:
            Ranked DataFrame
        """
        if scored_df.empty:
            return scored_df
        
        logger.info("Ranking opportunities...")
        
        # Sort by model score (descending)
        if 'model_score' in scored_df.columns:
            ranked_df = scored_df.sort_values('model_score', ascending=False).reset_index(drop=True)
        elif 'model_pop_probability' in scored_df.columns:
            ranked_df = scored_df.sort_values('model_pop_probability', ascending=False).reset_index(drop=True)
        else:
            logger.warning("No scoring columns available for ranking")
            return scored_df
        
        # Add rank
        ranked_df['opportunity_rank'] = range(1, len(ranked_df) + 1)
        
        # Select top opportunities
        if len(ranked_df) > max_positions:
            ranked_df = ranked_df.head(max_positions).copy()
            logger.info(f"Selected top {max_positions} opportunities")
        
        logger.info(f"Ranking completed: {len(ranked_df)} opportunities")
        return ranked_df
    
    def generate_scoring_report(self, scored_df: pd.DataFrame, 
                               ranked_df: pd.DataFrame,
                               snapshot_file: str) -> Dict[str, str]:
        """
        Generate comprehensive scoring report.
        
        Args:
            scored_df: DataFrame with all scored contracts
            ranked_df: DataFrame with ranked opportunities
            snapshot_file: Original snapshot file path
            
        Returns:
            Dictionary with report file paths
        """
        logger.info("Generating scoring report...")
        
        report_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save scored snapshot
        scored_filename = f"scored_snapshot_{timestamp}.csv"
        scored_path = os.path.join(self.output_dir, scored_filename)
        scored_df.to_csv(scored_path, index=False)
        report_paths['scored_snapshot'] = scored_path
        
        # Save ranked opportunities
        ranked_filename = f"ranked_opportunities_{timestamp}.csv"
        ranked_path = os.path.join(self.output_dir, ranked_filename)
        ranked_df.to_csv(ranked_path, index=False)
        report_paths['ranked_opportunities'] = ranked_path
        
        # Generate summary report
        summary_filename = f"scoring_summary_{timestamp}.txt"
        summary_path = os.path.join(self.output_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(f"ML Model Scoring Summary\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Snapshot: {snapshot_file}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Information:\n")
            f.write(f"Models loaded: {list(self.models.keys())}\n")
            f.write(f"Features used: {len(self.feature_columns)}\n")
            f.write(f"Training timestamp: {self.model_metadata.get('timestamp', 'Unknown')}\n\n")
            
            f.write("Scoring Results:\n")
            f.write(f"Total contracts scored: {len(scored_df)}\n")
            f.write(f"Top opportunities selected: {len(ranked_df)}\n\n")
            
            if 'model_pop_probability' in scored_df.columns:
                f.write("Classification Results:\n")
                avg_prob = scored_df['model_pop_probability'].mean()
                f.write(f"Average POP probability: {avg_prob:.4f}\n")
                high_prob_count = (scored_df['model_pop_probability'] > 0.7).sum()
                f.write(f"High probability contracts (>0.7): {high_prob_count}\n\n")
            
            if 'model_pnl_prediction' in scored_df.columns:
                f.write("Regression Results:\n")
                avg_pnl = scored_df['model_pnl_prediction'].mean()
                f.write(f"Average predicted PnL: {avg_pnl:.2f}\n")
                positive_pnl_count = (scored_df['model_pnl_prediction'] > 0).sum()
                f.write(f"Positive PnL predictions: {positive_pnl_count}\n\n")
            
            if 'model_score' in scored_df.columns:
                f.write("Combined Scoring:\n")
                top_scores = ranked_df['model_score'].head(5)
                f.write("Top 5 scores:\n")
                for i, score in enumerate(top_scores, 1):
                    f.write(f"  {i}. {score:.4f}\n")
        
        report_paths['summary'] = summary_path
        
        logger.info(f"Generated scoring report: {list(report_paths.values())}")
        return report_paths
    
    def score_snapshot(self, snapshot_file: str, 
                       model_timestamp: Optional[str] = None,
                       max_positions: int = 20) -> Dict[str, Any]:
        """
        Complete pipeline to score a snapshot.
        
        Args:
            snapshot_file: Path to snapshot JSON file
            model_timestamp: Specific model timestamp to use
            max_positions: Maximum positions to select
            
        Returns:
            Dictionary with scoring results
        """
        logger.info(f"Starting snapshot scoring pipeline for {snapshot_file}")
        
        try:
            # Step 1: Load trained models
            loaded_artifacts = self.load_trained_models(model_timestamp)
            
            # Step 2: Normalize snapshot
            normalized_df = self.normalize_snapshot(snapshot_file)
            
            # Step 3: Engineer features
            feature_df = self.engineer_features(normalized_df)
            
            # Step 4: Score with models
            scored_df = self.score_models(feature_df)
            
            # Step 5: Rank opportunities
            ranked_df = self.rank_opportunities(scored_df, max_positions)
            
            # Step 6: Generate reports
            report_paths = self.generate_scoring_report(
                scored_df, ranked_df, snapshot_file
            )
            
            # Compile results
            results = {
                'status': 'success',
                'snapshot_file': snapshot_file,
                'timestamp': datetime.now().isoformat(),
                'models_used': list(self.models.keys()),
                'total_contracts': len(scored_df),
                'opportunities_selected': len(ranked_df),
                'report_paths': report_paths,
                'model_metadata': self.model_metadata
            }
            
            logger.info("Snapshot scoring pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Snapshot scoring pipeline failed: {e}")
            return {
                'status': 'failed',
                'snapshot_file': snapshot_file,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Score current snapshot using trained ML models'
    )
    
    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to snapshot JSON file'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/model_store',
        help='Directory containing trained models (default: models/model_store)'
    )
    
    parser.add_argument(
        '--model-timestamp',
        type=str,
        help='Specific model timestamp to use (default: latest)'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        default=20,
        help='Maximum number of positions to select (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Check if snapshot file exists
    if not os.path.exists(args.snapshot):
        print(f"Error: Snapshot file not found: {args.snapshot}")
        sys.exit(1)
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Initialize scorer
    try:
        scorer = SnapshotScorer(args.model_dir)
    except Exception as e:
        print(f"Error initializing scorer: {e}")
        sys.exit(1)
    
    # Score snapshot
    print(f"Scoring snapshot: {args.snapshot}")
    print(f"Using models from: {args.model_dir}")
    if args.model_timestamp:
        print(f"Model timestamp: {args.model_timestamp}")
    print("=" * 50)
    
    results = scorer.score_snapshot(
        args.snapshot, 
        args.model_timestamp,
        args.max_positions
    )
    
    if results['status'] == 'success':
        print("\nScoring completed successfully!")
        print(f"Models used: {results['models_used']}")
        print(f"Total contracts scored: {results['total_contracts']}")
        print(f"Opportunities selected: {results['opportunities_selected']}")
        
        print("\nReports generated:")
        for report_type, path in results['report_paths'].items():
            print(f"  {report_type}: {path}")
        
        # Show top opportunities
        if 'ranked_opportunities' in results['report_paths']:
            ranked_path = results['report_paths']['ranked_opportunities']
            if os.path.exists(ranked_path):
                ranked_df = pd.read_csv(ranked_path)
                print(f"\nTop 5 opportunities:")
                for i, row in ranked_df.head().iterrows():
                    symbol = row.get('symbol', 'N/A')
                    strike = row.get('strike', 'N/A')
                    option_type = row.get('option_type', 'N/A')
                    score = row.get('model_score', 'N/A')
                    print(f"  {i+1}. {symbol} {strike} {option_type} - Score: {score:.4f}")
    else:
        print(f"\nScoring failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
