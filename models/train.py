#!/usr/bin/env python3
"""
ML Model Training Module for Mode B (ML-Live)
============================================

This module trains machine learning models on historical reconstructed option chains
for Mode B. It implements walk-forward validation, feature engineering, and model
training for both classification (POP_label) and regression (PnL/ROI) tasks.

Key Functions:
- load_training_data: Load historical data from reconstructed chains
- engineer_features: Create comprehensive feature set
- split_data: Implement walk-forward validation splits
- train_models: Train classifier and regressor models
- evaluate_models: Compute performance metrics
- save_models: Store trained models and metadata

Usage:
    python train.py --config configs/ml_experiment.yml
    python train.py --config configs/ml_experiment.yml --validate-only
"""

import os
import sys
import yaml
import json
import pickle
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    mean_squared_error, mean_absolute_error
)
import lightgbm as lgb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas import (
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


class MLModelTrainer:
    """
    Trains machine learning models on historical options data.
    
    This class implements the complete ML training pipeline for Mode B,
    including data loading, feature engineering, walk-forward validation,
    model training, and evaluation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the ML trainer.
        
        Args:
            config_path: Path to ML experiment configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Model store directory
        self.model_store_dir = "models/model_store"
        Path(self.model_store_dir).mkdir(parents=True, exist_ok=True)
        
        # Output directory
        self.output_dir = "outputs"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MLModelTrainer for {self.config.symbol}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and parse ML experiment configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['symbol', 'target', 'horizons', 'features', 'splits', 'models']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
        
        return config
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load historical training data from reconstructed chains.
        
        Returns:
            DataFrame with training data
        """
        logger.info("Loading training data from reconstructed chains...")
        
        # Try to load from daily chain norm parquet
        daily_norm_path = "reconstructed/parquet/daily_chain_norm.parquet"
        
        if not os.path.exists(daily_norm_path):
            raise FileNotFoundError(f"Training data not found: {daily_norm_path}")
        
        # Load data
        df = pd.read_parquet(daily_norm_path)
        
        # Filter for target symbol
        df = df[df['symbol'] == self.config['symbol']].copy()
        
        if df.empty:
            raise ValueError(f"No data found for symbol {self.config['symbol']}")
        
        # Filter for date range
        if 'splits' in self.config:
            splits = self.config['splits']
            if 'train_start' in splits and 'train_end' in splits:
                train_start = pd.to_datetime(splits['train_start'])
                train_end = pd.to_datetime(splits['train_end'])
                
                df = df[
                    (df['date_t'] >= train_start) &
                    (df['date_t'] <= train_end)
                ].copy()
        
        # Filter for valid horizons
        if 'horizons' in self.config:
            valid_horizons = self.config['horizons']
            df = df[df['ttm_days'].isin(valid_horizons)].copy()
        
        # Ensure we have labeled data (expired contracts)
        initial_count = len(df)
        df = df[df['POP_label'].notna()].copy()
        
        if df.empty:
            raise ValueError("No labeled data found after filtering")
        
        logger.info(f"Loaded {len(df)} training samples (from {initial_count} total)")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML training.
        
        Args:
            df: DataFrame with raw training data
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
        
        logger.info("Engineering features for ML training...")
        
        # Start with base features
        feature_df = df.copy()
        
        # 1. Base features (already in data)
        base_features = self.config['features'].get('base', [])
        logger.info(f"Using base features: {base_features}")
        
        # 2. Derived features
        derived_features = self.config['features'].get('derived', [])
        if derived_features:
            logger.info(f"Computing derived features: {derived_features}")
            feature_df = self._compute_derived_features(feature_df, derived_features)
        
        # 3. Interaction features
        interaction_features = self.config['features'].get('interactions', [])
        if interaction_features:
            logger.info(f"Computing interaction features: {interaction_features}")
            feature_df = self._compute_interaction_features(feature_df, interaction_features)
        
        # 4. Regime features
        regime_features = self.config['features'].get('regime', [])
        if regime_features:
            logger.info(f"Computing regime features: {regime_features}")
            feature_df = self._compute_regime_features(feature_df, regime_features)
        
        # 5. Time-based features
        feature_df = self._add_time_features(feature_df)
        
        # 6. Categorical encoding
        feature_df = self._encode_categorical_features(feature_df)
        
        # Final feature selection
        all_feature_columns = self._get_all_feature_columns(feature_df)
        logger.info(f"Total features available: {len(all_feature_columns)}")
        
        # Store feature list for later use
        self.feature_columns = all_feature_columns
        
        return feature_df
    
    def _compute_derived_features(self, df: pd.DataFrame, derived_features: List[str]) -> pd.DataFrame:
        """Compute derived features."""
        for feature in derived_features:
            if feature == "spread_pct" and 'bidPrice' in df.columns and 'askPrice' in df.columns:
                df['spread_pct'] = (df['askPrice'] - df['bidPrice']) / df['premium_t']
            
            elif feature == "oi_rank" and 'openInterest' in df.columns:
                df['oi_rank'] = df.groupby(['date_t', 'expiry_date'])['openInterest'].rank(pct=True)
            
            elif feature == "volume_rank" and 'totalTradedVolume' in df.columns:
                df['volume_rank'] = df.groupby(['date_t', 'expiry_date'])['totalTradedVolume'].rank(pct=True)
            
            elif feature == "iv_rank" and 'iv_est_t' in df.columns:
                df['iv_rank'] = df.groupby(['date_t', 'expiry_date'])['iv_est_t'].rank(pct=True)
        
        return df
    
    def _compute_interaction_features(self, df: pd.DataFrame, interaction_features: List[str]) -> pd.DataFrame:
        """Compute interaction features."""
        for interaction in interaction_features:
            if '*' in interaction:
                feature1, feature2 = interaction.split('*')
                if feature1 in df.columns and feature2 in df.columns:
                    df[f'interaction_{feature1}_{feature2}'] = df[feature1] * df[feature2]
        
        return df
    
    def _compute_regime_features(self, df: pd.DataFrame, regime_features: List[str]) -> pd.DataFrame:
        """Compute regime-based features."""
        for feature in regime_features:
            if feature == "volatility_regime" and 'rv_20d' in df.columns:
                # Create volatility regime buckets
                df['volatility_regime'] = pd.qcut(
                    df['rv_20d'], 
                    q=5, 
                    labels=['very_low', 'low', 'medium', 'high', 'very_high']
                )
            
            elif feature == "market_regime" and 'ret_5d' in df.columns:
                # Create market regime buckets
                df['market_regime'] = pd.cut(
                    df['ret_5d'],
                    bins=[-np.inf, -0.02, -0.01, 0.01, 0.02, np.inf],
                    labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
                )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'date_t' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date_t']).dt.dayofweek
            df['month'] = pd.to_datetime(df['date_t']).dt.month
            df['quarter'] = pd.to_datetime(df['date_t']).dt.quarter
        
        if 'expiry_date' in df.columns:
            df['days_to_expiry'] = (pd.to_datetime(df['expiry_date']) - pd.to_datetime(df['date_t'])).dt.days
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in df.columns and df[col].nunique() > 1:
                # Skip columns that are already encoded or have too many unique values
                if df[col].nunique() <= 20:  # Reasonable number of categories
                    df[f'{col}_encoded'] = self.label_encoder.fit_transform(df[col].astype(str))
        
        return df
    
    def _get_all_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of all available feature columns."""
        # Exclude target columns and metadata columns
        exclude_columns = [
            'POP_label', 'PnL', 'ROI', 'payoff_T', 'S_T',
            'symbol', 'date_t', 'expiry_date', 'option_type',
            'strike', 'premium_t', 'synthetic_flag'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Ensure all features are numeric
        numeric_features = []
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                logger.warning(f"Skipping non-numeric feature: {col}")
        
        return numeric_features
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets using walk-forward validation.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data using walk-forward validation...")
        
        # Sort by date
        df = df.sort_values('date_t').reset_index(drop=True)
        
        # Get split configuration
        splits = self.config['splits']
        train_start = pd.to_datetime(splits['train_start'])
        train_end = pd.to_datetime(splits['train_end'])
        val_start = pd.to_datetime(splits['valid_start'])
        val_end = pd.to_datetime(splits['valid_end'])
        
        # Split data
        train_df = df[
            (df['date_t'] >= train_start) &
            (df['date_t'] <= train_end)
        ].copy()
        
        val_df = df[
            (df['date_t'] >= val_start) &
            (df['date_t'] <= val_end)
        ].copy()
        
        # Test set (if specified)
        if 'test_start' in splits and 'test_end' in splits:
            test_start = pd.to_datetime(splits['test_start'])
            test_end = pd.to_datetime(splits['test_end'])
            test_df = df[
                (df['date_t'] >= test_start) &
                (df['date_t'] <= test_end)
            ].copy()
        else:
            test_df = pd.DataFrame()
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            Tuple of (X, y_classifier, y_regressor)
        """
        if df.empty:
            return np.array([]), np.array([]), np.array([])
        
        # Prepare features
        X = df[self.feature_columns].values
        
        # Prepare targets
        y_classifier = None
        y_regressor = None
        
        if 'classifier' in self.config['target']:
            target_col = self.config['target']['classifier']
            if target_col in df.columns:
                y_classifier = df[target_col].values
        
        if 'regressor' in self.config['target']:
            target_col = self.config['target']['regressor']
            if target_col in df.columns:
                y_regressor = df[target_col].values
        
        return X, y_classifier, y_regressor
    
    def train_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train classifier and regressor models.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Dictionary with trained models and metadata
        """
        logger.info("Training ML models...")
        
        models = {}
        
        # Prepare training data
        X_train, y_train_class, y_train_reg = self.prepare_features_and_targets(train_df)
        X_val, y_val_class, y_val_reg = self.prepare_features_and_targets(val_df)
        
        if X_train.size == 0:
            raise ValueError("No training data available")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train classifier
        if y_train_class is not None and 'classifier' in self.config['models']:
            logger.info("Training classifier model...")
            classifier_config = self.config['models']['classifier']
            
            if classifier_config['type'] == 'lightgbm':
                classifier = self._train_lightgbm_classifier(
                    X_train_scaled, y_train_class,
                    X_val_scaled, y_val_class,
                    classifier_config
                )
                models['classifier'] = classifier
        
        # Train regressor
        if y_train_reg is not None and 'regressor' in self.config['models']:
            logger.info("Training regressor model...")
            regressor_config = self.config['models']['regressor']
            
            if regressor_config['type'] == 'lightgbm':
                regressor = self._train_lightgbm_regressor(
                    X_train_scaled, y_train_reg,
                    X_val_scaled, y_val_reg,
                    regressor_config
                )
                models['regressor'] = regressor
        
        return models
    
    def _train_lightgbm_classifier(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   config: Dict[str, Any]) -> lgb.LGBMClassifier:
        """Train LightGBM classifier."""
        # Default parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Update with config parameters
        if 'parameters' in config:
            params.update(config['parameters'])
        
        # Create and train model
        model = lgb.LGBMClassifier(**params)
        
        # Train with validation data
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        return model
    
    def _train_lightgbm_regressor(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  config: Dict[str, Any]) -> lgb.LGBMRegressor:
        """Train LightGBM regressor."""
        # Default parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Update with config parameters
        if 'parameters' in config:
            params.update(config['parameters'])
        
        # Create and train model
        model = lgb.LGBMRegressor(**params)
        
        # Train with validation data
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        return model
    
    def evaluate_models(self, models: Dict[str, Any], val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate trained models on validation data.
        
        Args:
            models: Dictionary with trained models
            val_df: Validation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Prepare validation data
        X_val, y_val_class, y_val_reg = self.prepare_features_and_targets(val_df)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Evaluate classifier
        if 'classifier' in models and y_val_class is not None:
            classifier = models['classifier']
            y_pred_proba = classifier.predict_proba(X_val_scaled)[:, 1]
            y_pred = classifier.predict(X_val_scaled)
            
            # Classification metrics
            evaluation_results['classifier'] = {
                'roc_auc': roc_auc_score(y_val_class, y_pred_proba),
                'pr_auc': average_precision_score(y_val_class, y_pred_proba),
                'brier_score': brier_score_loss(y_val_class, y_pred_proba),
                'accuracy': (y_pred == y_val_class).mean()
            }
        
        # Evaluate regressor
        if 'regressor' in models and y_val_reg is not None:
            regressor = models['regressor']
            y_pred = regressor.predict(X_val_scaled)
            
            # Regression metrics
            evaluation_results['regressor'] = {
                'rmse': np.sqrt(mean_squared_error(y_val_reg, y_pred)),
                'mae': mean_absolute_error(y_val_reg, y_pred),
                'r2': regressor.score(X_val_scaled, y_val_reg)
            }
            
            # Rank correlation
            if len(y_val_reg) > 1:
                rank_corr = np.corrcoef(
                    np.argsort(y_val_reg), 
                    np.argsort(y_pred)
                )[0, 1]
                evaluation_results['regressor']['rank_correlation'] = rank_corr
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def save_models(self, models: Dict[str, Any], evaluation_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save trained models and metadata.
        
        Args:
            models: Dictionary with trained models
            evaluation_results: Model evaluation results
            
        Returns:
            Dictionary with saved file paths
        """
        logger.info("Saving trained models...")
        
        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_type, model in models.items():
            model_filename = f"model_{self.config['symbol']}_{model_type}_{timestamp}.pkl"
            model_path = os.path.join(self.model_store_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            saved_paths[f'{model_type}_model'] = model_path
        
        # Save scaler
        scaler_filename = f"scaler_{self.config['symbol']}_{timestamp}.pkl"
        scaler_path = os.path.join(self.model_store_dir, scaler_filename)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        saved_paths['scaler'] = scaler_path
        
        # Save feature list
        features_filename = f"feature_list_{self.config['symbol']}_{timestamp}.json"
        features_path = os.path.join(self.model_store_dir, features_filename)
        
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        saved_paths['feature_list'] = features_path
        
        # Save training manifest
        manifest = {
            'timestamp': timestamp,
            'symbol': self.config['symbol'],
            'config': self.config,
            'evaluation_results': evaluation_results,
            'feature_columns': self.feature_columns,
            'data_info': {
                'train_samples': len(self.train_df) if hasattr(self, 'train_df') else 0,
                'val_samples': len(self.val_df) if hasattr(self, 'val_df') else 0,
                'test_samples': len(self.test_df) if hasattr(self, 'test_df') else 0
            }
        }
        
        manifest_filename = f"training_manifest_{self.config['symbol']}_{timestamp}.json"
        manifest_path = os.path.join(self.model_store_dir, manifest_filename)
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        saved_paths['manifest'] = manifest_path
        
        logger.info(f"Saved {len(saved_paths)} model artifacts")
        return saved_paths
    
    def run_training(self) -> Dict[str, Any]:
        """
        Run the complete ML training pipeline.
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting ML training pipeline...")
        
        try:
            # Step 1: Load training data
            df = self.load_training_data()
            
            # Step 2: Engineer features
            feature_df = self.engineer_features(df)
            
            # Step 3: Split data
            train_df, val_df, test_df = self.split_data(feature_df)
            
            # Store for later use
            self.train_df = train_df
            self.val_df = val_df
            self.test_df = test_df
            
            # Step 4: Train models
            models = self.train_models(train_df, val_df)
            
            # Step 5: Evaluate models
            evaluation_results = self.evaluate_models(models, val_df)
            
            # Step 6: Save models
            saved_paths = self.save_models(models, evaluation_results)
            
            # Compile results
            results = {
                'status': 'success',
                'symbol': self.config['symbol'],
                'timestamp': datetime.now().isoformat(),
                'models_trained': list(models.keys()),
                'evaluation_results': evaluation_results,
                'saved_paths': saved_paths,
                'training_info': {
                    'train_samples': len(train_df),
                    'val_samples': len(val_df),
                    'test_samples': len(test_df),
                    'features_count': len(self.feature_columns)
                }
            }
            
            logger.info("ML training pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"ML training pipeline failed: {e}")
            return {
                'status': 'failed',
                'symbol': self.config['symbol'],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Train ML models on historical options data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to ML experiment configuration YAML file'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without training'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Initialize trainer
    try:
        trainer = MLModelTrainer(args.config)
    except Exception as e:
        print(f"Error initializing ML trainer: {e}")
        sys.exit(1)
    
    if args.validate_only:
        print("Configuration validation completed successfully!")
        print(f"Symbol: {trainer.config['symbol']}")
        print(f"Targets: {trainer.config['target']}")
        print(f"Horizons: {trainer.config['horizons']}")
        print(f"Features: {len(trainer.config['features'].get('base', []))} base features")
        return
    
    # Run training
    print(f"Starting ML training for {trainer.config['symbol']}")
    print("=" * 50)
    
    results = trainer.run_training()
    
    if results['status'] == 'success':
        print("\nTraining completed successfully!")
        print(f"Models trained: {results['models_trained']}")
        print(f"Training samples: {results['training_info']['train_samples']}")
        print(f"Validation samples: {results['training_info']['val_samples']}")
        print(f"Features used: {results['training_info']['features_count']}")
        
        print("\nEvaluation Results:")
        for model_type, metrics in results['evaluation_results'].items():
            print(f"\n{model_type.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print("\nModels saved to:")
        for artifact_type, path in results['saved_paths'].items():
            print(f"  {artifact_type}: {path}")
    else:
        print(f"\nTraining failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
