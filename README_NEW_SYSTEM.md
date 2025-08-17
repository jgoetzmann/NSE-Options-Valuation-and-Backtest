# NSE Options Valuation and Backtest Project - New Three-Mode System

## Overview

This project now implements a comprehensive three-execution-mode system for NSE options analysis, backtesting, and machine learning. The system leverages the existing valuation engine (`utils.py`) and provides three distinct modes of operation:

- **Mode A (SB-CS)**: Synthetic Backtest on Current Snapshot - For pipeline validation and exploratory ranking
- **Mode B (ML-Live)**: Train-and-Score ML on Historical Reconstructions; Score Current Snapshot - For machine learning model training and live scoring
- **Mode C (EOD-True)**: True EOD Backtest via Reconstructed Past Chains - For historically accurate backtesting

## Project Structure

```
NSE-Options-Valuation-and-Backtest/
├── data_pipeline/                    # Core data processing modules
│   ├── schemas.py                    # Central data schemas and validation
│   ├── normalize_snapshot.py         # Mode A: Normalize live JSON snapshots
│   ├── attach_underlier_features.py  # Attach underlier features (returns, volatility)
│   ├── compute_iv_and_greeks.py      # Compute IV and Greeks from premiums
│   └── make_labels.py                # Generate payoff, PnL, ROI, and POP labels
├── backtests/                        # Backtesting execution modules
│   ├── run_synthetic_on_snapshot.py  # Mode A: Synthetic backtest runner
│   ├── run_true_backtest.py          # Mode C: True EOD backtest runner
│   └── reports.py                    # Common KPIs, plots, and summaries
├── models/                           # Machine learning modules
│   ├── train.py                      # Mode B: Train ML models
│   ├── score_snapshot.py             # Mode B: Score current snapshots
│   └── model_store/                  # Serialized models and configs
├── reconstructed/                    # Historical data storage
│   ├── json/                         # Reconstructed per-day JSON chains
│   └── parquet/                      # Columnar normalized tables
├── snapshots/                        # Live data snapshots
│   └── json/                         # Timestamped live chain snapshots
├── configs/                          # Configuration files
│   ├── backtest_synth.yml            # Mode A configuration
│   ├── backtest_true.yml             # Mode C configuration
│   └── ml_experiment.yml             # Mode B configuration
└── outputs/                          # Results and reports
    ├── csv/                          # Processed CSV outputs
    ├── json/                         # Processed JSON outputs
    └── results_from_nse_valuations/  # Summary reports
```

## Installation and Dependencies

### Required Packages

```bash
pip install pandas numpy scipy requests yfinance pyyaml scikit-learn lightgbm
```

### Additional Dependencies

- **For IV inversion**: `scipy.optimize` (included with scipy)
- **For ML training**: `lightgbm`, `scikit-learn`
- **For configuration**: `pyyaml`

## Quick Start Guide

### Mode A: Synthetic Backtest (Recommended for First-Time Users)

Mode A is the easiest to get started with and validates the entire pipeline:

```bash
# 1. First, get a current snapshot using the existing scraper
python nse_options_scraper.py

# 2. Run synthetic backtest on the snapshot
python backtests/run_synthetic_on_snapshot.py outputs/json/chain_NIFTY_YYYYMMDD_HHMM.json
```

This will:
- Normalize the snapshot JSON to tabular format
- Attach underlier features (returns, volatility)
- Compute IV and Greeks
- Generate synthetic labels
- Run horizon simulations (3, 7, 30 days)
- Rank and select positions
- Generate comprehensive reports

### Mode B: Machine Learning Training and Scoring

Mode B requires historical data but provides ML-powered insights:

```bash
# 1. Train models on historical data (requires Mode C data)
python models/train.py --config configs/ml_experiment.yml

# 2. Score current snapshot with trained models
python models/score_snapshot.py --snapshot snapshot.json --model models/model_store/
```

### Mode C: True EOD Backtest

Mode C provides historically accurate backtesting:

```bash
# Run true EOD backtest (requires historical derivatives data)
python backtests/run_true_backtest.py --config configs/backtest_true.yml
```

## Detailed Usage

### Mode A: Synthetic Backtest (SB-CS)

**Purpose**: Validate end-to-end plumbing and explore feature importance without requiring historical data reconstruction.

**Inputs**:
- Live JSON snapshots from `nse_options_scraper.py`
- yfinance EOD underlier data for feature calculation

**Process Flow**:
1. **Normalize**: Convert JSON to tabular format
2. **Feature Engineering**: Attach underlier returns, volatility, regime indicators
3. **IV & Greeks**: Compute implied volatility and option Greeks
4. **Horizon Simulation**: Simulate entry scenarios at different time horizons
5. **Position Selection**: Rank and select optimal positions
6. **Reporting**: Generate comprehensive analysis reports

**Configuration**: `configs/backtest_synth.yml`

**Key Features**:
- Multiple entry horizons (3, 7, 30 days)
- Advanced filtering (OI, volume, spread, IV)
- Portfolio construction with position sizing
- Performance metrics and risk analysis
- Synthetic flag for clear identification

### Mode B: Machine Learning (ML-Live)

**Purpose**: Train interpretable ML models on historical data and score current snapshots for live trading decisions.

**Inputs**:
- Historical reconstructed data from Mode C
- Current snapshots for scoring

**Process Flow**:
1. **Data Preparation**: Feature engineering and preprocessing
2. **Model Training**: Train classifier (POP) and regressor (PnL)
3. **Validation**: Walk-forward validation with embargo periods
4. **Model Storage**: Save models, scalers, and metadata
5. **Live Scoring**: Score current snapshots with trained models
6. **Opportunity Ranking**: Generate ranked opportunity lists

**Configuration**: `configs/ml_experiment.yml`

**Key Features**:
- Multiple ML algorithms (LightGBM, XGBoost, CatBoost)
- Feature interactions and derived features
- Time-series aware validation
- Model interpretation (SHAP, feature importance)
- Ensemble methods and uncertainty quantification

### Mode C: True EOD Backtest (EOD-True)

**Purpose**: Create historically accurate, labeled datasets by reconstructing per-day option chains using official historical derivatives data.

**Inputs**:
- Official historical derivatives files (bhavcopy)
- yfinance underlier EOD data
- Trading calendar and holiday information

**Process Flow**:
1. **Data Reconstruction**: Build daily option chains from historical files
2. **Feature Computation**: Calculate IV, Greeks, and derived features
3. **Label Generation**: Compute realized payoffs and PnL
4. **Backtesting**: Run portfolio strategies with realistic constraints
5. **Performance Analysis**: Comprehensive risk and return metrics
6. **Data Export**: Save reconstructed data for ML training

**Configuration**: `configs/backtest_true.yml`

**Key Features**:
- Historical accuracy with official data sources
- Multiple data sources (bhavcopy, NSE historical)
- Data quality validation and integrity checks
- Realistic transaction costs and slippage
- Comprehensive performance metrics

## Configuration Files

### Backtest Configuration (Mode A & C)

```yaml
# Example: configs/backtest_synth.yml
symbol: "NIFTY"
horizons: [3, 7, 30]

filters:
  min_oi: 500
  min_premium: 2.0
  max_spread_pct: 0.08

costs:
  round_turn_bps: 60
  slippage_mode: "half_spread"

portfolio:
  daily_max_positions: 20
  rank_score: "pct_diff_times_confidence"
```

### ML Configuration (Mode B)

```yaml
# Example: configs/ml_experiment.yml
target:
  classifier: "POP_label"
  regressor: "PnL"

features:
  base: ["moneyness", "ttm_days", "iv_est_t", "delta", "gamma"]
  interactions: ["moneyness*ttm_days", "delta*theta"]

splits:
  train_start: "2022-01-01"
  train_end: "2024-06-30"
  valid_start: "2024-07-01"
  valid_end: "2024-12-31"
```

## Data Schema

The system uses a standardized schema across all modes:

### Core Contract Columns
- `date_t`: Trading date (as-of date)
- `expiry_date`: Contract expiry date
- `symbol`: Underlying symbol
- `option_type`: CE/PE
- `strike`: Strike price
- `premium_t`: Entry price

### Market Data Columns
- `openInterest`: Open interest
- `totalTradedVolume`: Trading volume
- `bidPrice`, `askPrice`: Bid/ask prices
- `impliedVolatility`: Implied volatility

### Computed Features
- `iv_est_t`: Estimated IV from premium
- `delta`, `gamma`, `theta`, `vega`, `rho`: Greeks
- `moneyness`: S_t / strike
- `ttm_days`: Time to maturity

### Labels (for ML)
- `payoff_T`: Option payoff at expiry
- `PnL`: Profit/Loss
- `ROI`: Return on Investment
- `POP_label`: Binary profit indicator

## Performance Metrics

### Portfolio Metrics
- **Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Risk**: Maximum drawdown, Value at Risk, volatility
- **Efficiency**: Win rate, profit factor, turnover

### ML Metrics
- **Classification**: ROC-AUC, PR-AUC, Brier score
- **Regression**: RMSE, MAE, rank information coefficient
- **Portfolio**: Information coefficient, rank correlation

## Advanced Features

### Feature Engineering
- **Base Features**: Moneyness, time decay, Greeks, market microstructure
- **Derived Features**: IV skew, volume-OI ratios, regime indicators
- **Interactions**: Moneyness × time, delta × theta, spread × OI rank

### Risk Management
- **Position Sizing**: Equal weight, volatility weight, confidence weight
- **Correlation Limits**: Maximum correlation between positions
- **Drawdown Controls**: Stop-loss and take-profit thresholds

### Data Quality
- **Validation**: Schema validation, data range checks
- **Imputation**: Missing value handling strategies
- **Outlier Detection**: Statistical outlier identification

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Data Quality**: Check for missing or invalid data in snapshots
3. **IV Convergence**: Some options may fail IV inversion; check convergence rates
4. **Memory Issues**: Large datasets may require batch processing

### Debug Mode

Enable detailed logging in configuration files:

```yaml
output:
  detailed_logging: true
  save_intermediate: true
```

### Validation

Use the built-in validation functions:

```python
from data_pipeline.schemas import validate_dataframe_schema

# Validate data quality
validation_result = validate_dataframe_schema(df, strict=False)
print(validation_result)
```

## Best Practices

### For Mode A (Synthetic)
- Use for system validation and feature exploration
- Clearly label results as synthetic
- Focus on relative rankings rather than absolute performance
- Test different parameter combinations

### For Mode B (ML)
- Use walk-forward validation for time-series data
- Avoid data leakage between train/validation sets
- Monitor feature importance stability
- Regular model retraining and validation

### For Mode C (True Backtest)
- Validate reconstructed data against official sources
- Use realistic transaction costs and slippage
- Account for trading holidays and market hours
- Document all assumptions and limitations

## Contributing

### Development Workflow
1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Include type hints for all functions
- Add comprehensive docstrings
- Write unit tests for new functionality

### Testing
- Run existing tests: `python -m pytest tests/`
- Validate data pipeline: `python data_pipeline/schemas.py`
- Test backtest modes with sample data

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration examples
3. Validate data quality and schema
4. Check system requirements and dependencies

---

**Note**: This system is designed for educational and research purposes. Always conduct thorough testing before using in live trading environments. The synthetic backtest results are not indicative of actual trading performance.
