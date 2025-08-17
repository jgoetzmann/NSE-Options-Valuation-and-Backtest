# Scripts Overview

This document provides a comprehensive overview of all available scripts in the NSE Options Valuation and Backtest Platform.

---

## Main Platform Scripts (Three-Mode System)

### Mode A: Synthetic Backtest
- **File**: `run_mode_a.py`
- **Purpose**: Daily opportunity screening and pipeline validation
- **Execution Time**: 2-5 minutes
- **Best For**: Daily analysis, system validation
- **Requirements**: Live NSE data snapshot
- **Outputs**: CSV results, performance summaries, portfolio analysis

### Mode B: Machine Learning
- **File**: `run_mode_b.py`
- **Purpose**: ML model training and live scoring
- **Execution Time**: 15-30 minutes
- **Best For**: Traders, researchers, ML enthusiasts
- **Requirements**: Historical data from Mode C
- **Outputs**: Trained models, scoring results, feature importance

### Mode C: True Backtest
- **File**: `run_mode_c.py`
- **Purpose**: Historical backtesting with reconstructed data
- **Execution Time**: 30-60 minutes
- **Best For**: Strategy validation, academic research
- **Requirements**: Historical derivatives data (bhavcopy files)
- **Outputs**: Backtest results, performance metrics, reconstructed data

---

## NSE Data Processing Scripts

### NSE Options Scraper
- **File**: `nse_options_scraper.py`
- **Purpose**: Fetch live options data from National Stock Exchange of India
- **Execution Time**: 1-3 minutes
- **Best For**: Data collection, market monitoring
- **Features**:
  - Real-time data collection from NSE API
  - Robust error handling and retry mechanisms
  - Data normalization and cleaning
  - Outputs both raw and processed JSON formats
- **Outputs**:
  - `raw_option_chain_<symbol>_<timestamp>.json`
  - `option_chain_slim_<symbol>_<timestamp>.json`

### Complete NSE Analysis Pipeline
- **File**: `nse_options_valuation_processor.py`
- **Purpose**: End-to-end NSE options analysis with comprehensive valuation
- **Execution Time**: 5-15 minutes
- **Best For**: Complete analysis, professional reporting
- **Features**:
  - Automatic data fetching and processing
  - Black-Scholes pricing and Greeks calculation
  - Enhanced valuation with confidence scoring
  - Portfolio construction and ranking
  - Professional reporting and CSV exports
- **Outputs**: Detailed CSV, performance summaries, portfolio analysis

---

## US Options Analysis Scripts

### Basic US Options Analysis
- **File**: `option_info_manual.py`
- **Purpose**: Quick Greeks calculation for individual US options
- **Execution Time**: Interactive (1-2 minutes)
- **Best For**: Quick analysis, learning options, basic risk assessment
- **Features**:
  - Interactive input interface
  - Real-time data from Yahoo Finance
  - Basic Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
  - Simple, focused output
- **Usage**: Interactive prompts for symbol, expiration, strike, type

### Advanced US Options Analysis
- **File**: `option_info_manual_valuation_processor.py`
- **Purpose**: Comprehensive US options analysis with advanced valuation
- **Execution Time**: Interactive (2-3 minutes)
- **Best For**: Detailed analysis, professional valuation, risk assessment
- **Features**:
  - All features from basic version
  - Theoretical vs. market price comparison
  - Advanced valuation algorithm with confidence scoring
  - Market condition analysis
  - Risk assessment and warnings
- **Usage**: Same interface as basic version, plus comprehensive analysis

---

## Utility and Support Scripts

### Data Pipeline Scripts
- **Location**: `data_pipeline/` directory
- **Purpose**: Core data processing and feature engineering
- **Scripts**:
  - `normalize_snapshot.py` - Live data normalization
  - `compute_iv_and_greeks.py` - IV calculation & Greeks
  - `compute_enhanced_features.py` - Advanced feature engineering
  - `make_labels.py` - ML label generation
  - `reconstruct_chain_from_eod.py` - Historical data reconstruction
  - `schemas.py` - Data validation and schemas

### Backtesting Scripts
- **Location**: `backtests/` directory
- **Purpose**: Backtesting execution and analysis
- **Scripts**:
  - `run_synthetic_on_snapshot.py` - Mode A backtest runner
  - `run_true_backtest.py` - Mode C backtest runner
  - `reports.py` - Performance analysis and reporting

### Machine Learning Scripts
- **Location**: `models/` directory
- **Purpose**: ML model training and scoring
- **Scripts**:
  - `train.py` - Model training (Mode B)
  - `score_snapshot.py` - Live scoring (Mode B)

---

## Script Selection Guide

### For New Users
1. **Start with**: `run_mode_a.py` (Mode A)
2. **Get data**: `nse_options_scraper.py`
3. **Learn options**: `option_info_manual.py`

### For Daily Traders
1. **Morning routine**: `nse_options_scraper.py` → `run_mode_a.py`
2. **Quick US check**: `option_info_manual.py`
3. **Detailed analysis**: `nse_options_valuation_processor.py`

### For Researchers
1. **Data collection**: `nse_options_scraper.py`
2. **Historical analysis**: `run_mode_c.py` (Mode C)
3. **ML exploration**: `run_mode_b.py` (Mode B)

### For Developers
1. **System validation**: `run_mode_a.py`
2. **Custom features**: Modify scripts in `data_pipeline/`
3. **Custom models**: Extend scripts in `models/`

---

## Important Notes

### Dependencies
- **Mode A**: Requires live NSE data (run scraper first)
- **Mode B**: Requires historical data from Mode C
- **Mode C**: Requires historical derivatives data files
- **US scripts**: Require internet connection for yfinance data

### Execution Order
1. **Data collection**: `nse_options_scraper.py`
2. **Basic analysis**: `run_mode_a.py` or `nse_options_valuation_processor.py`
3. **Historical analysis**: `run_mode_c.py` (if data available)
4. **ML analysis**: `run_mode_b.py` (after Mode C)

### File Locations
- **Scripts**: Project root directory
- **Data pipeline**: `data_pipeline/` directory
- **Backtests**: `backtests/` directory
- **Models**: `models/` directory
- **Outputs**: `outputs/` directory

---

## Troubleshooting Common Issues

### "No module found" errors
- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)

### "No snapshot data" errors
- Run `python nse_options_scraper.py` first
- Check `outputs/json/` directory

### "Historical data not found" errors
- Mode C requires historical derivatives data
- Check `reconstructed/` directory

### "ML models not found" errors
- Mode B requires training first
- Check `models/model_store/` directory

---

## Learning Path

### Week 1: Getting Started
1. Install and run Mode A
2. Understand basic outputs
3. Try different symbols

### Week 2: Deep Dive
1. Explore configuration options
2. Run complete NSE pipeline
3. Try US options analysis

### Week 3: Advanced Features
1. Run historical backtesting (Mode C)
2. Train ML models (Mode B)
3. Customize features and parameters

### Week 4: Production Use
1. Set up daily workflows
2. Optimize configurations
3. Integrate with trading systems

---

*For detailed usage instructions, see the main [README.md](README.md) and [GETTING_STARTED.md](GETTING_STARTED.md)*
