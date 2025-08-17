# NSE Options Valuation and Backtest Project
## Professional-Grade Options Analysis Platform with Three-Execution-Mode System

A comprehensive toolkit for options analysis that combines Black–Scholes pricing/Greeks, yfinance-based US options lookups, NSE (India) option-chain scraping and processing, and advanced backtesting with machine learning capabilities. This project provides enterprise-level options analysis with real-time data processing, advanced valuation algorithms, comprehensive testing frameworks, and three distinct execution modes.

## 🚀 **What You Can Do**

### **Core Capabilities**
- **Manual US Options Analysis**: Interactive Greeks calculation and pricing for individual US options
- **Advanced US Options Valuation**: Sophisticated valuation analysis with confidence scoring and market condition assessment
- **NSE Data Scraping**: Real-time option-chain data collection from National Stock Exchange of India
- **Comprehensive Data Processing**: Process large NSE datasets with professional-grade analysis
- **Advanced Valuation Engine**: Multi-factor valuation considering moneyness, time decay, volatility, and liquidity

### **Three-Execution-Mode System**
- **Mode A (SB-CS)**: Synthetic Backtest on Current Snapshot - For pipeline validation and exploratory ranking
- **Mode B (ML-Live)**: Train-and-Score ML on Historical Reconstructions; Score Current Snapshot - For machine learning model training and live scoring
- **Mode C (EOD-True)**: True EOD Backtest via Reconstructed Past Chains - For historically accurate backtesting

---

## 🏗️ **Project Architecture**

### **Core Engine**
- **`utils.py`**: Black-Scholes calculations, Greeks, and advanced valuation engine
- **`nse_options_scraper.py`**: NSE data scraping with robust error handling
- **`nse_options_valuation_processor.py`**: Complete NSE analysis pipeline

### **New Three-Mode System**
```
NSE-Options-Valuation-and-Backtest/
├── data_pipeline/                    # Core data processing modules
│   ├── schemas.py                    # Central data schemas and validation
│   ├── normalize_snapshot.py         # Mode A: Normalize live JSON snapshots
│   ├── attach_underlier_features.py  # Attach underlier features (returns, volatility)
│   ├── compute_iv_and_greeks.py      # Compute IV and Greeks from premiums
│   ├── compute_enhanced_features.py  # Enhanced mispricing and confidence scoring
│   ├── make_labels.py                # Generate payoff, PnL, ROI, and POP labels
│   └── reconstruct_chain_from_eod.py # Mode C: Historical data reconstruction
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
├── outputs/                          # Results and reports
│   ├── csv/                          # Processed CSV outputs
│   ├── json/                         # Processed JSON outputs
│   └── results_from_nse_valuations/  # Summary reports
└── run_mode_a.py, run_mode_b.py, run_mode_c.py  # Top-level execution scripts
```

---

## 📦 **Requirements & Installation**

### **Required Packages**
```bash
pip install pandas numpy scipy requests yfinance pyyaml scikit-learn lightgbm matplotlib seaborn tqdm
```

### **Additional Dependencies**
- **For IV inversion**: `scipy.optimize` (included with scipy)
- **For ML training**: `lightgbm`, `scikit-learn`
- **For configuration**: `pyyaml`
- **For visualization**: `matplotlib`, `seaborn`

---

## 🎯 **Three Execution Modes Explained**

### **Mode A: Synthetic Backtest on Current Snapshot (SB-CS)**
**Purpose**: Pipeline validation and exploratory ranking without requiring historical data

**What It Does**:
1. **Data Ingestion**: Takes live NSE options snapshot
2. **Feature Engineering**: Adds underlier returns, volatility, market regimes
3. **Options Valuation**: Computes IV, Greeks, and enhanced mispricing scores
4. **Horizon Simulation**: Simulates entry scenarios at 3, 7, and 30 days
5. **Portfolio Construction**: Ranks and selects top 20 positions
6. **Performance Analysis**: Generates comprehensive reports and metrics

**Key Features**:
- **Enhanced Valuation Engine**: Multi-factor scoring with confidence levels
- **Multiple Horizons**: 3, 7, and 30-day entry scenarios
- **Advanced Filtering**: OI, volume, spread, and IV-based selection
- **Portfolio Ranking**: Sophisticated scoring algorithms
- **Synthetic Labels**: For pipeline validation (not actual performance)

**Use Cases**:
- ✅ **System validation** and pipeline testing
- ✅ **Feature exploration** and importance analysis
- ✅ **Opportunity screening** in current market
- ✅ **Parameter tuning** for backtesting strategies
- ✅ **Educational purposes** and research

**Outputs**:
- Detailed CSV with all computed features
- Performance summary with key metrics
- Portfolio composition analysis
- Risk metrics and position rankings

---

### **Mode B: Machine Learning Training and Scoring (ML-Live)**
**Purpose**: Train interpretable ML models on historical data and score current snapshots

**What It Does**:
1. **Training Phase**:
   - Loads reconstructed historical data (from Mode C)
   - Engineers comprehensive features (base, derived, interactions)
   - Implements walk-forward validation for time series
   - Trains LightGBM classifier (POP) and regressor (PnL)
   - Evaluates model performance with multiple metrics
   - Saves models, scalers, and metadata

2. **Scoring Phase**:
   - Loads trained models and preprocessing artifacts
   - Normalizes current snapshot to match training format
   - Engineers features identical to training data
   - Generates predictions for profit probability and PnL
   - Ranks opportunities by combined model scores

**Key Features**:
- **Multiple ML Algorithms**: LightGBM, XGBoost, CatBoost support
- **Feature Engineering**: Base, derived, interaction, and regime features
- **Time-Series Validation**: Walk-forward validation with embargo periods
- **Model Interpretation**: SHAP values, feature importance, partial dependence
- **Ensemble Methods**: Weighted averaging and uncertainty quantification

**Use Cases**:
- ✅ **Live trading decisions** with ML-powered insights
- ✅ **Risk assessment** and position sizing
- ✅ **Portfolio optimization** using predictive models
- ✅ **Market regime detection** and adaptation
- ✅ **Performance prediction** and backtesting

**Outputs**:
- Trained ML models and preprocessing artifacts
- Feature importance and model interpretation
- Scoring reports with ranked opportunities
- Model performance metrics and validation results

---

### **Mode C: True EOD Backtest via Reconstructed Past Chains (EOD-True)**
**Purpose**: Create historically accurate, labeled datasets for true backtesting

**What It Does**:
1. **Data Reconstruction**:
   - Loads official historical derivatives files (bhavcopy)
   - Reconstructs daily option chains with full contract details
   - Attaches underlier EOD data from yfinance
   - Computes IV, Greeks, and derived features
   - Generates realized payoffs and PnL labels

2. **Backtesting**:
   - Applies realistic trading filters and constraints
   - Constructs portfolios using historical data
   - Computes comprehensive performance metrics
   - Analyzes risk-adjusted returns and drawdowns
   - Exports reconstructed data for ML training

**Key Features**:
- **Historical Accuracy**: Uses official derivatives data sources
- **Data Quality**: Comprehensive validation and integrity checks
- **Realistic Constraints**: Transaction costs, slippage, market hours
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios, max drawdown
- **Data Export**: JSON and parquet formats for further analysis

**Use Cases**:
- ✅ **Strategy validation** with historical accuracy
- ✅ **Risk assessment** using realized outcomes
- ✅ **ML training data** generation
- ✅ **Academic research** and backtesting
- ✅ **Performance attribution** and analysis

**Outputs**:
- Reconstructed historical option chains
- True backtest performance metrics
- Risk analysis and drawdown statistics
- Data quality reports and validation results

---

## 🚀 **Quick Start Guide**

### **For First-Time Users: Start with Mode A**
```bash
# 1. Get current NSE options snapshot
python nse_options_scraper.py

# 2. Run synthetic backtest (pipeline validation)
python run_mode_a.py
```

### **For ML Enthusiasts: Mode B (requires Mode C data first)**
```bash
# 1. Generate historical data (Mode C)
python run_mode_c.py

# 2. Train ML models
python run_mode_b.py
```

### **For True Backtesting: Mode C**
```bash
# Run true EOD backtest with historical data
python run_mode_c.py
```

---

## 📊 **Core Module: `utils.py`**

The shared financial utilities used across all scripts and modes:

### **Black-Scholes Functions**
- `black_scholes_price(S, K, T, r, sigma, option_type)`
  - Returns the Black–Scholes price for a European call/put
- `black_scholes_greeks(S, K, T, r, sigma, option_type)`
  - Returns a dict with delta, gamma, theta, vega, rho

### **Enhanced Valuation Functions**
- `option_valuation(theoretical_price, market_price, S=None, K=None, T=None, sigma=None, bid=None, ask=None, option_type='call', tolerance=0.05)`
  - **Multi-factor valuation** that adjusts tolerance using:
    - **Moneyness**: Distance from ATM (exponential decay)
    - **Time Decay**: Shorter time = higher tolerance
    - **Volatility**: Higher vol = higher tolerance
    - **Liquidity**: Wider spreads = higher tolerance
  - Returns `(rating, pct_diff, confidence)`
- `simple_option_valuation(theoretical_price, market_price, tolerance=0.05)`
  - Legacy simple valuation by percent difference

---

## 🔧 **Configuration Examples**

### **Mode A Configuration (`configs/backtest_synth.yml`)**
```yaml
symbol: "NIFTY"
horizons: [3, 7, 30]
filters:
  min_oi: 500
  min_premium: 2.0
  max_spread_pct: 0.08
portfolio:
  daily_max_positions: 20
  rank_score: "pct_diff_times_confidence"
```

### **Mode B Configuration (`configs/ml_experiment.yml`)**
```yaml
target:
  classifier: "POP_label"
  regressor: "PnL"
features:
  base: ["moneyness", "ttm_days", "iv_est_t", "delta", "gamma"]
  interactions: ["moneyness*ttm_days", "delta*theta"]
splits:
  train_start: "2022-01-01"
  train_end: "2024-06-30"
```

### **Mode C Configuration (`configs/backtest_true.yml`)**
```yaml
symbol: "NIFTY"
date_span:
  start: "2022-01-01"
  end: "2024-12-31"
reconstruction:
  data_source: "bhavcopy"
  cache_enabled: true
```

---

## 📈 **Data Schema & Features**

### **Core Contract Columns**
- `date_t`: Trading date (as-of date)
- `expiry_date`: Contract expiry date
- `symbol`: Underlying symbol
- `option_type`: CE/PE
- `strike`: Strike price
- `premium_t`: Entry price

### **Market Data Columns**
- `openInterest`: Open interest
- `totalTradedVolume`: Trading volume
- `bidPrice`, `askPrice`: Bid/ask prices
- `impliedVolatility`: Implied volatility

### **Computed Features**
- `iv_est_t`: Estimated IV from premium
- `delta`, `gamma`, `theta`, `vega`, `rho`: Greeks
- `moneyness`: S_t / strike
- `ttm_days`: Time to maturity
- `enhanced_confidence`: Multi-factor confidence score
- `enhanced_mispricing_pct`: Advanced mispricing percentage
- `enhanced_ranking_score`: Combined ranking score

### **Labels (for ML)**
- `payoff_T`: Option payoff at expiry
- `PnL`: Profit/Loss
- `ROI`: Return on Investment
- `POP_label`: Binary profit indicator

---

## 🎯 **Key Features & Capabilities**

### **1. Comprehensive Feature Engineering**
- **Base Features**: Moneyness, TTM, IV, Greeks, market microstructure
- **Derived Features**: IV skew, volume-OI ratios, regime indicators
- **Interaction Features**: Moneyness × TTM, delta × theta, spread × OI rank
- **Regime Features**: Volatility regime, market regime, momentum regime
- **Time Features**: Day of week, month, quarter, seasonal patterns

### **2. Advanced ML Capabilities**
- **Walk-forward validation** for time series data
- **Feature parity** between training and scoring
- **Multiple target variables** (classification + regression)
- **Model interpretability** and feature importance
- **Comprehensive evaluation metrics**

### **3. Robust Backtesting**
- **Multiple execution modes** for different use cases
- **Configurable filters** and constraints
- **Portfolio construction** algorithms
- **Risk metrics** and performance analysis
- **Cost sensitivity** analysis

### **4. Flexible Configuration**
- **YAML-based** configuration files
- **Parameter validation** and default handling
- **Mode-specific** settings and options
- **Easy customization** without code changes

---

## 📊 **Performance Metrics & KPIs**

### **Common Metrics Across All Modes**
- **Returns**: PnL, ROI, hit rate, cumulative returns
- **Risk**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR
- **Portfolio**: Position counts, composition analysis, turnover
- **Data Quality**: Missing data, validation results, convergence rates

### **Mode-Specific Metrics**
- **Mode A**: Synthetic performance indicators, enhanced ranking scores
- **Mode B**: ML model performance (ROC-AUC, RMSE, rank correlation)
- **Mode C**: Historical backtest performance, realized outcomes

---

## 🔒 **Safeguards & Best Practices**

### **Data Integrity**
- **Schema validation** at multiple pipeline stages
- **Feature parity checks** for ML models
- **Timestamp validation** and chronological ordering
- **Duplicate detection** and handling

### **Model Validation**
- **Walk-forward validation** for time series data
- **Feature availability** checks
- **Preprocessing consistency** across training/scoring
- **Model artifact versioning** and metadata

### **Performance Monitoring**
- **Comprehensive logging** and error handling
- **Progress tracking** for long operations
- **Memory-efficient** batch processing
- **Recovery mechanisms** for failed operations

---

## 🚨 **Important Notes & Limitations**

### **Mode A (Synthetic)**
- ⚠️ **Results are NOT indicative of actual trading performance**
- ⚠️ **All labels are "pending"** (no realized outcomes)
- ⚠️ **Use for pipeline validation and feature exploration only**
- ✅ **Perfect for system testing and parameter tuning**

### **Mode B (ML)**
- ⚠️ **Requires historical data from Mode C**
- ⚠️ **Models need regular retraining** for market adaptation
- ⚠️ **Feature engineering must match** between training/scoring
- ✅ **Provides ML-powered insights for live trading**

### **Mode C (True Backtest)**
- ⚠️ **Requires official historical derivatives data**
- ⚠️ **Data reconstruction can be time-consuming**
- ⚠️ **Limited by data availability** and quality
- ✅ **Historically accurate backtesting for research**

---

## 🎉 **Key Takeaways & Use Cases**

### **For Researchers & Academics**
- **Mode A**: Quick system validation and feature exploration
- **Mode C**: Historically accurate backtesting for papers
- **All Modes**: Comprehensive data analysis and visualization

### **For Traders & Analysts**
- **Mode A**: Daily opportunity screening and market analysis
- **Mode B**: ML-powered trading decisions and risk assessment
- **Mode C**: Strategy validation and performance attribution

### **For Developers & Engineers**
- **Mode A**: Pipeline testing and system validation
- **Mode B**: ML model development and deployment
- **Mode C**: Data quality validation and integrity checks

### **For Educational Purposes**
- **Mode A**: Understanding options valuation and Greeks
- **Mode B**: Learning ML applications in finance
- **Mode C**: Studying historical market behavior

---

## 🔧 **Troubleshooting & Support**

### **Common Issues**
1. **Import Errors**: Ensure all dependencies are installed
2. **Data Quality**: Check for missing or invalid data in snapshots
3. **IV Convergence**: Some options may fail IV inversion
4. **Memory Issues**: Large datasets may require batch processing

### **Debug Mode**
Enable detailed logging in configuration files:
```yaml
output:
  detailed_logging: true
  save_intermediate: true
```

### **Validation**
Use built-in validation functions:
```python
from data_pipeline.schemas import validate_dataframe_schema

# Validate data quality
validation_result = validate_dataframe_schema(df, strict=False)
print(validation_result)
```

---

## 📚 **Scripts and How to Run Them**

### **1) Manual US Option Info (Greeks Only)**
```bash
python option_info_manual.py
```
Interactive; fetches single option from yfinance and prints Greeks.

### **2) Manual US Option Info with Valuation**
```bash
python option_info_manual_Valuation_processor.py
```
Interactive; prints Greeks, theoretical price, valuation rating, and confidence.

### **3) NSE Scraper (Single Date, ALL Options)**
```bash
python nse_options_scraper.py
```
Fetches NSE option-chain JSON for chosen index and processes ALL options.

### **4) Complete NSE Options Analysis Pipeline**
```bash
python nse_options_valuation_processor.py
```
The flagship feature - automatically scrapes NSE data, runs comprehensive valuation analysis.

### **5) Three-Mode System Execution**
```bash
# Mode A: Synthetic Backtest
python run_mode_a.py

# Mode B: ML Training and Scoring
python run_mode_b.py

# Mode C: True EOD Backtest
python run_mode_c.py
```

---

## 🗺️ **Roadmap & Future Enhancements**

### **Short Term**
- Add more symbols and indices support
- Implement portfolio-level analysis
- Add more sophisticated volatility surface modeling
- Enhance confidence scoring with additional market factors

### **Medium Term**
- Web interface for data visualization
- Real-time data streaming capabilities
- Advanced risk management features
- Multi-asset portfolio optimization

### **Long Term**
- Cloud deployment and scaling
- API endpoints for external integration
- Advanced ML models (deep learning, reinforcement learning)
- Institutional-grade risk analytics

---

## 📄 **License**
MIT License - see LICENSE file for details.

---

## 🆘 **Support & Contributing**

### **For Issues and Questions**
1. Check the troubleshooting section above
2. Review configuration examples
3. Validate data quality and schema
4. Check system requirements and dependencies

### **Development Workflow**
1. Create feature branch from main
2. Implement changes with tests
3. Update documentation
4. Submit pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Include type hints for all functions
- Add comprehensive docstrings
- Write unit tests for new functionality

---

## 🎯 **Final Notes**

This system is designed for **educational and research purposes**. Always conduct thorough testing before using in live trading environments. The synthetic backtest results are not indicative of actual trading performance.

**Start with Mode A** to validate your system, then explore Mode C for historical analysis, and finally use Mode B for ML-powered insights. Each mode serves a specific purpose and builds upon the previous ones.

**Happy Options Trading and Analysis! 🚀📈**
