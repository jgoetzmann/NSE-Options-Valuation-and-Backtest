# NSE Options Valuation and Backtest Platform
## Professional-Grade Options Analysis with Three-Execution-Mode System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive, enterprise-level toolkit for options analysis that combines **Black-Scholes pricing/Greeks**, **NSE (India) option-chain scraping**, **advanced backtesting**, and **machine learning capabilities**. This platform provides professional-grade options analysis with real-time data processing, sophisticated valuation algorithms, and three distinct execution modes for different use cases.

## 🚀 **Quick Start (5 minutes)**

### **1. Installation**
```bash
git clone https://github.com/yourusername/NSE-Options-Valuation-and-Backtest.git
cd NSE-Options-Valuation-and-Backtest
pip install -r requirements.txt
```

### **2. Get Current Market Data**
```bash
python nse_options_scraper.py
```

### **3. Run Analysis (Choose Your Mode)**
```bash
# Mode A: Quick analysis of current market (Recommended for beginners)
python run_mode_a.py

# Mode B: ML-powered analysis (Requires historical data)
python run_mode_b.py

# Mode C: Historical backtesting (Requires historical data)
python run_mode_c.py
```

---

## 🎯 **What This Platform Does**

### **Core Capabilities**
- **📊 Real-time NSE Data**: Live option-chain scraping from National Stock Exchange of India
- **🧮 Professional Valuation**: Black-Scholes pricing with advanced Greeks calculation
- **🤖 Machine Learning**: ML-powered opportunity scoring and risk assessment
- **📈 Advanced Backtesting**: Three execution modes for different analysis needs
- **🔍 Feature Engineering**: 50+ sophisticated features for options analysis
- **📋 Portfolio Construction**: Intelligent position selection and sizing algorithms

### **Three-Execution-Mode System**
| Mode | Purpose | Best For | Time Required |
|------|---------|----------|---------------|
| **Mode A** | Pipeline validation & opportunity screening | Daily analysis, beginners | 2-5 minutes |
| **Mode B** | ML training & live scoring | Traders, researchers | 15-30 minutes |
| **Mode C** | Historical backtesting | Strategy validation, research | 30-60 minutes |

---

## 🏗️ **Project Architecture**

```
NSE-Options-Valuation-and-Backtest/
├── 📁 data_pipeline/           # Core data processing modules
│   ├── schemas.py              # Data validation and schemas
│   ├── normalize_snapshot.py   # Live data normalization
│   ├── compute_iv_and_greeks.py # IV calculation & Greeks
│   ├── compute_enhanced_features.py # Advanced feature engineering
│   └── make_labels.py          # ML label generation
├── 📁 backtests/               # Backtesting execution modules
│   ├── run_synthetic_on_snapshot.py  # Mode A runner
│   ├── run_true_backtest.py          # Mode C runner
│   └── reports.py                    # Performance analysis
├── 📁 models/                   # Machine learning modules
│   ├── train.py                # Model training (Mode B)
│   ├── score_snapshot.py       # Live scoring (Mode B)
│   └── model_store/            # Trained models
├── 📁 configs/                  # Configuration files
│   ├── backtest_synth.yml      # Mode A settings
│   ├── backtest_true.yml       # Mode C settings
│   └── ml_experiment.yml       # Mode B settings
├── 📁 outputs/                  # Results and reports
├── 📁 reconstructed/            # Historical data storage
└── 🚀 run_mode_a.py, run_mode_b.py, run_mode_c.py  # Execution scripts
```

---

## 🎯 **Execution Modes Explained**

### **Mode A: Synthetic Backtest (SB-CS) ⚡ FAST**
**Purpose**: Daily opportunity screening and pipeline validation

**What You Get**:
- ✅ **Live market analysis** in 2-5 minutes
- ✅ **Top 20 opportunities** ranked by sophisticated algorithms
- ✅ **Enhanced valuation scores** with confidence levels
- ✅ **Multiple time horizons** (3, 7, 30 days)
- ✅ **Professional reports** and visualizations

**Perfect For**:
- Daily market analysis
- Opportunity screening
- System validation
- Educational purposes

**Example Output**:
```
🎯 TOP 5 OPPORTUNITIES (Mode A)
1. NIFTY 19500 CE (7 days) - Score: 8.7/10, Confidence: 85%
2. NIFTY 19400 PE (3 days) - Score: 8.2/10, Confidence: 78%
3. NIFTY 19600 CE (30 days) - Score: 7.9/10, Confidence: 72%
```

---

### **Mode B: Machine Learning (ML-Live) 🤖 SMART**
**Purpose**: ML-powered trading decisions and risk assessment

**What You Get**:
- ✅ **Trained ML models** on historical data
- ✅ **Probability of profit** predictions
- ✅ **Expected PnL** estimates
- ✅ **Feature importance** analysis
- ✅ **Confidence scoring** for each prediction

**Perfect For**:
- Live trading decisions
- Risk assessment
- Portfolio optimization
- Advanced analysis

**Example Output**:
```
🤖 ML PREDICTIONS (Mode B)
NIFTY 19500 CE: POP: 73%, Expected PnL: ₹45, Confidence: 82%
NIFTY 19400 PE: POP: 68%, Expected PnL: ₹38, Confidence: 79%
```

---

### **Mode C: True Backtest (EOD-True) 📊 ACCURATE**
**Purpose**: Historically accurate strategy validation

**What You Get**:
- ✅ **Real historical performance** data
- ✅ **True PnL and returns** calculations
- ✅ **Risk metrics** (Sharpe, Sortino, max drawdown)
- ✅ **Portfolio analysis** and attribution
- ✅ **Data for ML training**

**Perfect For**:
- Strategy validation
- Academic research
- Risk assessment
- Performance attribution

**Example Output**:
```
📊 BACKTEST RESULTS (Mode C)
Total Return: 23.4% | Sharpe Ratio: 1.67
Max Drawdown: -8.2% | Win Rate: 64.3%
```

---

## 📦 **Installation & Setup**

### **System Requirements**
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

### **Installation Steps**
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/NSE-Options-Valuation-and-Backtest.git
cd NSE-Options-Valuation-and-Backtest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import pandas, numpy, scipy, yfinance; print('✅ All dependencies installed!')"
```

### **First Run Setup**
```bash
# 1. Get current market data
python nse_options_scraper.py

# 2. Run your first analysis (Mode A recommended)
python run_mode_a.py

# 3. Check results in outputs/ folder
```

---

## 🔧 **Configuration & Customization**

### **Mode A Configuration** (`configs/backtest_synth.yml`)
```yaml
symbol: "NIFTY"                    # Index to analyze
horizons: [3, 7, 30]             # Analysis timeframes
filters:
  min_oi: 500                     # Minimum open interest
  min_premium: 2.0                # Minimum premium (₹)
  max_spread_pct: 0.08           # Maximum bid-ask spread
portfolio:
  daily_max_positions: 20         # Maximum positions per day
  rank_score: "pct_diff_times_confidence"  # Ranking formula
```

### **Mode B Configuration** (`configs/ml_experiment.yml`)
```yaml
target:
  classifier: "POP_label"         # Profit probability target
  regressor: "PnL"               # PnL prediction target
features:
  base: ["moneyness", "ttm_days", "iv_est_t", "delta"]
  interactions: ["moneyness*ttm_days", "delta*theta"]
splits:
  train_start: "2022-01-01"      # Training period start
  train_end: "2024-06-30"        # Training period end
```

### **Mode C Configuration** (`configs/backtest_true.yml`)
```yaml
symbol: "NIFTY"
date_span:
  start: "2022-01-01"            # Backtest start date
  end: "2024-12-31"              # Backtest end date
reconstruction:
  data_source: "bhavcopy"        # Historical data source
  cache_enabled: true            # Enable data caching
```

---

## 📊 **Data Schema & Features**

### **Core Contract Data**
| Column | Description | Example |
|--------|-------------|---------|
| `symbol` | Underlying symbol | "NIFTY" |
| `strike` | Strike price | 19500 |
| `option_type` | Call/Put | "CE" |
| `expiry_date` | Expiry date | "2025-08-28" |
| `premium_t` | Current premium | 45.50 |

### **Computed Features**
| Feature | Description | Range |
|---------|-------------|-------|
| `moneyness` | S/K ratio | 0.5 - 2.0 |
| `ttm_days` | Days to expiry | 1 - 365 |
| `iv_est_t` | Implied volatility | 0.1 - 2.0 |
| `delta`, `gamma`, `theta`, `vega` | Greeks | Various |
| `enhanced_confidence` | Multi-factor confidence | 0.0 - 1.0 |

### **ML Labels**
| Label | Description | Type |
|-------|-------------|------|
| `POP_label` | Profit probability | Binary (0/1) |
| `PnL` | Profit/Loss | Continuous |
| `ROI` | Return on Investment | Percentage |

---

## 🚀 **Usage Examples**

### **Daily Market Analysis (Mode A)**
```bash
# 1. Get fresh market data
python nse_options_scraper.py

# 2. Run analysis
python run_mode_a.py

# 3. Check results
ls outputs/csv/synthetic_backtest_*.csv
ls outputs/results_from_nse_valuations/synthetic_summary_*.txt
```

### **ML Model Training (Mode B)**
```bash
# 1. Ensure you have historical data
ls reconstructed/parquet/

# 2. Train models
python run_mode_b.py

# 3. Check trained models
ls models/model_store/
```

### **Historical Backtesting (Mode C)**
```bash
# 1. Run backtest
python run_mode_c.py

# 2. Check results
ls outputs/csv/true_backtest_*.csv
ls outputs/true_backtest_summary_*.txt
```

---

## 📚 **All Available Scripts and How to Run Them**

### **🚀 Three-Mode System (Main Platform)**
```bash
# Mode A: Synthetic Backtest (Recommended for beginners)
python run_mode_a.py

# Mode B: ML Training and Scoring (Requires historical data)
python run_mode_b.py

# Mode C: True EOD Backtest (Historical validation)
python run_mode_c.py
```

### **📊 NSE Data Collection and Analysis**
```bash
# NSE Options Scraper - Get live market data
python nse_options_scraper.py

# Complete NSE Analysis Pipeline - End-to-end processing
python nse_options_valuation_processor.py
```

### **🇺🇸 US Options Analysis Tools**
```bash
# Basic US Options Analysis (Greeks only)
python option_info_manual.py

# Advanced US Options Analysis (with valuation)
python option_info_manual_valuation_processor.py
```

---

## 🎯 **Script Details and Use Cases**

### **Three-Mode System Scripts**
| Script | Purpose | Best For | Time Required |
|--------|---------|----------|---------------|
| `run_mode_a.py` | Daily opportunity screening | Beginners, daily analysis | 2-5 minutes |
| `run_mode_b.py` | ML model training & scoring | Traders, researchers | 15-30 minutes |
| `run_mode_c.py` | Historical backtesting | Strategy validation | 30-60 minutes |

### **NSE Data Processing Scripts**
| Script | Purpose | Output | Use Case |
|--------|---------|--------|----------|
| `nse_options_scraper.py` | Fetch live NSE data | Raw & processed JSON | Data collection |
| `nse_options_valuation_processor.py` | Complete analysis pipeline | CSV + reports | End-to-end analysis |

### **US Options Analysis Scripts**
| Script | Purpose | Features | Best For |
|--------|---------|----------|----------|
| `option_info_manual.py` | Basic Greeks calculation | Delta, Gamma, Theta, Vega, Rho | Quick analysis |
| `option_info_manual_valuation_processor.py` | Advanced valuation | Greeks + pricing + confidence | Detailed analysis |

---

## 📈 **Performance Metrics & KPIs**

### **Common Metrics Across All Modes**
- **Returns**: PnL, ROI, hit rate, cumulative returns
- **Risk**: Sharpe ratio, Sortino ratio, maximum drawdown, VaR
- **Portfolio**: Position counts, composition analysis, turnover
- **Data Quality**: Missing data, validation results, convergence rates

### **Mode-Specific Metrics**
| Mode | Key Metrics | Focus |
|------|-------------|-------|
| **A** | Enhanced ranking scores, synthetic performance | Opportunity screening |
| **B** | ROC-AUC, RMSE, rank correlation | ML model performance |
| **C** | Historical returns, realized outcomes | Strategy validation |

---

## 🔒 **Safeguards & Best Practices**

### **Data Integrity**
- ✅ **Schema validation** at multiple pipeline stages
- ✅ **Feature parity checks** for ML models
- ✅ **Timestamp validation** and chronological ordering
- ✅ **Duplicate detection** and handling

### **Model Validation**
- ✅ **Walk-forward validation** for time series data
- ✅ **Feature availability** checks
- ✅ **Preprocessing consistency** across training/scoring
- ✅ **Model artifact versioning** and metadata

### **Performance Monitoring**
- ✅ **Comprehensive logging** and error handling
- ✅ **Progress tracking** for long operations
- ✅ **Memory-efficient** batch processing
- ✅ **Recovery mechanisms** for failed operations

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

## 🎯 **Use Cases & Target Users**

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

### **Common Issues & Solutions**

#### **1. Import Errors**
```bash
# Solution: Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, scipy; print('✅ OK')"
```

#### **2. No Snapshot Data**
```bash
# Solution: Run scraper first
python nse_options_scraper.py

# Check outputs/json/ folder
ls outputs/json/
```

#### **3. ML Training Fails**
```bash
# Solution: Check historical data
ls reconstructed/parquet/

# If empty, run Mode C first or get sample data
```

#### **4. Memory Issues**
```yaml
# In config files, reduce batch sizes
processing:
  batch_size: 1000  # Reduce from default
  max_memory_gb: 4  # Limit memory usage
```

### **Debug Mode**
Enable detailed logging in configuration files:
```yaml
output:
  detailed_logging: true
  save_intermediate: true
```

### **Validation Commands**
```python
# Validate data quality
from data_pipeline.schemas import validate_dataframe_schema
validation_result = validate_dataframe_schema(df, strict=False)
print(validation_result)
```

---

## 📚 **Advanced Usage**

### **Custom Feature Engineering**
```python
# Add custom features in compute_enhanced_features.py
def add_custom_features(df):
    df['custom_ratio'] = df['volume'] / df['openInterest']
    df['volatility_regime'] = np.where(df['iv_est_t'] > 0.3, 'high', 'low')
    return df
```

### **Custom ML Models**
```python
# Extend models/train.py for custom algorithms
from sklearn.ensemble import RandomForestClassifier

class CustomMLTrainer(MLModelTrainer):
    def train_classifier(self, X, y):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
        return model
```

### **Custom Backtesting Strategies**
```python
# Extend backtests/run_true_backtest.py
class CustomBacktestRunner(TrueBacktestRunner):
    def custom_portfolio_construction(self, opportunities):
        # Implement custom logic
        return selected_positions
```

---

## 🔧 **Individual Script Deep Dive**

### **1. NSE Options Scraper (`nse_options_scraper.py`)**
**Purpose**: Fetch live options data from National Stock Exchange of India

**Features**:
- Real-time data collection from NSE API
- Robust error handling and retry mechanisms
- Data normalization and cleaning
- Outputs both raw and processed JSON formats

**Usage**:
```bash
# Basic usage (defaults to NIFTY index)
python nse_options_scraper.py

# Custom symbol and type
python nse_options_scraper.py --symbol BANKNIFTY --kind indices
```

**Outputs**:
- `raw_option_chain_<symbol>_<timestamp>.json` - Raw API response
- `option_chain_slim_<symbol>_<timestamp>.json` - Processed data

---

### **2. Complete NSE Analysis Pipeline (`nse_options_valuation_processor.py`)**
**Purpose**: End-to-end NSE options analysis with comprehensive valuation

**Features**:
- Automatic data fetching and processing
- Black-Scholes pricing and Greeks calculation
- Enhanced valuation with confidence scoring
- Portfolio construction and ranking
- Professional reporting and CSV exports

**Usage**:
```bash
# Run complete analysis
python nse_options_valuation_processor.py

# Custom symbol
python nse_options_valuation_processor.py --symbol FINNIFTY
```

**Outputs**:
- Detailed CSV with all computed features
- Performance summary reports
- Portfolio composition analysis
- Risk metrics and rankings

---

### **3. Basic US Options Analysis (`option_info_manual.py`)**
**Purpose**: Quick Greeks calculation for individual US options

**Features**:
- Interactive input interface
- Real-time data from Yahoo Finance
- Basic Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Simple, focused output

**Usage**:
```bash
python option_info_manual.py
# Follow interactive prompts for:
# - Ticker symbol (e.g., AAPL)
# - Expiration date (YYYY-MM-DD)
# - Strike price
# - Option type (call/put)
```

**Best For**: Quick analysis, learning options, basic risk assessment

---

### **4. Advanced US Options Analysis (`option_info_manual_valuation_processor.py`)**
**Purpose**: Comprehensive US options analysis with advanced valuation

**Features**:
- All features from basic version
- Theoretical vs. market price comparison
- Advanced valuation algorithm with confidence scoring
- Market condition analysis
- Risk assessment and warnings

**Usage**:
```bash
python option_info_manual_valuation_processor.py
# Same interactive interface as basic version
# Plus comprehensive valuation analysis
```

**Best For**: Detailed analysis, professional valuation, risk assessment

---

### **5. Three-Mode Execution Scripts**
**Purpose**: Main platform execution with different analysis modes

**Mode A (`run_mode_a.py`)**:
- Synthetic backtesting on current snapshots
- Pipeline validation and opportunity screening
- Fast execution (2-5 minutes)
- Perfect for daily analysis

**Mode B (`run_mode_b.py`)**:
- ML model training on historical data
- Live scoring of current opportunities
- Requires historical data from Mode C
- Best for ML-powered trading decisions

**Mode C (`run_mode_c.py`)**:
- Historical backtesting with reconstructed data
- True performance validation
- Data generation for ML training
- Best for strategy validation and research

---

## 🗺️ **Roadmap & Future Enhancements**

### **Short Term (1-3 months)**
- ✅ **Multi-symbol support** (BANKNIFTY, FINNIFTY, equity options)
- ✅ **Enhanced volatility surface** modeling
- ✅ **Portfolio-level analysis** and optimization
- ✅ **Real-time data streaming** capabilities

### **Medium Term (3-6 months)**
- 🔄 **Web interface** for data visualization
- 🔄 **Advanced risk management** features
- 🔄 **Multi-asset portfolio** optimization
- 🔄 **API endpoints** for external integration

### **Long Term (6+ months)**
- 📋 **Cloud deployment** and scaling
- 📋 **Deep learning models** (LSTM, Transformers)
- 📋 **Reinforcement learning** for strategy optimization
- 📋 **Institutional-grade risk analytics**

---

## 📄 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 **Key Takeaways**

This platform is designed for **educational and research purposes**. Always conduct thorough testing before using in live trading environments.

**Start with Mode A** to validate your system, then explore Mode C for historical analysis, and finally use Mode B for ML-powered insights. Each mode serves a specific purpose and builds upon the previous ones.

**Happy Options Trading and Analysis! 🚀📈**

