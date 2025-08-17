# 🚀 Getting Started Guide

## Quick Start (5 minutes)

This guide will get you up and running with the NSE Options Valuation and Backtest Platform in just 5 minutes.

### **Step 1: Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/NSE-Options-Valuation-and-Backtest.git

# Navigate to the project directory
cd NSE-Options-Valuation-and-Backtest

# Install dependencies
pip install -r requirements.txt
```

### **Step 2: Get Market Data**
```bash
# Fetch current NSE options data
python nse_options_scraper.py
```

### **Step 3: Run Your First Analysis**
```bash
# Start with Mode A (recommended for beginners)
python run_mode_a.py
```

### **Step 4: Check Results**
```bash
# View your results
ls outputs/csv/
ls outputs/results_from_nse_valuations/
```

---

## 🎯 What Just Happened?

1. **Data Collection**: The scraper fetched live options data from NSE
2. **Analysis**: Mode A analyzed the data and found opportunities
3. **Results**: Generated CSV files with detailed analysis and summary reports

---

## 📊 Understanding Your First Results

### **CSV Files** (Detailed Analysis)
- **3-day horizon**: Short-term opportunities
- **7-day horizon**: Medium-term opportunities  
- **30-day horizon**: Long-term opportunities

### **Summary Files** (Key Insights)
- Top opportunities ranked by score
- Performance metrics and statistics
- Portfolio composition analysis

---

## 🔧 Customizing Your Analysis

### **Modify Configuration**
```bash
# Copy the example config
cp configs/example_config.yml configs/my_config.yml

# Edit the file with your preferences
# Use any text editor or IDE
```

### **Key Settings to Adjust**
```yaml
# In configs/my_config.yml
symbol: "NIFTY"                    # Change to BANKNIFTY or FINNIFTY
horizons: [5, 10, 45]             # Customize time horizons
filters:
  min_oi: 1000                     # Increase minimum open interest
  max_spread_pct: 0.05             # Tighter spread requirement
```

---

## 🚀 Next Steps

### **For Beginners**
1. ✅ **Run Mode A** (you just did this!)
2. 🔄 **Experiment with different settings** in the config file
3. 📊 **Analyze results** and understand the metrics
4. 🎯 **Try different symbols** (BANKNIFTY, FINNIFTY)

### **For Advanced Users**
1. 📚 **Read the full README.md** for comprehensive documentation
2. 🤖 **Try Mode B** for ML-powered analysis (requires historical data)
3. 📈 **Use Mode C** for historical backtesting
4. 🔧 **Customize the code** for your specific needs

---

## 🛠️ **Additional Tools Available**

### **NSE Data Processing**
- **`nse_options_scraper.py`** - Get live NSE options data
- **`nse_options_valuation_processor.py`** - Complete end-to-end analysis

### **US Options Analysis**
- **`option_info_manual.py`** - Quick Greeks calculation for US options
- **`option_info_manual_valuation_processor.py`** - Advanced US options valuation

### **When to Use Each Tool**
- **Daily analysis**: Use Mode A (`run_mode_a.py`)
- **Get fresh data**: Use `nse_options_scraper.py`
- **Quick US options check**: Use `option_info_manual.py`
- **Professional US analysis**: Use `option_info_manual_valuation_processor.py`
- **Complete NSE pipeline**: Use `nse_options_valuation_processor.py`

---

## 🆘 Need Help?

### **Common Issues**
- **"No module found"**: Run `pip install -r requirements.txt`
- **"No snapshot data"**: Run `python nse_options_scraper.py` first
- **"Empty results"**: Check your configuration filters

---

## 📚 Learning Resources

### **Understanding Options**
- **Greeks**: Delta, Gamma, Theta, Vega, Rho
- **Moneyness**: In-the-money, At-the-money, Out-of-the-money
- **Implied Volatility**: Market's expectation of future volatility

### **Platform Features**
- **Feature Engineering**: 50+ sophisticated options features
- **Portfolio Construction**: Intelligent position selection
- **Risk Management**: Comprehensive risk metrics
- **Machine Learning**: ML-powered opportunity scoring

---

## 🎉 Congratulations!

You've successfully:
- ✅ Installed the platform
- ✅ Fetched live market data
- ✅ Run your first analysis
- ✅ Generated professional reports

**You're now ready to explore options trading opportunities with professional-grade tools!**

---

## 🔄 Daily Workflow

### **Morning Routine**
```bash
# 1. Get fresh market data
python nse_options_scraper.py

# 2. Run analysis
python run_mode_a.py

# 3. Review opportunities
# Check outputs/results_from_nse_valuations/
```

### **Weekly Review**
```bash
# 1. Analyze performance trends
# Review CSV files for patterns

# 2. Adjust parameters
# Modify configs/backtest_synth.yml

# 3. Validate changes
# Run Mode A again with new settings
```

---

*Happy Options Trading! 🚀📈*

For more information, see the main [README.md](README.md) file.
