# NSE Options Valuation and Backtest

A Python toolkit for analyzing NSE (National Stock Exchange of India) options data with Black-Scholes pricing, Greeks calculation, and basic backtesting capabilities. This project scrapes live options data from NSE, computes theoretical valuations using the Black-Scholes model, and provides three different analysis modes for different use cases.

The platform includes data collection scripts for NSE options chains, valuation processors that calculate implied volatility and Greeks, and backtesting modules for historical analysis. It also contains a theoretical framework for training machine learning models on historical options data to improve backtesting accuracy, though this component wasn't fully implemented due to GPU computational requirements. The codebase is organized into data pipeline modules, backtesting scripts, and configuration files that allow customization of analysis parameters.

To get started, install the dependencies with `pip install -r requirements.txt`, run `python nse_options_scraper.py` to fetch current market data, and then use `python run_mode_a.py` for quick analysis or `python run_mode_c.py` for historical backtesting. 
