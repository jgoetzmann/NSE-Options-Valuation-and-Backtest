#!/usr/bin/env python3
"""
Basic US Options Analysis - Greeks Calculator
============================================

This script provides a simple interactive interface for calculating basic Greeks for individual US options.
It's designed as a lightweight tool for quick option analysis using yfinance data.

Key Features:
- Interactive input for symbol, expiration, strike, and option type
- Real-time market data from yfinance API
- Basic Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Simple, focused output for quick analysis

Usage:
    python option_info_manual.py
    # Enter: ticker symbol (e.g., AAPL)
    # Enter: expiration date (YYYY-MM-DD)
    # Enter: strike price
    # Enter: option type (call/put)

Output:
- Basic option information (last price, implied volatility)
- All Greeks values with 4 decimal precision

Dependencies:
- yfinance: For real-time US options data
- utils: For Black-Scholes Greeks calculations
- datetime: For date handling and time calculations

Note: This is the basic version. For advanced valuation analysis including theoretical pricing,
confidence scoring, and market condition assessment, use option_info_manual_with_valuation.py instead.

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import yfinance as yf
from utils import black_scholes_greeks
from datetime import datetime

# User input
symbol = input('Enter ticker symbol (e.g., AAPL): ').strip().upper()
expiration = input('Enter expiration date (YYYY-MM-DD): ').strip()
strike = float(input('Enter strike price: ').strip())
option_type = input('Enter option type (call/put): ').strip().lower()

# Fetch option chain
ticker = yf.Ticker(symbol)
try:
    opt_chain = ticker.option_chain(expiration)
    options = opt_chain.calls if option_type == 'call' else opt_chain.puts
except Exception as e:
    print(f'Error fetching option chain: {e}')
    exit(1)

# Find the specific contract
row = options[options['strike'] == strike]
if row.empty:
    print('No option found for that strike.')
    exit(1)
row = row.iloc[0]

# Extract required data
S = ticker.history(period='1d')['Close'][-1]
K = strike
T = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days / 365.0
r = 0.05  # Assume 5% risk-free rate
sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not (row['impliedVolatility'] is None or row['impliedVolatility'] == 0) else 0.2

# Calculate Greeks
greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)

# Print info
print(f'\nOption info for {symbol} {option_type.upper()} {K} expiring {expiration}')
print(f'Last price: {row["lastPrice"]}')
print(f'Implied Volatility: {sigma:.2%}')
print('Greeks:')
for k, v in greeks.items():
    print(f'  {k.capitalize()}: {v:.4f}')