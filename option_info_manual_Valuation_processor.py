#!/usr/bin/env python3
"""
Interactive US Options Analysis with Advanced Valuation
====================================================

This script provides an interactive interface for analyzing individual US options using yfinance data.
It goes beyond basic Greeks calculation to provide comprehensive valuation analysis including:
- Real-time option data fetching from Yahoo Finance
- Black-Scholes theoretical pricing
- Complete Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Advanced valuation rating system with confidence scoring
- Market condition analysis and risk assessment

Key Features:
- Interactive input for symbol, expiration, strike, and option type
- Real-time market data from yfinance API
- Theoretical vs. market price comparison
- Sophisticated valuation algorithm considering:
  * Moneyness (distance from ATM)
  * Time decay effects
  * Volatility environment
  * Bid-ask spread liquidity
  * Market confidence factors
- Risk assessment and market condition warnings
- Professional-grade output formatting

Usage:
    python option_info_manual_with_valuation.py
    # Enter: ticker symbol (e.g., AAPL)
    # Enter: expiration date (YYYY-MM-DD)
    # Enter: strike price
    # Enter: option type (call/put)

Output:
- Comprehensive option analysis with all Greeks
- Theoretical vs. market price comparison
- Valuation rating (undervalued/overvalued/fair)
- Confidence level assessment
- Market insights and risk warnings

Dependencies:
- yfinance: For real-time US options data
- utils: For Black-Scholes calculations and valuation
- datetime: For date handling and time calculations

This is the enhanced version of option_info_manual.py, adding sophisticated
valuation capabilities beyond basic Greeks calculation.

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import yfinance as yf
from utils import black_scholes_greeks, black_scholes_price, option_valuation
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
S = ticker.history(period='1d')['Close'].iloc[-1]  # Fixed deprecation warning
K = strike
T = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days / 365.0
r = 0.05  # Assume 5% risk-free rate
sigma = row['impliedVolatility'] if 'impliedVolatility' in row and not (row['impliedVolatility'] is None or row['impliedVolatility'] == 0) else 0.2
market_price = row['lastPrice']

# Get bid/ask for liquidity factor
bid = row.get('bid', None)
ask = row.get('ask', None)

# Calculate Greeks
greeks = black_scholes_greeks(S, K, T, r, sigma, option_type)
# Calculate theoretical price
theoretical_price = black_scholes_price(S, K, T, r, sigma, option_type)
# Enhanced valuation with all factors
eval_rating, pct_diff, confidence = option_valuation(
    theoretical_price, market_price, S, K, T, sigma, bid, ask, option_type
)

# Print comprehensive info
print(f'\n{"="*60}')
print(f'OPTION ANALYSIS: {symbol} {option_type.upper()} {K} expiring {expiration}')
print(f'{"="*60}')
print(f'Spot Price:     ${S:.2f}')
print(f'Strike Price:   ${K:.2f}')
print(f'Market Price:   ${market_price:.4f}')
print(f'Time to Expiry: {T:.3f} years ({T*365:.1f} days)')
print(f'Risk-free Rate: {r:.1%}')
print(f'Implied Vol:    {sigma:.2%}')
if bid and ask:
    print(f'Bid/Ask:       ${bid:.4f}/${ask:.4f}')
    spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
    print(f'Spread:        {spread_pct:.1f}%')

print(f'\n{"-"*40}')
print('GREEKS:')
print(f'{"-"*40}')
for k, v in greeks.items():
    print(f'{k.capitalize():<8}: {v:>10.4f}')

print(f'\n{"-"*40}')
print('VALUATION ANALYSIS:')
print(f'{"-"*40}')
print(f'Theoretical Price: ${theoretical_price:.4f}')
print(f'Market Price:      ${market_price:.4f}')
print(f'Difference:        ${theoretical_price - market_price:.4f}')
print(f'Percentage Diff:   {pct_diff*100:+.2f}%')
print(f'Valuation Rating:  {eval_rating.upper()}')
print(f'Confidence Level:  {confidence:.1%}')

# Additional insights
print(f'\n{"-"*40}')
print('INSIGHTS:')
print(f'{"-"*40}')
if abs(pct_diff) > 0.5:
    print(f"⚠️  Large price discrepancy detected ({pct_diff*100:.1f}%)")
if confidence < 0.3:
    print(f"⚠️  Low confidence rating - consider market conditions")
if T < 0.01:
    print(f"⚠️  Very short time to expiry - high time decay risk")
if sigma > 0.5:
    print(f"⚠️  High volatility - increased uncertainty")
if bid and ask and (ask - bid) / ((ask + bid) / 2) > 0.1:
    print(f"⚠️  Wide bid-ask spread - potential liquidity issues")

print(f'\n{"="*60}')