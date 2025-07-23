import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    S = stock.history(period='1d')['Close'].iloc[-1]

    hist = stock.history(period="90d")
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    sigma = returns.std() * np.sqrt(252)  # Annualized

    return S, sigma

# ==== CONFIGURE HERE ====
ticker = "AAPL"
strike_price = 200
expiry_date = "2025-12-20"
r = 0.045  # Risk-free rate
# =========================

S, sigma = get_stock_data(ticker)
T = (datetime.strptime(expiry_date, "%Y-%m-%d") - datetime.today()).days / 365

call_price = black_scholes(S, strike_price, T, r, sigma, 'call')
put_price = black_scholes(S, strike_price, T, r, sigma, 'put')

print(f"Ticker: {ticker}")
print(f"Current Price: ${S:.2f}")
print(f"Volatility (annualized): {sigma:.2%}")
print(f"Time to Expiry: {T:.2f} years")
print(f"Call Option Price: ${call_price:.2f}")
print(f"Put Option Price: ${put_price:.2f}")
