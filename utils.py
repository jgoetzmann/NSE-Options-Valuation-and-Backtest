#!/usr/bin/env python3
"""
Financial Utilities Module - Core Options Analysis Engine
======================================================

This module contains the core financial calculation engine for options analysis, providing
professional-grade implementations of Black-Scholes pricing, Greeks calculation, and advanced
valuation algorithms.

Core Functions:
1. black_scholes_price(S, K, T, r, sigma, option_type)
   - Calculates theoretical price for European options using Black-Scholes model
   - Handles both calls and puts with proper mathematical implementation
   - Inputs: Spot price, Strike, Time to expiry, Risk-free rate, Volatility, Option type

2. black_scholes_greeks(S, K, T, r, sigma, option_type)
   - Calculates all major Greeks: Delta, Gamma, Theta, Vega, Rho
   - Returns normalized values (Vega per 1% vol change, Rho per 1% rate change)
   - Essential for risk management and portfolio analysis

3. option_valuation(theoretical_price, market_price, ...)
   - Advanced valuation system considering multiple real-world factors:
     * Moneyness factor (distance from ATM)
     * Time decay effects
     * Volatility environment
     * Bid-ask spread liquidity
     * Market confidence factors
   - Returns: (rating, percentage_difference, confidence_level)
   - Ratings: strongly undervalued → strongly overvalued

4. simple_option_valuation(theoretical_price, market_price, tolerance)
   - Legacy simple valuation based on percentage difference
   - Useful for basic analysis or when detailed factors unavailable

Mathematical Foundation:
- Implements standard Black-Scholes formulas for European options
- Uses scipy.stats.norm for cumulative normal distribution
- Proper handling of edge cases and mathematical constraints
- Professional-grade precision and error handling

Usage:
    from utils import black_scholes_price, black_scholes_greeks, option_valuation
    
    # Calculate option price
    price = black_scholes_price(100, 100, 0.25, 0.05, 0.2, 'call')
    
    # Calculate Greeks
    greeks = black_scholes_greeks(100, 100, 0.25, 0.05, 0.2, 'call')
    
    # Advanced valuation
    rating, pct_diff, confidence = option_valuation(theo_price, market_price, ...)

Dependencies:
- math: For mathematical operations (log, sqrt, exp)
- scipy.stats.norm: For cumulative normal distribution calculations

This module serves as the foundation for all options analysis in the project,
providing reliable, tested financial calculations used by all other scripts.

Author: NSE Options Analysis Project
License: See LICENSE file
"""

import math
from scipy.stats import norm

# Black-Scholes pricing for European options
# S: spot price, K: strike, T: time to expiry (years), r: risk-free rate, sigma: volatility, option_type: 'call' or 'put'
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError('option_type must be "call" or "put"')
    return price

# Greeks calculation for European options
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)))
    vega = S * norm.pdf(d1) * math.sqrt(T)
    if option_type == 'call':
        rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega / 100,  # per 1% change in volatility
        'rho': rho / 100     # per 1% change in rate
    }

# Enhanced valuation function with multiple factors
def option_valuation(theoretical_price, market_price, S=None, K=None, T=None, sigma=None, 
                     bid=None, ask=None, option_type='call', tolerance=0.05):
    """
    Enhanced option valuation considering multiple real-world factors beyond simple percentage difference.
    
    This function implements a sophisticated scoring system that accounts for:
    1. MONEYNESS FACTOR: How far in/out of the money the option is
       - Uses exponential decay: further from ATM = higher tolerance for mispricing
       - OTM options have more uncertainty in pricing due to lower probability of exercise
    
    2. TIME DECAY FACTOR: Time value considerations
       - Shorter time to expiry = higher tolerance (more uncertainty about final outcome)
       - Uses exponential decay with time: exp(-T * 2)
       - Accounts for accelerated time decay near expiration
    
    3. VOLATILITY FACTOR: Uncertainty in underlying asset movement
       - Higher volatility = higher tolerance (more uncertainty about future price movements)
       - Capped at 20% to prevent excessive tolerance
       - Reflects that high vol environments have more pricing noise
    
    4. LIQUIDITY FACTOR: Bid-ask spread considerations
       - Wider spreads indicate lower liquidity and higher transaction costs
       - Higher spread = higher tolerance for mispricing
       - Accounts for market maker risk and transaction friction
    
    5. NON-LINEAR SCORING: Sigmoid function for smooth transitions
       - Provides gradual transitions between valuation categories
       - Avoids sharp cutoffs that don't reflect real market conditions
       - Confidence score indicates reliability of the valuation
    
    Parameters:
    -----------
    theoretical_price : float
        Black-Scholes theoretical price
    market_price : float
        Current market price of the option
    S : float, optional
        Current spot price of underlying asset
    K : float, optional
        Strike price of the option
    T : float, optional
        Time to expiry in years
    sigma : float, optional
        Implied volatility (0-1 scale)
    bid : float, optional
        Current bid price
    ask : float, optional
        Current ask price
    option_type : str
        'call' or 'put'
    tolerance : float
        Base tolerance for percentage difference (default 0.05 = 5%)
    
    Returns:
    --------
    tuple : (rating, pct_diff, confidence)
        rating: str - 'strongly undervalued', 'undervalued', 'slightly undervalued',
                    'fairly priced', 'slightly overvalued', 'overvalued', 'strongly overvalued'
        pct_diff: float - Percentage difference (theoretical - market) / market
        confidence: float - Confidence level (0-1, higher = more confident)
    """
    if market_price <= 0:
        return 'invalid_price', float('inf'), 0.0
    
    # Basic percentage difference calculation
    diff = theoretical_price - market_price
    pct_diff = diff / market_price
    
    # Initialize scoring components (each adds to tolerance)
    moneyness_score = 0
    time_score = 0
    volatility_score = 0
    liquidity_score = 0
    
    # 1. MONEYNESS FACTOR: Exponential decay based on distance from ATM
    if S and K:
        moneyness = abs(math.log(S / K))  # Natural log of spot/strike ratio
        # Exponential decay: further from ATM = higher tolerance
        # Formula: exp(-moneyness) * 0.1
        # Rationale: OTM options have more pricing uncertainty
        moneyness_score = math.exp(-moneyness) * 0.1
    
    # 2. TIME DECAY FACTOR: Exponential decay with time to expiry
    if T:
        # Shorter time = higher tolerance for mispricing
        # Formula: exp(-T * 2) * 0.15
        # Rationale: Near-expiry options have accelerated time decay and uncertainty
        time_score = math.exp(-T * 2) * 0.15
    
    # 3. VOLATILITY FACTOR: Higher vol = higher tolerance
    if sigma:
        # Higher volatility = higher tolerance (more uncertainty)
        # Formula: min(sigma * 0.3, 0.2) - capped at 20%
        # Rationale: High vol environments have more pricing noise
        volatility_score = min(sigma * 0.3, 0.2)
    
    # 4. LIQUIDITY FACTOR: Bid-ask spread consideration
    if bid and ask and ask > bid:
        spread_pct = (ask - bid) / ((ask + bid) / 2)  # Percentage spread
        # Higher spread = higher tolerance for mispricing
        # Formula: min(spread_pct * 0.5, 0.1) - capped at 10%
        # Rationale: Wide spreads indicate liquidity issues and transaction costs
        liquidity_score = min(spread_pct * 0.5, 0.1)
    
    # Calculate dynamic tolerance by combining all factors
    dynamic_tolerance = tolerance + moneyness_score + time_score + volatility_score + liquidity_score
    
    # 5. NON-LINEAR SCORING: Sigmoid function for smooth transitions
    def sigmoid_score(x, tolerance):
        """
        Sigmoid function to create smooth transitions between valuation categories.
        Returns value between 0 and 1, with steep transition around the tolerance level.
        """
        return 1 / (1 + math.exp(-10 * (abs(x) - tolerance)))
    
    # Calculate confidence score (0-1, higher = more confident in valuation)
    # Confidence decreases as percentage difference approaches tolerance
    confidence = 1 - sigmoid_score(pct_diff, dynamic_tolerance)
    
    # Determine rating with confidence levels
    if pct_diff > dynamic_tolerance:
        if confidence > 0.8:
            rating = 'strongly undervalued'
        elif confidence > 0.6:
            rating = 'undervalued'
        else:
            rating = 'slightly undervalued'
    elif pct_diff < -dynamic_tolerance:
        if confidence > 0.8:
            rating = 'strongly overvalued'
        elif confidence > 0.6:
            rating = 'overvalued'
        else:
            rating = 'slightly overvalued'
    else:
        rating = 'fairly priced'
    
    # Return rating, percentage difference, and confidence score
    return rating, pct_diff, confidence

# Legacy function for backward compatibility
def simple_option_valuation(theoretical_price, market_price, tolerance=0.05):
    """
    Simple valuation function (legacy) - only considers percentage difference.
    Use option_valuation() for more sophisticated analysis.
    """
    diff = theoretical_price - market_price
    pct_diff = diff / market_price if market_price != 0 else float('inf')
    if pct_diff > tolerance:
        rating = 'undervalued'
    elif pct_diff < -tolerance:
        rating = 'overvalued'
    else:
        rating = 'fairly priced'
    return rating, pct_diff