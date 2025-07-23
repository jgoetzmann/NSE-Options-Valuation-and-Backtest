# nse_bs_daily.py  ──  Black‑Scholes valuation on NSE option chain (no API key)
import requests, json, time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm


# ─────── CONFIG ──────────────────────────────────────────────────────────────
SYMBOL          = "NIFTY"     # or "BANKNIFTY", "FINNIFTY", etc.
RISK_FREE       = 0.063       # 10‑year G‑Sec ~6.3 % (set once per day)
DEV_THRESHOLD   = 10          # % deviation for Over/Under label
SAVE_PATH       = "nse_option_pricing_nifty.csv"
# ─────────────────────────────────────────────────────────────────────────────


def black_scholes(S, K, T, r, sigma, opt_type="call"):
    """Black–Scholes theoretical price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def get_nse_chain(symbol="NIFTY"):
    """Download option‑chain JSON from nseindia.com (no key, spoof headers)."""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.nseindia.com/option-chain",
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()
    # First call to home page sets cookies that the JSON endpoint wants
    session.get("https://www.nseindia.com", headers=hdrs, timeout=10)
    time.sleep(0.5)  # polite pause
    resp = session.get(url, headers=hdrs, timeout=10)
    resp.raise_for_status()

    return resp.json()


def main():
    data = get_nse_chain(SYMBOL)

    spot_price   = data["records"]["underlyingValue"]        # current index price
    all_expiries = data["records"]["expiryDates"]
    expiry_str   = all_expiries[0]                           # choose nearest expiry
    expiry_date  = datetime.strptime(expiry_str, "%d-%b-%Y")
    T            = (expiry_date - datetime.today()).days / 365

    rows = []
    for item in data["records"]["data"]:
        strike = item["strikePrice"]

        for side, opt_key in [("CE", "call"), ("PE", "put")]:
            if side not in item:
                continue
            opt = item[side]
            # Skip illiquid rows
            if opt.get("lastPrice") is None:
                continue

            ltp   = float(opt["lastPrice"])
            iv    = float(opt.get("impliedVolatility") or 0) / 100  # % ➔ decimal
            iv    = iv if iv > 0 else 0.25                          # fallback vol 25 %
            theo  = black_scholes(spot_price, strike, T, RISK_FREE, iv, opt_key)
            dev   = ltp - theo
            dev_pct = dev / theo * 100 if theo else 0

            if   dev_pct >  DEV_THRESHOLD: verdict = "Overpriced"
            elif dev_pct < -DEV_THRESHOLD: verdict = "Underpriced"
            else:                          verdict = "Fairly Priced"

            rows.append(
                {
                    "Option": side,
                    "Strike": strike,
                    "Expiry": expiry_str,
                    "Spot": round(spot_price, 2),
                    "LTP": round(ltp, 2),
                    "Theo": round(theo, 2),
                    "IV(%)": round(iv * 100, 2),
                    "T(yrs)": round(T, 4),
                    "Dev": round(dev, 2),
                    "Dev(%)": round(dev_pct, 2),
                    "Verdict": verdict,
                }
            )

    df = pd.DataFrame(rows).sort_values(["Strike", "Option"])
    df.to_csv(SAVE_PATH, index=False)
    print(f"Saved {len(df)} rows ➜ {SAVE_PATH}")


if __name__ == "__main__":
    main()
