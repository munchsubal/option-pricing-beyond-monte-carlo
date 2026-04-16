# data_loader.py

from datetime import datetime
import yfinance as yf
import numpy as np

import config

def get_option_chain_data():
    stock = yf.Ticker(config.ticker)

    S0 = stock.history(period="1d")["Close"].iloc[-1]

    hist = stock.history(period="1y")["Close"]
    returns = np.log(hist / hist.shift(1)).dropna()
    sigma = returns.std() * np.sqrt(252)

    expirations = stock.options
    if not expirations:
        raise ValueError(f"No option expirations available for ticker {config.ticker}")

    today = datetime.today()
    target_days = config.target_dte

    expiry = min(
        expirations,
        key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - target_days),
    )

    option_chain = stock.option_chain(expiry)
    calls = option_chain.calls

    # compute T
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    dte = max((expiry_date - today).days, 0)
    T = max(dte / 365, 0.001)

    return {
        "S0": S0,
        "sigma": sigma,
        "T": T,
        "dte": dte,
        "expiry": expiry,
        "calls": calls,
    }