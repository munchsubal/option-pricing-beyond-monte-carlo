# main.py

import numpy as np

from data_loader import get_option_chain_data
from models.jump_diff import simulate_jump_diffusion, find_params_lbfgsb
from monte_carlo import price_option
from models.gbm import simulate_gbm
import config

if __name__ == "__main__":
    data = get_option_chain_data()

    S0 = data["S0"]
    sigma = data["sigma"]
    T = data["T"]
    dte = data["dte"]
    expiry = data["expiry"]
    calls = data["calls"]

    # sort strikes
    calls = calls.sort_values(by="strike")

    # keep only tradable calls for display
    display_calls = calls[calls["lastPrice"] > 0].reset_index(drop=True)

    # sample strikes across the chain (left wing, ATM, right wing)
    n_display = min(config.display_n_options, len(display_calls))
    if n_display == 0:
        raise ValueError("No call options with non-zero market price for display")
    sample_idx = np.linspace(0, len(display_calls) - 1, n_display, dtype=int)
    sample_calls = display_calls.iloc[sample_idx]

    lbfgsb_params = find_params_lbfgsb(
        data,
        config.steps,
        config.r,
        config.calibration_n_sim,
        initial_params=(config.init_lambda_j, config.init_mu_j, config.init_sigma_j),
    )

    if lbfgsb_params is None:
        lambda_j, mu_j, sigma_j = config.lambda_j, config.mu_j, config.sigma_j
        print("L-BFGS-B calibration failed/unavailable, using default jump params")
    else:
        lambda_j, mu_j, sigma_j = lbfgsb_params

    print(
        f"Calibrated JD params -> lambda: {lambda_j:.3f}, mu_j: {mu_j:.3f}, sigma_j: {sigma_j:.3f}"
    )

    print(f"Ticker: {config.ticker} | Expiry: {expiry} | DTE: {dte}")
    print(f"Stock Price (S0): {S0:.2f}\n")
    print("Strike | Market | MC Price | JD Price | JD-MC")

    for _, row in sample_calls.iterrows():
        K = row["strike"]
        market_price = row["lastPrice"]

        if market_price == 0:
            continue

        # Monte Carlo pricing (GBM) - assuming constant risk free interest rate of 5%
        mc_price = price_option(
            simulate_gbm,
            S0, K, config.r, sigma, T,
            config.steps, config.n_sim
        )

        jd_price = price_option(
            simulate_jump_diffusion,
            S0, K, config.r, sigma, T,
            config.steps,
            config.n_sim,
            lambda_j=lambda_j,
            mu_j=mu_j,
            sigma_j=sigma_j,
        )

        spread = jd_price - mc_price
        print(
            f"{K:.2f} | {market_price:.2f} | "
            f"{mc_price:.2f} | {jd_price:.2f} | {spread:.2f}"
        )