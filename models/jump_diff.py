import numpy as np

import config
from monte_carlo import price_option

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


def find_params_lbfgsb(data, steps, r, n_sim, initial_params=None):
    if minimize is None:
        return None

    S0 = data["S0"]
    sigma = data["sigma"]
    T = data["T"]
    calls = data["calls"].sort_values(by="strike")

    liquid_calls = calls[calls["lastPrice"] > 0].reset_index(drop=True)
    n_sample = min(config.calibration_n_options, len(liquid_calls))
    if n_sample == 0:
        return None

    sample_idx = np.linspace(0, len(liquid_calls) - 1, n_sample, dtype=int)
    sample_calls = liquid_calls.iloc[sample_idx]

    if initial_params is None:
        initial_params = (
            config.init_lambda_j,
            config.init_mu_j,
            config.init_sigma_j,
        )

    bounds = [config.lambda_bounds, config.mu_bounds, config.sigma_j_bounds]

    def objective(x):
        lam, mu, sig = x
        if lam <= 0 or sig <= 0:
            return 1e12

        # Fixed seed per objective call keeps optimization stable.
        np.random.seed(config.lbfgsb_seed)
        total_sq_error = 0.0

        for _, row in sample_calls.iterrows():
            K = row["strike"]
            market_price = float(row["lastPrice"])

            jd_price = price_option(
                simulate_jump_diffusion,
                S0,
                K,
                r,
                sigma,
                T,
                steps,
                n_sim,
                lambda_j=lam,
                mu_j=mu,
                sigma_j=sig,
            )

            total_sq_error += (jd_price - market_price) ** 2

        return total_sq_error

    result = minimize(
        objective,
        x0=np.array(initial_params, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": config.lbfgsb_max_iter},
    )

    if not result.success:
        return None

    return tuple(result.x)


def simulate_jump_diffusion(S0, r, sigma, T, steps, lambda_j=None, mu_j=None, sigma_j=None):
    dt = T / steps
    S = S0

    if lambda_j is None:
        lambda_j = config.lambda_j
    if mu_j is None:
        mu_j = config.mu_j
    if sigma_j is None:
        sigma_j = config.sigma_j

    # Merton risk-neutral adjustment: kappa = E[e^Y] - 1 for Y ~ N(mu_j, sigma_j^2)
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift = (r - lambda_j * kappa - 0.5 * sigma**2) * dt

    for _ in range(steps):
        Z = np.random.randn()
        diffusion = drift + sigma * np.sqrt(dt) * Z

        Nj = np.random.poisson(lambda_j * dt)
        if Nj > 0:
            Y = np.random.normal(mu_j, sigma_j, Nj)
            jump_sum = np.sum(Y)
        else:
            jump_sum = 0.0

        S = S * np.exp(diffusion + jump_sum)

    return S