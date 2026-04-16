# monte_carlo.py

import numpy as np

def price_option(model_func, S0, K, r, sigma, T, steps, n_sim, **model_kwargs):
    payoffs = []

    for _ in range(n_sim):
        path_or_price = model_func(S0, r, sigma, T, steps, **model_kwargs)
        ST = path_or_price[-1] if isinstance(path_or_price, (list, np.ndarray)) else path_or_price
        payoff = max(ST - K, 0)
        payoffs.append(payoff)

    price = np.exp(-r * T) * np.mean(payoffs)
    return price