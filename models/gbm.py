import numpy as np

def simulate_gbm(S0, r, sigma, T, steps):
    dt = T / steps
    S = S0
    prices = [S]

    for _ in range(steps):
        Z = np.random.normal()
        S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(S)

    return prices