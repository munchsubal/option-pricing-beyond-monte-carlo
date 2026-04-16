# Option Pricing: GBM vs Merton Jump-Diffusion

## Theory reference (paper)
This project's theoretical foundation for jump-diffusion option pricing follows the paper:

[1] Karmanpartap Singh Sidhu, Pranshi Saxena, "Beyond Black-Scholes: A Computational Framework for Option Pricing Using Heston, GARCH, and Jump Diffusion Models" arXiv,  2026, arXiv:2604.06068.

## Theme : 
This project prices European call options from live market data using Monte Carlo simulation and compares two models: GBM and Merton Jump-Diffusion. GBM assumes continuous random movement. Merton adds random jumps to capture sharp moves. Jump parameters are calibrated with L-BFGS-B so model prices are as close as possible to market prices.

## How?
"Does adding jumps improve pricing versus plain GBM for this ticker and expiry range?"

This is answered by comparing, for each strike:
- Market price
- GBM Monte Carlo price
- Jump-Diffusion Monte Carlo price
- Spread = JD - GBM

Lower model error to market is what matters most.

## Core math

### 1) Monte Carlo option price
For a call option:

$$
C_0 = e^{-rT} \; \mathbb{E}[\max(S_T - K, 0)]
$$

The code approximates this expectation by simulating many paths and averaging payoff.

### 2) GBM dynamics

$$
S_{t+\Delta t} = S_t \exp\left((r - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z\right),\; Z\sim \mathcal{N}(0,1)
$$

### 3) Merton Jump-Diffusion dynamics

$$
S_{t+\Delta t} = S_t \exp\left((r - \lambda\kappa - \tfrac{1}{2}\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z + \sum_{k=1}^{N_t}Y_k\right)
$$

where:
- $N_t \sim \text{Poisson}(\lambda\Delta t)$ is number of jumps in the step
- $Y_k \sim \mathcal{N}(\mu_j, \sigma_j^2)$ are jump sizes (in log-return terms)
- $\kappa = \mathbb{E}[e^Y]-1 = e^{\mu_j + 0.5\sigma_j^2} - 1$

The $-\lambda\kappa$ term is the risk-neutral drift adjustment.

### 4) Calibration objective (L-BFGS-B)
You fit $(\lambda,\mu_j,\sigma_j)$ by minimizing squared pricing error:

$$
\min_{\lambda,\mu_j,\sigma_j} \sum_i \left(C_i^{JD} - C_i^{mkt}\right)^2
$$

with parameter bounds from config.

## End-to-end flow when running main.py
1. Load market data from yfinance.
2. Compute spot price $S_0$, historical volatility $\sigma$, selected expiry, and maturity $T$.
3. Keep liquid calls and sample strikes across the chain.
4. Run L-BFGS-B calibration for jump parameters.
5. If calibration fails/unavailable, use default jump parameters.
6. Price each strike with GBM and JD via Monte Carlo.
7. Print comparison table.

## File-by-file map
- main.py
Orchestrates everything: data -> calibration -> pricing -> output table.

- config.py
All knobs: ticker, rates, simulation sizes, calibration settings, optimizer bounds, init/fallback jump params.

- data_loader.py
Fetches stock and option chain, chooses expiry near target DTE, computes $T$ and historical volatility.

- monte_carlo.py
Generic pricing engine. Takes any model simulator, computes discounted expected payoff.

- models/gbm.py
GBM path simulator (baseline model).

- models/jump_diff.py
Contains:
1. find_params_lbfgsb: calibrates jump params.
2. simulate_jump_diffusion: simulates paths with diffusion + jumps.

## How to read the output table
- Market: observed option price
- MC Price: GBM price
- JD Price: Jump-Diffusion price
- JD-MC: how much jumps changed the price

Important: negative JD-MC is not automatically good or bad. Good means model error to market is smaller.

## Current strengths
- Uses bounded L-BFGS-B calibration.
- Clear fallback behavior when calibration fails.

## Current limitations (important for interviews)
- Uses last traded option price (can be noisy/stale).
- Call-only calibration.
- Calibration can be slow because optimizer repeatedly calls Monte Carlo.
- Results depend on simulation settings (paths, steps, selected strikes).

## Fast run vs quality run
- Fast debug mode: small `n_sim`, `steps`, `calibration_n_sim`, `calibration_n_options`.
- Better quality mode: increase those values gradually.

## Run command
```bash
python3 main.py
```

