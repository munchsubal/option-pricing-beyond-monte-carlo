ticker = "COIN"

# config.py
r = 0.05      # risk-free rate
steps = 100
n_sim = 200

# Option chain selection controls
target_dte = 45
display_n_options = 15

# L-BFGS-B initialization (as in your algorithm)
init_lambda_j = 0.1
init_mu_j = 0.02
init_sigma_j = 0.05

# Default jump parameters (used as fallback if calibration fails)
lambda_j = 1.0
mu_j = -0.1
sigma_j = 0.3

# Calibration controls
calibration_n_sim = 200
calibration_n_options = 6

# L-BFGS-B optimizer controls
lbfgsb_seed = 42
lbfgsb_max_iter = 8
lambda_bounds = (0.01, 5.0)
mu_bounds = (-0.5, 0.5)
sigma_j_bounds = (0.01, 1.0)