"""
simulation.py
=============

Monte Carlo engine for the second half of the project.

`portfolio.py` answers: *which weights should I hold?* (Markowitz).
This file answers: *what could happen to that portfolio, and what would it
cost to insure it?* (Monte Carlo risk, option pricing, hedging).

Monte Carlo in one sentence: instead of solving a formula, we simulate
thousands of possible futures, then read the answer off the distribution of
outcomes. If 5% of the futures lose more than 28%, then the 95% VaR is 28%.

Map of this file:

    Section                               What it covers
    -----------------------------------   -----------------------------------
    1. Expected returns for simulation    historical / CAPM / flat drift
    2. Path simulation (GBM)              correlated paths, single-asset paths
    3. Risk metrics                       VaR, CVaR, drawdown, probabilities
    4. Model checking                     historical VaR, VaR backtest
    5. Black-Scholes                      closed-form prices and Greeks
    6. Monte Carlo option pricing         European, Asian, convergence
    7. Monte Carlo Greeks                 bump-and-revalue, pathwise delta
    8. Protective put                     single-stock puts vs index put
    9. Delta hedging                      discrete rebalancing, hedging error

Notation (same as portfolio.py where possible):
    mu     = expected annual return per asset (simple returns)
    Sigma  = annual covariance matrix
    sigma  = annual volatility of one asset
    rf / r = risk-free rate (10-year G-Sec yield)
    T      = time to expiry in years
    S      = spot price, K = strike price
    N      = number of simulated paths

One important distinction, used throughout:

    REAL-WORLD measure  -> used for RISK. Assets drift at their expected
                           return mu. Answers "what might I actually earn?"
    RISK-NEUTRAL measure-> used for OPTION PRICING. Assets drift at the
                           risk-free rate r, never at mu. Answers "what is
                           this option worth today?" Using mu to price an
                           option is the classic beginner's mistake: the
                           no-arbitrage argument (hedging the option with the
                           stock) removes mu from the answer entirely.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np                 # numerical maths: vectors, matrices
import pandas as pd                # tables of data (like an Excel sheet)
from scipy.stats import norm       # the normal distribution (N(x) in Black-Scholes)

# Trading days in an NSE year: the same convention as portfolio.py, so the
# two files can share mu and Sigma without any rescaling.
TRADING_DAYS = 252


# ===========================================================================
# 1. EXPECTED RETURNS USED FOR THE SIMULATION
# ===========================================================================
# The simulated outcomes are only as good as the expected returns we feed in.
# The Markowitz notebook showed the problem clearly: the training-period mean
# for the Max Sharpe portfolio was 84% a year, which no one would forecast.
# So this section offers three ways to set the drift, and the notebook and
# dashboard let you switch between them.
# ---------------------------------------------------------------------------
def historical_drift(returns):
    """
    Expected returns taken straight from past data: average daily return x 252.
    Identical to portfolio.py's annualise(), repeated here so the choice of
    drift is visible in one place.

    WARNING (this is the point of offering alternatives): a two-year sample
    from a strong rally produces expected returns of 50-100% a year. Those are
    sample averages, not forecasts. Simulating with them makes almost every
    future look profitable, which flatters the portfolio and understates risk.
    """
    return returns.mean() * TRADING_DAYS


def capm_drift(returns, benchmark_returns, rf, equity_premium=0.055):
    """
    Expected returns from the CAPM:

        E[R_i] = rf + beta_i * (equity risk premium)

    Instead of asking "what did this stock return?", the CAPM asks "how much
    market risk does it carry?" A stock with a beta of 1.5 is expected to earn
    1.5 times the market's excess return, whatever its own past happened to be.

    Parameters
    ----------
    returns           : DataFrame of daily stock returns
    benchmark_returns : Series of daily Nifty 50 returns (same dates)
    rf                : risk-free rate, e.g. 0.07
    equity_premium    : the extra return expected from equities over the
                        risk-free rate. 5.5% is a common estimate for India
                        (Damodaran's country-risk-adjusted ERP is in this area).

    Returns
    -------
    mu    : Series of expected annual returns
    betas : Series of betas, kept because the index hedge needs them
    """
    betas = compute_betas(returns, benchmark_returns)
    mu = rf + betas * equity_premium
    return mu, betas


def flat_drift(columns, rf, equity_premium=0.055):
    """
    The simplest assumption: every stock is expected to earn rf + premium.
    Useful as a sanity check, because it removes the effect of stock-specific
    forecasts entirely. Any difference between portfolios then comes only from
    their risk and correlation structure, not from the return estimates.
    """
    return pd.Series(rf + equity_premium, index=columns)


def compute_betas(returns, benchmark_returns):
    """
    Beta of each stock against the benchmark:

        beta_i = covariance(r_i, r_market) / variance(r_market)

    Beta says how much the stock tends to move for a 1% move in the Nifty.
    It is used twice in this project: for CAPM expected returns, and to size
    the Nifty put in the hedging section.
    """
    # Line the two up on the same dates, so we never compare mismatched days.
    aligned = returns.join(benchmark_returns.rename("benchmark"), how="inner")
    market = aligned["benchmark"]
    market_var = market.var()

    betas = {}
    for col in returns.columns:
        betas[col] = aligned[col].cov(market) / market_var
    return pd.Series(betas)


def portfolio_beta(weights, betas):
    """
    Beta of the whole portfolio = the weighted average of the stock betas.
    This is the hedge ratio: a portfolio with a beta of 1.3 needs 1.3 times
    its own value in Nifty puts to be hedged against a market fall.
    """
    return float((weights * betas[weights.index]).sum())


# ===========================================================================
# 2. PATH SIMULATION (GEOMETRIC BROWNIAN MOTION)
# ===========================================================================
# GBM is the standard model behind Black-Scholes. In words: returns are random
# and normally distributed, so prices are lognormal and can never go negative.
#
#     S(t + dt) = S(t) * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
#
# The (- 0.5*sigma^2) term is the Ito correction. Without it, the average
# simulated price would drift ABOVE exp(mu*t), because exponentiating a random
# number is not the same as exponentiating its average (Jensen's inequality).
# With it, E[S(T)/S(0)] = exp(mu*T) exactly, which is what "expected return
# of mu" should mean.
#
# For several assets at once we need their moves to be correlated, which is
# what the Cholesky factor does (see below).
# ---------------------------------------------------------------------------
def cholesky_factor(Sigma):
    """
    Split the covariance matrix into L such that  L @ L' = Sigma.

    Why: np.random.standard_normal gives INDEPENDENT random numbers Z. But our
    stocks move together. Multiplying by L turns independent draws into
    correlated ones with exactly the covariance we asked for:

        correlated shocks = L @ Z,   covariance(L @ Z) = L @ L' = Sigma

    It is the matrix version of "multiply a standard normal by sigma to get a
    normal with that volatility".

    A covariance matrix estimated from data is occasionally not quite positive
    definite (rounding, or more assets than observations), which makes the
    Cholesky decomposition fail. If that happens we nudge the diagonal up by a
    tiny amount ("jitter") until it succeeds.
    """
    S = np.asarray(Sigma, dtype=float)
    jitter = 0.0
    for _ in range(6):
        try:
            return np.linalg.cholesky(S + jitter * np.eye(len(S)))
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-12)
    raise np.linalg.LinAlgError("Covariance matrix is not positive definite")


def simulate_portfolio(mu, Sigma, weights, horizon_years=1.0,
                       n_steps=TRADING_DAYS, n_paths=10_000,
                       initial_value=1_000_000.0, seed=42):
    """
    Simulate many possible futures for a BUY-AND-HOLD portfolio.

    The portfolio is bought on day 0 and never rebalanced, exactly like the
    out-of-sample test in the Markowitz notebook:

        Value(t) = V0 * sum_i [ w_i * S_i(t) / S_i(0) ]

    Parameters
    ----------
    mu            : Series of expected annual returns (one per asset)
    Sigma         : annual covariance matrix (DataFrame, same asset order)
    weights       : Series of portfolio weights. Assets with a weight of zero
                    are still simulated, which is how the Nifty is included:
                    it is needed for the index hedge but is not held.
    horizon_years : how far into the future to simulate (1.0 = one year)
    n_steps       : time steps (252 = daily for a one-year horizon)
    n_paths       : number of simulated futures
    initial_value : starting portfolio value in rupees (default Rs 10 lakh)
    seed          : fixes the random numbers so results are reproducible

    Returns
    -------
    dict with:
        "values"     : array (n_paths, n_steps+1) of portfolio values over time
        "relatives"  : array (n_paths, n_assets) of each asset's FINAL price
                       divided by its starting price (needed for option payoffs)
        "assets"     : the asset names, in column order
        "weights"    : the weights used
        "initial"    : the starting value

    Memory note: we deliberately do NOT keep every asset's full path (that
    would be paths x steps x assets numbers, hundreds of MB). We step through
    time keeping only the current prices, and store the portfolio value at each
    step. That is all the risk metrics need, and it keeps the dashboard light.
    """
    assets = list(mu.index)
    w = np.asarray(weights[assets], dtype=float)
    mu_vec = np.asarray(mu, dtype=float)

    # Per-step size: one trading day is 1/252 of a year.
    dt = horizon_years / n_steps
    L = cholesky_factor(Sigma)

    # Drift per step, including the Ito correction described above.
    variances = np.diag(np.asarray(Sigma, dtype=float))
    drift = (mu_vec - 0.5 * variances) * dt

    rng = np.random.default_rng(seed)

    # log_growth holds each asset's cumulative log return so far, per path.
    # Starting at zero means every asset starts at a price relative of 1.0.
    log_growth = np.zeros((n_paths, len(assets)))

    values = np.empty((n_paths, n_steps + 1))
    values[:, 0] = initial_value

    for step in range(1, n_steps + 1):
        # One batch of independent standard normal shocks ...
        z = rng.standard_normal((n_paths, len(assets)))
        # ... turned into correlated shocks by the Cholesky factor.
        # (z @ L.T is the row-wise version of L @ z.)
        log_growth += drift + np.sqrt(dt) * (z @ L.T)

        # Price relatives today, then the portfolio value: sum of w_i * growth_i
        relatives = np.exp(log_growth)
        values[:, step] = initial_value * (relatives @ w)

    return {
        "values": values,
        "relatives": np.exp(log_growth),   # final price relatives per asset
        "assets": assets,
        "weights": pd.Series(w, index=assets),
        "initial": initial_value,
    }


def simulate_asset_paths(S0, drift, sigma, T=1.0, n_steps=TRADING_DAYS,
                         n_paths=10_000, seed=42, antithetic=False):
    """
    Simulate ONE asset (used for option pricing and delta hedging).

    Parameters
    ----------
    S0        : starting price
    drift     : expected annual return. Use the RISK-FREE RATE for pricing,
                and the real expected return only when simulating real-world
                outcomes (e.g. the P&L of a hedging strategy).
    sigma     : annual volatility
    T         : years to expiry
    n_steps   : steps per path. Use 1 for a European option (only the final
                price matters), and 252 for path-dependent options like Asians.
    antithetic: if True, every random draw Z is also used as -Z. Paths then
                come in mirror-image pairs, which cancels out some of the
                sampling noise and gives a more accurate price for the same
                number of paths. n_paths must be even.

    Returns
    -------
    array (n_paths, n_steps+1) of prices, starting at S0.
    """
    dt = T / n_steps
    drift_step = (drift - 0.5 * sigma ** 2) * dt
    vol_step = sigma * np.sqrt(dt)

    rng = np.random.default_rng(seed)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")
        half = rng.standard_normal((n_paths // 2, n_steps))
        z = np.vstack([half, -half])     # each path and its mirror image
    else:
        z = rng.standard_normal((n_paths, n_steps))

    # Cumulative sum along the time axis turns per-step log returns into the
    # whole path, in one vectorised operation (no Python loop needed).
    log_path = np.cumsum(drift_step + vol_step * z, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_path)
    return paths


# ===========================================================================
# 3. RISK METRICS
# ===========================================================================
# These turn a cloud of simulated outcomes into the numbers a risk report
# actually quotes.
# ---------------------------------------------------------------------------
def value_at_risk(terminal_values, initial_value, confidence=0.95):
    """
    Value at Risk: the loss that is exceeded only (1 - confidence) of the time.

    "95% VaR of 28%" means: in the worst 5% of simulated futures the loss was
    worse than 28%. It is a threshold, NOT a worst case.

    We compute it as a percentile of the simulated outcomes, which is the
    natural Monte Carlo approach and needs no normality assumption about the
    final distribution (GBM's terminal values are lognormal, not normal).

    Returns VaR as a POSITIVE fraction of the initial value, so 0.28 = a 28%
    loss. A negative number would mean even the bad cases made money.
    """
    losses = 1.0 - np.asarray(terminal_values) / initial_value
    return float(np.percentile(losses, confidence * 100))


def conditional_value_at_risk(terminal_values, initial_value, confidence=0.95):
    """
    CVaR (also called Expected Shortfall): the AVERAGE loss in the cases that
    are worse than VaR.

    VaR says "how bad does it get before the worst 5%".
    CVaR says "and how bad is it when you are in that 5%".

    CVaR is the more informative of the two, and unlike VaR it is coherent
    (sub-additive: merging two portfolios can never increase it), which is why
    Basel's market-risk framework moved to Expected Shortfall.
    """
    losses = 1.0 - np.asarray(terminal_values) / initial_value
    threshold = np.percentile(losses, confidence * 100)
    tail = losses[losses >= threshold]
    return float(tail.mean())


def max_drawdown_paths(values):
    """
    Worst peak-to-trough fall WITHIN each simulated path.

    Terminal VaR only looks at the finish line. Drawdown looks at the journey:
    a path can end flat after falling 40% on the way, and an investor would
    have lived through that fall.

    Returns an array with one drawdown per path, as a negative fraction
    (-0.40 = a 40% fall from the peak).
    """
    values = np.asarray(values)
    # np.maximum.accumulate gives the running peak along each row.
    running_peak = np.maximum.accumulate(values, axis=1)
    drawdowns = values / running_peak - 1.0
    return drawdowns.min(axis=1)


def simulation_summary(sim, rf=0.07, confidence_levels=(0.95, 0.99),
                       target_return=None):
    """
    One row of headline statistics for a simulated portfolio.

    Includes the mean and median (which differ because the distribution is
    skewed: a few very good paths pull the mean above the median), VaR and
    CVaR at each confidence level, drawdown, and the probabilities that make
    the result easy to explain to a non-specialist.
    """
    values = sim["values"]
    initial = sim["initial"]
    terminal = values[:, -1]
    growth = terminal / initial - 1.0

    row = {
        "Mean return": float(growth.mean()),
        "Median return": float(np.median(growth)),
        "Std dev of return": float(growth.std(ddof=1)),
        "5th percentile": float(np.percentile(growth, 5)),
        "95th percentile": float(np.percentile(growth, 95)),
    }

    for c in confidence_levels:
        row[f"VaR {c:.0%}"] = value_at_risk(terminal, initial, c)
        row[f"CVaR {c:.0%}"] = conditional_value_at_risk(terminal, initial, c)

    drawdowns = max_drawdown_paths(values)
    row["Median max drawdown"] = float(np.median(drawdowns))
    row["Worst max drawdown"] = float(drawdowns.min())

    row["P(loss)"] = float((growth < 0).mean())
    row["P(beat risk-free)"] = float((growth > rf).mean())
    if target_return is not None:
        row[f"P(beat {target_return:.0%})"] = float((growth > target_return).mean())

    return row


# ===========================================================================
# 4. MODEL CHECKING
# ===========================================================================
# GBM assumes normally distributed returns. Real equity returns have FAT TAILS:
# crashes happen far more often than a normal distribution allows. These
# functions measure how badly that assumption fails, rather than leaving it as
# a footnote.
# ---------------------------------------------------------------------------
def historical_var(portfolio_returns, confidence=0.95, horizon_days=1):
    """
    VaR read directly from history, with no model at all.

    We add up actual returns over every overlapping window of `horizon_days`
    days, then take the percentile of those. Overlapping windows are used
    because non-overlapping ones would leave very few observations (a year of
    data has only one non-overlapping annual window).

    Comparing this with the GBM number is the honest check: if the historical
    VaR is much larger, the model is understating the tails.
    """
    r = pd.Series(portfolio_returns).dropna()

    if horizon_days == 1:
        window_returns = r
    else:
        # Compound each window properly: (1+r1)(1+r2)... - 1
        window_returns = (1 + r).rolling(horizon_days).apply(
            np.prod, raw=True).dropna() - 1

    return float(-np.percentile(window_returns, (1 - confidence) * 100))


def gbm_var(mu_p, sigma_p, confidence=0.95, horizon_days=1):
    """
    VaR implied by the GBM model, straight from the formula rather than by
    simulation. With lognormal prices the loss threshold is:

        VaR = 1 - exp( (mu - sigma^2/2)*h + sigma*sqrt(h)*z )

    where h is the horizon in years and z is the normal quantile (e.g. -1.645
    for 95%). Having both the formula and the simulation is a useful check:
    with enough paths they should agree closely, which confirms the simulation
    code is correct.
    """
    h = horizon_days / TRADING_DAYS
    z = norm.ppf(1 - confidence)
    growth = np.exp((mu_p - 0.5 * sigma_p ** 2) * h + sigma_p * np.sqrt(h) * z)
    return float(1 - growth)


def var_backtest(portfolio_returns, var_1day, confidence=0.95):
    """
    Test the VaR number against what actually happened.

    The rule is simple: count the days when the actual loss was bigger than the
    VaR. These are "exceptions" or "breaches". If the 95% VaR is honest, about
    5% of days should breach it: no more (the model understates risk), and not
    far fewer (it overstates risk and wastes capital).

    Regulators do exactly this. Basel's traffic-light test counts breaches over
    250 trading days, with a bank's capital multiplier rising once it has too
    many.

    Parameters
    ----------
    portfolio_returns : daily returns of the portfolio over the test period
    var_1day          : the one-day VaR being tested, as a positive fraction
    confidence        : the confidence level that VaR was computed at. It must
                        be passed in rather than inferred from the test data,
                        because inferring it would make the test compare the
                        data with itself and always appear to pass.

    Returns a dict with the counts, plus the dates and sizes of the breaches
    so the notebook can mark them on a chart.
    """
    r = pd.Series(portfolio_returns).dropna()
    breaches = r[r < -var_1day]

    n_days = len(r)
    expected = n_days * (1 - confidence)

    return {
        "n_days": n_days,
        "n_breaches": len(breaches),
        "breach_rate": len(breaches) / n_days,
        "expected_rate": 1 - confidence,
        "breach_dates": breaches.index,
        "breach_returns": breaches,
        "expected_breaches": expected,
        "var_1day": var_1day,
        "confidence": confidence,
    }


# ===========================================================================
# 5. BLACK-SCHOLES (the closed-form benchmark)
# ===========================================================================
# Black-Scholes gives the exact price of a European option under GBM. We use
# it for two things: to check that the Monte Carlo pricer is correct, and to
# provide exact Greeks to compare the simulated Greeks against.
#
#     Call = S*N(d1) - K*e^(-rT)*N(d2)
#     Put  = K*e^(-rT)*N(-d2) - S*N(-d1)
#
#     d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
#     d2 = d1 - sigma*sqrt(T)
#
# N() is the cumulative normal distribution: norm.cdf() in Python.
# ---------------------------------------------------------------------------
def _d1_d2(S, K, T, r, sigma):
    """The two standardised distances used by every Black-Scholes formula."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Exact European option price.

    Reading the call formula: S*N(d1) is the expected value of the shares you
    receive if you exercise, and K*e^(-rT)*N(d2) is the present value of the
    strike you pay, with N(d2) being the risk-neutral probability of exercise.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    elif option_type == "put":
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    The Greeks: how the option's price responds to each input.

        delta = dPrice/dS      shares to hold to hedge one option
        gamma = d(delta)/dS    how fast delta changes (the rehedging cost)
        vega  = dPrice/dsigma  sensitivity to volatility
        theta = dPrice/dT      time decay
        rho   = dPrice/dr      sensitivity to interest rates

    Conventions used here (stated because they vary between textbooks and
    trading desks):
        - all values are RAW partial derivatives, per 1.0 change in the input
        - "vega_1pct" and "rho_1pct" rescale to a 1 percentage-point move,
          which is how a desk quotes them
        - "theta_per_day" is the loss of value over one calendar day
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)
    pdf_d1 = norm.pdf(d1)          # the bell curve's height at d1

    # Gamma and vega are identical for calls and puts: both depend only on how
    # much probability sits near the strike, not on which side you own.
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 - r * K * discount * norm.cdf(d2))
        rho = K * T * discount * norm.cdf(d2)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1.0          # always between -1 and 0
        theta = (-S * pdf_d1 * sigma / (2 * np.sqrt(T))
                 + r * K * discount * norm.cdf(-d2))
        rho = -K * T * discount * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "vega_1pct": float(vega * 0.01),
        "rho_1pct": float(rho * 0.01),
        "theta_per_day": float(theta / 365.0),
    }


# ===========================================================================
# 6. MONTE CARLO OPTION PRICING
# ===========================================================================
# The recipe is always the same three steps:
#   1. Simulate many price paths under the RISK-NEUTRAL measure (drift = r)
#   2. Work out the payoff on each path
#   3. Average the payoffs and discount back at the risk-free rate
#
# Its weakness is accuracy: the error shrinks only as 1/sqrt(N), so 100 times
# more paths gives 10 times more accuracy. Its strength is generality: step 2
# can be any payoff at all, including ones with no formula.
# ---------------------------------------------------------------------------
def option_payoffs(paths, K, kind="european_call"):
    """
    Payoff at expiry for each simulated path.

        european : uses only the FINAL price          max(S_T - K, 0)
        asian    : uses the AVERAGE price of the path max(avg(S) - K, 0)

    The Asian option is the reason Monte Carlo exists in derivatives pricing.
    The average of lognormal prices is not itself lognormal, so there is no
    exact Black-Scholes-style formula for the arithmetic Asian option. Averaging
    also makes it cheaper than a European option (an average wanders less than
    a final price) and much harder to manipulate near expiry, which is why
    Asians are common in commodity and currency hedging.
    """
    S_T = paths[:, -1]
    # The average excludes the starting price, which is known today and is not
    # part of the averaging period.
    S_avg = paths[:, 1:].mean(axis=1)

    if kind == "european_call":
        return np.maximum(S_T - K, 0.0)
    elif kind == "european_put":
        return np.maximum(K - S_T, 0.0)
    elif kind == "asian_call":
        return np.maximum(S_avg - K, 0.0)
    elif kind == "asian_put":
        return np.maximum(K - S_avg, 0.0)
    raise ValueError(f"Unknown option kind: {kind}")


def monte_carlo_option(S0, K, T, r, sigma, kind="european_call",
                       n_paths=100_000, n_steps=None, seed=42,
                       antithetic=False):
    """
    Price an option by simulation, with an honest error estimate.

    Because the price is an AVERAGE of random payoffs, it is itself random.
    Quoting it without a standard error is like quoting a poll without a margin
    of error. The standard error is  std(payoffs) / sqrt(N), and the 95%
    confidence interval is the price plus or minus 1.96 standard errors.

    Parameters
    ----------
    kind      : "european_call", "european_put", "asian_call", "asian_put"
    n_steps   : defaults to 1 for European options (only the final price
                matters, so simulating daily steps would waste time) and 252
                for Asian options (the whole path is needed).
    antithetic: mirror-image paths, for a more accurate price at the same cost.

    Returns
    -------
    dict with the price, standard error, 95% confidence interval and settings.
    """
    if n_steps is None:
        n_steps = 1 if kind.startswith("european") else TRADING_DAYS

    # Risk-neutral drift: the risk-free rate, NOT the expected return.
    paths = simulate_asset_paths(S0, r, sigma, T, n_steps, n_paths, seed,
                                 antithetic=antithetic)
    payoffs = option_payoffs(paths, K, kind)

    discounted = np.exp(-r * T) * payoffs
    price = float(discounted.mean())

    if antithetic:
        # With mirror-image pairs the paths are not independent, so the plain
        # standard deviation would understate the error. The correct unit of
        # observation is the AVERAGE of each pair.
        half = len(discounted) // 2
        pair_means = 0.5 * (discounted[:half] + discounted[half:])
        stderr = float(pair_means.std(ddof=1) / np.sqrt(half))
    else:
        stderr = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))

    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - 1.96 * stderr,
        "ci_high": price + 1.96 * stderr,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "kind": kind,
    }


def convergence_study(S0, K, T, r, sigma, kind="european_call",
                      path_counts=(100, 500, 1_000, 5_000, 10_000,
                                   50_000, 100_000, 500_000),
                      seed=42, antithetic=False):
    """
    Price the same option with increasing numbers of paths, to show the error
    shrinking as 1/sqrt(N).

    For a European option we also report the exact Black-Scholes price and the
    actual error, which is the clearest possible check that the simulation is
    unbiased: the errors should fall inside the confidence interval roughly 95%
    of the time, and shrink by a factor of about 3.16 (the square root of 10)
    each time the number of paths is multiplied by ten.
    """
    exact = None
    if kind.startswith("european"):
        exact = black_scholes_price(
            S0, K, T, r, sigma, "call" if "call" in kind else "put")

    rows = []
    for n in path_counts:
        # A different seed per run, so each row is an independent experiment
        # rather than the same random numbers reused.
        result = monte_carlo_option(S0, K, T, r, sigma, kind, n_paths=n,
                                    seed=seed + n, antithetic=antithetic)
        row = {
            "Paths": n,
            "MC price": result["price"],
            "Std error": result["stderr"],
            "CI low": result["ci_low"],
            "CI high": result["ci_high"],
        }
        if exact is not None:
            row["Black-Scholes"] = exact
            row["Error"] = result["price"] - exact
            row["Inside CI"] = result["ci_low"] <= exact <= result["ci_high"]
        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# 7. MONTE CARLO GREEKS
# ===========================================================================
# For European options the Greeks have formulas, so simulation is unnecessary.
# For an Asian option they do not, and a desk still has to hedge it. These
# functions show the standard simulation approaches and let us verify them on
# the European case, where the right answer is known.
# ---------------------------------------------------------------------------
def mc_greeks_bump(S0, K, T, r, sigma, kind="european_call",
                   n_paths=200_000, n_steps=None, seed=42,
                   bump_S=0.01, bump_sigma=0.01, bump_T=1 / 365, bump_r=0.0001):
    """
    Greeks by "bump and revalue": nudge one input, reprice, and measure how
    much the price moved. This is finite differences applied to a simulation.

        delta = [P(S+h) - P(S-h)] / (2h)
        gamma = [P(S+h) - 2*P(S) + P(S-h)] / h^2

    The critical detail is COMMON RANDOM NUMBERS: every repricing uses the same
    seed, so the same random shocks. Without this, the difference between two
    prices would be dominated by simulation noise rather than by the bump, and
    gamma in particular would be pure noise. With it, the noise largely cancels
    and the estimates are stable.

    bump_S is given as a FRACTION of the spot (0.01 = 1%), because a 1-rupee
    bump means something different for a Rs 50 stock and a Rs 5,000 stock.
    """
    if n_steps is None:
        n_steps = 1 if kind.startswith("european") else TRADING_DAYS

    def price_at(S=S0, vol=sigma, mat=T, rate=r):
        # Same seed every time = common random numbers.
        return monte_carlo_option(S, K, mat, rate, vol, kind,
                                  n_paths=n_paths, n_steps=n_steps,
                                  seed=seed)["price"]

    h = S0 * bump_S
    base = price_at()
    up, down = price_at(S=S0 + h), price_at(S=S0 - h)

    delta = (up - down) / (2 * h)
    gamma = (up - 2 * base + down) / (h ** 2)

    # Vega: bump volatility up and down by one percentage point.
    vega = (price_at(vol=sigma + bump_sigma)
            - price_at(vol=sigma - bump_sigma)) / (2 * bump_sigma)

    # Theta: shorten the time to expiry by one day and see what the option
    # loses. Note the direction: theta is the change in value as CALENDAR TIME
    # passes, which is the opposite sign to the change as MATURITY lengthens.
    # An option with less time left is worth less, so theta is normally
    # negative for a long position.
    theta = (price_at(mat=T - bump_T) - base) / bump_T if T > bump_T else np.nan

    # Rho: bump the interest rate.
    rho = (price_at(rate=r + bump_r) - price_at(rate=r - bump_r)) / (2 * bump_r)

    return {
        "price": base,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vega_1pct": vega * 0.01,
        "rho_1pct": rho * 0.01,
        "theta_per_day": theta / 365.0,
    }


def mc_delta_pathwise(S0, K, T, r, sigma, option_type="call",
                      n_paths=200_000, seed=42):
    """
    Delta by the "pathwise" method: differentiate the payoff itself instead of
    repricing twice.

    For a European call, the payoff is max(S_T - K, 0), and S_T is proportional
    to S_0, so:

        d(payoff)/dS_0 = 1{S_T > K} * S_T / S_0

    Taking the average of that (discounted) gives delta directly from a single
    simulation. It is both faster and much more accurate than bumping, because
    there is no cancellation of two nearly equal numbers.

    The catch, worth knowing: it requires the payoff to be differentiable in
    the usual sense almost everywhere. It works for calls and puts, but fails
    for a digital option, whose payoff jumps at the strike. Gamma also cannot
    be obtained this way for a call, because delta's derivative is a spike at
    the strike, so desks mix methods in practice.
    """
    paths = simulate_asset_paths(S0, r, sigma, T, n_steps=1,
                                 n_paths=n_paths, seed=seed)
    S_T = paths[:, -1]

    if option_type == "call":
        indicator = (S_T > K).astype(float)
    elif option_type == "put":
        indicator = -(S_T < K).astype(float)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    estimates = np.exp(-r * T) * indicator * S_T / S0
    delta = float(estimates.mean())
    stderr = float(estimates.std(ddof=1) / np.sqrt(n_paths))
    return {"delta": delta, "stderr": stderr}


# ===========================================================================
# 8. PROTECTIVE PUT (portfolio insurance)
# ===========================================================================
# A protective put is simply: hold the portfolio, and buy puts on it. The put
# pays out when the portfolio falls, which truncates the left tail of the
# distribution. The premium is the cost of that protection, paid whether or
# not it is ever needed.
#
# Two ways to do it, and the comparison is the interesting part:
#
#   1. A put on EACH stock. Protection exactly matches what you hold, but you
#      pay for eight separate options. Each one pays out when its own stock
#      falls, even if the portfolio as a whole is fine, so you are buying more
#      protection than the portfolio needs. Single-stock volatility is also
#      much higher than index volatility, and premium rises with volatility.
#
#   2. One put on the NIFTY, sized by the portfolio's beta. Far cheaper,
#      because index volatility is low (diversification has already removed
#      the stock-specific risk). But it only pays out if the INDEX falls. If
#      the basket falls while the Nifty rises, the hedge pays nothing. That
#      mismatch is BASIS RISK, and it is the central trade-off here.
#
# Financing convention used below: the premium is borrowed at the risk-free
# rate on day 0 and repaid at expiry, so the full amount stays invested in the
# portfolio and the cost appears as premium * e^(rT) at the end. This keeps
# the hedged and unhedged portfolios directly comparable.
# ---------------------------------------------------------------------------
def single_stock_put_hedge(sim, Sigma, weights, strike_pct, T, r):
    """
    Buy a put on every stock in the portfolio, each struck at `strike_pct` of
    its own starting price (0.95 = 5% out of the money).

    Because simulate_portfolio() returns price RELATIVES (final price divided
    by starting price), we can work entirely in relative terms: every stock
    starts at 1.0 and the strike is simply strike_pct. The number of units
    held of stock i is (w_i * V0) / S_i0, so its payoff contribution is
    w_i * V0 * max(strike_pct - relative_i, 0).
    """
    assets = list(weights.index)
    V0 = sim["initial"]
    relatives = sim["relatives"]

    # Each stock's annual volatility is the square root of its own variance.
    vols = pd.Series(np.sqrt(np.diag(np.asarray(Sigma))), index=sim["assets"])

    payoff = np.zeros(len(relatives))
    premium = 0.0

    for i, asset in enumerate(sim["assets"]):
        w = float(weights.get(asset, 0.0))
        if w <= 0:
            continue    # no holding, nothing to hedge

        # Payoff of this stock's put, in rupees, across all paths.
        payoff += w * V0 * np.maximum(strike_pct - relatives[:, i], 0.0)

        # Premium for the same position: Black-Scholes on a stock priced at
        # 1.0 with strike strike_pct, scaled by the rupees held in that stock.
        premium += w * V0 * black_scholes_price(
            1.0, strike_pct, T, r, float(vols[asset]), "put")

    return {"payoff": payoff, "premium": premium, "label": "Single-stock puts"}


def index_put_hedge(sim, index_name, index_vol, beta, strike_pct, T, r):
    """
    Buy ONE put on the Nifty 50, sized by the portfolio's beta.

    Hedge notional = beta * portfolio value. A portfolio with a beta of 1.3
    tends to fall 1.3% when the index falls 1%, so it needs 1.3 times its own
    value in index puts to offset that move.

    The index must be included in the simulation (with a weight of zero) so
    that its path is correlated with the stocks in the same way it was in the
    data. That correlation is exactly what decides whether the hedge works.
    """
    V0 = sim["initial"]
    idx = sim["assets"].index(index_name)
    index_relatives = sim["relatives"][:, idx]

    notional = beta * V0
    payoff = notional * np.maximum(strike_pct - index_relatives, 0.0)
    premium = notional * black_scholes_price(
        1.0, strike_pct, T, r, index_vol, "put")

    return {"payoff": payoff, "premium": premium,
            "label": f"Nifty put (beta {beta:.2f})"}


def hedged_outcomes(sim, hedge, r, T):
    """
    Combine the unhedged simulation with a hedge's payoffs and premium.

        hedged value = portfolio value + put payoff - premium * e^(rT)

    Returns the hedged terminal values, so the same risk metrics
    (VaR, CVaR, probability of loss) can be applied to both and compared.
    """
    terminal = sim["values"][:, -1]
    financing_cost = hedge["premium"] * np.exp(r * T)
    return terminal + hedge["payoff"] - financing_cost


def compare_hedges(sim, hedges, r, T, confidence=0.95):
    """
    A table comparing the unhedged portfolio with each hedge.

    The point of the table: a hedge should reduce CVaR a lot and reduce the
    mean a little. If it reduces the mean as much as it reduces the tail, it
    is not worth buying.
    """
    V0 = sim["initial"]
    rows = {"Unhedged": sim["values"][:, -1]}
    for hedge in hedges:
        rows[hedge["label"]] = hedged_outcomes(sim, hedge, r, T)

    premiums = {"Unhedged": 0.0}
    premiums.update({h["label"]: h["premium"] for h in hedges})

    table = {}
    for label, terminal in rows.items():
        growth = terminal / V0 - 1.0
        table[label] = {
            "Premium (% of value)": premiums[label] / V0,
            "Mean return": float(growth.mean()),
            "Median return": float(np.median(growth)),
            f"VaR {confidence:.0%}": value_at_risk(terminal, V0, confidence),
            f"CVaR {confidence:.0%}": conditional_value_at_risk(terminal, V0, confidence),
            "5th percentile": float(np.percentile(growth, 5)),
            "P(loss)": float((growth < 0).mean()),
            "Worst case": float(growth.min()),
        }
    return pd.DataFrame(table).T


# ===========================================================================
# 9. DELTA HEDGING
# ===========================================================================
# Black-Scholes assumes the option seller rehedges CONTINUOUSLY. In reality
# hedging happens at discrete times: daily, weekly, or when delta moves past a
# threshold. This section simulates a trader who has sold a call and hedges it
# at fixed intervals, and measures what that approximation costs.
#
# Two results come out of it, both worth being able to state in an interview:
#
#   1. With rehedging n times, the standard deviation of the hedging error
#      falls roughly as 1/sqrt(n) (Boyle & Emanuel, 1980). Hedging twice as
#      often only reduces the error by about 30%.
#   2. The profit or loss depends on REALISED volatility versus the volatility
#      the option was PRICED at. Selling at 40% and realising 25% is
#      profitable; the P&L accumulates through gamma, which is why traders
#      describe this as "gamma scalping" or being "short gamma".
# ---------------------------------------------------------------------------
def delta_hedge_simulation(S0, K, T, r, sigma_pricing, sigma_realised=None,
                           mu_real=None, rebalance_every=1,
                           n_steps=TRADING_DAYS, n_paths=2_000, seed=42,
                           option_type="call"):
    """
    Sell one option, hedge it, and see what is left at expiry.

    The mechanics, step by step:
      1. Sell the option and receive the Black-Scholes premium (priced at
         sigma_pricing). Put the cash in the bank at the risk-free rate.
      2. Buy `delta` shares as a hedge, borrowing the money to do so.
      3. At each rebalance date, recompute delta with the time remaining and
         adjust the shareholding, paying or receiving cash.
      4. At expiry, sell the shares, pay the option's payoff, and see what
         remains. Under perfect continuous hedging that remainder would be
         exactly zero.

    Parameters
    ----------
    sigma_pricing  : the volatility the option was sold at (implied volatility)
    sigma_realised : the volatility the stock actually moves at. Defaults to
                     sigma_pricing (a "fair" market). Set it lower to see the
                     seller profit, higher to see the seller lose.
    mu_real        : the real-world drift of the stock. Defaults to r. The
                     hedging error's SPREAD barely depends on this, which is
                     itself the point: delta hedging removes directional risk.
    rebalance_every: 1 = daily, 5 = weekly, 21 = monthly.

    Returns
    -------
    dict with the P&L per path (as a fraction of the premium received, so the
    numbers are comparable across settings), plus the premium and settings.
    """
    if sigma_realised is None:
        sigma_realised = sigma_pricing
    if mu_real is None:
        mu_real = r

    dt = T / n_steps

    # Step 1: the premium received for selling the option.
    premium = black_scholes_price(S0, K, T, r, sigma_pricing, option_type)

    # The stock's actual path uses the REALISED volatility and the real-world
    # drift; the hedge is computed with the PRICING volatility, because that is
    # all the trader knows when quoting.
    paths = simulate_asset_paths(S0, mu_real, sigma_realised, T, n_steps,
                                 n_paths, seed)

    # Step 2: the initial hedge.
    delta = np.full(n_paths, black_scholes_greeks(
        S0, K, T, r, sigma_pricing, option_type)["delta"])
    shares = delta.copy()
    cash = premium - shares * S0          # premium in, shares bought

    for step in range(1, n_steps):
        # Cash earns (or costs) interest every step, whether we trade or not.
        cash *= np.exp(r * dt)

        if step % rebalance_every != 0:
            continue                      # not a rebalancing date: do nothing

        S_t = paths[:, step]
        time_left = T - step * dt

        # New delta for every path at once.
        d1 = ((np.log(S_t / K) + (r + 0.5 * sigma_pricing ** 2) * time_left)
              / (sigma_pricing * np.sqrt(time_left)))
        new_delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0

        # Buy or sell the difference, paying for it out of cash.
        cash -= (new_delta - shares) * S_t
        shares = new_delta

    # Final step's interest.
    cash *= np.exp(r * dt)

    # Step 4: unwind. Sell the shares, settle the option.
    S_T = paths[:, -1]
    payoff = (np.maximum(S_T - K, 0.0) if option_type == "call"
              else np.maximum(K - S_T, 0.0))
    pnl = cash + shares * S_T - payoff

    return {
        "pnl": pnl,
        "pnl_pct_of_premium": pnl / premium,
        "premium": premium,
        "rebalance_every": rebalance_every,
        "n_rebalances": n_steps // rebalance_every,
        "sigma_pricing": sigma_pricing,
        "sigma_realised": sigma_realised,
        "terminal_prices": S_T,
    }


def hedging_error_study(S0, K, T, r, sigma, frequencies=(1, 5, 21),
                        n_steps=TRADING_DAYS, n_paths=2_000, seed=42,
                        option_type="call"):
    """
    Run the hedge at several rebalancing frequencies and tabulate the results.

    The column to look at is "Std dev (% of premium)". It should fall roughly
    in proportion to 1/sqrt(number of rebalances), which is the discrete-hedging
    result from Boyle & Emanuel (1980). The mean P&L stays near zero at every
    frequency: hedging less often is not biased, just noisier.
    """
    rows = []
    for freq in frequencies:
        res = delta_hedge_simulation(S0, K, T, r, sigma, rebalance_every=freq,
                                     n_steps=n_steps, n_paths=n_paths,
                                     seed=seed, option_type=option_type)
        pnl = res["pnl_pct_of_premium"]
        rows.append({
            "Rebalance every (days)": freq,
            "Number of rebalances": res["n_rebalances"],
            "Mean P&L (% of premium)": float(pnl.mean()),
            "Std dev (% of premium)": float(pnl.std(ddof=1)),
            "5th percentile": float(np.percentile(pnl, 5)),
            "95th percentile": float(np.percentile(pnl, 95)),
            "Worst path": float(pnl.min()),
        })
    return pd.DataFrame(rows)
