"""
portfolio.py
============

Helper functions for the Markowitz Efficient Frontier project.

The notebook (markowitz_frontier.ipynb) tells the story and draws the charts.
This file holds the reusable "engine" so that the notebook stays readable.

Everything here maps to something you already know from Excel / paper:

    Excel / paper                         Python (this file)
    -----------------------------------   -----------------------------------
    Price sheet                           download_prices()
    =P1/P0 - 1                            compute_returns()
    AVERAGE * 252, COVARIANCE.S * 252     annualise()
    SUMPRODUCT / MMULT for return & risk  portfolio_performance()
    Solver (min variance)                 min_variance_portfolio()
    Solver (max Sharpe)                   max_sharpe_portfolio()
    Solver for each target return         efficient_frontier()

Notation used throughout (same as the textbook):
    w     = vector of portfolio weights          (one number per stock)
    mu    = vector of expected annual returns    (one number per stock)
    Sigma = annual covariance matrix             (stocks x stocks table)
    rf    = risk-free rate (10-year G-Sec yield)

    Portfolio return      = w' * mu
    Portfolio variance    = w' * Sigma * w
    Portfolio volatility  = sqrt(variance)
    Sharpe ratio          = (return - rf) / volatility
"""

# ---------------------------------------------------------------------------
# Imports: bring in the libraries we need.
# ---------------------------------------------------------------------------
from pathlib import Path          # handles file paths in a Windows/Mac-safe way

import numpy as np                # numerical maths: vectors, matrices
import pandas as pd               # tables of data (like an Excel sheet)
from scipy.optimize import minimize          # the Python equivalent of Solver
from sklearn.covariance import LedoitWolf    # shrinkage covariance estimator

# Number of trading days in a year on the NSE (industry convention).
# Used to turn daily numbers into annual numbers.
TRADING_DAYS = 252

# Optimiser settings, shared by every optimisation below.
#   ftol    = how precise the answer must be (smaller = more precise)
#   maxiter = the maximum number of steps before giving up
SOLVER_OPTIONS = {"ftol": 1e-12, "maxiter": 1000}


# ===========================================================================
# 1. DATA
# ===========================================================================
def download_prices(tickers, start, end, cache_file=None):
    """
    Download daily adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers    : list of Yahoo symbols, e.g. ["HCLTECH.NS", "BEL.NS"]
    start, end : dates as text, e.g. "2023-09-01"
    cache_file : optional path to a CSV. If the file exists we read it instead
                 of downloading again. This makes the project reproducible:
                 anyone re-running it gets exactly the same numbers.

    Returns
    -------
    A pandas DataFrame: one row per trading day, one column per stock.
    """
    # If we already saved the data earlier, just load it from disk.
    if cache_file is not None and Path(cache_file).exists():
        print(f"Loading cached prices from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    # yfinance is imported here (not at the top) so the rest of the file
    # still works offline once the CSV exists.
    import yfinance as yf

    print(f"Downloading {len(tickers)} tickers from Yahoo Finance ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts prices for splits and dividends
        progress=False,     # hide the progress bar
    )

    # yf.download returns several price fields (Open, High, Low, Close, Volume).
    # We only need "Close", which is already adjusted because auto_adjust=True.
    prices = raw["Close"]

    # Keep the columns in the same order the user supplied them.
    prices = prices[tickers]

    # Save a copy so next time we don't need the internet.
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_file)
        print(f"Saved prices to {cache_file}")

    return prices


def check_price_jumps(prices, threshold=0.25):
    """
    Look for suspicious one-day moves (e.g. an un-adjusted stock split).

    A real stock rarely moves more than 20-25% in one day. If a split was not
    adjusted, the price would suddenly "fall off a cliff" (e.g. -80%), which
    would ruin our return and risk estimates.

    Parameters
    ----------
    prices    : DataFrame of prices
    threshold : flag any daily move bigger than this (0.25 = 25%)

    Returns
    -------
    A DataFrame listing every flagged (date, stock, daily return).
    """
    # Daily percentage change for every stock.
    daily = prices.pct_change()

    # .stack() turns the table into a long list of (date, stock) -> value,
    # which makes filtering easy.
    long_format = daily.stack()

    # Keep only moves whose absolute size is above the threshold.
    flagged = long_format[long_format.abs() > threshold]

    # Turn it into a neat table for printing.
    return flagged.rename("daily_return").reset_index()


# ===========================================================================
# 2. RETURNS AND ESTIMATES
# ===========================================================================
def compute_returns(prices):
    """
    Simple daily returns: r_t = P_t / P_(t-1) - 1   (same as Excel =B3/B2-1)

    Why simple returns and not log returns?
    A portfolio's return is the weighted SUM of the stocks' simple returns
    (w' * r). That is only exactly true for simple returns, and Markowitz
    relies on it, so simple returns are the correct choice here.
    """
    # pct_change() computes P_t / P_(t-1) - 1 for every column.
    # dropna() removes the first row (it has no previous price) and any
    # rows where a stock was not trading.
    return prices.pct_change().dropna()


def annualise(returns, cov_method="sample"):
    """
    Estimate the two inputs Markowitz needs: mu (expected returns) and
    Sigma (covariance matrix), both in ANNUAL terms.

    Parameters
    ----------
    returns    : DataFrame of daily returns
    cov_method : "sample"       -> ordinary covariance (Excel COVARIANCE.S)
                 "ledoit_wolf"  -> shrinkage covariance (see note below)

    Note on Ledoit-Wolf shrinkage
    -----------------------------
    The sample covariance matrix is noisy, especially with few years of data
    and many stocks. The optimiser treats that noise as if it were real and
    produces extreme weights ("error maximisation", Michaud 1989).

    Ledoit & Wolf (2004) fix this by blending the sample matrix with a simple,
    very stable "target" matrix:

        Sigma_shrunk = delta * Target + (1 - delta) * Sigma_sample

    The shrinkage intensity delta (between 0 and 1) is chosen automatically
    to minimise the expected estimation error.

    Returns
    -------
    mu    : pandas Series, annual expected return per stock
    Sigma : pandas DataFrame, annual covariance matrix
    """
    # Expected return = average daily return x 252 (Excel: AVERAGE * 252)
    mu = returns.mean() * TRADING_DAYS

    if cov_method == "sample":
        # Ordinary sample covariance, scaled to annual (Excel: COVARIANCE.S * 252)
        Sigma = returns.cov() * TRADING_DAYS

    elif cov_method == "ledoit_wolf":
        # Fit the Ledoit-Wolf estimator on the daily returns.
        lw = LedoitWolf().fit(returns.values)
        # lw.covariance_ is a plain NumPy array; wrap it back into a labelled
        # table so we keep the stock names, then scale to annual.
        Sigma = pd.DataFrame(
            lw.covariance_ * TRADING_DAYS,
            index=returns.columns,
            columns=returns.columns,
        )
        # Print how much shrinkage was applied (useful for the write-up).
        print(f"Ledoit-Wolf shrinkage intensity (delta): {lw.shrinkage_:.3f}")

    else:
        raise ValueError("cov_method must be 'sample' or 'ledoit_wolf'")

    return mu, Sigma


# ===========================================================================
# 3. PORTFOLIO MATHS
# ===========================================================================
def portfolio_performance(weights, mu, Sigma, rf=0.0):
    """
    Return, volatility and Sharpe ratio of ONE portfolio.

    The '@' symbol is matrix multiplication in Python (Excel: MMULT).
        weights @ mu              -> w' * mu           (portfolio return)
        weights @ Sigma @ weights -> w' * Sigma * w    (portfolio variance)
    """
    # np.asarray makes sure we are working with plain numbers (no labels).
    w = np.asarray(weights)
    ret = w @ np.asarray(mu)
    vol = np.sqrt(w @ np.asarray(Sigma) @ w)
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


def random_portfolios(mu, Sigma, n_portfolios=10_000, rf=0.0, seed=42):
    """
    Monte Carlo simulation: generate many random long-only portfolios.

    This is NOT how we find the frontier (the optimiser does that). It is a
    visual check: the random dots should all sit on or BELOW the frontier.

    Parameters
    ----------
    n_portfolios : how many random portfolios to create
    seed         : fixes the random numbers so results are reproducible

    Returns
    -------
    DataFrame with columns: Return, Volatility, Sharpe, plus one column of
    weights per stock.
    """
    rng = np.random.default_rng(seed)     # random number generator
    n_assets = len(mu)

    # Dirichlet distribution: produces random weights that are all >= 0
    # and always add up to exactly 1. Perfect for long-only portfolios.
    weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    # Compute return and volatility for ALL portfolios at once (vectorised).
    # This is much faster than a loop over 10,000 portfolios.
    rets = weights @ mu.values
    # einsum computes w' * Sigma * w for every row of 'weights' in one go.
    vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, Sigma.values, weights))
    sharpes = (rets - rf) / vols

    # Put everything in one table.
    results = pd.DataFrame(weights, columns=mu.index)
    results.insert(0, "Sharpe", sharpes)
    results.insert(0, "Volatility", vols)
    results.insert(0, "Return", rets)
    return results


# ===========================================================================
# 4. OPTIMISATION (the "Solver" part)
# ===========================================================================
def _base_setup(n_assets, max_weight):
    """
    Common Solver settings shared by all optimisations.

    Returns
    -------
    start       : starting guess (equal weights: 1/n each)
    bounds      : each weight must be between 0 and max_weight
                  (0 = no short selling, max_weight = concentration cap)
    constraints : weights must add up to 1 (fully invested)
    """
    start = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, max_weight)] * n_assets
    # "eq" means the function must equal zero: sum(w) - 1 = 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    return start, bounds, constraints


def min_variance_portfolio(mu, Sigma, max_weight=1.0):
    """
    Global Minimum Variance (GMV) portfolio: the left-most point of the
    frontier. It ignores expected returns entirely and only minimises risk.

        minimise   w' Sigma w
        subject to sum(w) = 1,  0 <= w <= max_weight
    """
    n = len(mu)
    start, bounds, constraints = _base_setup(n, max_weight)

    # The objective function the optimiser tries to make as small as possible.
    def variance(w):
        return w @ Sigma.values @ w

    # SLSQP = Sequential Least Squares Programming, a standard algorithm for
    # problems with both bounds and equality constraints (like Excel's GRG).
    result = minimize(variance, start, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options=SOLVER_OPTIONS)

    # Return the weights as a labelled Series (stock name -> weight).
    return pd.Series(result.x, index=mu.index)


def max_sharpe_portfolio(mu, Sigma, rf, max_weight=1.0):
    """
    Tangency (Maximum Sharpe Ratio) portfolio: the point where the Capital
    Market Line touches the frontier.

        maximise   (w' mu - rf) / sqrt(w' Sigma w)
        subject to sum(w) = 1,  0 <= w <= max_weight

    Optimisers only MINIMISE, so we minimise the NEGATIVE Sharpe ratio,
    which is the same as maximising the Sharpe ratio.
    """
    n = len(mu)
    start, bounds, constraints = _base_setup(n, max_weight)

    def negative_sharpe(w):
        ret, vol, sharpe = portfolio_performance(w, mu, Sigma, rf)
        return -sharpe

    result = minimize(negative_sharpe, start, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options=SOLVER_OPTIONS)
    return pd.Series(result.x, index=mu.index)


def efficient_return(mu, Sigma, target_return, max_weight=1.0):
    """
    The minimum-risk portfolio that earns exactly 'target_return'.
    Running this for many targets traces out the efficient frontier.

        minimise   w' Sigma w
        subject to w' mu = target_return
                   sum(w) = 1,  0 <= w <= max_weight
    """
    n = len(mu)
    start, bounds, constraints = _base_setup(n, max_weight)

    # Add the extra constraint: portfolio return must equal the target.
    constraints = constraints + [
        {"type": "eq", "fun": lambda w: w @ mu.values - target_return}
    ]

    def variance(w):
        return w @ Sigma.values @ w

    result = minimize(variance, start, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options=SOLVER_OPTIONS)

    # result.success tells us whether the optimiser found a valid answer.
    # (Some targets are impossible, e.g. above the best stock's return.)
    if not result.success:
        return None
    return pd.Series(result.x, index=mu.index)


def efficient_frontier(mu, Sigma, n_points=60, max_weight=1.0):
    """
    Trace the efficient frontier.

    Steps:
      1. Start at the Global Minimum Variance portfolio's return
         (anything below it is on the INEFFICIENT lower half).
      2. End at the highest achievable return.
      3. For n_points targets in between, solve efficient_return().

    Returns
    -------
    DataFrame with columns Return, Volatility, plus the weights of each
    frontier portfolio (one row per point on the curve).
    """
    # Lowest return on the efficient part of the curve.
    gmv = min_variance_portfolio(mu, Sigma, max_weight)
    low = gmv @ mu

    # Highest achievable return. With no cap this is simply the best stock.
    # With a cap (e.g. 40%), it is found by filling the best stocks first.
    high = _max_achievable_return(mu, max_weight)

    rows = []
    # np.linspace creates n_points evenly spaced target returns.
    for target in np.linspace(low, high, n_points):
        w = efficient_return(mu, Sigma, target, max_weight)
        if w is None:          # skip targets the optimiser could not solve
            continue
        ret, vol, _ = portfolio_performance(w, mu, Sigma)
        rows.append({"Return": ret, "Volatility": vol, **w.to_dict()})

    return pd.DataFrame(rows)


def _max_achievable_return(mu, max_weight):
    """
    Highest return possible under the weight cap: put max_weight into the
    best stock, then the next best, and so on until weights sum to 1.
    (Small helper so the frontier does not ask for impossible targets.)
    """
    remaining = 1.0
    total = 0.0
    for r in mu.sort_values(ascending=False):
        take = min(max_weight, remaining)
        total += take * r
        remaining -= take
        if remaining <= 1e-12:
            break
    # Stay a hair below the true maximum so the optimiser can solve it.
    return total - 1e-6


# ===========================================================================
# 5. OUT-OF-SAMPLE EVALUATION (back-test)
# ===========================================================================
def buy_and_hold_value(prices, weights, initial=100.0):
    """
    Value of a portfolio that is bought on the first day and then held
    (no rebalancing), starting from 'initial' (e.g. Rs 100).

    Each stock's value = initial * weight * (P_t / P_0)
    Portfolio value    = sum over stocks
    """
    # Normalise every price series so it starts at 1.0
    growth = prices / prices.iloc[0]
    # Multiply each stock's growth by its weight and add across stocks.
    # .values makes sure the weights line up with the columns in order.
    return initial * (growth * weights[prices.columns].values).sum(axis=1)


def performance_summary(value_series, rf):
    """
    Key performance statistics for a portfolio value series.

    Returns a dict with:
      Annual Return   : compound annual growth rate (CAGR)
      Annual Vol      : annualised standard deviation of daily returns
      Sharpe          : (Annual Return - rf) / Annual Vol
      Max Drawdown    : worst peak-to-trough fall (e.g. -0.30 = -30%)
      Total Return    : end value / start value - 1
    """
    daily = value_series.pct_change().dropna()
    n_days = len(daily)

    total_return = value_series.iloc[-1] / value_series.iloc[0] - 1
    # CAGR: (1 + total)^(252 / days) - 1
    annual_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1
    annual_vol = daily.std() * np.sqrt(TRADING_DAYS)
    sharpe = (annual_return - rf) / annual_vol

    # Drawdown: how far below its previous highest value the portfolio is.
    running_peak = value_series.cummax()
    drawdown = value_series / running_peak - 1
    max_drawdown = drawdown.min()

    return {
        "Annual Return": annual_return,
        "Annual Vol": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Total Return": total_return,
    }
