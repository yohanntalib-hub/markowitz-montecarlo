"""
app.py
======

Streamlit dashboard for the Monte Carlo notebook.

The notebook tells the story with fixed assumptions. This dashboard lets a
reader change those assumptions and watch the answers move, which is the
fastest way to build intuition for what actually drives the numbers.

Run it locally with:

    .venv\\Scripts\\streamlit.exe run app.py

Streamlit in one paragraph, since it is the only unfamiliar library here: the
whole script re-runs from top to bottom every time a widget changes. That
would mean re-simulating on every click, so the expensive functions are marked
with @st.cache_data, which stores the result and returns it instantly when the
inputs are unchanged.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import streamlit as st

import portfolio as pf
import simulation as sim

# ---------------------------------------------------------------------------
# Page setup and constants (kept identical to the notebook)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Monte Carlo portfolio risk: India AI basket",
                   page_icon="🎲", layout="wide")

TICKERS = ["HCLTECH.NS", "TATAELXSI.NS", "KAYNES.NS", "CGPOWER.NS",
           "DIXON.NS", "NETWEB.NS", "E2E.NS", "TATACOMM.NS"]
NAMES = {t: t.replace(".NS", "") for t in TICKERS}
START, END, SPLIT_DATE = "2023-09-01", "2026-09-17", "2025-09-01"
RF = 0.07
EQUITY_PREMIUM = 0.055
INITIAL_VALUE = 1_000_000

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "lines.linewidth": 2,
    "font.family": ["Segoe UI", "DejaVu Sans"], "figure.dpi": 110,
})


def pct_axis(ax, which="both"):
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))


def rupee_axis(ax, which="x"):
    fmt = mtick.FuncFormatter(lambda v, _: f"{v/100000:,.1f}L")
    (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------
# Data and portfolios: computed once, then cached
# ---------------------------------------------------------------------------
@st.cache_data
def load_everything():
    """
    Load the cached prices, rebuild the three portfolios from the training
    period, and estimate everything the tabs need.

    @st.cache_data means this runs once per session, not on every click.
    """
    prices = pf.download_prices(TICKERS, START, END, cache_file="data/prices.csv")
    prices = prices.rename(columns=NAMES).dropna()

    bench = pf.download_prices(["^NSEI"], START, END, cache_file="data/nifty50.csv")
    bench = bench.rename(columns={"^NSEI": "NIFTY"}).reindex(prices.index).ffill()

    all_prices = prices.join(bench)
    train_all = all_prices.loc[all_prices.index < SPLIT_DATE]
    test_all = all_prices.loc[all_prices.index >= SPLIT_DATE]

    train_returns = pf.compute_returns(train_all)
    stock_returns = train_returns[prices.columns]
    bench_returns = train_returns["NIFTY"]

    mu_hist, Sigma = pf.annualise(stock_returns)
    betas = sim.compute_betas(stock_returns, bench_returns)
    mu_capm, _ = sim.capm_drift(stock_returns, bench_returns, RF, EQUITY_PREMIUM)
    mu_flat = sim.flat_drift(prices.columns, RF, EQUITY_PREMIUM)

    portfolios = {
        "GMV (minimum variance)": pf.min_variance_portfolio(mu_hist, Sigma),
        "Max Sharpe (tangency)": pf.max_sharpe_portfolio(mu_hist, Sigma, RF),
        "Equal weight (1/N)": pd.Series(1 / len(prices.columns),
                                        index=prices.columns),
    }

    return {
        "prices": prices,
        "Sigma": Sigma,
        "Sigma_all": train_returns.cov() * 252,
        "mu": {"CAPM (recommended)": mu_capm,
               "Historical average": mu_hist,
               "Flat (risk-free + premium)": mu_flat},
        "mu_nifty_hist": bench_returns.mean() * 252,
        "betas": betas,
        "portfolios": portfolios,
        "stock_returns": stock_returns,
        "test_returns": pf.compute_returns(test_all)[prices.columns],
        "split_date": SPLIT_DATE,
    }


@st.cache_data
def run_simulation(portfolio_name, drift_name, horizon_years, n_paths, seed):
    """
    Simulate one portfolio. Cached on its arguments, so moving an unrelated
    slider (for example the confidence level) does not re-run it.

    Note what is passed in: plain strings and numbers, not DataFrames. Cache
    keys have to be hashable, and this also keeps the cache small.
    """
    data = load_everything()
    Sigma_all = data["Sigma_all"]

    mu_stocks = data["mu"][drift_name]
    mu_all = mu_stocks.copy()
    mu_all["NIFTY"] = (data["mu_nifty_hist"] if drift_name == "Historical average"
                       else RF + EQUITY_PREMIUM)
    mu_all = mu_all[Sigma_all.columns]

    weights = data["portfolios"][portfolio_name].reindex(
        Sigma_all.columns).fillna(0.0)

    n_steps = max(21, int(round(252 * horizon_years)))
    return sim.simulate_portfolio(mu_all, Sigma_all, weights,
                                  horizon_years=horizon_years, n_steps=n_steps,
                                  n_paths=n_paths, initial_value=INITIAL_VALUE,
                                  seed=seed)


data = load_everything()

# ---------------------------------------------------------------------------
# Sidebar: the controls shared by every tab
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

portfolio_name = st.sidebar.selectbox(
    "Portfolio", list(data["portfolios"].keys()), index=1,
    help="Weights come from the Markowitz notebook, estimated on Sep 2023 - Aug 2025.")

drift_name = st.sidebar.selectbox(
    "Expected returns (drift)", list(data["mu"].keys()), index=0,
    help="The single most important assumption in the whole simulation.")

horizon_months = st.sidebar.slider("Horizon (months)", 1, 36, 12, step=1)
horizon_years = horizon_months / 12

n_paths = st.sidebar.select_slider(
    "Simulated paths", options=[1_000, 5_000, 10_000, 20_000, 50_000],
    value=10_000,
    help="More paths means a more precise answer, and a slower one. "
         "The error falls as 1/sqrt(N).")

confidence = st.sidebar.select_slider(
    "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
    format_func=lambda c: f"{c:.0%}")

seed = st.sidebar.number_input("Random seed", value=42, step=1,
                               help="Change it to draw a different set of "
                                    "random futures and see how stable the "
                                    "numbers are.")

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Rs {INITIAL_VALUE:,} invested, buy and hold, no rebalancing. "
    f"Risk-free rate {RF:.1%} (10-year G-Sec). Prices to 16 Sep 2026."
)

# A warning that appears exactly when it is relevant: the historical drift is
# the assumption most likely to mislead, so the dashboard says so on the spot.
if drift_name == "Historical average":
    weights = data["portfolios"][portfolio_name]
    implied = float(weights @ data["mu"]["Historical average"][weights.index])
    st.sidebar.warning(
        f"**Historical drift selected.** This portfolio is being simulated "
        f"with an expected return of **{implied:.0%} a year**, the average of "
        f"a two-year AI rally rather than a forecast. Expect the risk numbers "
        f"to look unrealistically comfortable: almost every simulated future "
        f"makes money. Chopra & Ziemba (1993) found errors in expected returns "
        f"do about ten times the damage of errors in variances."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Monte Carlo risk, option pricing and hedging")
st.caption(
    "Eight NSE-listed stocks across India's AI and semiconductor supply chain. "
    "Portfolio weights from Markowitz mean-variance optimisation; everything "
    "here simulates what could happen to them next."
)

tab_risk, tab_options, tab_greeks, tab_hedge, tab_delta = st.tabs(
    ["Portfolio risk", "Option pricer", "Greeks", "Protective put",
     "Delta hedging"])

# ===========================================================================
# TAB 1: PORTFOLIO RISK
# ===========================================================================
with tab_risk:
    s = run_simulation(portfolio_name, drift_name, horizon_years, n_paths, seed)
    terminal = s["values"][:, -1]
    growth = terminal / INITIAL_VALUE - 1

    var = sim.value_at_risk(terminal, INITIAL_VALUE, confidence)
    cvar = sim.conditional_value_at_risk(terminal, INITIAL_VALUE, confidence)
    drawdowns = sim.max_drawdown_paths(s["values"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR {confidence:.0%}", f"{var:.1%}",
              help="The loss exceeded in the worst (1 - confidence) of futures. "
                   "A threshold, not a worst case.")
    c2.metric(f"CVaR {confidence:.0%}", f"{cvar:.1%}",
              help="The average loss in those worst cases. Always larger than "
                   "VaR, and the more informative of the two.")
    c3.metric("Probability of loss", f"{(growth < 0).mean():.0%}")
    c4.metric("Median max drawdown", f"{np.median(drawdowns):.1%}",
              help="The worst peak-to-trough fall during the period, in the "
                   "typical simulated future.")

    st.markdown(
        f"In **{(1-confidence)*100:.0f}%** of the {n_paths:,} simulated futures, "
        f"Rs {INITIAL_VALUE:,} falls below **Rs {INITIAL_VALUE*(1-var):,.0f}**. "
        f"When it does, the average outcome is **Rs {INITIAL_VALUE*(1-cvar):,.0f}**."
    )

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        days = np.arange(s["values"].shape[1])
        for i in range(min(200, n_paths)):
            ax.plot(days, s["values"][i], color=SERIES[0], alpha=0.06,
                    linewidth=0.6)
        for lo, hi, shade in [(5, 95, 0.18), (25, 75, 0.3)]:
            ax.fill_between(days, np.percentile(s["values"], lo, axis=0),
                            np.percentile(s["values"], hi, axis=0),
                            color=SERIES[1], alpha=shade, linewidth=0)
        ax.plot(days, np.percentile(s["values"], 50, axis=0), color=SERIES[1],
                label="Median")
        ax.axhline(INITIAL_VALUE, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title("Simulated futures")
        ax.set_xlabel("Trading days ahead")
        ax.set_xlim(0, days[-1])
        rupee_axis(ax, "y")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    with right:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        cut = INITIAL_VALUE * (1 - var)
        ax.hist(terminal[terminal >= cut], bins=60, color=SERIES[0], alpha=0.85)
        ax.hist(terminal[terminal < cut], bins=25, color=SERIES[7], alpha=0.9,
                label=f"Worst {1-confidence:.0%}")
        ax.axvline(INITIAL_VALUE, color=MUTED, linestyle="--", linewidth=1)
        ax.axvline(cut, color=SERIES[7], linewidth=1.5)
        ax.set_title("Where the money ends up")
        ax.set_xlabel("Value at the end of the horizon")
        ax.set_xlim(0, np.percentile(terminal, 99.5))
        rupee_axis(ax, "x")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    with st.expander("Full statistics, and the same run for all three portfolios"):
        table = pd.DataFrame({
            name: sim.simulation_summary(
                run_simulation(name, drift_name, horizon_years, n_paths, seed),
                RF, confidence_levels=(confidence,))
            for name in data["portfolios"]
        }).T
        st.dataframe(table.style.format("{:.2%}"), width="stretch")

    with st.expander("Is the model honest? Historical VaR and a backtest"):
        st.markdown(
            "GBM assumes normally distributed returns. Real returns have fat "
            "tails. Two checks: compare the model's VaR with VaR taken "
            "straight from past returns, then count how often the **test "
            "period** (Sep 2025 onwards, never seen by the model) broke it."
        )
        w = data["portfolios"][portfolio_name]
        mu_p = float(w @ data["mu"][drift_name][w.index])
        sigma_p = float(np.sqrt(w @ data["Sigma"] @ w))
        train_port = (data["stock_returns"] * w).sum(axis=1)
        test_port = (data["test_returns"] * w).sum(axis=1)

        rows = []
        for h in (1, 10):
            rows.append({
                "Horizon (days)": h,
                "GBM (model)": sim.gbm_var(mu_p, sigma_p, confidence, h),
                "Historical": sim.historical_var(train_port, confidence, h),
            })
        var_df = pd.DataFrame(rows)
        var_df["Model / Historical"] = var_df["GBM (model)"] / var_df["Historical"]
        st.dataframe(var_df.style.format({"GBM (model)": "{:.2%}",
                                          "Historical": "{:.2%}",
                                          "Model / Historical": "{:.2f}"}),
                     width="stretch", hide_index=True)

        bt = sim.var_backtest(test_port,
                              sim.gbm_var(mu_p, sigma_p, confidence, 1),
                              confidence)
        st.markdown(
            f"**Backtest:** the 1-day {confidence:.0%} VaR was breached on "
            f"**{bt['n_breaches']} of {bt['n_days']}** test days "
            f"({bt['breach_rate']:.1%}), against **{bt['expected_breaches']:.0f}** "
            f"expected ({1-confidence:.0%}). "
            + ("More breaches than allowed: the model is understating risk."
               if bt["n_breaches"] > bt["expected_breaches"] else
               "Within the expected range.")
        )

# ===========================================================================
# TAB 2: OPTION PRICER
# ===========================================================================
with tab_options:
    st.markdown(
        "Options are priced under the **risk-neutral measure**: the drift is "
        "the risk-free rate, never the stock's expected return. Two portfolios "
        "with the same payoff must cost the same, and that argument never "
        "mentions mu. The sidebar's drift setting deliberately has no effect here."
    )

    c1, c2, c3 = st.columns(3)
    underlying = c1.selectbox("Underlying", list(data["prices"].columns), index=4)
    option_family = c2.selectbox("Option type",
                                 ["European call", "European put",
                                  "Asian call", "Asian put"])
    maturity = c3.slider("Maturity (years)", 0.25, 2.0, 1.0, step=0.25)

    spot = float(data["prices"][underlying].loc[:SPLIT_DATE].iloc[-1])
    hist_vol = float(np.sqrt(data["Sigma"].loc[underlying, underlying]))

    c1, c2, c3 = st.columns(3)
    strike_pct = c1.slider("Strike (% of spot)", 70, 130, 100, step=5) / 100
    vol = c2.slider("Volatility", 0.10, 1.20, round(hist_vol, 2), step=0.01,
                    help=f"Historical volatility for {underlying} is "
                         f"{hist_vol:.1%}. NSE implied volatility is not "
                         f"freely available, so history is the default.")
    option_paths = c3.select_slider("Paths",
                                    options=[1_000, 10_000, 50_000, 100_000,
                                             200_000], value=50_000)

    strike = strike_pct * spot
    kind = option_family.lower().replace(" ", "_")
    opt_type = "call" if "call" in kind else "put"

    mc = sim.monte_carlo_option(spot, strike, maturity, RF, vol, kind,
                                n_paths=option_paths, seed=int(seed))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"Rs {spot:,.0f}")
    c2.metric("Strike", f"Rs {strike:,.0f}")
    c3.metric("Monte Carlo price", f"Rs {mc['price']:,.2f}",
              help=f"95% confidence interval: Rs {mc['ci_low']:,.2f} to "
                   f"Rs {mc['ci_high']:,.2f}")
    if kind.startswith("european"):
        bs = sim.black_scholes_price(spot, strike, maturity, RF, vol, opt_type)
        c4.metric("Black-Scholes (exact)", f"Rs {bs:,.2f}",
                  delta=f"{mc['price'] - bs:+,.2f} vs simulation")
    else:
        euro_kind = kind.replace("asian", "european")
        euro = sim.black_scholes_price(spot, strike, maturity, RF, vol, opt_type)
        c4.metric("European equivalent", f"Rs {euro:,.2f}",
                  delta=f"{mc['price'] - euro:+,.2f} vs Asian",
                  help="The Asian option is cheaper: an average price varies "
                       "less than a final price, giving an effective "
                       "volatility of about sigma/sqrt(3).")

    st.caption(
        f"Simulated price Rs {mc['price']:,.2f} +/- {1.96*mc['stderr']:,.2f} "
        f"(95% CI, {option_paths:,} paths). The price is an average of random "
        f"payoffs, so it is itself random: quoting it without an error bar "
        f"would be like quoting a poll with no margin of error."
    )

    if kind.startswith("european"):
        st.subheader("Convergence")
        st.markdown(
            "Monte Carlo error falls as $1/\\sqrt{N}$: 100 times more paths "
            "buys only 10 times the accuracy. The exact price should stay "
            "inside the confidence band, which is the evidence the estimator "
            "is unbiased rather than merely precise."
        )

        @st.cache_data
        def cached_convergence(spot, strike, maturity, vol, kind, seed):
            return sim.convergence_study(spot, strike, maturity, RF, vol, kind,
                                         path_counts=(100, 500, 1_000, 5_000,
                                                      10_000, 50_000, 100_000),
                                         seed=int(seed))

        conv = cached_convergence(spot, strike, maturity, vol, kind, seed)

        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.fill_between(conv["Paths"], conv["CI low"], conv["CI high"],
                            color=SERIES[0], alpha=0.2, label="95% CI")
            ax.plot(conv["Paths"], conv["MC price"], color=SERIES[0],
                    marker="o", markersize=4, label="Monte Carlo")
            ax.axhline(conv["Black-Scholes"].iloc[0], color=SERIES[1],
                       linestyle="--", label="Black-Scholes")
            ax.set_xscale("log")
            ax.set_xlabel("Paths (log scale)")
            ax.set_ylabel("Price (Rs)")
            ax.set_title("Converging on the exact price")
            ax.legend(fontsize=8)
            st.pyplot(fig, clear_figure=True)
        with right:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            ax.plot(conv["Paths"], conv["Std error"], color=SERIES[0],
                    marker="o", markersize=4, label="Actual")
            ref = conv["Std error"].iloc[0] * np.sqrt(
                conv["Paths"].iloc[0] / conv["Paths"])
            ax.plot(conv["Paths"], ref, color=SERIES[1], linestyle="--",
                    label="1/sqrt(N)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("Paths (log scale)")
            ax.set_ylabel("Standard error")
            ax.set_title("A straight line on log axes")
            ax.legend(fontsize=8)
            st.pyplot(fig, clear_figure=True)

        st.dataframe(conv.style.format({
            "MC price": "{:,.3f}", "Std error": "{:,.3f}", "CI low": "{:,.3f}",
            "CI high": "{:,.3f}", "Black-Scholes": "{:,.3f}",
            "Error": "{:+,.3f}"}), width="stretch", hide_index=True)

# ===========================================================================
# TAB 3: GREEKS
# ===========================================================================
with tab_greeks:
    st.markdown(
        "The Greeks say how the price responds to each input, which is how a "
        "position is hedged. European Greeks have exact formulas, so they are "
        "used here to check the simulated ones. Asian Greeks have no formula, "
        "which is the point: a desk that sells them still has to hedge them."
    )

    c1, c2, c3 = st.columns(3)
    g_underlying = c1.selectbox("Underlying ", list(data["prices"].columns),
                                index=4, key="greek_underlying")
    g_family = c2.selectbox("Option", ["European call", "European put",
                                       "Asian call", "Asian put"],
                            key="greek_family")
    g_strike_pct = c3.slider("Strike (% of spot)", 70, 130, 100, step=5,
                             key="greek_strike") / 100

    g_spot = float(data["prices"][g_underlying].loc[:SPLIT_DATE].iloc[-1])
    g_vol = float(np.sqrt(data["Sigma"].loc[g_underlying, g_underlying]))
    g_strike = g_strike_pct * g_spot
    g_kind = g_family.lower().replace(" ", "_")
    g_type = "call" if "call" in g_kind else "put"

    @st.cache_data
    def cached_greeks(spot, strike, vol, kind, seed):
        return sim.mc_greeks_bump(spot, strike, 1.0, RF, vol, kind,
                                  n_paths=100_000,
                                  n_steps=1 if kind.startswith("european") else 252,
                                  seed=int(seed))

    mc_g = cached_greeks(g_spot, g_strike, g_vol, g_kind, seed)

    if g_kind.startswith("european"):
        bs_g = sim.black_scholes_greeks(g_spot, g_strike, 1.0, RF, g_vol, g_type)
        table = pd.DataFrame({
            "Black-Scholes (exact)": {k: bs_g[k] for k in
                                      ["delta", "gamma", "vega", "theta", "rho"]},
            "Monte Carlo (bump)": {k: mc_g[k] for k in
                                   ["delta", "gamma", "vega", "theta", "rho"]},
        })
        table["Difference"] = (table["Monte Carlo (bump)"]
                               - table["Black-Scholes (exact)"])
        st.dataframe(table.style.format("{:,.4f}"), width="stretch")

        pw = sim.mc_delta_pathwise(g_spot, g_strike, 1.0, RF, g_vol, g_type,
                                   n_paths=100_000, seed=int(seed))
        st.caption(
            f"Pathwise delta {pw['delta']:.4f} (+/- {1.96*pw['stderr']:.4f}) "
            f"against the exact {bs_g['delta']:.4f}. The pathwise method "
            f"differentiates the payoff instead of repricing twice, so it is "
            f"faster and more accurate, but it needs a payoff without jumps."
        )
    else:
        st.dataframe(pd.DataFrame({"Monte Carlo (bump)": {
            k: mc_g[k] for k in ["delta", "gamma", "vega", "theta", "rho"]}
        }).style.format("{:,.4f}"), width="stretch")
        st.caption("No closed form exists for an arithmetic Asian option, so "
                   "there is nothing to check these against: simulation is "
                   "the answer, not the approximation.")

    st.subheader("How the Greeks change with the spot price")
    spots = np.linspace(0.6 * g_spot, 1.4 * g_spot, 60)
    curves = pd.DataFrame(
        [sim.black_scholes_greeks(s_, g_strike, 1.0, RF, g_vol, g_type)
         for s_ in spots], index=spots)

    cols = st.columns(3)
    for col, greek, title in zip(
            cols, ["delta", "gamma", "vega"],
            ["Delta: shares per option", "Gamma: how fast delta moves",
             "Vega: sensitivity to volatility"]):
        with col:
            fig, ax = plt.subplots(figsize=(4.4, 3.2))
            ax.plot(spots, curves[greek], color=SERIES[0])
            ax.axvline(g_strike, color=MUTED, linestyle="--", linewidth=1)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Spot (Rs)")
            st.pyplot(fig, clear_figure=True)

    st.caption("Gamma and vega both peak at the strike: that is where the "
               "outcome is least certain, so hedging costs the most and "
               "volatility matters most. Both fall to zero once an option is "
               "far enough in or out of the money for its fate to be settled.")

# ===========================================================================
# TAB 4: PROTECTIVE PUT
# ===========================================================================
with tab_hedge:
    st.markdown(
        "Hold the portfolio, buy puts on it. The put cuts off the left tail; "
        "the premium is paid whether or not it is needed. Two ways to do it, "
        "and the comparison is the interesting part."
    )

    c1, c2 = st.columns([1, 2])
    hedge_strike = c1.slider("Put strike (% of value)", 80, 100, 95, step=5) / 100
    c2.markdown(
        "**Single-stock puts** match the holdings exactly but cost far more, "
        "because single-stock volatility is high and eight separate options "
        "insure more than the portfolio needs. **One Nifty put**, sized by the "
        "portfolio's beta, is much cheaper but pays out only if the *index* "
        "falls — leaving **basis risk**."
    )

    s_h = run_simulation(portfolio_name, drift_name, horizon_years, n_paths, seed)
    w_h = data["portfolios"][portfolio_name]
    beta_p = sim.portfolio_beta(w_h, data["betas"])
    nifty_vol = float(np.sqrt(data["Sigma_all"].loc["NIFTY", "NIFTY"]))

    h_stocks = sim.single_stock_put_hedge(s_h, data["Sigma_all"], w_h,
                                          hedge_strike, horizon_years, RF)
    h_index = sim.index_put_hedge(s_h, "NIFTY", nifty_vol, beta_p,
                                  hedge_strike, horizon_years, RF)

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio beta to Nifty", f"{beta_p:.2f}",
              help="The index hedge notional is beta x portfolio value.")
    c2.metric("Premium: single-stock puts",
              f"{h_stocks['premium']/INITIAL_VALUE:.2%}",
              help=f"Rs {h_stocks['premium']:,.0f}")
    c3.metric("Premium: one Nifty put",
              f"{h_index['premium']/INITIAL_VALUE:.2%}",
              help=f"Rs {h_index['premium']:,.0f}",
              delta=f"{h_stocks['premium']/h_index['premium']:.0f}x cheaper")

    table = sim.compare_hedges(s_h, [h_stocks, h_index], RF, horizon_years,
                               confidence=confidence)
    st.dataframe(table.style.format("{:.2%}"), width="stretch")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        outcomes = {
            "Unhedged": s_h["values"][:, -1],
            h_stocks["label"]: sim.hedged_outcomes(s_h, h_stocks, RF, horizon_years),
            h_index["label"]: sim.hedged_outcomes(s_h, h_index, RF, horizon_years),
        }
        bins = np.linspace(0, np.percentile(outcomes["Unhedged"], 99), 100)
        for i, (label, values) in enumerate(outcomes.items()):
            ax.hist(values, bins=bins, color=SERIES[i], alpha=0.5, label=label)
        ax.axvline(INITIAL_VALUE, color=INK, linestyle="--", linewidth=1.2)
        ax.set_title("Outcomes with and without the hedge")
        ax.set_xlabel("Value at the end of the horizon")
        rupee_axis(ax, "x")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    with right:
        # Payoff diagram, drawn from this portfolio's own premium.
        fig, ax = plt.subplots(figsize=(6, 4))
        outcome_range = np.linspace(0.4, 1.8, 200) * INITIAL_VALUE
        growth_range = outcome_range / INITIAL_VALUE
        cost = h_stocks["premium"] * np.exp(RF * horizon_years)
        payoff = INITIAL_VALUE * np.maximum(hedge_strike - growth_range, 0.0)

        ax.plot(outcome_range, outcome_range - INITIAL_VALUE, color=MUTED,
                linestyle="--", label="Unhedged")
        ax.plot(outcome_range, outcome_range + payoff - cost - INITIAL_VALUE,
                color=SERIES[2], linewidth=2.4, label="Protective put")
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.axvline(hedge_strike * INITIAL_VALUE, color=SERIES[7],
                   linestyle=":", linewidth=1.4, label="Strike")
        ax.set_title("Payoff at expiry (single-stock puts)")
        ax.set_xlabel("Portfolio value")
        ax.set_ylabel("Profit or loss")
        rupee_axis(ax, "x"); rupee_axis(ax, "y")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    base_cvar = sim.conditional_value_at_risk(s_h["values"][:, -1],
                                              INITIAL_VALUE, confidence)
    idx_cvar = sim.conditional_value_at_risk(
        sim.hedged_outcomes(s_h, h_index, RF, horizon_years),
        INITIAL_VALUE, confidence)
    st.info(
        f"**Basis risk in one number.** The Nifty put costs "
        f"{h_index['premium']/INITIAL_VALUE:.2%} and moves CVaR from "
        f"{base_cvar:.1%} to {idx_cvar:.1%}. This basket's correlation with "
        f"the index is low, so a hedge that pays only when the index falls "
        f"can be close to no hedge at all for this portfolio."
    )

# ===========================================================================
# TAB 5: DELTA HEDGING
# ===========================================================================
with tab_delta:
    st.markdown(
        "Black-Scholes assumes the seller rehedges **continuously**. Nobody "
        "can. Here a trader sells a one-year at-the-money call and rehedges at "
        "a fixed interval; what is left at expiry is the cost of that "
        "approximation. Under perfect continuous hedging it would be zero."
    )

    c1, c2, c3 = st.columns(3)
    d_underlying = c1.selectbox("Underlying  ", list(data["prices"].columns),
                                index=4, key="delta_underlying")
    rebalance = c2.select_slider("Rehedge every", options=[1, 2, 5, 21, 63],
                                 value=5,
                                 format_func=lambda d: {1: "Day", 2: "2 days",
                                                        5: "Week", 21: "Month",
                                                        63: "Quarter"}[d])
    d_spot = float(data["prices"][d_underlying].loc[:SPLIT_DATE].iloc[-1])
    d_vol = float(np.sqrt(data["Sigma"].loc[d_underlying, d_underlying]))
    realised_vol = c3.slider("Volatility actually realised", 0.10, 1.20,
                             round(d_vol, 2), step=0.01,
                             help=f"The option is always SOLD at "
                                  f"{d_vol:.0%} (historical volatility). Move "
                                  f"this to see the seller win or lose.")

    @st.cache_data
    def cached_hedge(spot, vol, realised, freq, seed):
        return sim.delta_hedge_simulation(spot, spot, 1.0, RF, vol,
                                          sigma_realised=realised,
                                          rebalance_every=freq, n_paths=2_000,
                                          seed=int(seed))

    res = cached_hedge(d_spot, d_vol, realised_vol, rebalance, seed)
    pnl = res["pnl_pct_of_premium"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium received", f"Rs {res['premium']:,.2f}")
    c2.metric("Mean P&L", f"{pnl.mean():+.1%} of premium")
    c3.metric("Std dev of P&L", f"{pnl.std(ddof=1):.1%}",
              help="The cost of hedging discretely. It falls as 1/sqrt(n) in "
                   "the number of rehedges.")
    c4.metric("Probability of a loss", f"{(pnl < 0).mean():.0%}")

    if realised_vol < d_vol * 0.98:
        st.success(
            f"Sold at {d_vol:.0%}, realised {realised_vol:.0%}. The seller "
            f"collected more premium than the hedge cost to run: profit comes "
            f"through gamma, as the daily rehedging buys low and sells high."
        )
    elif realised_vol > d_vol * 1.02:
        st.error(
            f"Sold at {d_vol:.0%}, realised {realised_vol:.0%}. The stock moved "
            f"more than the option was priced for, so rehedging cost more than "
            f"the premium. This is what being short gamma feels like."
        )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(pnl, bins=60, color=SERIES[0], alpha=0.8)
        ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
        ax.axvline(pnl.mean(), color=SERIES[1], linewidth=1.6, label="Mean")
        ax.set_title("Hedging P&L across 2,000 paths")
        ax.set_xlabel("Profit or loss, as % of the premium")
        pct_axis(ax, "x")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    with right:
        @st.cache_data
        def cached_freq_study(spot, vol, seed):
            return sim.hedging_error_study(spot, spot, 1.0, RF, vol,
                                           frequencies=(1, 2, 5, 21, 63),
                                           n_paths=2_000, seed=int(seed))

        freq = cached_freq_study(d_spot, d_vol, seed)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(freq["Number of rebalances"], freq["Std dev (% of premium)"],
                color=SERIES[0], marker="o", markersize=5, label="Simulated")
        ref = (freq["Std dev (% of premium)"].iloc[0]
               * np.sqrt(freq["Number of rebalances"].iloc[0]
                         / freq["Number of rebalances"]))
        ax.plot(freq["Number of rebalances"], ref, color=SERIES[1],
                linestyle="--", label="1/sqrt(n)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Rehedges per year (log scale)")
        ax.set_ylabel("Std dev of P&L")
        ax.set_title("Boyle-Emanuel: error falls as 1/sqrt(n)")
        ax.legend(fontsize=8)
        st.pyplot(fig, clear_figure=True)

    st.dataframe(freq.style.format({
        "Mean P&L (% of premium)": "{:.2%}", "Std dev (% of premium)": "{:.2%}",
        "5th percentile": "{:.2%}", "95th percentile": "{:.2%}",
        "Worst path": "{:.2%}"}), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with NumPy, SciPy, pandas, Matplotlib and Streamlit. Prices from "
    "Yahoo Finance to 16 Sep 2026. Educational project, not investment advice: "
    "returns are simulated under geometric Brownian motion, which has no fat "
    "tails, no volatility clustering and no jumps."
)
