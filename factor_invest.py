import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize as sci_min
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "INTC",
    "JPM", "BAC", "GS", "MS", "BLK", "AXP",
    "JNJ", "UNH", "PFE", "ABBV", "MRK",
    "AMZN", "WMT", "COST", "PG", "KO",
    "XOM", "CVX", "COP",
    "CAT", "HON", "GE"
]

print(f"Factor Investing Model")
print(f"Universe: {len(UNIVERSE)} stocks")
print(f"Period: {START_DATE} to {END_DATE}")

# STEP 1: Fetch price data
print("\nFetching price data...")
prices = yf.download(UNIVERSE, start=START_DATE, end=END_DATE, progress=False)["Close"]
prices = prices.dropna(axis=1, thresh=int(len(prices)*0.95))
prices = prices.ffill().dropna()
print(f"Clean universe: {prices.shape[1]} stocks, {prices.shape[0]} trading days")

returns = prices.pct_change().dropna()
monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna()

# STEP 2: Construct factors
print("\nConstructing factors...")
momentum = monthly_prices.shift(1).pct_change(11).dropna()
value = (prices / prices.rolling(252).max()).dropna()
low_vol = returns.rolling(252).std().dropna() * np.sqrt(252)
quality = returns.rolling(126).mean() / returns.rolling(126).std()
quality = quality.dropna()

common_dates = momentum.index.intersection(value.index).intersection(low_vol.index).intersection(quality.index)
momentum = momentum.loc[common_dates]
value = value.loc[common_dates]
low_vol = low_vol.loc[common_dates]
quality = quality.loc[common_dates]
print(f"Aligned dates: {len(common_dates)} ({common_dates[0].date()} to {common_dates[-1].date()})")

# PERFORMANCE METRICS FUNCTION
def performance_metrics(returns, name, silent=False):
    ann_return = (1 + returns).prod() ** (12/len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol
    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_return / downside.item() if downside.item() > 0 else 0
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = drawdown.min()
    if not silent:
        print(f"\n{name}:")
        print(f"  Ann. Return:  {ann_return*100:.2f}%")
        print(f"  Ann. Vol:     {ann_vol*100:.2f}%")
        print(f"  Sharpe:       {sharpe:.3f}")
        print(f"  Sortino:      {sortino:.3f}")
        print(f"  Max Drawdown: {max_dd*100:.2f}%")
    return {"name": name, "return": ann_return, "vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd}

# BUILD FACTOR PORTFOLIO FUNCTION
def build_factor_portfolio(factor_scores, returns, top_n=10,
                           higher_is_better=True, sizing="equal"):
    portfolio_returns = []
    dates = []

    for i in range(len(factor_scores) - 1):
        date = factor_scores.index[i]
        next_date = factor_scores.index[i + 1]
        scores = factor_scores.iloc[i].dropna()

        if higher_is_better:
            selected = scores.nlargest(top_n).index
        else:
            selected = scores.nsmallest(top_n).index

        if next_date not in monthly_returns.index:
            continue

        month_rets = monthly_returns.loc[next_date, selected].dropna()
        if len(month_rets) == 0:
            continue

        if sizing == "equal":
            weights = np.ones(len(month_rets)) / len(month_rets)

        elif sizing == "factor_score":
            raw_scores = scores[month_rets.index]
            if not higher_is_better:
                raw_scores = 1 / raw_scores
            raw_scores = raw_scores - raw_scores.min() + 1e-6
            weights = (raw_scores / raw_scores.sum()).values

        elif sizing == "risk_parity":
            if date in low_vol.index:
                vols = low_vol.loc[date, month_rets.index].dropna()
                if len(vols) == 0:
                    weights = np.ones(len(month_rets)) / len(month_rets)
                else:
                    inv_vol = 1 / vols
                    weights = (inv_vol / inv_vol.sum()).values
            else:
                weights = np.ones(len(month_rets)) / len(month_rets)

        elif sizing == "mvo":
            lookback = monthly_returns.loc[:date].tail(24)[month_rets.index].dropna(axis=1)
            if lookback.shape[0] < 6 or lookback.shape[1] < 2:
                weights = np.ones(len(month_rets)) / len(month_rets)
            else:
                mu = lookback.mean().values
                cov = lookback.cov().values
                n = len(mu)
                def neg_sharpe(w):
                    ret = w @ mu
                    vol = np.sqrt(w @ cov @ w)
                    return -ret / vol if vol > 0 else 0
                w0 = np.ones(n) / n
                bounds = [(0, 0.4)] * n
                constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
                res = sci_min(neg_sharpe, w0, method="SLSQP",
                             bounds=bounds, constraints=constraints)
                weights = res.x if res.success else w0

        portfolio_returns.append((month_rets.values * weights).sum())
        dates.append(next_date)

    return pd.Series(portfolio_returns, index=dates)

# STEP 3: Build factor portfolios
print("\nBuilding factor portfolios...")
momentum_returns    = build_factor_portfolio(momentum, monthly_returns, higher_is_better=True)
value_returns       = build_factor_portfolio(value, monthly_returns, higher_is_better=False)
lowvol_returns      = build_factor_portfolio(low_vol, monthly_returns, higher_is_better=False)
quality_returns     = build_factor_portfolio(quality, monthly_returns, higher_is_better=True)
multifactor_returns = (momentum_returns + value_returns + lowvol_returns + quality_returns) / 4

spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)["Close"]
spy_monthly = spy.resample("ME").last().pct_change().dropna()
spy_aligned = spy_monthly.reindex(momentum_returns.index).dropna().squeeze()

print(f"Portfolio periods: {len(momentum_returns)}")

# STEP 4: Performance metrics
print("\nPerformance Metrics:")
metrics = []
metrics.append(performance_metrics(momentum_returns, "Momentum"))
metrics.append(performance_metrics(value_returns, "Value"))
metrics.append(performance_metrics(lowvol_returns, "Low Volatility"))
metrics.append(performance_metrics(quality_returns, "Quality"))
metrics.append(performance_metrics(multifactor_returns, "Multi-Factor"))
metrics.append(performance_metrics(spy_aligned, "SPY Benchmark"))

# STEP 5: Position sizing comparison on Quality factor
print("\nTesting position sizing methods on Quality factor...")
eq_returns  = build_factor_portfolio(quality, monthly_returns, higher_is_better=True, sizing="equal")
fs_returns  = build_factor_portfolio(quality, monthly_returns, higher_is_better=True, sizing="factor_score")
rp_returns  = build_factor_portfolio(quality, monthly_returns, higher_is_better=True, sizing="risk_parity")
mvo_returns = build_factor_portfolio(quality, monthly_returns, higher_is_better=True, sizing="mvo")

print("\nPosition Sizing Comparison (Quality Factor):")
for rets, name in [(eq_returns, "Equal Weight"),
                   (fs_returns, "Factor Score"),
                   (rp_returns, "Risk Parity"),
                   (mvo_returns, "MVO")]:
    performance_metrics(rets, name)

# STEP 6: Plot equity curves
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

cum_momentum = (1 + momentum_returns).cumprod()
cum_value    = (1 + value_returns).cumprod()
cum_lowvol   = (1 + lowvol_returns).cumprod()
cum_quality  = (1 + quality_returns).cumprod()
cum_multi    = (1 + multifactor_returns).cumprod()
cum_spy      = (1 + spy_aligned).cumprod()

axes[0].plot(cum_momentum, label=f"Momentum (23.7%)", linewidth=2)
axes[0].plot(cum_value, label=f"Value (20.4%)", linewidth=2)
axes[0].plot(cum_lowvol, label=f"Low Vol (9.9%)", linewidth=2)
axes[0].plot(cum_quality, label=f"Quality (25.9%)", linewidth=2)
axes[0].plot(cum_multi, label=f"Multi-Factor (20.3%)", linewidth=2, linestyle="--")
axes[0].plot(cum_spy, label=f"SPY (15.1%)", linewidth=2, color="black", linestyle="--")
axes[0].set_title("Factor Portfolio Equity Curves (2017-2025)")
axes[0].set_ylabel("Growth of $1")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

def drawdown_series(returns):
    cum = (1 + returns).cumprod()
    return (cum - cum.cummax()) / cum.cummax()

axes[1].fill_between(momentum_returns.index, drawdown_series(momentum_returns)*100, alpha=0.3, label="Momentum")
axes[1].fill_between(quality_returns.index, drawdown_series(quality_returns)*100, alpha=0.3, label="Quality")
axes[1].fill_between(multifactor_returns.index, drawdown_series(multifactor_returns)*100, alpha=0.3, label="Multi-Factor")
axes[1].plot(drawdown_series(spy_aligned)*100, color="black", linewidth=2, linestyle="--", label="SPY")
axes[1].set_title("Drawdowns")
axes[1].set_ylabel("Drawdown (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("factor_equity_curves.png")
plt.show()

# STEP 7: Out-of-sample test on 2025
print("\nFetching 2025 out-of-sample data...")
prices_oos = yf.download(UNIVERSE, start="2025-01-01", end="2025-12-31", progress=False)["Close"]
prices_oos = prices_oos.ffill().dropna()

returns_oos = prices_oos.pct_change().dropna()
monthly_prices_oos = prices_oos.resample("ME").last()
monthly_returns_oos = monthly_prices_oos.pct_change().dropna()

# Reconstruct factors on 2025 data
# Use combined prices (2015-2025) for rolling calculations so we have history
prices_full = yf.download(UNIVERSE, start="2015-01-01", end="2025-12-31", progress=False)["Close"]
prices_full = prices_full.ffill().dropna()
returns_full = prices_full.pct_change().dropna()
monthly_prices_full = prices_full.resample("ME").last()
monthly_returns_full = monthly_prices_full.pct_change().dropna()

momentum_full = monthly_prices_full.shift(1).pct_change(11).dropna()
value_full    = (prices_full / prices_full.rolling(252).max()).dropna()
low_vol_full  = returns_full.rolling(252).std().dropna() * np.sqrt(252)
quality_full  = returns_full.rolling(126).mean() / returns_full.rolling(126).std()
quality_full  = quality_full.dropna()

# Filter to 2025 only
oos_dates = monthly_returns_oos.index
momentum_oos = momentum_full.reindex(oos_dates).dropna()
value_oos    = value_full.reindex(oos_dates).dropna()
low_vol_oos  = low_vol_full.reindex(oos_dates).dropna()
quality_oos  = quality_full.reindex(oos_dates).dropna()

common_oos = momentum_oos.index.intersection(value_oos.index).intersection(low_vol_oos.index).intersection(quality_oos.index)
momentum_oos = momentum_oos.loc[common_oos]
value_oos    = value_oos.loc[common_oos]
low_vol_oos  = low_vol_oos.loc[common_oos]
quality_oos  = quality_oos.loc[common_oos]

print(f"OOS periods: {len(common_oos)} months")

# Run same factor score strategy on 2025
def build_oos_portfolio(factor_scores, monthly_rets, top_n=10, higher_is_better=True):
    portfolio_returns = []
    dates = []
    for i in range(len(factor_scores) - 1):
        date = factor_scores.index[i]
        next_date = factor_scores.index[i + 1]
        scores = factor_scores.iloc[i].dropna()
        selected = scores.nlargest(top_n).index if higher_is_better else scores.nsmallest(top_n).index
        if next_date not in monthly_rets.index:
            continue
        month_rets = monthly_rets.loc[next_date, selected].dropna()
        if len(month_rets) == 0:
            continue
        # Factor score weighting (winner from Step 5)
        raw_scores = scores[month_rets.index]
        if not higher_is_better:
            raw_scores = 1 / raw_scores
        raw_scores = raw_scores - raw_scores.min() + 1e-6
        weights = (raw_scores / raw_scores.sum()).values
        portfolio_returns.append((month_rets.values * weights).sum())
        dates.append(next_date)
    return pd.Series(portfolio_returns, index=dates)

mom_oos  = build_oos_portfolio(momentum_oos, monthly_returns_full, higher_is_better=True)
val_oos  = build_oos_portfolio(value_oos, monthly_returns_full, higher_is_better=False)
lv_oos   = build_oos_portfolio(low_vol_oos, monthly_returns_full, higher_is_better=False)
qual_oos = build_oos_portfolio(quality_oos, monthly_returns_full, higher_is_better=True)
multi_oos = (mom_oos + val_oos + lv_oos + qual_oos) / 4

spy_oos = yf.download("SPY", start="2025-01-01", end="2025-12-31", progress=False)["Close"]
spy_oos_monthly = spy_oos.resample("ME").last().pct_change().dropna().squeeze()
spy_oos_aligned = spy_oos_monthly.reindex(mom_oos.index).dropna()

print("\nOut-of-Sample 2025 Results:")
for rets, name in [(mom_oos, "Momentum"),
                   (val_oos, "Value"),
                   (lv_oos, "Low Volatility"),
                   (qual_oos, "Quality"),
                   (multi_oos, "Multi-Factor"),
                   (spy_oos_aligned, "SPY Benchmark")]:
    performance_metrics(rets, name)

# Plot OOS equity curves
plt.figure(figsize=(14, 6))
plt.plot((1 + mom_oos).cumprod(), label="Momentum", linewidth=2)
plt.plot((1 + val_oos).cumprod(), label="Value", linewidth=2)
plt.plot((1 + lv_oos).cumprod(), label="Low Volatility", linewidth=2)
plt.plot((1 + qual_oos).cumprod(), label="Quality", linewidth=2)
plt.plot((1 + multi_oos).cumprod(), label="Multi-Factor", linewidth=2, linestyle="--")
plt.plot((1 + spy_oos_aligned).cumprod(), label="SPY", linewidth=2, color="black", linestyle="--")
plt.title("Out-of-Sample 2025 — Factor Portfolios vs SPY")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("oos_2025.png")
plt.show()