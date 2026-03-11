# Factor Investing Model

Built a multi-factor stock selection model on a 30-stock universe using Momentum, Value, Low Volatility, and Quality factors — then compared four position sizing methods.

Factor investing is the backbone of most systematic hedge funds. The idea is that certain stock characteristics (momentum, quality etc.) reliably predict future returns.

## What it does
- Constructs 4 factors and rebalances monthly
- Compares equal weight, factor score, MVO, and risk parity sizing
- Tests out-of-sample on 2025

## Results (2017-2024)
| Strategy | Sharpe | vs SPY |
|---|---|---|
| Quality | 1.39 | +0.46 |
| Momentum | 1.28 | +0.35 |
| SPY | 0.93 | — |

Factor score weighting beat all others — Sharpe 1.82.

## Stack
Python, yfinance, pandas, numpy, scipy, matplotlib