## Ancilla
_cash out or crash out_

---

Ancilla contains some personal tools for quantitative finance.

Currently featuring:
- A Polygon.io data wrapper with retries & automatic caching
- Basic visualizations for IV surfaces, liquidity, and price data
- Backtesting for equities and options
  - Batch requests and data caching for speed
  - Configurable strategy frequency (30min, 1hour, etc.)
  - Models slippage, commissions, price impact
  - Fill probability based on volume (or deterministic fills for testing)
  - Adjusts stock prices for splits and dividends, pays dividends
  - Monthly interest using accurate rates
  - Automatic ITM options exercise and assignment
  - Detailed trade history visualizations, performance metrics, logs

Requires *Polygon.io* starter subscriptions for data access.
Requires a *FRED* API key for interest rates.
