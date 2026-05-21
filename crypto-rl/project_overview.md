# Crypto RL Trading Bot — Complete Project Overview

## What This Project Is

This is a **reinforcement learning trading bot** that learns to trade Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) using historical daily price data from 2020 to 2024. The bot is trained using a state-of-the-art RL algorithm called **PPO (Proximal Policy Optimization)** and is benchmarked against four classical trading strategies. The entire system is wrapped in a web application — a FastAPI backend and a React dashboard — so results can be visualized interactively.

**This is an educational project. It is not financial advice.**

---

## High-Level Architecture

The project is organized in five distinct layers that work together end-to-end:

```
Raw Market Data (yfinance)
        ↓
  Technical Indicators (ta library)
        ↓
  CryptoTradingEnv (Gymnasium)   ←→   PPO Agent (Stable-Baselines3)
        ↓
  Baselines (4 classical strategies)
        ↓
  FastAPI Backend (port 8000)
        ↓
  React + Vite Frontend (port 3000)
```

| Layer | Technology | Purpose |
|---|---|---|
| Data | yfinance, pandas, ta | Download and process market data |
| Environment | Gymnasium | Simulate the trading market for the agent |
| Agent | Stable-Baselines3 PPO | The RL algorithm that learns to trade |
| Baselines | Pure Python | Classical strategies to compare against |
| API | FastAPI | Serve backtest results and metrics |
| UI | React, Vite, Plotly.js | Interactive dashboard |

---

## Directory Structure

```
projectimplementation/
├── data/
│   ├── raw/              ← Downloaded OHLCV CSVs (gitignored)
│   └── processed/        ← Indicator-enriched CSVs with train/val/test split column
├── env/
│   ├── __init__.py       ← SEED = 42
│   ├── crypto_trading_env.py   ← The Gymnasium trading environment
│   └── reward.py         ← Rolling-Sharpe reward function
├── agents/
│   ├── ppo_agent.py      ← PPO wrapper (build / train / save / load / predict)
│   └── callbacks.py      ← TensorBoard + checkpoint callbacks
├── baselines/
│   ├── base.py           ← Abstract base class
│   ├── buy_and_hold.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   ├── random_agent.py
│   └── metrics.py        ← Sharpe, max DD, win rate, total return, Calmar
├── api/
│   ├── main.py           ← FastAPI app, CORS, model preloading
│   ├── routes.py         ← 6 endpoints
│   ├── schemas.py        ← Pydantic response models
│   └── cache.py          ← In-memory result cache
├── scripts/
│   ├── download_data.py
│   ├── compute_indicators.py
│   ├── smoke_test_env.py
│   ├── train_ppo.py
│   ├── run_baselines.py
│   └── evaluate.py
├── notebooks/
│   └── train_colab.ipynb ← Google Colab training workflow
├── models/
│   └── ppo_btc_final.zip ← Pre-trained PPO model
├── results/
│   ├── comparison_2024.csv
│   └── equity_curves.png
├── tests/
│   ├── test_env.py
│   ├── test_baselines.py
│   └── test_metrics.py
├── frontend/             ← React + Vite SPA
└── requirements.txt
```

---

## Phase 1 — Data Pipeline

### Data Source

Market data is downloaded using **yfinance**, which pulls free daily OHLCV (Open, High, Low, Close, Volume) bars from Yahoo Finance. Three assets are supported:

- `BTC-USD` — Bitcoin
- `ETH-USD` — Ethereum
- `SOL-USD` — Solana

Data spans **2020-01-01 to 2024-12-31**, giving approximately 1,826 trading days per asset.

```bash
python scripts/download_data.py --assets btc eth sol
```

### Data Splits

The data is divided into three non-overlapping periods to prevent overfitting:

| Split | Period | Purpose |
|---|---|---|
| Train | 2020-01-01 → 2022-12-31 | Agent learns from this data |
| Val | 2023-01-01 → 2023-12-31 | Used to monitor training progress |
| Test | 2024-01-01 → 2024-12-31 | Final unseen evaluation |

### Technical Indicators

Five indicators are computed using the `ta` (Technical Analysis) library:

| Indicator | Description | Computation |
|---|---|---|
| **RSI** | Relative Strength Index | Momentum oscillator (14-period window) |
| **MACD Diff** | MACD histogram | Difference between fast (12) and slow (26) EMA, smoothed by signal (9) |
| **BB Width** | Bollinger Band width | (Upper − Lower) / Middle band, 20-period, 2 std devs |
| **EMA50 Dist** | Distance from 50-period EMA | (Close − EMA50) / EMA50 |
| **Vol Z-score** | Volume z-score | (Volume − 20d mean) / 20d std |

**Critical:** Indicators are computed per split using only data within that split. This prevents **look-ahead bias** — no future information leaks into the training data.

```bash
python scripts/compute_indicators.py
```

---

## Phase 2 — The Trading Environment

The environment (`env/crypto_trading_env.py`) is the heart of the RL system. It follows the **Gymnasium API** (the modern successor to OpenAI Gym).

### Observation Space

The agent receives a flat vector of **265 numbers** at each time step:

```
obs = [
    OHLCV_window,      # 50 bars × 5 features = 250 values (z-scored)
    RSI,               # 1 value
    MACD_diff,         # 1 value
    BB_width,          # 1 value
    EMA50_dist,        # 1 value
    vol_zscore,        # 1 value
    position,          # {0, 1} — flat or long
    net_worth_norm,    # (net_worth / initial_balance) - 1
    unrealized_pnl,    # asset_value / net_worth
    peak_pct,          # (net_worth / peak) - 1
    var5,              # 5th percentile of last 20 daily returns
]
# Total: 50*5 + 5 + 5 = 260... actually 265 = 250 + 5 + 5
```

The OHLCV window is **z-scored using training split statistics only**, so the same normalization is applied to val/test without leaking information.

### Action Space

The agent can choose from **5 discrete actions** at each time step:

| Action | ID | Effect |
|---|---|---|
| Buy 25% | 0 | Spend 25% of available cash |
| Buy 50% | 1 | Spend 50% of available cash |
| Hold | 2 | Do nothing |
| Sell 25% | 3 | Sell 25% of held asset |
| Sell 50% | 4 | Sell 50% of held asset |

Fractional buying/selling (vs. all-in/all-out) gives the agent more nuanced control, avoiding the boom-bust patterns of binary strategies.

### Portfolio Bookkeeping

The environment tracks:
- `self.balance` — cash in USD
- `self.asset_held` — units of BTC/ETH/SOL held
- `self.net_worths` — history of total portfolio value
- `self.peak_value` — highest portfolio value ever reached
- `self.trades` — list of trade dictionaries

Net worth at each step: `net_worth = balance + asset_held × current_price`

### Termination Conditions

- **Terminated** (`terminated=True`): Portfolio drops below 50% of the initial balance — the bot has lost too much money.
- **Truncated** (`truncated=True`): The episode has run through all available data.

### Reward Function

The reward function (`env/reward.py`) is designed to encourage risk-adjusted returns rather than raw profit:

```
reward = rolling_sharpe - drawdown_penalty - transaction_cost
```

**1. Rolling Sharpe (base reward):**
```
sharpe = √252 × (mean_return - risk_free_daily) / std_return
```
Computed over the last 30 portfolio returns. The √252 factor annualizes the daily Sharpe ratio. Risk-free rate is 4% annual (4%/252 daily).

**2. Drawdown Penalty:**
```
drawdown = (peak - current_value) / peak
dd_penalty = max(0, drawdown - 0.20) × 2.0
```
No penalty for drawdowns under 20%. Above that threshold, penalty scales linearly at 2× to strongly discourage large losses.

**3. Transaction Cost:**
```
tx_cost = 0.001 × trade_notional
```
0.1% fee per trade discourages excessive trading.

**4. Clipping:**
The final reward is clipped to `[-10, 10]` and any non-finite values (from near-zero variance) are replaced with 0.

---

## Phase 3 — The PPO Agent

The agent (`agents/ppo_agent.py`) wraps **Stable-Baselines3's PPO** implementation with trading-optimized hyperparameters adapted from **FinRL's validated configuration**.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `policy` | MlpPolicy | Standard multilayer perceptron |
| `n_steps` | 2048 | Steps collected before each update |
| `batch_size` | 256 | Mini-batch size for gradient updates |
| `n_epochs` | 10 | Gradient passes over each batch |
| `learning_rate` | 2.5e-4 | FinRL validated value |
| `gamma` | 0.99 | Discount factor — values future rewards |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_range` | 0.2 | PPO clipping parameter |
| `ent_coef` | 0.01 | Entropy bonus — encourages exploration |
| `vf_coef` | 0.5 | Value function loss weight |
| `net_arch` | [128, 128] | Two hidden layers of 128 neurons each |

### Vectorized Environments

Training uses **4 parallel environments** (`DummyVecEnv` with `n_envs=4`). Each environment runs the same asset but is independently seeded. This increases data throughput and reduces correlation between collected transitions.

### Training Workflow

Training runs on **Google Colab** (free GPU/CPU), not locally. The notebook:

1. Clones the repository
2. Downloads market data via yfinance
3. Computes indicators
4. Trains PPO for 500,000 timesteps per asset (~20-60 min on T4 GPU)
5. Downloads the model `.zip` files

Pre-trained models are committed to the repo: `models/ppo_btc_final.zip`, etc.

### Callbacks

- **MetricsCallback**: Logs per-step reward and rollout statistics (min, mean, max reward) to TensorBoard.
- **CheckpointCallback**: Saves model snapshots every 10,000 steps to `models/checkpoints/`.

---

## Phase 4 — Classical Baselines

Four classical strategies are implemented as the comparison benchmark. All share the same `BaselineStrategy` abstract base class and are evaluated on the same 2024 test split with the same $10,000 starting capital.

### Buy and Hold
Buy all-in on day 1, hold until the end. This is the classic passive strategy and often surprisingly hard to beat. Just two trades total.

### Mean Reversion
- Buy when price drops more than 2% below its 20-day moving average
- Sell when price rises above its 20-day moving average

Assumes prices tend to return to their average. Works in ranging markets, poorly in trending ones.

### Momentum
- Buy when price has risen for 5 consecutive days
- Sell when price has fallen for 3 consecutive days

Assumes trends continue. Works in trending markets, poorly in choppy ones.

### Random Agent
Uniformly samples a random action `{0, 1, 2, 3, 4}` at each step, seeded with `np.random.default_rng(42)` for reproducibility.

### Performance Metrics

All strategies are evaluated using five metrics from `baselines/metrics.py`:

| Metric | Description | Formula |
|---|---|---|
| **Sharpe Ratio** | Risk-adjusted return | √252 × (mean_daily_return − rf) / std_daily_return |
| **Max Drawdown** | Worst peak-to-trough loss | max((peak − value) / peak) |
| **Total Return** | Overall profit/loss | (final − initial) / initial |
| **Win Rate** | Fraction of profitable sell trades | profitable_sells / total_sells |
| **Calmar Ratio** | Annualized return / max drawdown | ann_return / max_drawdown |

---

## Phase 5 — FastAPI Backend

The API (`api/main.py`, `api/routes.py`) serves backtest results to the frontend.

### Model Preloading

All three PPO models are loaded once at startup via the `lifespan` context manager. Subsequent requests use the in-memory model without re-loading from disk.

### Endpoints

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /api/backtest` | `asset`, `agent` | Metrics, trade log, equity curve |
| `GET /api/compare` | `asset` | All 5 strategies compared |
| `GET /api/training-curves` | `asset` | Episode reward history |
| `GET /api/portfolio-history` | `asset`, `agent` | Portfolio value time series |
| `GET /api/disclaimer` | — | Disclaimer text |
| `GET /api/paper-trading` | `asset`, `agent`, `days` | Live recent data backtest |

### Caching

Results are cached in a simple in-memory dict keyed by `(endpoint, asset, agent)`. Since backtests on historical data are deterministic, there is no invalidation — results are computed once per server session. Paper-trading uses 5-minute cache buckets to periodically refresh live data.

### CORS

The API allows cross-origin requests from `http://localhost:3000` (the Vite dev server) so the frontend and backend can run independently on different ports.

---

## Phase 6 — React Frontend

The frontend (`frontend/`) is a React + Vite single-page application that calls the API and renders interactive charts.

### Components

| Component | Purpose |
|---|---|
| `AssetAgentSelector` | Dropdowns to pick asset (BTC/ETH/SOL) and agent |
| `EquityCurve` | Plotly.js line chart of portfolio value over time |
| `ComparisonTable` | Table of all 5 strategies × 5 metrics |
| `TrainingCurves` | Episode reward line chart from training |
| `TradeLog` | Paginated table of individual trades |
| `DisclaimerPage` | Legal disclaimer text |

---

## Reproducibility

The entire project is deterministic at `SEED = 42` (defined in `env/__init__.py`):

- Environment reset uses this seed
- PPO training uses `seed=42`
- Random baseline uses `np.random.default_rng(42)`

All three splits are fixed date ranges, so anyone re-running the full pipeline gets identical results.

---

## Reference Projects

This project adapts code and ideas from five open-source repositories:

| Reference | What Was Used |
|---|---|
| **RLTrader** (notadamking) | Environment skeleton, portfolio bookkeeping, buy-and-hold baseline pattern |
| **RL-Bitcoin-trading-bot** | Indicator computation patterns |
| **RL-eth-agent** | Trade log dictionary shape |
| **FinRL** (AI4Finance) | PPO hyperparameters, vectorized env pattern, metrics formulas, data split helper |
| **BTC_RL_Trading_Bot** | General reference |

AI assistance: **Claude Code** was used for planning, scaffolding, code authoring, and review. See `docs/ai_disclosure.md`.

---

## Running the Full Stack

```bash
# 1. Data
python scripts/download_data.py --assets btc eth sol
python scripts/compute_indicators.py

# 2. Verify env
python scripts/smoke_test_env.py

# 3. Baselines
python scripts/run_baselines.py --asset btc --year 2024

# 4. Evaluate (needs pre-trained model in models/)
python scripts/evaluate.py --asset btc

# 5. Backend
uvicorn api.main:app --reload --port 8000

# 6. Frontend
cd frontend && npm run dev   # http://localhost:3000
```
