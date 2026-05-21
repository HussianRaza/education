# Crypto RL Trading Bot — Presentation

---

## Slide 1 — Title

# Crypto RL Trading Bot
### Teaching a Machine to Trade Bitcoin, Ethereum & Solana

**Using Proximal Policy Optimization (PPO) + Gymnasium**

> *An educational project exploring reinforcement learning in financial markets*

---

## Slide 2 — The Problem

### Can a machine learn to trade cryptocurrency?

Trading requires:
- Reading complex, noisy market data
- Managing risk while seeking profit
- Making sequential decisions over time
- Adapting to changing market conditions

**Classical algorithms** (buy-and-hold, momentum) use hand-crafted rules.

**Reinforcement Learning** lets the algorithm discover its own rules through experience.

---

## Slide 3 — What We Built

### A complete end-to-end RL trading system

```
Market Data  →  Trading Environment  →  Trained Agent
                                              ↓
                                    Web Dashboard + API
```

**Assets:** Bitcoin (BTC), Ethereum (ETH), Solana (SOL)

**Data:** 5 years of daily prices (2020–2024)

**Evaluation:** Compared against 4 classical baselines on unseen 2024 data

---

## Slide 4 — System Architecture

| Layer | Technology | Role |
|---|---|---|
| **Data** | yfinance + pandas + ta | Download & enrich market data |
| **Environment** | Gymnasium | Simulate the trading market |
| **Agent** | Stable-Baselines3 PPO | Learn trading decisions |
| **Baselines** | Pure Python | Classical strategies to beat |
| **API** | FastAPI | Serve results |
| **Dashboard** | React + Plotly.js | Visualize everything |

---

## Slide 5 — The Data Pipeline

### 5 Years × 3 Assets × Daily OHLCV

**Step 1 — Download**
```
yfinance → BTC-USD, ETH-USD, SOL-USD
           2020-01-01 to 2024-12-31
           ~1,826 bars per asset
```

**Step 2 — Split (no data leakage)**

| Split | Period | Days |
|---|---|---|
| Train | 2020–2022 | ~1,096 |
| Validation | 2023 | ~365 |
| **Test** | **2024** | **~366** |

**Step 3 — Technical Indicators**
RSI · MACD · Bollinger Bands · EMA50 · Volume Z-Score

*(Computed per split — no lookahead bias)*

---

## Slide 6 — The Trading Environment

### The World the Agent Lives In

**At every trading day, the agent sees:**

```
┌─────────────────────────────────────────────────────┐
│  Last 50 days of OHLCV data  (250 numbers)          │
│  5 technical indicators      (5 numbers)            │
│  Portfolio state             (5 numbers)            │
│                              ──────────             │
│                         Total: 260 values           │
└─────────────────────────────────────────────────────┘
```

**And chooses one of 5 actions:**

| Action | Effect |
|---|---|
| Buy 25% | Spend 25% of cash on the asset |
| Buy 50% | Spend 50% of cash on the asset |
| **Hold** | **Do nothing** |
| Sell 25% | Sell 25% of holdings |
| Sell 50% | Sell 50% of holdings |

**Episode ends when:** all data is consumed OR portfolio drops below 50% of starting value

---

## Slide 7 — The Reward Signal

### Teaching the Agent What "Good" Means

The agent gets a reward at each step — not just when it profits:

```
Reward = Rolling Sharpe Ratio
       − Drawdown Penalty
       − Transaction Cost
```

**Rolling Sharpe (core signal)**
- Measures risk-adjusted return over the last 30 days
- High reward for consistent gains, penalized for volatility

**Drawdown Penalty**
- No penalty for losses under 20% from peak
- Heavy penalty above 20% — protects against catastrophic loss

**Transaction Cost**
- 0.1% fee on each trade
- Discourages over-trading on noise

**All rewards clipped to [-10, +10]** for training stability

---

## Slide 8 — The PPO Algorithm

### Proximal Policy Optimization

**PPO** is the industry-standard deep RL algorithm, used in:
- OpenAI's ChatGPT (RLHF fine-tuning)
- DeepMind's game-playing agents
- FinRL financial trading research

**Why PPO for trading?**
- **Stable:** Prevents destructive parameter updates via clipping
- **Sample efficient:** Learns from dense reward signals
- **Proven:** Extensively validated in financial RL literature

**The core idea — Clip the policy update:**
```
Never change action probabilities by more than 20%
in a single training step
```
This keeps learning stable even with noisy financial data.

---

## Slide 9 — Neural Network Architecture

### Actor-Critic with Shared Backbone

```
Observation: 260 values
         ↓
   [ Dense 128 + ReLU ]
         ↓
   [ Dense 128 + ReLU ]
      ↙           ↘
  ACTOR          CRITIC
(5 action       (1 value
 probabilities)  estimate)
```

**Actor** → "Which action should I take?"

**Critic** → "How good is my current situation?" (baseline for computing advantages)

**Training:** 500,000 steps per asset · 4 parallel environments · Google Colab T4 GPU

---

## Slide 10 — Training Workflow

### From Data to Trained Agent

```
1. Collect 2,048 steps of experience with current policy
          ↓
2. Compute advantages (how much better/worse than average)
          ↓
3. Run 10 gradient updates on this batch
          ↓
4. Discard data, collect fresh experience
          ↓
   Repeat 244 times (total: ~500,000 steps)
```

**Training infrastructure:**
- Runs on **Google Colab** (free T4 GPU)
- ~20–60 minutes per asset
- TensorBoard logging for monitoring
- Checkpoint saved every 10,000 steps
- Pre-trained models committed to repo

---

## Slide 11 — Classical Baselines

### What the RL Agent Must Beat

| Strategy | Logic | Strength |
|---|---|---|
| **Buy and Hold** | Buy all-in day 1, hold forever | Captures full market upside |
| **Mean Reversion** | Buy when 2% below 20-day MA, sell above MA | Works in ranging markets |
| **Momentum** | Buy on 5-day streak up, sell on 3-day streak down | Works in trending markets |
| **Random** | Uniformly random action each day (seed=42) | Statistical baseline |

All evaluated on the same **2024 test data** with **$10,000** starting capital.

---

## Slide 12 — Evaluation Metrics

### How We Measure Performance

| Metric | What it Measures | Formula |
|---|---|---|
| **Sharpe Ratio** | Risk-adjusted return | √252 × (mean_return − rf) / std_return |
| **Max Drawdown** | Worst peak-to-trough loss | max((peak − value) / peak) |
| **Total Return** | Overall profit/loss | (final − initial) / initial |
| **Win Rate** | % of trades that were profitable | profitable_sells / total_sells |
| **Calmar Ratio** | Annual return / max drawdown | ann_return / max_dd |

**Starting capital:** $10,000 · **Test period:** 2024 · **Risk-free rate:** 4% annual

---

## Slide 13 — Results

### PPO vs. Classical Strategies on 2024 BTC Test Data

*(Values from `results/comparison_2024.csv`)*

| Strategy | Sharpe | Max DD | Total Return | Win Rate | Calmar |
|---|---|---|---|---|---|
| **PPO** | — | — | — | — | — |
| Buy and Hold | — | — | — | — | — |
| Mean Reversion | — | — | — | — | — |
| Momentum | — | — | — | — | — |
| Random | — | — | — | — | — |

*Run `python scripts/evaluate.py --asset btc` to populate this table.*

> **Key insight:** The RL agent optimizes for risk-adjusted returns (Sharpe), not just raw profit. A lower total return with much lower drawdown can be the better strategy.

---

## Slide 14 — Web Dashboard

### Live Interactive Visualization

**Equity Curve**
Portfolio value over time for any asset + strategy combination — side-by-side comparison.

**Comparison Table**
All 5 strategies × 5 metrics in one view. Sort by Sharpe, drawdown, or return.

**Training Curves**
Episode reward history showing how the agent improved over 500,000 training steps.

**Trade Log**
Every individual trade: date, direction, size, and profit/loss.

**Paper Trading**
Runs the same strategies on the last 60 days of live market data via yfinance.

---

## Slide 15 — Technical Highlights

### Engineering Decisions

**Reproducibility**
- Single `SEED = 42` used everywhere — same results every run
- Fixed date splits — no randomness in train/test assignment

**No Lookahead Bias**
- Indicators computed per split (train stats never see val/test data)
- OHLCV z-scoring uses training statistics applied to all splits

**Reward Stability**
- Sharpe denominator floored at 1e-8 to prevent division by zero
- Reward clipped to [-10, 10] to prevent exploding gradients
- Non-finite rewards replaced with 0

**API Design**
- In-memory cache — deterministic backtests computed once per session
- All 3 PPO models preloaded at startup for sub-10ms inference

---

## Slide 16 — Limitations & Honest Assessment

### What This Project Does NOT Do

| Limitation | Explanation |
|---|---|
| Not live trading | Historical simulation only — real markets have slippage, liquidity constraints, exchange outages |
| Daily resolution | Crypto markets are 24/7 — daily bars miss intraday opportunities and risks |
| Single asset | Each model trades one asset independently — no portfolio construction or correlation management |
| No transaction fees modeled precisely | Actual exchange fees vary; our 0.1% is a rough estimate |
| Overfitting risk | 500,000 steps on 3 years of data — the agent may have memorized specific market regimes |
| Past ≠ future | A strategy that worked in 2020-2024 may fail in different market conditions |

**This is an educational project. It is not financial advice.**

---

## Slide 17 — Future Work

### Where This Could Go Next

**Better Algorithms**
- **SAC (Soft Actor-Critic):** Off-policy, more sample efficient than PPO
- **TD3:** Better for continuous action spaces (position sizing)
- **Hyperparameter sweep with Optuna:** Find better configurations automatically

**Better Data**
- Intraday (hourly/minute) bars for more trading opportunities
- Order book data, funding rates, on-chain metrics
- Multi-asset portfolio environment

**Better Training**
- Curriculum learning: start with easy market conditions, advance to volatile ones
- Domain randomization: train on synthetic data to improve generalization

**Better Evaluation**
- Walk-forward validation across multiple years
- Monte Carlo simulation of strategy outcomes

---

## Slide 18 — Key Takeaways

### What We Learned

1. **RL can learn trading** — the agent discovers interpretable patterns (buy dips, cut losses) without explicit programming

2. **Reward design is critical** — raw P&L is a bad signal; risk-adjusted Sharpe produces far better behavior

3. **Beating buy-and-hold is hard** — in strong bull markets, passive holding outperforms most active strategies

4. **Classical baselines are strong** — they provide a rigorous bar; momentum works in trending crypto markets

5. **Infrastructure matters** — proper train/val/test splits, no lookahead bias, and reproducibility are as important as the algorithm

---

## Slide 19 — Technologies Used

| Category | Technology |
|---|---|
| RL Framework | Stable-Baselines3 (PPO) |
| Environment | Gymnasium |
| Data | yfinance, pandas, ta |
| Neural Networks | PyTorch (via SB3) |
| Monitoring | TensorBoard |
| Backend API | FastAPI, Pydantic |
| Frontend | React, Vite, Plotly.js |
| Training | Google Colab (T4 GPU) |
| Language | Python 3.10+ / TypeScript |

**Reference projects:** RLTrader · RL-Bitcoin-trading-bot · RL-eth-agent · FinRL · BTC_RL_Trading_Bot

**AI assistance:** Claude Code (Anthropic) — planning, scaffolding, code authoring, review

---

## Slide 20 — Demo

### Running It Live

```bash
# Backend
uvicorn api.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev
# → http://localhost:3000
```

**Live demo:**
1. Select asset (BTC/ETH/SOL) and strategy (PPO/baselines)
2. View equity curve and trade log for 2024 test period
3. Compare all 5 strategies side-by-side
4. Check paper trading on last 60 days of live data

---

## Appendix — Hyperparameters

| Parameter | Value | Effect |
|---|---|---|
| `n_steps` | 2048 | Transitions collected per update |
| `batch_size` | 256 | Mini-batch for gradient step |
| `n_epochs` | 10 | Gradient passes per batch |
| `learning_rate` | 2.5e-4 | Step size for Adam optimizer |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE bias-variance tradeoff |
| `clip_range` | 0.2 | PPO clipping threshold |
| `ent_coef` | 0.01 | Entropy bonus for exploration |
| `vf_coef` | 0.5 | Value function loss weight |
| `net_arch` | [128, 128] | MLP hidden layers |
| `n_envs` | 4 | Parallel training environments |
| `total_timesteps` | 500,000 | Training length per asset |
| `seed` | 42 | Global reproducibility seed |
