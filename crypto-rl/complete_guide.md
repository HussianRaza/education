# Crypto RL Trading Bot — Complete Guide with Viva Questions

---

# PART 1 — PROJECT OVERVIEW

---

## What This Project Is

This is a **reinforcement learning trading bot** that learns to trade Bitcoin (BTC), Ethereum (ETH), and Solana (SOL) using 5 years of historical daily price data (2020–2024). The bot is trained using **PPO (Proximal Policy Optimization)** — one of the most powerful and stable RL algorithms available — and its performance is compared against four classical trading strategies.

The entire system includes:
- A custom **Gymnasium trading environment** that simulates the market
- A **PPO agent** trained on Google Colab (free GPU)
- **Four classical baselines** (buy-and-hold, mean reversion, momentum, random)
- A **FastAPI backend** serving backtest results
- A **React + Plotly.js dashboard** for interactive visualization

**This is an educational project. It is not financial advice.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   yfinance → Raw OHLCV CSVs → ta indicators → Split CSVs       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT LAYER                            │
│         CryptoTradingEnv (Gymnasium)                            │
│   Observation: 260-dim vector | Actions: Discrete(5)            │
│   Reward: Rolling Sharpe - Drawdown Penalty - Tx Cost           │
└──────────────┬──────────────────────────────────────────────────┘
               ↓                              ↓
┌──────────────────────────┐    ┌─────────────────────────────────┐
│      PPO AGENT           │    │         4 BASELINES             │
│  Stable-Baselines3       │    │  Buy&Hold · Mean Rev            │
│  MlpPolicy [128,128]     │    │  Momentum · Random              │
│  500,000 timesteps       │    │                                 │
└──────────────┬───────────┘    └──────────────┬──────────────────┘
               └──────────────┬────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       API LAYER                                 │
│      FastAPI on port 8000 — 6 endpoints, in-memory cache        │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FRONTEND LAYER                              │
│      React + Vite + Plotly.js on port 3000                      │
│  Equity Curve · Comparison Table · Trade Log · Paper Trading    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
projectimplementation/
├── data/
│   ├── raw/                    ← Downloaded OHLCV CSVs (gitignored)
│   └── processed/              ← Indicator-enriched CSVs with split column
├── env/
│   ├── __init__.py             ← SEED = 42
│   ├── crypto_trading_env.py   ← The Gymnasium trading environment
│   └── reward.py               ← Rolling-Sharpe reward function
├── agents/
│   ├── ppo_agent.py            ← PPO wrapper (build/train/save/load/predict)
│   └── callbacks.py            ← TensorBoard + checkpoint callbacks
├── baselines/
│   ├── base.py                 ← Abstract base class
│   ├── buy_and_hold.py
│   ├── mean_reversion.py
│   ├── momentum.py
│   ├── random_agent.py
│   └── metrics.py              ← Sharpe, max DD, win rate, total return, Calmar
├── api/
│   ├── main.py                 ← FastAPI app, CORS, model preloading
│   ├── routes.py               ← 6 endpoints
│   ├── schemas.py              ← Pydantic response models
│   └── cache.py                ← In-memory result cache
├── scripts/
│   ├── download_data.py
│   ├── compute_indicators.py
│   ├── smoke_test_env.py
│   ├── train_ppo.py
│   ├── run_baselines.py
│   └── evaluate.py
├── notebooks/
│   └── train_colab.ipynb       ← Google Colab training workflow
├── models/
│   └── ppo_btc_final.zip       ← Pre-trained PPO model
└── results/
    ├── comparison_2024.csv
    └── equity_curves.png
```

---

## Phase 1 — Data Pipeline

### Data Source

Market data is downloaded using **yfinance**, which pulls free daily OHLCV (Open, High, Low, Close, Volume) bars from Yahoo Finance. Three assets are supported: `BTC-USD`, `ETH-USD`, `SOL-USD`. Data spans 2020-01-01 to 2024-12-31 — approximately 1,826 bars per asset.

### Data Splits

The data is divided into three non-overlapping time periods:

| Split | Period | Purpose |
|---|---|---|
| **Train** | 2020-01-01 → 2022-12-31 | Agent learns from this data |
| **Validation** | 2023-01-01 → 2023-12-31 | Monitor training for overfitting |
| **Test** | 2024-01-01 → 2024-12-31 | Final unseen evaluation |

### Technical Indicators

Five indicators are computed using the `ta` (Technical Analysis) library:

| Indicator | Window | What it Captures |
|---|---|---|
| **RSI** (Relative Strength Index) | 14 periods | Overbought/oversold momentum |
| **MACD Diff** | 12/26/9 EMA | Trend direction and momentum |
| **BB Width** (Bollinger Bands) | 20 periods, 2σ | Market volatility regime |
| **EMA50 Distance** | 50 periods | Price deviation from medium-term trend |
| **Volume Z-Score** | 20-period rolling | Unusual volume (precedes big moves) |

**Critical design decision:** Indicators are computed **per split**, using only data within that split. This prevents **look-ahead bias** — no future information leaks into past training data.

---

## Phase 2 — The Trading Environment

### Gymnasium API

The environment follows the **Gymnasium API** (the maintained successor to OpenAI Gym):

```python
env = CryptoTradingEnv(df, initial_balance=10_000)

obs, info = env.reset(seed=42)
# obs: numpy array of shape (260,)
# info: empty dict

obs, reward, terminated, truncated, info = env.step(action)
# action: integer 0-4
# reward: float
# terminated: bool — portfolio collapsed below 50% of initial
# truncated: bool — end of data reached
# info: {"net_worth", "balance", "asset_held", "step"}
```

### Observation Space — `Box(260,)`

The 260-dimensional observation vector the agent receives at each time step:

```
┌─────────────────────────────────────────────────────────┐
│  OHLCV WINDOW (250 values)                              │
│  Last 50 bars × 5 features (open, high, low, close, vol)│
│  Z-scored with TRAINING split mean/std only             │
├─────────────────────────────────────────────────────────┤
│  TECHNICAL INDICATORS (5 values)                        │
│  RSI · MACD diff · BB width · EMA50 dist · vol_zscore   │
├─────────────────────────────────────────────────────────┤
│  PORTFOLIO STATE (5 values)                             │
│  position · net_worth_norm · unrealized_pnl             │
│  peak_pct · VaR(5%)                                     │
└─────────────────────────────────────────────────────────┘
```

**Portfolio state features:**
- `position` — 0.0 (flat) or 1.0 (holding)
- `net_worth_norm` — (net_worth / initial_balance) − 1.0
- `unrealized_pnl` — asset_value / net_worth (fraction of portfolio in open position)
- `peak_pct` — (net_worth / peak_value) − 1.0 (how far below the all-time high)
- `var5` — 5th percentile of last 20 daily returns (simple Value at Risk estimate)

### Action Space — `Discrete(5)`

| ID | Action | Effect |
|---|---|---|
| 0 | Buy 25% | Spend 25% of available cash on the asset |
| 1 | Buy 50% | Spend 50% of available cash on the asset |
| 2 | Hold | Do nothing |
| 3 | Sell 25% | Sell 25% of held asset units |
| 4 | Sell 50% | Sell 50% of held asset units |

Fractional actions give the agent nuanced control. It can scale into a position over multiple days rather than making all-or-nothing bets.

### Portfolio Bookkeeping

```python
net_worth = self.balance + self.asset_held × current_price
```

The environment tracks `balance` (USD cash), `asset_held` (units), `net_worths` (history), `peak_value` (running maximum), and `trades` (list of trade dicts).

### Termination Conditions

- **Terminated:** `net_worth < 0.5 × initial_balance` — the agent has lost more than 50%.
- **Truncated:** `current_step >= end_of_data` — the episode ran through all available bars.

### Reward Function

```python
reward = rolling_sharpe - drawdown_penalty - transaction_cost
reward = clip(reward, -10, +10)
```

**1. Rolling Sharpe (base signal):**
```
sharpe = √252 × (mean_30_step_return − rf_daily) / std_30_step_return
rf_daily = 4% / 252 = 0.0001587
```
Computed over the last 30 portfolio returns. Rewards consistent, low-volatility gains over risky one-off profits.

**2. Drawdown Penalty:**
```
drawdown = (peak_value − current_value) / peak_value
dd_penalty = max(0, drawdown − 0.20) × 2.0
```
No penalty for drawdowns under 20%. Above 20%, the penalty grows linearly at 2× rate.

**3. Transaction Cost:**
```
tx_cost = 0.001 × trade_notional
```
0.1% fee per trade. Discourages excessive trading on noise.

**4. Safety clipping:**
Any non-finite reward (caused by near-zero variance in Sharpe denominator) is replaced with 0. Final reward is clipped to `[−10, +10]`.

---

## Phase 3 — The PPO Agent

### Hyperparameters

```python
PPO_HYPERPARAMS = dict(
    policy        = "MlpPolicy",
    n_steps       = 2048,       # steps collected before each update
    batch_size    = 256,        # mini-batch size
    n_epochs      = 10,         # gradient passes per collected batch
    learning_rate = 2.5e-4,    # Adam optimizer step size
    gamma         = 0.99,       # discount factor
    gae_lambda    = 0.95,       # GAE bias-variance tradeoff
    clip_range    = 0.2,        # PPO clipping parameter (ε)
    ent_coef      = 0.01,       # entropy bonus for exploration
    vf_coef       = 0.5,        # value function loss weight
    policy_kwargs = dict(net_arch=[128, 128]),
    seed          = 42,
)
```

### Neural Network

```
Input: 260 values
    ↓
[Dense 128, ReLU]
    ↓
[Dense 128, ReLU]
   ↙         ↘
Actor       Critic
[Dense 5]   [Dense 1]
[Softmax]   (value estimate)
```

### Training Workflow

```
Collect 2,048 steps (4 envs × 512 each)
    ↓
Compute advantages using GAE
    ↓
Run 10 gradient updates on mini-batches of 256
    ↓
Discard data, collect fresh experience
    ↓
Repeat ≈ 244 times → 500,000 total timesteps
```

### Training Infrastructure

Training runs on **Google Colab** (free T4 GPU), not locally. The `notebooks/train_colab.ipynb` notebook automates the full pipeline. Pre-trained models are committed to the repo as `models/ppo_btc_final.zip`.

---

## Phase 4 — Classical Baselines

| Strategy | Entry Rule | Exit Rule | Market Regime |
|---|---|---|---|
| **Buy and Hold** | Buy all-in day 1 | Sell last day | Bull markets |
| **Mean Reversion** | Price < MA20 × 0.98 | Price > MA20 | Ranging markets |
| **Momentum** | 5-day consecutive up streak | 3-day consecutive down streak | Trending markets |
| **Random** | Random action (seed=42) | Random | None — statistical baseline |

### Performance Metrics

| Metric | Formula |
|---|---|
| Sharpe Ratio | √252 × (mean_return − rf) / std_return |
| Max Drawdown | max((peak − value) / peak) |
| Total Return | (final − initial) / initial |
| Win Rate | profitable_sells / total_sells |
| Calmar Ratio | annualized_return / max_drawdown |

---

## Phase 5 — FastAPI Backend

### Endpoints

| Endpoint | Parameters | Returns |
|---|---|---|
| `GET /api/backtest` | `asset`, `agent` | Metrics + trade log + equity curve |
| `GET /api/compare` | `asset` | All 5 strategies compared |
| `GET /api/training-curves` | `asset` | Episode reward history |
| `GET /api/portfolio-history` | `asset`, `agent` | Portfolio value time series |
| `GET /api/disclaimer` | — | Disclaimer text |
| `GET /api/paper-trading` | `asset`, `agent`, `days` | Live data backtest |

All 3 PPO models are preloaded at startup. Results are cached in memory — backtests on historical data are deterministic, so they are computed once per session.

---

## Phase 6 — React Frontend

The dashboard (`frontend/`) provides:
- **Asset + Agent selector** — dropdowns for BTC/ETH/SOL and PPO/baselines
- **Equity curve** — portfolio value over the 2024 test year (Plotly.js)
- **Comparison table** — all 5 strategies × 5 metrics side-by-side
- **Training curves** — episode reward over 500,000 training steps
- **Trade log** — paginated table of every individual trade
- **Paper trading** — same strategies run on the last N days of live market data

---

---

# PART 2 — REINFORCEMENT LEARNING CONCEPTS IN DEPTH

---

## Concept 1 — What is Reinforcement Learning?

Reinforcement Learning is a machine learning paradigm where an **agent** learns to make decisions through interaction with an **environment**. Unlike supervised learning (which requires labeled examples) and unsupervised learning (which finds structure in data), RL learns from **reward signals** — numerical feedback indicating whether the agent's actions were good or bad.

```
          action aₜ
Agent ──────────────→ Environment
  ↑                        ↓
  └──── reward rₜ + state sₜ₊₁ ──┘
```

**The RL loop:**
1. Agent observes state `s`
2. Agent selects action `a`
3. Environment transitions to new state `s'`
4. Environment emits reward `r`
5. Agent updates its policy based on (s, a, r, s')
6. Repeat until episode ends

**Why RL for trading?** Trading is an inherently sequential decision-making problem. Each day the agent decides to buy, sell, or hold. The reward (profit or loss) is often delayed — buying today might not pay off for weeks. This delayed feedback structure is exactly what RL is designed to handle.

---

## Concept 2 — The Markov Decision Process (MDP)

Every RL problem is formally defined as a **Markov Decision Process (MDP)**: the tuple `(S, A, T, R, γ)`.

| Symbol | Name | This Project |
|---|---|---|
| `S` | State space | 260-dimensional Box (OHLCV + indicators + portfolio) |
| `A` | Action space | Discrete(5) — buy/hold/sell fractions |
| `T(s, a, s')` | Transition function | The market — determined by historical data |
| `R(s, a)` | Reward function | Rolling Sharpe − DD penalty − tx cost |
| `γ` | Discount factor | 0.99 |

**The Markov Property:** "The future depends only on the current state, not on history." Formally: `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_1, ..., s_t, a_1, ..., a_t)`

In practice, raw market price alone violates this — tomorrow's price does depend on recent history. We resolve this by including the **50-bar OHLCV window** in the observation vector. With enough recent history embedded in the state, the Markov property is approximately satisfied.

**Episodic vs Continuing tasks:**
- **Episodic:** The agent runs from a start state to a terminal state (one trading year = one episode). This project is episodic.
- **Continuing:** The agent runs forever with no natural endpoint.

---

## Concept 3 — Policy

A **policy** `π` is the agent's decision-making strategy. It maps states to actions.

**Deterministic policy:** `a = π(s)` — in the same state, always takes the same action.

**Stochastic policy:** `a ~ π(a|s)` — samples from a probability distribution over actions. For Discrete(5), this is a categorical distribution over {0, 1, 2, 3, 4}.

**In this project:**
- During training: stochastic policy → samples actions from the distribution → **exploration**
- During inference: `deterministic=True` → takes the action with highest probability → **exploitation**

The PPO actor network outputs **logits** for each of the 5 actions, which are passed through a softmax to produce probabilities:

```
obs → [128] → [128] → [5 logits] → softmax → [p₀, p₁, p₂, p₃, p₄]
```

During training, an action is sampled: `a ~ Categorical(p₀...p₄)`
During inference: `a = argmax(p₀...p₄)`

**Optimal policy:** The policy `π*` that maximizes expected cumulative discounted reward. Training is the process of finding `π*`.

---

## Concept 4 — Value Functions

Value functions estimate the expected cumulative reward from a given state (or state-action pair), providing a "map" of how good each situation is.

### State Value Function V(s)

```
V^π(s) = E_π [ Σ γᵏ rₜ₊ₖ | sₜ = s ]
```

"Starting from state `s` and following policy `π`, how much total discounted reward do I expect?"

In trading: V(s) answers "given this market observation and portfolio state, how valuable is this situation?"

### Action-Value Function Q(s, a)

```
Q^π(s, a) = E_π [ Σ γᵏ rₜ₊ₖ | sₜ = s, aₜ = a ]
```

"Taking action `a` in state `s`, then following `π`, how much reward do I expect?"

In trading: Q(s, buy_25%) vs Q(s, hold) tells you whether buying is better than holding right now.

### Advantage Function A(s, a)

```
A(s, a) = Q(s, a) - V(s)
```

"How much better is action `a` compared to the average action the policy would take in state `s`?"

- `A > 0` → this action is better than average → increase its probability
- `A < 0` → this action is worse than average → decrease its probability
- `A ≈ 0` → this action is about average → leave probability unchanged

The advantage function is the **core training signal** in PPO. It tells the optimizer exactly which actions to reinforce and which to suppress, without requiring a high or absolute target value.

### Bellman Equation

Value functions satisfy the Bellman equation — a recursive definition:

```
V^π(s) = Σ_a π(a|s) [ R(s,a) + γ Σ_s' T(s,a,s') V^π(s') ]
```

In words: "The value of a state equals the immediate reward plus the discounted value of the next state, averaged over all possible actions and transitions."

In practice, PPO estimates `V(s)` with a neural network (the critic) and updates it toward the observed returns using the Bellman equation as a training target.

---

## Concept 5 — Proximal Policy Optimization (PPO)

PPO is a **policy gradient** algorithm that directly optimizes the policy by gradient ascent. It is the industry-standard algorithm used in OpenAI's ChatGPT (RLHF), DeepMind robotics, and extensively validated in financial RL research (FinRL).

### Why PPO over Other Algorithms?

| Algorithm | Type | Trading Suitability |
|---|---|---|
| **PPO** | On-policy, policy gradient | Stable, well-validated in finance |
| DQN | Off-policy, value-based | Unstable with reward shaping |
| A3C | On-policy, async | More complex, similar performance |
| SAC | Off-policy, policy gradient | More sample efficient but harder to tune |
| DDPG | Off-policy, policy gradient | Designed for continuous actions |

PPO's key advantages:
1. **Stability:** The clipping mechanism prevents catastrophically large updates
2. **On-policy:** Each update uses freshly generated data — no stale experience
3. **Simplicity:** Fewer hyperparameters than SAC/DDPG; easier to tune
4. **Proven:** FinRL's benchmarks show PPO competitive or superior in financial tasks

### The Vanilla Policy Gradient Problem

The REINFORCE algorithm (vanilla policy gradient) updates the policy as:

```
∇L = E[ ∇ log π(a|s) × G ]
```

Where `G` is the return. Problem: if `G` is large and we took a bad action, this causes a huge bad update. The policy can collapse — becoming so bad it never recovers.

**PPO's solution:** Limit how much the policy can change in one update.

### PPO's Clipped Objective

Define the probability ratio:

```
r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

This ratio measures how much the new policy differs from the old one (the one that collected the data). If `r = 1`, the policy hasn't changed. If `r = 2`, the new policy is twice as likely to take action `a` as the old one.

**Unclipped objective (dangerous):**
```
L_unclipped = E[ r(θ) × A(s,a) ]
```
Maximizing this can push `r` to extreme values — destructive updates.

**PPO's clipped objective (safe):**
```
L_CLIP = E[ min( r(θ)×A , clip(r(θ), 1-ε, 1+ε)×A ) ]
```

With `ε = 0.2` (the `clip_range` hyperparameter):

- If advantage `A > 0` (good action): `r` is clipped at `1 + 0.2 = 1.2`. The policy can increase the action probability by at most 20%.
- If advantage `A < 0` (bad action): `r` is clipped at `1 - 0.2 = 0.8`. The policy can decrease the action probability by at most 20%.

The `min` ensures we take whichever is more conservative — clipped or unclipped — so we never make optimistically large updates.

**Intuition:** PPO acts like a cautious investor. Even if the evidence strongly suggests a certain action is great, it only increases its probability moderately. Overconfident policy changes are just as dangerous in RL as overconfident trades.

### Total PPO Loss

```
L_total = L_CLIP − ent_coef × H(π) + vf_coef × L_value
```

| Term | Coefficient | Purpose |
|---|---|---|
| `L_CLIP` | — | Maximize expected advantage (the main RL objective) |
| `H(π)` | 0.01 | Entropy bonus — encourage exploration |
| `L_value` | 0.5 | Value function loss (MSE between predicted and actual returns) |

### Training Loop in Detail

```
For each iteration (total_timesteps / n_steps iterations):
    1. ROLLOUT PHASE
       For n_steps = 2048 steps across n_envs = 4 environments:
           obs, action, reward, value, log_prob → stored in buffer
    
    2. COMPUTE RETURNS AND ADVANTAGES
       Use GAE (λ=0.95) to compute Aₜ for each transition
       Compute return Gₜ = Aₜ + V(sₜ) as value training target
    
    3. UPDATE PHASE (n_epochs = 10 passes)
       Shuffle buffer
       For each mini-batch of size batch_size = 256:
           Compute new log_probs and values for same (s, a) pairs
           Compute ratio r = exp(log_prob_new - log_prob_old)
           Compute L_CLIP, L_value, H(π)
           Gradient step on -L_total (Adam optimizer)
    
    4. Discard buffer → go to step 1
```

---

## Concept 6 — Generalized Advantage Estimation (GAE)

GAE addresses the **credit assignment problem**: how do we correctly assign blame or credit for a reward that arrived N steps after the action that caused it?

### Two Extremes

**Monte Carlo returns (λ=1, high variance):**
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γᴺrᴺ
A_t = G_t - V(s_t)
```
Unbiased (exact) but extremely noisy. A single lucky or unlucky episode can dominate the gradient.

**TD(0) (λ=0, high bias):**
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)   ← one-step TD error
A_t = δ_t
```
Low variance but biased — relies on the accuracy of the value function estimate, which is imperfect early in training.

### GAE — The Best of Both

```
A_t^GAE = Σ_{k=0}^{∞} (γλ)^k δ_{t+k}

where δ_{t+k} = r_{t+k} + γV(s_{t+k+1}) - V(s_{t+k})
```

With `λ = 0.95`:
- `(γλ)^0 = 1.0` → immediate TD error fully counted
- `(γλ)^1 = 0.94` → one-step-ahead TD error weighted 94%
- `(γλ)^5 = 0.74` → five-step-ahead TD error weighted 74%
- `(γλ)^20 = 0.28` → 20-step-ahead TD error weighted 28%

Distant rewards matter but are downweighted. The agent correctly learns that a buying decision today is responsible for a profit that materializes over the next 20 days.

**In trading context:** The agent buys BTC on day 1. Prices rise over 14 days and it sells. GAE distributes credit back through all 14 hold decisions — each one "earned" a portion of the eventual profit — not just the final sell.

---

## Concept 7 — Discount Factor (γ = 0.99)

The discount factor determines how much the agent values future rewards relative to immediate rewards:

```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... = Σ γᵏ r_{t+k}
```

With `γ = 0.99`:

| Steps into future | Discount |
|---|---|
| 1 day | 0.99 (99% of face value) |
| 7 days | 0.93 |
| 30 days | 0.74 |
| 100 days | 0.37 |
| 365 days | 0.026 |

**Why not γ = 1.0 (equal weight all future rewards)?** Without discounting:
- Infinite horizon problems have infinite returns (mathematically undefined)
- The agent has no preference for getting rewards sooner — but time value of money is real

**Why not γ = 0.5 (very short-sighted)?** The agent would ignore whether a trade ultimately becomes profitable after a week.

**γ = 0.99** means the agent cares a lot about returns over the next month but heavily discounts anything beyond 6 months. This is appropriate for a daily trading bot.

---

## Concept 8 — Reward Shaping in Detail

**Reward shaping** is the design of the reward function to accelerate learning. Poor reward design is the single most common cause of RL training failure.

### Why Raw P&L Fails

**Problem 1 — Sparsity:** Profit is only realized at sell events. If sells are rare, the agent goes thousands of steps with reward = 0 and learns nothing.

**Problem 2 — Risk blindness:** Doubling returns by taking 10× the risk is not better. Raw P&L doesn't penalize volatility.

**Problem 3 — Short-term bias:** An agent rewarded only on realized P&L will trade frequently to realize small gains, ignoring large unrealized positions.

### The Rolling Sharpe Reward

```python
REWARD_WINDOW = 30
ANNUALIZATION = 252 ** 0.5  # ≈ 15.87
RISK_FREE_DAILY = 0.04 / 252  # 4% annual ≈ 0.000159 daily

window = net_worths[-30:]
returns = diff(window) / window[:-1]   # daily portfolio returns

sharpe = ANNUALIZATION × (mean(returns) - RISK_FREE_DAILY) / std(returns)
```

**Why Sharpe as reward?**
- Dense: computed at every step, not just at sells
- Risk-adjusted: penalizes volatility directly
- Meaningful scale: Sharpe = 1.5 means good; = -1 means bad; agent learns this scale

**The 30-step window** is a balance:
- Too short (5 steps): noisy, dominated by single price swings
- Too long (100 steps): sluggish, doesn't respond to strategy changes

### The Drawdown Penalty

```python
DD_THRESHOLD = 0.20
DD_SCALE = 2.0

drawdown = max(0, (peak_value - current_value) / peak_value)
dd_penalty = max(0, drawdown - DD_THRESHOLD) × DD_SCALE
```

Designed as a **threshold penalty** rather than a continuous one:
- Drawdowns under 20% are free — normal fluctuations should not be penalized
- Above 20%, penalty grows linearly at 2× — strongly discourages catastrophic loss
- The threshold mimics real risk management: a 5% drawdown is normal, a 30% drawdown requires intervention

### The Transaction Cost

```python
TX_COST_RATE = 0.001  # 0.1%
tx_cost = TX_COST_RATE × trade_notional
```

Without this, the agent discovers it can achieve high Sharpe by trading on every tiny fluctuation (each individual trade looks profitable in isolation). The 0.1% fee makes excessive trading costly, forcing the agent to only trade when it has strong conviction.

### Reward Clipping

```python
reward = 0.0 if not np.isfinite(reward) else reward
reward = np.clip(reward, -10.0, 10.0)
```

**Why clip?** The Sharpe denominator is `std(returns)`. When volatility is near zero (e.g., a long hold period in a stable market), `std ≈ 0` and `sharpe → ±∞`. A reward of ±10,000 would completely dominate one gradient update, collapsing the policy.

---

## Concept 9 — Exploration vs. Exploitation

The fundamental dilemma in RL:

- **Exploration:** Try new, uncertain actions to discover potentially better strategies
- **Exploitation:** Use the currently best-known strategy to maximize immediate reward

**The dilemma:** Always exploiting means you never discover better strategies. Always exploring means you never use what you've learned.

### PPO's Exploration Mechanisms

**1. Stochastic policy:** During training, actions are sampled from the probability distribution rather than taken greedily. A policy with action probabilities `[0.25, 0.2, 0.1, 0.2, 0.25]` explores all five actions.

**2. Entropy bonus (`ent_coef=0.01`):**
```
Entropy H(π) = -Σ π(a|s) log π(a|s)
```
High entropy = uniform distribution = maximum exploration. Low entropy = peaked distribution = exploitation. By adding `+0.01 × H(π)` to the objective, the policy is rewarded for staying uncertain, preventing it from prematurely committing to a suboptimal strategy.

**3. Clip range prevents collapse:** Without clipping, a lucky sequence of trades could push the policy to near-certainty on one action very quickly. Clipping ensures the policy changes gradually, maintaining some exploration even as it converges.

**At inference time:** `deterministic=True` selects `argmax(π)` — pure exploitation. The stochastic exploration is a training artifact.

---

## Concept 10 — Vectorized Environments

```python
vec_env = make_vec_env(env_factory, n_envs=4, vec_env_cls=DummyVecEnv, seed=42)
```

A vectorized environment runs `n_envs=4` independent copies of the trading environment simultaneously. Each call to `vec_env.step([a0, a1, a2, a3])` executes one action in each environment and returns 4 sets of results.

**Benefits:**
1. **Throughput:** Collect 4× as many transitions per wall-clock second
2. **Decorrelation:** Each environment is at a different point in its episode, so transitions in the rollout buffer are less correlated → more stable gradient estimates
3. **Variance reduction:** Advantages averaged over more diverse experience

**DummyVecEnv vs SubprocVecEnv:**
- `DummyVecEnv`: runs all envs sequentially in one process. No multiprocessing overhead. Appropriate when the env is fast (as here — it's pure numpy).
- `SubprocVecEnv`: each env runs in a separate subprocess — true parallelism. Better when the env has slow I/O or heavy computation.

---

## Concept 11 — On-Policy vs Off-Policy Learning

**On-policy:** The agent learns from data generated by its current policy. Data is discarded after each update.

**Off-policy:** The agent learns from data generated by any policy (including old versions). Old data is stored in a replay buffer and reused.

**PPO is on-policy.** After collecting 2,048 steps with the current policy and running 10 gradient updates, all that data is thrown away. Fresh data is collected with the updated policy.

**Why on-policy for trading?**
- Financial markets are non-stationary (conditions change). Fresh on-policy data is more relevant.
- Off-policy methods (DQN, SAC) can use stale data from old market conditions, which may mislead the policy.
- On-policy methods have lower risk of distribution shift between the data-generating policy and the learning policy.

**The tradeoff:** On-policy methods are less sample efficient — they cannot reuse old data. The 500,000 timestep budget is used entirely for fresh rollouts.

---

## Concept 12 — The Actor-Critic Architecture

PPO uses an **actor-critic** architecture — two neural networks working together:

```
                  Observation s (260 values)
                           ↓
                  Shared Backbone (optional)
                  [Dense 128, ReLU] → [Dense 128, ReLU]
                      ↙                 ↘
            ACTOR                      CRITIC
     [Dense 5]                        [Dense 1]
     [Softmax]                        (no activation)
         ↓                                ↓
   π(a|s): action                    V(s): value
   probabilities                     estimate
```

**Actor (policy network):** Outputs a probability distribution over actions. This is the part that actually makes trading decisions. Updated by the clipped policy gradient.

**Critic (value network):** Outputs a scalar estimate of `V(s)`. Does not make decisions — it provides a baseline for computing advantages. Updated by minimizing `(V(s) - G_t)²` where `G_t` is the actual return.

**Why have a critic at all?** Without the critic, we'd need Monte Carlo returns to compute advantages — requiring the episode to complete before any learning happens. The critic provides a "running estimate" of future returns, enabling learning within an episode.

In SB3's `MlpPolicy`, the actor and critic share the same hidden layers by default. This parameter sharing improves data efficiency — both networks learn useful representations from the same features simultaneously.

---

## Concept 13 — Observation Normalization and Data Leakage

### Z-Score Normalization

Raw OHLCV prices range from $0.50 (SOL in early 2020) to $70,000+ (BTC in late 2021). Without normalization, the neural network would learn to be sensitive to absolute price levels rather than patterns. Z-scoring fixes this:

```python
z = (x - μ_train) / σ_train
```

**The critical rule:** `μ_train` and `σ_train` are computed **only on the training split (2020–2022)** and then frozen. The same statistics are applied to val and test data.

### Why This Matters — Data Leakage

If you compute normalization statistics on the full dataset (including test), you give the model indirect information about future data:

- The mean and std of test data "leak" into the model's input representation
- The model sees subtly different (correctly scaled) inputs during evaluation
- This inflates test performance and gives a false sense of generalization

This is **data leakage** — a fundamental error in ML evaluation. The gold standard is: treat the test set as if it doesn't exist until the very final evaluation. All preprocessing decisions (scaling, indicator computation) must be made using only training data.

### Indicators and Lookahead Bias

Similarly, technical indicators are computed per split. If you compute the 50-period EMA using the full dataset, a training bar would be influenced by future bars that haven't "happened yet" from the agent's perspective. This is **lookahead bias**.

In this project: `scripts/compute_indicators.py` processes each split independently. The train split's indicators use only 2020-2022 data. The test split's indicators use only 2024 data.

---

## Concept 14 — Reproducibility and Seeding

Every source of randomness is controlled:

```python
SEED = 42  # env/__init__.py

env.reset(seed=SEED)               # Gymnasium environment RNG
PPO(seed=SEED)                     # SB3: model init + data shuffling
np.random.default_rng(SEED)        # Random baseline
```

Fixed date splits (not random train/test assignment) ensure the exact same 2,836 bars are always in train, 365 in val, 366 in test.

**Why reproducibility matters:**
- Academic papers must be reproducible
- Debugging is impossible without determinism
- Comparing results between runs requires identical conditions

---

## Concept 15 — Evaluation Metrics in Depth

### Sharpe Ratio

```
Sharpe = √252 × (μ_daily − rf_daily) / σ_daily
```

- `μ_daily`: mean daily portfolio return
- `σ_daily`: standard deviation of daily returns
- `rf_daily = 4% / 252`: daily risk-free rate
- `√252`: annualization factor (252 trading days per year)

**Interpretation:**
| Sharpe | Assessment |
|---|---|
| < 0 | Strategy destroys value relative to risk-free |
| 0–1 | Acceptable but mediocre |
| 1–2 | Good — institutional quality |
| > 2 | Exceptional (very hard to sustain) |

**Why annualize?** Allows comparison across strategies with different time horizons.

### Maximum Drawdown

```
MDD = max_t [ (peak_{0..t} - value_t) / peak_{0..t} ]
```

Answers: "What was the worst peak-to-trough decline an investor would have experienced?"

- `MDD = 0.10` → at worst, 10% below peak (mild)
- `MDD = 0.50` → at worst, lost half the portfolio from peak (severe)

**Why it matters:** Total return doesn't capture the psychological and practical difficulty of a large drawdown. A 50% drawdown requires a 100% gain just to break even.

### Calmar Ratio

```
Calmar = Annualized_Return / Max_Drawdown
```

Higher is better. A Calmar of 2.0 means: for each unit of maximum drawdown risk, the strategy earned 2 units of annualized return. Favors strategies that generate good returns without catastrophic losses.

### Win Rate

```
Win Rate = profitable_sell_trades / total_sell_trades
```

Fraction of sell events that closed at a profit. High win rate alone is insufficient — a strategy that wins 90% of trades but loses 10× on each loss is terrible (this is the risk of stop-loss hunting).

---

## Concept 16 — TensorBoard Monitoring

During training, the `MetricsCallback` records:

```python
self.logger.record("train/reward", float(reward))           # per step
self.logger.record("train/reward_min", ...)                 # per rollout
self.logger.record("train/reward_mean", ...)
self.logger.record("train/reward_max", ...)
```

**What the curves tell you:**

| Pattern | Interpretation |
|---|---|
| Reward increases steadily | Agent is learning successfully |
| Reward plateaus early | Stuck in local optimum; try higher `ent_coef` |
| Reward unstable/oscillating | Learning rate too high or `clip_range` too large |
| Reward crashes mid-training | Policy collapsed; reduce `learning_rate` |
| Reward never improves | Reward function is broken; debug reward values |

The training curves saved to `results/training_curves_{asset}.json` are served by `/api/training-curves` and displayed in the dashboard.

---

---

# PART 3 — VIVA QUESTIONS AND ANSWERS

---

## Section A — Conceptual RL Questions

**Q1. What is the difference between reinforcement learning and supervised learning?**

Supervised learning trains on labeled (input, correct_output) pairs. The correct answer is always known. Reinforcement learning has no labels — the agent must discover good actions through trial and error, guided only by a reward signal. In trading, there are no correct labeled actions; the agent must learn what "buy at this market state" is worth through experience.

---

**Q2. What is a Markov Decision Process? Does the trading environment satisfy the Markov property?**

An MDP is a formal framework for sequential decision-making: `(S, A, T, R, γ)`. The Markov property states that the next state depends only on the current state and action, not on history. 

Raw price alone violates this (tomorrow's price depends on the last several days' trend). We resolve this by embedding a 50-bar OHLCV window in the observation. With enough recent history in the state, the Markov property is approximately satisfied — the current observation contains all practically relevant past information.

---

**Q3. What is the difference between a deterministic and stochastic policy? Why does PPO use stochastic during training?**

A deterministic policy always maps the same state to the same action. A stochastic policy outputs a probability distribution and samples from it. PPO uses stochastic during training for **exploration** — the agent needs to occasionally try non-greedy actions to discover better strategies. Without exploration, the policy converges to the first decent strategy it finds, which may be a local optimum. At inference time, we switch to deterministic (greedy) to exploit the learned policy.

---

**Q4. What is the advantage function and why is it used instead of raw returns?**

The advantage function `A(s,a) = Q(s,a) - V(s)` measures how much better action `a` is compared to the average action in state `s`. Using raw returns `G_t` as the gradient signal has high variance — a lucky episode produces large positive gradients for all actions taken, even bad ones. Subtracting the baseline `V(s)` removes the "luck" component, leaving only the signal about which specific actions were better or worse than average. This dramatically reduces gradient variance without introducing bias (the baseline doesn't change the expected gradient).

---

**Q5. Explain PPO's clipping mechanism in plain language.**

Imagine you're training a policy and the data shows that buying 25% was a great decision (high advantage). Vanilla policy gradient would increase the probability of buying by a large amount — potentially making the policy overconfident and ignoring other actions.

PPO says: "Even if the data strongly suggests buying is great, only increase its probability by at most 20% from where it was before this update." This prevents the policy from changing so drastically that it stops exploring and enters a bad regime it can't escape from. The `clip_range=0.2` is this 20% limit.

---

**Q6. What is Generalized Advantage Estimation (GAE)?**

GAE is a method to compute advantage estimates that balances bias and variance. Two extreme approaches exist: Monte Carlo returns (low bias, high variance — waits for the episode to end) and TD(0) (low variance, high bias — relies on an imperfect value function). GAE interpolates between them using `λ = 0.95`:

```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
```

Each δ is a one-step TD error. This sums up TD errors with exponentially decreasing weights, so distant rewards matter but contribute less. In trading, this correctly credits a buy decision made 14 days before a profitable sell — GAE distributes that credit backward through all the hold steps.

---

**Q7. Why is the discount factor set to 0.99 instead of 1.0?**

`γ = 1.0` means all future rewards are valued equally, creating mathematical problems (infinite sums for non-terminating tasks) and making the agent indifferent between rewards now vs. much later. `γ = 0.99` means a reward 100 steps away is worth only 37% of an immediate reward. This is realistic for trading — a profit next week is more certain than a profit next year, and the agent should prioritize near-term performance while still caring about medium-term outcomes. It also ensures the infinite sum `Σ γᵏrₖ` converges.

---

**Q8. What is the difference between `terminated` and `truncated` in Gymnasium?**

In Gymnasium 0.26+ (used here), `step()` returns a 5-tuple including both `terminated` and `truncated`:
- `terminated=True`: The episode ended due to a **natural terminal condition** — the portfolio dropped below 50% of initial balance. The environment reached a state where the episode legitimately ends.
- `truncated=True`: The episode ended due to an **external constraint** — we ran out of data (current_step ≥ end_of_data). The episode would continue but we artificially cut it off.

This distinction matters for value estimation: in terminated episodes, the value of the next state is 0 (there is no next state). In truncated episodes, the value of the next state is not 0 — we just stopped observing it. PPO uses this distinction in the advantage computation via the bootstrap value.

---

**Q9. What is the entropy bonus and what happens without it?**

The entropy bonus is `+ ent_coef × H(π)` added to the PPO objective. `H(π) = -Σ π(a|s) log π(a|s)` is maximized when the policy is uniform (maximum uncertainty) and minimized when the policy is deterministic.

With `ent_coef=0.01`, the policy is rewarded for staying somewhat uncertain — preventing it from collapsing to a single action early in training before it has adequately explored.

Without the entropy bonus: the policy often converges prematurely. For example, if the first few episodes show that holding is better than selling, the policy might collapse to near-100% probability of holding — missing that buy timing matters critically in a bull market.

---

**Q10. What is the credit assignment problem? How does PPO address it?**

Credit assignment is the problem of determining which past actions were responsible for a reward received now. In trading: if the agent buys BTC on day 1, holds for 20 days, and sells at a profit on day 21 — which actions (buy, hold×19, sell) deserve credit?

PPO addresses this through:
1. **GAE (λ=0.95):** Propagates credit backward through time, weighting immediate decisions more but still crediting distant ones
2. **Discount factor (γ=0.99):** Ensures decisions made recently are weighted more heavily than very old decisions
3. **Rollout of 2,048 steps:** The policy observes consequences of its decisions within the same rollout before updating

---

## Section B — Project-Specific Questions

**Q11. Why did you choose PPO over DQN for this project?**

Three reasons. First, the action space is small and discrete (5 actions) but reward shaping is complex — PPO's policy gradient approach handles shaped rewards better than value-based DQN, which can diverge with strong reward shaping. Second, PPO produces stochastic policies that naturally explore, whereas DQN requires explicit ε-greedy exploration. Third, FinRL's benchmarks show PPO consistently outperforming DQN in financial trading tasks with similar hyperparameter budgets.

---

**Q12. Why use Discrete(5) actions instead of Discrete(2) (just buy/sell) or a continuous action space?**

Discrete(2) forces all-or-nothing decisions, which is unrealistic and encourages boom-bust patterns. The agent might go all-in on every buy signal, which amplifies losses when the signal is wrong.

Discrete(5) allows fractional buying and selling — the agent can scale into and out of positions gradually. This more closely mirrors professional trading practice and gives the agent nuanced control.

A continuous action space (e.g., "buy X% of portfolio") would be more expressive but requires a different algorithm (SAC, DDPG, TD3) and is harder to train stably. The 5-action discrete approach is the simplest formulation that captures meaningful position sizing.

---

**Q13. Explain the observation vector design. Why 50-bar OHLCV window?**

The 50-bar window is chosen to satisfy the Markov property approximately: 50 trading days captures roughly 2.5 months of daily patterns. This covers:
- Short-term momentum (5–10 days)
- Medium-term trends (20–50 days)
- Key technical indicator periods (RSI: 14, EMA: 50)

Shorter windows (10 bars) miss medium-term trends. Longer windows (100+ bars) make the observation too large, slowing training and potentially including irrelevant old information. 50 bars is the standard window used across most financial RL literature.

The OHLCV window is z-scored with training statistics to prevent absolute price levels from dominating the neural network.

---

**Q14. Why is the Sharpe ratio used as the reward signal rather than profit?**

Sharpe ratio is preferred because:
1. **Dense feedback:** Computed at every step (even during holding periods), not just at sell events. This gives the agent constant learning signal.
2. **Risk adjustment:** Sharpe penalizes volatility. An agent maximizing Sharpe learns to generate consistent returns rather than gambling on big wins.
3. **Prevents over-trading:** Without risk adjustment, the agent might buy and sell every day to realize tiny gains. Sharpe's std penalty discourages high-frequency trading that introduces noise.
4. **Scale invariance:** Sharpe has a meaningful universal scale (>1 is good, >2 is excellent), making the reward easier for the neural network to interpret.

---

**Q15. How does the drawdown penalty in the reward function work?**

```python
drawdown = (peak_value - current_value) / peak_value
dd_penalty = max(0, drawdown - 0.20) × 2.0
```

A threshold of 20% is set — drawdowns below this are "free" (no penalty). This reflects the reality that small fluctuations are normal and shouldn't be penalized. Above 20%, the penalty grows linearly at 2× rate, creating a strong incentive to cut losses before they become catastrophic.

The threshold design encourages the agent to hold through normal volatility but aggressively protect capital during major downturns. Without this, the agent might sell too quickly on normal dips (excessive caution) or hold too long during crashes (excessive optimism).

---

**Q16. What is look-ahead bias and how is it prevented?**

Look-ahead bias occurs when a model is trained or evaluated using information that wouldn't have been available at the time of the decision. In trading: if the RSI computed on day 100 uses price data from days 1–100, that's fine. But if the scaler normalizing day 100's price was fit using data from days 1–365 (including future data), the model indirectly "knows" that prices will be higher in Q4, biasing it.

Prevention in this project:
1. **Technical indicators:** Computed per split using only within-split data
2. **OHLCV normalization:** `μ_train` and `σ_train` computed on training data only, then frozen and applied to val/test
3. **Data split:** Strict chronological split — no shuffling, no data from after the training period

---

**Q17. How are the baselines evaluated fairly against the PPO agent?**

All strategies — PPO and all four baselines — are evaluated on:
- **Same data:** 2024 test split (never seen during training)
- **Same starting capital:** $10,000
- **Same metrics:** Sharpe, max DD, total return, win rate, Calmar
- **Same metric implementation:** Single `baselines/metrics.py` module shared by all — no metric drift

The only difference is the strategy itself. This controlled comparison ensures that performance differences are due to the strategy, not evaluation methodology.

---

**Q18. What is the termination condition and why 50% of initial balance?**

The environment terminates early (`terminated=True`) when net worth drops below `0.5 × initial_balance` — a 50% loss. This serves two purposes:

1. **Training efficiency:** An agent that has lost 50% has likely failed. Continuing the episode teaches the agent only bad strategies (since it has little capital left to work with). Early termination sends a clear failure signal.

2. **Reward signal:** By ending the episode early, the episode's total return is severely negative, strongly discouraging the policy that led to this outcome via the advantage function.

50% was chosen as the threshold because: a 50% loss is catastrophic but not impossible to recover from with correct strategy. A higher threshold (e.g., 90% loss) would allow too much damage before terminating. A lower threshold (e.g., 10% loss) would terminate too many episodes early, preventing the agent from learning to trade through normal market volatility.

---

**Q19. Why are 4 parallel environments used for training?**

Vectorized environments with `n_envs=4` provide:
1. **More experience per step:** 4 transitions collected per `step()` call instead of 1
2. **Decorrelation:** Each of the 4 environments is at a different point in its episode. When data is collected into the rollout buffer, consecutive transitions come from different environments at different market states — reducing correlation that would otherwise harm gradient quality
3. **Better advantage estimates:** With more diverse transitions, the batch statistics (mean advantage, etc.) are less noisy

`n_envs=4` with `n_steps=2048` means each environment contributes 512 transitions per rollout.

---

**Q20. How does caching work in the API and why is it safe?**

The API uses a simple in-memory Python dict keyed by `(endpoint, asset, agent)`:

```python
cache = {}
cache[("backtest", "btc", "ppo")] = result
```

This caching is safe because backtests on historical data are **fully deterministic** — given the same model, same data split, and same seed, the backtest always produces identical results. There is no need for cache invalidation during a server session (the models and data don't change).

Paper trading uses time-bucketed cache keys (`bucket = int(time.time() / 300)`) so live market data refreshes every 5 minutes while still caching within that window.

---

## Section C — Technical and Implementation Questions

**Q21. What is the Gymnasium 5-tuple step return and why was it changed from the old gym?**

Old OpenAI Gym (≤0.25) returned a 4-tuple: `(obs, reward, done, info)`. Gymnasium 0.26+ returns a 5-tuple: `(obs, reward, terminated, truncated, info)`.

The split matters for value estimation. If an episode ended because the agent failed (`terminated=True`), the bootstrap value of the next state is 0 — there's nothing after a terminal state. If it ended because the data ran out (`truncated=True`), the environment would have continued — the next state has real value that should be bootstrapped. Using a single `done=True` for both cases causes incorrect value estimates in the truncated case, leading to biased advantage estimates near episode boundaries.

---

**Q22. What are the 5 technical indicators used and what market information does each capture?**

| Indicator | Information |
|---|---|
| RSI (14-period) | Momentum oscillator. Values near 70+ suggest overbought (potential reversal down). Values near 30 suggest oversold (potential bounce). |
| MACD Diff (12/26/9) | Trend and momentum. The difference between fast and slow EMA, smoothed by signal line. Positive = bullish momentum, negative = bearish. |
| BB Width (20-period, 2σ) | Volatility. Wide bands = high volatility regime. Narrow bands often precede big moves (volatility compression). |
| EMA50 Distance | Medium-term trend. (Close - EMA50) / EMA50. Positive = price above trend (bullish), negative = below trend. |
| Volume Z-Score | Unusual volume. Computed as (vol - 20d mean) / 20d std. High z-score = unusual volume spike, often associated with strong directional moves. |

---

**Q23. How is the transaction cost in the reward function justified?**

Real cryptocurrency exchanges charge fees of 0.05%–0.5% per trade (market maker/taker rates). The 0.1% fee in this project is a reasonable midpoint estimate. It serves two purposes:

1. **Realism:** Makes the simulation more accurate to real trading conditions
2. **Regularization:** Without tx costs, the agent has no incentive to avoid unnecessary trades. The fee creates a threshold — only trade when the expected gain exceeds 0.1% of the trade size.

In real trading, tx costs compound significantly with frequent trading: 0.1% per day × 252 days = ~28% drag per year. This simulation makes the agent learn this implicitly.

---

**Q24. What is the difference between DummyVecEnv and SubprocVecEnv?**

Both create vectorized environments running `n_envs` copies in parallel, but:

- `DummyVecEnv`: All environments run in the **same Python process**, called sequentially in a loop. Zero multiprocessing overhead. Best for fast environments (like this one — pure numpy arithmetic).
- `SubprocVecEnv`: Each environment runs in a **separate subprocess** with inter-process communication via pipes. True CPU parallelism (bypasses Python's GIL). Better for slow environments with I/O, simulation, or heavy computation.

Since `CryptoTradingEnv` is very fast (each step is a few numpy operations), `DummyVecEnv` is used by default. `SubprocVecEnv` is available as an option for Colab's multi-core setup where the subprocess overhead is amortized over longer episodes.

---

**Q25. Why is training done on Google Colab rather than locally?**

PyTorch, Stable-Baselines3, and the training loop require a GPU to run efficiently. Training 500,000 timesteps for 3 assets on a CPU would take many hours. Google Colab provides free access to NVIDIA T4 GPUs, reducing training time to 20-60 minutes per asset.

Additionally, this avoids the requirement for the user to have a high-end local machine. The workflow is: train on Colab, download the model `.zip` file, commit it to the repo. Inference (prediction during backtesting and API serving) runs on a tiny MLP (two 128-neuron layers) and completes in under 10ms on any modern CPU.

---

**Q26. What happens if the PPO model file is missing when the API starts?**

The API's `lifespan` context manager attempts to load all 3 models at startup. If a model file is missing, that asset's model is set to `None` in `app.state`. When a request comes in for that asset with agent=ppo, the route handler checks `getattr(request.app.state, f"ppo_{asset}", None)` and raises `HTTPException(404)`. The other assets with successfully loaded models continue to work. This graceful degradation allows the API to serve baselines for all assets even if PPO models are partially missing.

---

**Q27. How does the mean reversion baseline decide when to buy or sell?**

```python
# Buy when price is 2% below 20-day moving average
if close < ma20 × 0.98 and position == 0:
    BUY ALL-IN

# Sell when price rises back above moving average
if close > ma20 and position > 0:
    SELL ALL
```

The 2% threshold prevents trading on tiny fluctuations around the MA. The strategy assumes prices are pulled toward their mean — when they deviate significantly downward, it's a buying opportunity; when they return to average, take profit. This works in **ranging markets** but performs poorly in **trending markets** where prices sustainably deviate from the historical average.

---

**Q28. What is the Calmar ratio and when is it more useful than the Sharpe ratio?**

```
Calmar = Annualized_Return / Max_Drawdown
```

The Calmar ratio focuses specifically on the relationship between return and the worst case loss scenario. It's more useful than Sharpe when:
1. **Returns are non-normal:** Sharpe assumes normally distributed returns. Crypto has fat tails and skewness. Calmar only cares about the single worst drawdown, not the distribution.
2. **Tail risk is the primary concern:** For a risk-averse investor, the maximum loss is more important than average volatility. Two strategies with identical Sharpe can have very different Calmar if one had a single catastrophic drawdown.

Example: Two strategies with 15% annual return. Strategy A had a 10% max drawdown (Calmar=1.5). Strategy B had a 40% max drawdown (Calmar=0.375). Sharpe might be similar; Calmar correctly identifies Strategy A as far superior.

---

**Q29. What are the main risks of deploying this as a live trading system?**

This project has several fundamental limitations for live deployment:

1. **Historical simulation ≠ live trading:** No slippage (large orders move the price), no liquidity constraints, no exchange downtime, no API rate limits.
2. **Overfitting to 2020–2024:** The agent may have memorized specific market regimes (COVID crash, 2021 bull run, 2022 bear market). Different future conditions could produce very different results.
3. **Daily resolution:** Crypto is 24/7 and extremely volatile intraday. A daily model misses intraday price swings that can be very significant.
4. **No position sizing beyond 25%/50% fractions:** No portfolio-level risk management (e.g., Kelly criterion, VaR-based sizing).
5. **Single asset:** No diversification across assets or asset classes.
6. **Model staleness:** Market regimes change; a model trained through 2024 should be periodically retrained.

---

**Q30. What would you change if you had more time/compute?**

Strong answer covers multiple dimensions:

**Algorithm improvements:**
- **SAC (Soft Actor-Critic):** Off-policy, more sample efficient, maximum entropy framework — often outperforms PPO in financial tasks
- **Hyperparameter optimization with Optuna:** Systematically search `γ`, `lr`, `n_steps`, `ent_coef` rather than using manually chosen values

**Environment improvements:**
- **Intraday resolution:** Hourly bars would give 8,760 bars/year instead of 365
- **Multi-asset environment:** Let the agent allocate across BTC/ETH/SOL simultaneously, enabling portfolio effects
- **More realistic transaction costs:** Model slippage, bid-ask spread, and liquidity constraints

**Data improvements:**
- **Alternative data:** Crypto fear/greed index, on-chain metrics (active addresses, exchange flows), social sentiment
- **Order book features:** Bid-ask depth provides information about near-term price direction

**Evaluation improvements:**
- **Walk-forward validation:** Train on 2020-2021, test on 2022; retrain on 2020-2022, test on 2023; etc. — more robust than a single train/test split
- **Monte Carlo simulation:** Bootstrap the test period to estimate distribution of outcomes

---

## Section D — Quick-Fire Conceptual Questions

**Q31. What is the Bellman equation?**
A recursive definition of the value function: `V(s) = E[r + γV(s')]`. The value of a state equals the expected immediate reward plus the discounted value of the next state. Used to train the critic network.

**Q32. What is a rollout buffer?**
The data structure storing `(s, a, r, V(s), log_prob)` tuples collected during the rollout phase. In PPO, it holds `n_steps × n_envs` transitions. Cleared after each update cycle.

**Q33. What is the difference between model-free and model-based RL?**
Model-free RL (PPO, DQN) learns directly from experience without modeling the environment dynamics. Model-based RL learns a model of `T(s,a,s')` and uses it for planning. PPO is model-free — it doesn't model the market; it just learns from observed transitions.

**Q34. What is entropy in the context of policy?**
`H(π) = -Σ π(a|s) log π(a|s)`. Measures uncertainty/randomness in the policy. Maximum entropy = uniform distribution (agent equally likely to take any action). Minimum entropy = deterministic policy. PPO maximizes entropy as a regularizer to prevent premature convergence.

**Q35. Why is `n_epochs=10` used in PPO?**
After collecting a rollout, PPO runs 10 passes over the data to squeeze out as much learning as possible before discarding it (since it is on-policy and data is expensive to collect). More epochs would risk over-optimizing on the same batch, causing the policy to drift too far from the data-generating policy (which the clipping mechanism already limits). 10 is a standard value from PPO's original paper and FinRL's tuning.

**Q36. What does `gae_lambda=0.95` control?**
The exponential decay rate for GAE. `λ=0` → one-step TD (low variance, high bias). `λ=1` → full Monte Carlo (high variance, low bias). `λ=0.95` is the standard choice: heavily weights near-term TD errors while still incorporating information from further into the future.

**Q37. What is the purpose of the critic's value loss (`vf_coef=0.5`)?**
The critic must accurately predict `V(s)` to compute good advantage estimates. Without training the critic, advantages would be based on poor value estimates, degrading policy gradient quality. The `vf_coef=0.5` scales the value loss relative to the policy loss so both networks are updated appropriately.

**Q38. What is the learning rate and what happens if it's too high/low?**
`lr=2.5e-4` is the Adam optimizer step size. Too high → training diverges, policy collapses, rewards crash. Too low → training is slow, may converge to suboptimal local optima within the timestep budget. `2.5e-4` is FinRL's validated value for trading environments.

**Q39. How does the `check_env()` utility help?**
`stable_baselines3.common.env_checker.check_env(env)` verifies that the environment conforms to the Gymnasium API: correct observation/action space types, proper reset/step signatures, 5-tuple step return, space bounds respected. Catches API mismatches that would fail silently during training (e.g., wrong dtype in observation, missing info dict key).

**Q40. What would indicate overfitting in this RL project?**
If the agent achieves a Sharpe ratio of 2.5 on the training data but only 0.2 on the 2024 test data, that is strong evidence of overfitting — the policy memorized specific training market patterns rather than learning generalizable trading principles. This is particularly risky if the training period (2021 bull market) was very different from the test period (2024 post-ETF-approval market). Prevention: more training data, higher `ent_coef` for more exploration, regularization.

---

*End of Complete Guide*
