# Reinforcement Learning Concepts Used in This Project

This document explains every RL concept, term, and technique used in the Crypto Trading Bot project — from first principles to how each is applied here.

---

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment**. The agent takes **actions**, the environment transitions to a new **state**, and the agent receives a **reward** signal. Over time, the agent learns a **policy** — a mapping from states to actions — that maximizes cumulative reward.

```
Agent ──action──→ Environment
  ↑                    ↓
  └──── reward + state ←┘
```

**Key difference from supervised learning:** There are no labeled examples. The agent must discover which actions lead to good outcomes through trial and error.

**Key difference from unsupervised learning:** There is a clear objective signal — the reward.

### Why RL for Trading?

Trading is a sequential decision-making problem. At each market bar, the agent observes the current market state and must decide to buy, sell, or hold. The outcome of this decision isn't known until future price movements reveal whether it was profitable. This delayed feedback and sequential structure maps naturally to the RL framework.

---

## 2. The Markov Decision Process (MDP)

All RL problems are formalized as a **Markov Decision Process (MDP)**, defined by the tuple `(S, A, T, R, γ)`:

| Symbol | Name | In This Project |
|---|---|---|
| `S` | State space | 265-dimensional observation vector (OHLCV window + indicators + portfolio state) |
| `A` | Action space | {0, 1, 2, 3, 4} — buy 25%, buy 50%, hold, sell 25%, sell 50% |
| `T` | Transition function | The market — next price is determined by historical data |
| `R` | Reward function | Rolling Sharpe − drawdown penalty − transaction cost |
| `γ` | Discount factor | 0.99 — future rewards are slightly less valuable |

**The Markov property:** The next state depends only on the current state and action, not on the full history. In this project, we partially satisfy this by including a 50-bar OHLCV window in the observation — giving the agent enough history so that the current observation approximately summarizes all relevant past information.

---

## 3. Policy

A **policy** `π` is the agent's strategy: given a state `s`, it specifies which action `a` to take.

- **Deterministic policy:** `a = π(s)` — always takes the same action in the same state
- **Stochastic policy:** `a ~ π(a|s)` — samples an action from a probability distribution

PPO (used here) maintains a **stochastic policy** during training (for exploration) but switches to **deterministic** action selection at inference time (`predict(obs, deterministic=True)`).

---

## 4. Value Functions

Value functions estimate how good it is to be in a particular state (or state-action pair).

### State Value Function V(s)
`V(s) = E[sum of future discounted rewards starting from state s under policy π]`

This answers: "Starting from this market observation, how much total reward can I expect if I follow the current policy?"

### Action-Value Function Q(s, a)
`Q(s, a) = E[reward from taking action a in state s, then following policy π]`

This answers: "If I buy 25% right now, and then continue trading optimally, what reward do I expect?"

### Advantage Function A(s, a)
`A(s, a) = Q(s, a) - V(s)`

This answers: "How much better is action `a` compared to the average action the policy would take in state `s`?" Positive advantage → action is better than average. Negative → worse.

PPO uses the advantage function extensively — it is the core signal telling the policy which actions to increase or decrease probability for.

---

## 5. Gymnasium Environment (OpenAI Gym API)

The project uses **Gymnasium** (the maintained fork of OpenAI Gym) as the standard interface for the trading environment.

### Core API

```python
env = CryptoTradingEnv(df, initial_balance=10_000)

obs, info = env.reset(seed=42)          # Start a new episode
obs, reward, terminated, truncated, info = env.step(action)  # Take one step
```

### Episode
An **episode** is one complete run of the environment from reset to termination. In trading, one episode = trading through the entire dataset once (typically one year). The PPO agent runs many episodes during training.

### Observation Space
Defined as `gymnasium.spaces.Box(low=-inf, high=inf, shape=(265,), dtype=float32)`. A flat numpy array the agent observes at each step.

### Action Space
Defined as `gymnasium.spaces.Discrete(5)`. Five discrete choices at each step.

### Termination vs Truncation (Gymnasium 0.26+ API)
- `terminated=True`: The episode ended naturally due to a terminal condition — portfolio dropped below 50% of initial balance.
- `truncated=True`: The episode was cut short because the data ran out (end of the year).

This two-flag system is a key API change from old OpenAI Gym (which used a single `done` flag).

---

## 6. Proximal Policy Optimization (PPO)

PPO is the RL algorithm used to train the trading bot. It is a **policy gradient** method — it directly optimizes the policy by gradient ascent on expected return.

### Why PPO?

| Property | Why it Matters for Trading |
|---|---|
| On-policy | Uses freshly collected data — ensures the policy stays close to what generated the data |
| Stable updates | The clipping mechanism prevents destructive parameter updates |
| Sample efficient | Works well in environments where each step is computationally cheap |
| No Q-function needed | Avoids instabilities from Q-function approximation errors |
| Proven in finance | FinRL uses PPO as its primary agent; extensive validation in trading literature |

### PPO Core Idea: The Clipped Objective

Standard policy gradient (REINFORCE) can make excessively large updates that destabilize training. PPO constrains updates by clipping the probability ratio:

```
ratio = π_new(a|s) / π_old(a|s)

L_CLIP = E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]
```

Where:
- `ratio` measures how much the new policy differs from the old one
- `A` is the advantage estimate
- `ε = 0.2` (the `clip_range` hyperparameter)

**Interpretation:** If an action had positive advantage, increase its probability — but not by more than 20% relative to the old policy. If an action had negative advantage, decrease its probability — but not by more than 20%. This keeps updates stable.

### Actor-Critic Architecture

PPO uses an **actor-critic** architecture:

- **Actor** (policy network): Takes observation → outputs action probabilities. This is the trading decision maker.
- **Critic** (value network): Takes observation → outputs estimated value V(s). This provides the baseline for computing advantages.

Both networks share the same MLP backbone: `[128 neurons] → [128 neurons]`.

```
Observation (265 values)
        ↓
   [Dense 128]
        ↓
   [Dense 128]
      ↙    ↘
  Actor   Critic
  (5 logits)  (1 value)
```

### Generalized Advantage Estimation (GAE)

Rather than computing advantages using raw Monte Carlo returns (high variance) or one-step TD errors (high bias), PPO uses **GAE** with `λ = 0.95`:

```
A_t = Σ (γλ)^k × δ_{t+k}
```

Where `δ_t = r_t + γV(s_{t+1}) - V(s_t)` is the TD error.

- `λ = 0` → pure TD (low variance, high bias)
- `λ = 1` → pure Monte Carlo (high variance, low bias)
- `λ = 0.95` → a good balance, used here

**For trading:** GAE helps the agent correctly attribute credit when a profitable trade results from decisions made several steps earlier (e.g., buying before a rally).

### Rollout Buffer

PPO is an **on-policy** algorithm. During each iteration:
1. Collect `n_steps = 2048` steps of experience using the current policy
2. Compute advantages for all collected transitions
3. Run `n_epochs = 10` gradient updates on this batch
4. Discard the data and collect fresh experience

This avoids the stale data problem of off-policy methods.

### Entropy Regularization

The PPO loss includes an entropy bonus:

```
L_total = L_CLIP - ent_coef × entropy + vf_coef × L_value
```

With `ent_coef = 0.01`, the policy is penalized for being too deterministic. This encourages **exploration** — the agent is rewarded for occasionally trying actions it's uncertain about, rather than always taking the greedy action.

---

## 7. Reward Shaping

**Reward shaping** means designing the reward function so the agent learns useful behavior faster. Raw P&L (profit and loss) is a poor reward signal because:

- It's sparse (profits/losses only realized at sell events)
- It encourages excessive risk-taking
- It ignores volatility (doubling returns at 10× risk is not better)

### Rolling Sharpe Reward

The reward in this project measures **risk-adjusted performance** at every step:

```python
sharpe = √252 × (mean_30_step_return - risk_free_daily) / std_30_step_return
```

Computed over the last 30 portfolio returns. This gives the agent a continuous feedback signal about whether its current approach is generating good risk-adjusted returns.

**Why 30 steps?** Short enough to be responsive to recent performance; long enough to have a meaningful variance estimate.

### Drawdown Penalty

```python
drawdown = (peak - current_value) / peak
dd_penalty = max(0, drawdown - 0.20) × 2.0
```

Drawdowns under 20% are free — small losses are acceptable. Beyond 20%, the penalty grows linearly with severity. This trains the agent to protect capital and avoid catastrophic losses.

### Transaction Cost

```python
tx_cost = 0.001 × trade_notional
```

0.1% fee on each trade discourages over-trading. Without this, the agent might learn to trade every bar even if it provides no signal value.

### Clipping

```python
reward = clip(reward, -10, 10)
```

Clips extreme rewards caused by near-zero variance in the Sharpe denominator. Without clipping, a single episode with unusually low volatility could generate a reward of ±10,000, completely dominating the gradient update.

---

## 8. Observation Design

The observation vector is engineered to give the agent everything it needs to make informed decisions:

### OHLCV Window (250 values)
50 bars × 5 features (Open, High, Low, Close, Volume). Provides the agent with a "picture" of recent market structure — trends, volatility regimes, and volume patterns.

**Z-scoring:** Each OHLCV feature is standardized using training split mean and standard deviation:
```
z = (x - μ_train) / σ_train
```
This prevents price scale from dominating the neural network. The same statistics are applied to val/test (never refit on test data — that would be data leakage).

### Technical Indicators (5 values)
- **RSI:** Momentum oscillator. Values near 0/100 signal potential reversals.
- **MACD Diff:** Trend and momentum. Positive = bullish, negative = bearish.
- **BB Width:** Volatility measure. Wide bands = high volatility.
- **EMA50 Dist:** How far price is from its 50-day trend.
- **Vol Z-score:** Unusual volume can precede significant price moves.

### Portfolio State (5 values)
- **Position:** 0 (flat) or 1 (long) — lets the agent know its current exposure
- **Net worth normalized:** (net_worth / initial_balance) − 1 — relative P&L
- **Unrealized PnL:** Fraction of portfolio in the open position
- **Peak pct:** How far below the all-time high — helps the agent know when to protect profits
- **VaR(5%):** 5th percentile of last 20-day returns — risk awareness

---

## 9. Vectorized Environments

```python
vec_env = make_vec_env(env_factory, n_envs=4, vec_env_cls=DummyVecEnv)
```

**Vectorized environments** run multiple independent copies of the same environment in parallel. Each call to `vec_env.step(actions)` takes 4 actions (one per environment) and returns 4 sets of observations, rewards, and done flags.

**Benefits:**
- More transitions per wall-clock second
- Decorrelates experience — the 4 environments are at different points in the episode
- Reduces variance in gradient estimates

**DummyVecEnv vs SubprocVecEnv:**
- `DummyVecEnv`: runs all environments in the same process sequentially — simpler, lower overhead for fast envs
- `SubprocVecEnv`: spawns separate processes — true parallelism, better for slow envs (e.g., those with I/O)

The trading env is fast (just numpy arithmetic), so `DummyVecEnv` is used locally. `SubprocVecEnv` is available as an option for Colab's multi-core setup.

---

## 10. MlpPolicy

The `MlpPolicy` in Stable-Baselines3 is a standard multilayer perceptron policy. The architecture used here:

```
Input: 265-dimensional observation
→ Linear(265, 128) + ReLU
→ Linear(128, 128) + ReLU
→ Actor head: Linear(128, 5) + Softmax   [outputs action probabilities]
→ Critic head: Linear(128, 1)            [outputs state value estimate]
```

The `net_arch=[128, 128]` hyperparameter controls the hidden layer sizes. This is large enough to capture complex market patterns but small enough for fast inference (sub-10ms on CPU).

---

## 11. Exploration vs Exploitation

The fundamental dilemma in RL:

- **Exploration:** Try new actions to discover potentially better strategies
- **Exploitation:** Use the current best-known strategy to maximize reward

PPO manages this via:
1. **Stochastic policy:** During training, actions are sampled from the policy distribution rather than taken greedily. A policy with high entropy (uncertain) explores more.
2. **Entropy bonus (`ent_coef=0.01`):** Adds a reward for maintaining policy uncertainty, preventing premature convergence to suboptimal strategies.
3. **Clip range (`clip_range=0.2`):** Prevents the policy from changing so drastically that it stops exploring entire regions of action space.

**At inference time** (`deterministic=True`): the agent takes the action with highest probability — pure exploitation.

---

## 12. Discount Factor (γ = 0.99)

The discount factor determines how much the agent values future rewards vs immediate rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

With `γ = 0.99`:
- A reward 100 steps in the future is worth `0.99^100 ≈ 0.37` of an immediate reward
- A reward 365 steps in the future (1 year) is worth `0.99^365 ≈ 0.026`

**Trading implication:** The agent cares heavily about near-term returns but is not myopic. Long-term value (holding through a rally) is still incentivized, just weighted slightly less than immediate returns.

---

## 13. Episode vs Timestep

- **Timestep:** One call to `env.step(action)` — one trading day in this project
- **Episode:** One complete run from reset to termination — roughly one year of trading
- **Total timesteps:** The total number of `step()` calls across all training episodes — 500,000 here

At 365 timesteps per episode, 500,000 total timesteps ≈ 1,370 complete episodes.

---

## 14. TensorBoard Logging

During training, the `MetricsCallback` logs to TensorBoard:

- `train/reward` — reward at each step
- `train/reward_min`, `train/reward_mean`, `train/reward_max` — rollout statistics

**Why this matters:** These curves reveal:
- **Reward increasing over training?** Agent is learning.
- **Reward plateaued?** Training has converged (or is stuck in a local optimum).
- **Reward volatile or crashing?** Learning instability — may need to tune hyperparameters.

---

## 15. Train/Val/Test Split in RL

In supervised learning, the train/val/test split prevents overfitting to training data. In RL, the concern is slightly different: **overfitting to the specific market conditions in the training period**.

- **Train (2020-2022):** The agent learns from 3 years of BTC/ETH/SOL data including the 2020 COVID crash, 2021 bull run, and 2022 bear market.
- **Val (2023):** Used to monitor for overfitting during training.
- **Test (2024):** The agent has never seen this data. Performance here is the honest measure of generalization.

**Key discipline:** Hyperparameters are never tuned on the test set. The test set is touched exactly once — for final evaluation.

---

## 16. Sharpe Ratio as Optimization Target

The Sharpe ratio is central to this project both as a reward signal and as an evaluation metric:

```
Sharpe = √252 × (mean_daily_return - rf_daily) / std_daily_return
```

The √252 annualizes it (252 trading days per year). A Sharpe of:
- `< 0`: Strategy loses money relative to risk-free rate
- `0–1`: Acceptable but not great
- `1–2`: Good
- `> 2`: Excellent (hard to achieve consistently)

**Why Sharpe instead of total return?** Total return doesn't account for risk. A strategy that returns 50% by taking 90% drawdowns is far worse than one that returns 30% with a 15% max drawdown. Sharpe penalizes volatility, rewarding consistency.

---

## 17. Max Drawdown

```
Max Drawdown = max((peak_value - current_value) / peak_value)
```

Represents the worst peak-to-trough loss an investor would have experienced. A critical risk metric for any trading strategy:

- `0.10` = at worst, you were 10% below your peak (tolerable)
- `0.50` = at worst, you lost half your portfolio value from peak (very bad)

The reward function directly penalizes drawdowns above 20% to train the agent to avoid them.

---

## 18. The Credit Assignment Problem

A fundamental challenge in RL: if the agent buys BTC on day 1 and sells profitably on day 30, which of the 30 daily "hold" decisions deserve credit for the profit?

PPO addresses this through:
1. **GAE (λ=0.95):** Propagates credit backward through time by weighting TD errors
2. **Discount factor (γ=0.99):** Ensures that recent decisions are weighted more heavily
3. **Rollout-level updates:** PPO collects 2,048 steps before updating, so the policy sees the consequences of its decisions in context

In trading, this means the agent can learn patterns like "accumulating a position over multiple days before a rally is rewarded even though individual hold days have small immediate rewards."

---

## 19. Reproducibility (SEED = 42)

Every source of randomness is seeded:
- `env.reset(seed=42)` — Gymnasium environment
- `PPO(seed=42)` — model initialization and data shuffling
- `np.random.default_rng(42)` — random baseline
- Fixed date splits — no random train/test splitting

This ensures that running the full pipeline twice produces identical results, which is essential for academic reproducibility.

---

## Summary Table

| Concept | Where Used | Key Value |
|---|---|---|
| MDP | Entire system | Formalizes the trading problem |
| Policy | PPO actor network | `MlpPolicy` [128, 128] |
| Value function | PPO critic network | Enables advantage estimation |
| Advantage (GAE) | PPO update | λ = 0.95 |
| Discount factor | PPO | γ = 0.99 |
| Reward shaping | `env/reward.py` | Sharpe + DD penalty + tx cost |
| Clip range | PPO objective | ε = 0.2 |
| Entropy bonus | PPO exploration | coef = 0.01 |
| Vectorized envs | Training | n_envs = 4 |
| Observation space | `CryptoTradingEnv` | Box(265,) |
| Action space | `CryptoTradingEnv` | Discrete(5) |
| Termination | `CryptoTradingEnv` | Portfolio < 50% initial |
| Total timesteps | Training | 500,000 per asset |
| TensorBoard | `MetricsCallback` | reward, min, mean, max |
