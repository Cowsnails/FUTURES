# Deep Research Prompt: High-Probability Intraday Price Action Setups

## Instructions for Claude Deep Research

I need you to conduct exhaustive research into **intraday price action setups** used by professional scalpers, day traders, and quantitative researchers. I'm building an automated signal testing framework that will run **multiple setups simultaneously** on futures markets (MNQ/NQ, MES/ES, MGC/GC) and track their real-world win rates over weeks/months.

**My current system architecture:**
- 1-minute bar data with real-time WebSocket streaming from Interactive Brokers
- Available indicators: VWAP (with ±1σ/±2σ bands), EMA-9, RSI-7, ADX-10, ATR-10, Choppiness Index, Volume vs 20-SMA, Supertrend
- Available levels: Previous Day High/Low/Close, Overnight High/Low, Opening Range High/Low
- Session awareness: Pre-market, Opening Range (first 15 min), AM Session, Lunch, PM Session, Close
- Bracket resolution tracking: entry → stop/target/trailing/timeout with full MFE/MAE/R-multiple stats
- Hold times I want to test: **1 minute to 4 hours** (wide range of timeframes)

**What I need from this research:**

---

## PART 1: SETUP CATALOG (15-25 distinct setups)

For each setup, provide ALL of the following:

### A. Setup Identity
- **Name**: Clear, descriptive name (e.g., "VWAP Mean Reversion Bounce", "Opening Range Breakout")
- **Category**: Which category does it fall into? (Mean Reversion, Momentum/Breakout, Level-Based, Pattern-Based, Volatility-Based, Session-Specific)
- **Expected Hold Time**: How long should this trade typically last? (1-5 min, 5-15 min, 15-60 min, 1-4 hours)
- **Expected Win Rate**: Based on academic research and practitioner evidence, what win rate range is realistic?
- **Risk:Reward Profile**: What R:R ratio works best? (e.g., 1:1 for high-win-rate setups, 1:3 for breakout setups)

### B. Entry Conditions (EXACT and PROGRAMMABLE)
- List every condition that must be true to enter
- Use concrete numbers, not vague descriptions (e.g., "RSI crosses below 30 then back above 30 within 3 bars" not "RSI oversold bounce")
- Specify which indicators/levels are required
- Specify the session/time filters (when does this setup work best? when should it be avoided?)
- Specify regime filters (trending vs ranging vs choppy — when does this setup activate?)

### C. Exit Logic
- **Stop Loss**: Exact placement rule (ATR-based? Level-based? Fixed? How many ATR multiples?)
- **Take Profit**: Exact target rule
- **Trailing Stop**: If applicable, exact trailing mechanism
- **Time Stop**: Maximum bars to hold before forced exit
- **Partial Exits**: Any scaling out rules?

### D. Edge Case Filters (What PREVENTS the trade)
- Conditions that would invalidate the setup even if entry criteria are met
- Chop filters, volatility filters, news/event filters
- Conflicting signal filters

### E. Evidence & Rationale
- Why does this edge exist? (market microstructure, behavioral finance, institutional order flow)
- Any academic papers, books, or well-known practitioners who document this setup
- Known failure modes and market conditions where it stops working

---

## PART 2: SETUP CATEGORIES TO COVER

Research setups across ALL of these categories. I need diversity — not 15 variations of the same VWAP bounce.

### Category 1: VWAP-Based Setups (3-4 setups)
- Mean reversion to VWAP from ±1σ or ±2σ bands
- VWAP breakout/breakdown and retest
- VWAP cross with momentum confirmation
- First touch of VWAP after gap open
- Research: How does VWAP band distance correlate with bounce probability? What z-score thresholds produce the best win rates?

### Category 2: Opening Range / Session-Based (3-4 setups)
- Opening Range Breakout (ORB) — what timeframe for the OR produces best results? (5 min? 15 min? 30 min?)
- Opening Range Breakdown and failure
- A-B-C morning reversal pattern
- London/Asia session range breakout into US session
- Research: Toby Crabel's work on opening range breakouts. What percentage of the day's range is established in the first 15/30/60 minutes?

### Category 3: Level-Based Setups (3-4 setups)
- Previous Day High/Low rejection
- Previous Day High/Low breakout and retest
- Overnight High/Low sweep and reversal
- Round number / psychological level bounces
- Research: How often does price test and reject vs break through PDH/PDL? What confirmation signals improve the win rate?

### Category 4: Momentum / Trend Continuation (3-4 setups)
- EMA pullback in trend (EMA-9 touch during strong ADX)
- RSI divergence reversal (bullish/bearish divergence)
- Volume spike breakout (2x+ average volume with directional move)
- ADX thrust (ADX crossing above 25 from below with directional bias)
- Supertrend flip confirmation
- Research: What ADX levels indicate tradeable trends vs chop? What RSI divergence criteria produce reliable reversals?

### Category 5: Volatility-Based Setups (2-3 setups)
- Bollinger Band squeeze breakout (Choppiness Index drop + directional move)
- ATR expansion trade (volatility expansion from compression)
- Volatility contraction pattern (inside bars / narrow range bars before breakout)
- Research: John Carter's TTM Squeeze, Bollinger on Bollinger Bands. What's the measured probability of direction after a squeeze?

### Category 6: Microstructure / Institutional Setups (2-3 setups)
- Liquidity sweep and reversal (stop hunt above/below key levels)
- Fair Value Gap fill (imbalance candles)
- Absorption / failed auction (high volume at level without price movement)
- Research: ICT concepts, order flow mechanics. Which of these can be detected from OHLCV data alone (no Level 2)?

---

## PART 3: COMPARATIVE ANALYSIS

After cataloging all setups, provide:

1. **Win Rate Ranking**: Order all setups by expected win rate (highest to lowest)
2. **R:R Ranking**: Order by expected R-multiple per trade
3. **Expectancy Ranking**: Order by (win_rate × avg_win) - (loss_rate × avg_loss)
4. **Session Compatibility Matrix**: Which setups work in which sessions?
5. **Regime Compatibility Matrix**: Which setups work in trending/ranging/volatile/quiet markets?
6. **Correlation Analysis**: Which setups tend to fire together vs independently? (Important for not over-concentrating signals)
7. **Hold Time Distribution**: Group by expected hold time so I can see short (1-5 min) vs medium (5-30 min) vs long (30 min - 4 hours)

---

## PART 4: IMPLEMENTATION PRIORITY

Rank the setups by:
1. **Ease of implementation** (can I build this with OHLCV + the indicators I already have?)
2. **Robustness** (likely to maintain edge over months, not curve-fit to recent data)
3. **Independence** (doesn't overlap with other setups — adds diversification value)

Give me a recommended "Phase 1" batch (5-7 setups to implement first) and "Phase 2" batch (remaining setups).

---

## PART 5: SPECIFIC NUMBERS I NEED

For each setup, where possible, cite:
- **Academic/backtest win rates** with sample sizes
- **Optimal stop loss distance** (in ATR multiples or fixed points)
- **Optimal target distance** (in ATR multiples or fixed points)
- **Optimal max hold time** (in bars/minutes)
- **Minimum ADX/ATR/Volume thresholds** for the setup to be valid
- **Session filters** (what hours of the day does this work?)
- **Failure rate** in choppy/ranging conditions

I don't want theoretical setups — I want setups that have measurable, documented edges in intraday futures/equity index markets.

---

## IMPORTANT CONSTRAINTS

- All setups must be detectable from **OHLCV data + standard indicators only** (no Level 2, no order book, no tick data beyond 1-minute bars)
- All setups must work on **futures markets** (MNQ, MES, MGC specifically — liquid, electronic, nearly 24-hour)
- All entries and exits must be **fully programmable** with exact rules (no discretionary judgment)
- Setups should be **independent enough** that running 15-20 simultaneously doesn't just generate correlated signals
- Focus on setups with documented edges from **academic research, professional trading literature, or large-sample backtests** — not social media / YouTube trader patterns
- For each setup, be honest about the **realistic win rate** — I'd rather have accurate 55% expectations than inflated 80% claims
