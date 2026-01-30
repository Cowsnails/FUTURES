# Implementation Roadmap: 23 Setup Detectors + Stats Page

## Architecture Overview

```
                    BACKEND (Python - runs on your 5090)
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   on_bar_update(OHLCV)                                  │
    │         │                                               │
    │         ▼                                               │
    │   ┌──────────────┐                                      │
    │   │ SetupManager │  (new module: setup_detectors.py)    │
    │   │              │                                      │
    │   │  ┌─────────────────────┐                            │
    │   │  │ Setup 1: ORB        │──┐                         │
    │   │  │ Setup 2: VWAP MR    │  │                         │
    │   │  │ Setup 3: PDH Break  │  │  Each emits:            │
    │   │  │ Setup 4: NR7+ORB    │  │  {setup_id, direction,  │
    │   │  │ Setup 5: EMA Pull   │  ├─▶ entry_price, stop,   │
    │   │  │ Setup 6: TTM Sqz    │  │   target, max_bars,    │
    │   │  │ Setup 7: ON Sweep   │  │   confidence, reason}  │
    │   │  │ ...                 │  │                         │
    │   │  │ Setup 23: Absorb.   │──┘                         │
    │   │  └─────────────────────┘                            │
    │   └──────────┬───────────────────────────────────────── │
    │              │                                          │
    │              ▼                                          │
    │   ┌──────────────────┐     ┌──────────────────────┐     │
    │   │ bracket_signals  │────▶│ bracket_resolutions  │     │
    │   │ + setup_name col │     │ (existing system)    │     │
    │   └──────────────────┘     └──────────────────────┘     │
    │              │                                          │
    │              ▼                                          │
    │   ┌──────────────────┐                                  │
    │   │ /api/stats/full  │  ← adds per-setup breakdown     │
    │   │ /api/stats/setups│  ← NEW endpoint                  │
    │   └──────────────────┘                                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
                         │
                         ▼
              FRONTEND (stats.html only)
    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │   Setup Leaderboard Table                               │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │ # │ Setup Name       │ WR%  │ Trades │ Avg R │  │   │
    │   │ 1 │ ORB              │ 62%  │ 47     │ +0.3R │  │   │
    │   │ 2 │ VWAP MR ±2σ     │ 58%  │ 83     │ +0.2R │  │   │
    │   │ 3 │ PDH/PDL Break    │ 71%  │ 31     │ +0.5R │  │   │
    │   │ ...                                             │   │
    │   └─────────────────────────────────────────────────┘   │
    │                                                         │
    │   Per-Setup Detail Cards (expandable)                    │
    │   ┌─────────────────────────────────────────────────┐   │
    │   │ ORB - Opening Range Breakout                    │   │
    │   │ Win Rate: 62% │ Total R: +14.2R │ Trades: 47   │   │
    │   │ Avg MFE: 1.2R │ Avg MAE: 0.5R  │ Avg Hold: 23 │   │
    │   │ Best Session: AM │ Worst: Lunch  │ SQN: 1.8    │   │
    │   │ Today: 3/5 (60%) │ 7d: 18/29 │ 30d: 47 total  │   │
    │   └─────────────────────────────────────────────────┘   │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
```

---

## Phase 0: Database + Infrastructure (DO FIRST)

**Goal**: Add `setup_name` column to existing tables, create the SetupManager base class, wire into bar update pipeline.

### 0.1 — Database Migration
- Add `setup_name TEXT DEFAULT 'confluence_v1'` to `bracket_signals` table
- Add index: `CREATE INDEX idx_bracket_sig_setup ON bracket_signals(setup_name)`
- Backfill existing signals: `UPDATE bracket_signals SET setup_name = 'confluence_v1' WHERE setup_name IS NULL`

### 0.2 — Setup Detector Base Class (`backend/setup_detectors.py`)
```python
class SetupDetector:
    """Base class for all setup detectors."""
    name: str           # e.g. "orb_breakout"
    display_name: str   # e.g. "Opening Range Breakout"
    category: str       # e.g. "session", "vwap", "level", "momentum", "volatility", "micro"
    hold_time: str      # e.g. "15-60min"

    def update(self, bar: BarData, indicators: dict, levels: dict, session: str) -> Optional[Signal]
    def get_state(self) -> dict  # internal state for debugging

class Signal:
    setup_name: str
    direction: str      # "LONG" or "SHORT"
    entry_price: float
    stop_price: float
    target_price: float
    stop_atr_mult: float
    target_atr_mult: float
    max_bars: int
    confidence: float
    reason: str

class SetupManager:
    """Runs all setup detectors on each bar, feeds signals to stats tracker."""
    detectors: List[SetupDetector]

    def process_bar(self, bar_data, indicators, levels, session) -> List[Signal]
```

### 0.3 — Wire SetupManager into `app.py`
- Instantiate `SetupManager` alongside `StatsManager`
- In `on_bar_update()`: call `setup_manager.process_bar()` on every new bar
- Each emitted signal → `stats_manager.process_setup_signal(signal)` → insert into `bracket_signals` with `setup_name`
- Existing bracket resolution logic handles the rest (stop/target/trailing tracking unchanged)

### 0.4 — Stats API: Per-Setup Endpoint
- New endpoint: `GET /api/stats/setups` → returns per-setup breakdown
- Query: `GROUP BY setup_name` on `bracket_signals JOIN bracket_resolutions`
- Returns: `{setup_name: {win_rate, total_trades, avg_r, total_r, avg_mfe, avg_mae, ...}}`

### 0.5 — Stats Page: Setup Leaderboard Section
- New section on `stats.html` below existing bracket cards
- Sortable table: Setup Name | Category | Win Rate | Trades | Avg R | Total R | Expectancy | SQN
- Color-coded: green for positive expectancy, red for negative
- Click to expand per-setup detail card

---

## Phase 1: Foundation Setups (7 setups)

Each setup is a class inheriting from `SetupDetector`. All run on the backend, all feed into the same bracket resolution system.

### 1.1 — Opening Range Breakout (ORB)
- **State to track**: OR_High, OR_Low (from 9:30-9:45 ET bars), OR complete flag
- **Indicator needs**: ADX, Volume, VWAP (all available)
- **Complexity**: Medium — needs time-window aggregation for OR definition
- **Notes**: Most academically validated setup. Implement 15-min OR first, can test 5/30-min variants later by parameterizing.

### 1.2 — VWAP Mean Reversion from ±2σ
- **State to track**: VWAP, StdDev bands (already computed by scalping engine)
- **Indicator needs**: VWAP ±2σ, RSI, Volume, ADX
- **Complexity**: Low — all indicators exist, just need threshold checks
- **Notes**: Your current setup already uses VWAP but as one factor among many. This isolates it as a standalone signal.

### 1.3 — PDH/PDL Breakout Continuation
- **State to track**: PDH, PDL levels (already in `levels` object)
- **Indicator needs**: Volume, price vs level
- **Complexity**: Low — levels already computed, need breakout + retest detection
- **Notes**: 67-81% continuation rate is the strongest statistical edge in the list.

### 1.4 — NR7 + ORB Combination
- **State to track**: Daily ranges for last 7 days, NR7 flag
- **Indicator needs**: Daily OHLC history (need to add daily bar tracking)
- **Complexity**: Medium — needs daily bar aggregation to compute NR7
- **Notes**: Pre-condition (NR7 day) dramatically improves ORB win rate.

### 1.5 — EMA-9 Pullback in Trend
- **State to track**: EMA-9, trend state (ADX > 25 + DI direction)
- **Indicator needs**: EMA-9, ADX, +DI/-DI, Choppiness Index (all available)
- **Complexity**: Low — straightforward indicator threshold checks
- **Notes**: The "Holy Grail" setup from Linda Raschke.

### 1.6 — TTM Squeeze
- **State to track**: Bollinger Bands (20,2), Keltner Channels (20,1.5), squeeze state
- **Indicator needs**: Need to ADD Bollinger Bands and Keltner Channels to indicator stack
- **Complexity**: Medium — need new indicators, momentum histogram calculation
- **Notes**: Most complex indicator setup but well-documented. Squeeze state is binary.

### 1.7 — Overnight High/Low Sweep Reversal
- **State to track**: ONH, ONL (already in `levels` object), sweep detected flag
- **Indicator needs**: Volume, price action at levels
- **Complexity**: Medium — needs sweep detection logic (wick beyond + close back inside)
- **Notes**: 68% win rate with clear stop placement.

---

## Phase 2: Advanced Setups (8 setups)

### 2.1 — ORB Failure Reversal
- Depends on ORB detector (Phase 1.1) — detects when ORB fails
- **Complexity**: Low once ORB exists — just invert on failure

### 2.2 — A-B-C Morning Reversal
- Needs swing point detection algorithm (local highs/lows)
- Fibonacci retracement calculation
- **Complexity**: High — pattern matching with geometric requirements

### 2.3 — RSI Divergence Reversal
- Needs divergence detection: compare price swing lows/highs vs RSI swing lows/highs
- **Complexity**: Medium — swing detection + comparison logic

### 2.4 — Volume Spike Breakout
- Simple threshold checks: Volume > 2x SMA, price breaking 5-bar range
- **Complexity**: Low

### 2.5 — PDH/PDL Rejection
- Reverse of PDH/PDL Breakout — detect approach + rejection candle pattern
- **Complexity**: Medium — needs candle pattern recognition (pin bar, engulfing)

### 2.6 — VWAP Breakout and Retest
- Multi-step: breakout detection → wait for pullback → retest entry
- **Complexity**: Medium — state machine with breakout/pullback/retest phases

### 2.7 — ADX Thrust
- ADX crossing above 25 from below + directional bias
- **Complexity**: Low

### 2.8 — First VWAP Touch After Gap
- Gap detection (compare open vs previous close)
- Track first VWAP touch
- **Complexity**: Medium — needs gap detection + "first touch" state

---

## Phase 3: Remaining Setups (8 setups)

### 3.1 — ATR Expansion Trade
### 3.2 — NR4/NR7 Standalone Breakout
### 3.3 — VCP (Intraday Adaptation)
### 3.4 — VWAP Cross with Momentum
### 3.5 — Round Number Bounce
### 3.6 — Liquidity Sweep and Reversal
### 3.7 — Fair Value Gap Fill
### 3.8 — Absorption Proxy

These have weaker evidence bases or more complex detection requirements. Implement after Phases 1-2 are running and collecting data.

---

## Shared Infrastructure Needed

### New Indicators to Add
| Indicator | Needed By | Priority |
|-----------|-----------|----------|
| Bollinger Bands (20, 2.0 SD) | TTM Squeeze | Phase 1 |
| Keltner Channels (20, 1.5 ATR) | TTM Squeeze | Phase 1 |
| Swing High/Low Detection | A-B-C, RSI Div, Liquidity Sweep | Phase 2 |
| Daily Bar Aggregation | NR7/NR4 | Phase 1 |
| Gap Detection | First VWAP Touch | Phase 2 |
| Candle Pattern Recognition | PDH/PDL Rejection, FVG | Phase 2 |

### Session/Time Utilities (already exist, may need extension)
- `get_session_type(timestamp)` — exists
- `get_trading_date(timestamp)` — exists
- Opening Range window tracking — NEW (9:30-9:45 ET aggregation)
- Overnight session detection — partially exists via ONH/ONL levels

### Cooldown Management
- Each setup needs independent cooldown tracking
- Prevent: Setup A fires, then Setup B fires on same bar for same direction = over-concentrated
- Solution: Per-setup cooldown (60s minimum between same-setup signals) + optional global directional cooldown

---

## Stats Page Additions

### New Section: "Setup Leaderboard"
```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SETUP LEADERBOARD                                              Sort by: ▼  │
├────┬──────────────────────┬──────────┬────────┬────────┬────────┬───────────┤
│  # │ Setup                │ Win Rate │ Trades │ Avg R  │ Total R│Expectancy │
├────┼──────────────────────┼──────────┼────────┼────────┼────────┼───────────┤
│  1 │ PDH/PDL Break Cont.  │   71.0%  │   31   │ +0.52R │ +16.1R│  +0.38R   │
│  2 │ ORB (15-min)         │   62.0%  │   47   │ +0.31R │ +14.6R│  +0.19R   │
│  3 │ VWAP MR ±2σ         │   58.3%  │   83   │ +0.18R │ +14.9R│  +0.11R   │
│  4 │ ON H/L Sweep Rev     │   65.0%  │   20   │ +0.44R │  +8.8R│  +0.29R   │
│  5 │ EMA-9 Pullback       │   55.0%  │   60   │ +0.12R │  +7.2R│  +0.06R   │
│  6 │ Confluence v1 (orig) │   38.1%  │   33   │ -0.18R │  -5.9R│  -0.18R   │
│ ...│ ...                  │   ...    │  ...   │  ...   │  ...   │   ...     │
└────┴──────────────────────┴──────────┴────────┴────────┴────────┴───────────┘
```

### New Section: "Setup Detail Cards" (expandable per setup)
- Win rate (today / 7d / 30d / all-time)
- Total R, Avg R, SQN
- Avg MFE / MAE / ETD
- Avg bars held
- Best/worst session
- Quality score
- Recent trades list (last 10)
- Category badge + hold time badge

### New Section: "Category Performance"
- Grouped by category (VWAP, Session, Level, Momentum, Volatility, Micro)
- Shows aggregate stats per category
- Helps identify which TYPES of setups work on your instruments

### New Section: "Regime × Setup Matrix"
- Heatmap: rows = setups, columns = regimes (trending/ranging/volatile/quiet)
- Cell color = expectancy in that regime
- Instantly shows which setups work in which conditions

---

## File Changes Summary

| File | Changes |
|------|---------|
| `backend/setup_detectors.py` | **NEW** — Base class + all 23 setup detector classes |
| `backend/setup_indicators.py` | **NEW** — Bollinger Bands, Keltner, swing detection, candle patterns |
| `backend/stats_tracker.py` | Add `setup_name` to schema, add per-setup query methods |
| `backend/app.py` | Instantiate SetupManager, wire into `on_bar_update`, new `/api/stats/setups` endpoint |
| `frontend/templates/stats.html` | Add leaderboard table, detail cards, category grouping, regime matrix |

---

## Implementation Order

```
Week 1: Phase 0 (infrastructure)
  ├── 0.1 DB migration (setup_name column)
  ├── 0.2 SetupDetector base class + SetupManager
  ├── 0.3 Wire into app.py bar update pipeline
  ├── 0.4 /api/stats/setups endpoint
  └── 0.5 Stats page leaderboard section

Week 2-3: Phase 1 (7 foundation setups)
  ├── 1.1 ORB
  ├── 1.2 VWAP Mean Reversion ±2σ
  ├── 1.3 PDH/PDL Breakout
  ├── 1.4 NR7 + ORB
  ├── 1.5 EMA-9 Pullback
  ├── 1.6 TTM Squeeze (needs new indicators)
  └── 1.7 ON H/L Sweep Reversal

Week 4-5: Phase 2 (8 advanced setups)
  ├── 2.1-2.8 (see above)
  └── Stats page: detail cards, category grouping

Week 6+: Phase 3 (8 remaining setups)
  ├── 3.1-3.8 (see above)
  └── Stats page: regime matrix, correlation analysis
```

---

## Key Design Decisions

1. **All detection runs on backend** — your 5090 handles the compute, frontend just displays stats
2. **Each setup is independent** — can enable/disable individual setups without affecting others
3. **Reuse existing bracket resolution** — no new exit tracking code needed, just tag signals with `setup_name`
4. **Stats page only** — no chart overlays until you identify winners after weeks of data
5. **Per-setup cooldowns** — prevent signal spam from any single detector
6. **Indicator sharing** — SetupManager computes indicators once, passes to all detectors (not 23x redundant computation)
