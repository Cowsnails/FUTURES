"""
Trading Statistics Tracker

Tracks performance of three signal sources:
1. Scalping Decision Engine (trade signals: BUY/SELL/NO_TRADE)
2. Daily Pattern Matcher (RTH session predictions)
3. Overnight Pattern Matcher (overnight session predictions)

Uses SQLite with WAL mode for live operations, buffered writes,
signal state machine with 2-bar confirmation, and multi-horizon
synthetic outcome tracking.
"""

import json
import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

from .signal_resolution import (
    Direction, ResolutionOutcome, TrailingStopState,
    BracketParams, MFEMAETracker, TrailingStopTracker,
    SignalResolution, BarData, resolve_signal_on_bar,
    score_signal_quality, calculate_sqn, WinRateStats,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

STATS_DIR = Path.home() / "trading_stats"
DB_PATH = STATS_DIR / "live" / "signals.db"

# Signal state machine
# BUY/SELL only appear when confluence >= 0.50 (ranging) or 0.60 (trending)
# so no additional threshold needed — if action is BUY/SELL, it's valid.
MIN_SIGNAL_GAP_SECONDS = 60    # Minimum gap between confirmed signals (cooldown)

# Outcome measurement horizons (minutes)
OUTCOME_HORIZONS = [1, 5, 15]

# Win thresholds (points in signal direction)
WIN_THRESHOLD_1M = 5.0
WIN_THRESHOLD_5M_ATR_MULT = 0.10
WIN_THRESHOLD_15M_ATR_MULT = 0.25

# Buffer settings
BUFFER_SIZE = 50
FLUSH_INTERVAL = 5.0  # seconds


# ═══════════════════════════════════════════════════════════════════════════
# SESSION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_trading_date(timestamp_s: int) -> str:
    """Get CME trading date from unix timestamp (ET-as-UTC)."""
    dt = datetime.utcfromtimestamp(timestamp_s)
    hour = dt.hour
    if hour >= 18:  # After 6PM = next day's session
        dt = dt + timedelta(days=1)
    return dt.strftime('%Y-%m-%d')


def get_session_type(timestamp_s: int) -> str:
    """Determine session type from unix timestamp (ET-as-UTC)."""
    dt = datetime.utcfromtimestamp(timestamp_s)
    hour, minute = dt.hour, dt.minute
    t = hour * 60 + minute
    if 570 <= t < 975:  # 9:30 AM - 4:15 PM
        return 'rth'
    elif 1020 <= t < 1080:  # 5:00 PM - 6:00 PM
        return 'maintenance'
    else:
        return 'overnight'


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════

class SignalState(Enum):
    IDLE = "idle"
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    ACTIVE = "active"
    RESOLVED = "resolved"
    EXPIRED = "expired"


@dataclass
class ActiveSignal:
    """Tracks a single signal through its lifecycle."""
    signal_id: str
    source: str  # 'trade_signal'
    state: SignalState = SignalState.IDLE
    direction: str = ''  # 'long' or 'short'

    # Confirmation tracking
    confirmation_bars: int = 0
    detection_bars: int = 0

    # Timestamps & prices
    detected_at: int = 0       # unix timestamp
    confirmed_at: int = 0
    entry_price: float = 0.0
    entry_bar_time: int = 0

    # Signal data
    confluence_score: float = 0.0
    indicator_data: dict = field(default_factory=dict)

    # Outcome tracking
    outcome_prices: dict = field(default_factory=dict)
    peak_favorable: float = 0.0   # MFE
    peak_adverse: float = 0.0     # MAE
    bars_since_entry: int = 0
    resolved: bool = False
    resolution_type: str = ''


class TradeSignalTracker:
    """
    State machine for scalping decision engine signals.

    Dual-mode tracking:
    1. Legacy time-snapshot resolution (backward compatible)
    2. New bracket-order resolution with ATR-based stops/targets

    Instant confirmation — signals are fleeting (last seconds to 1 min).
    Cooldown prevents duplicate fires within MIN_SIGNAL_GAP_SECONDS.
    """

    def __init__(self):
        self._signal_counter = 0
        self.current_signal: Optional[ActiveSignal] = None
        self.pending_outcomes: List[ActiveSignal] = []  # Legacy outcome tracking
        self._last_confirmed_time: int = 0

        # Bracket resolution tracking
        self.active_bracket_signals: Dict[str, SignalResolution] = {}
        self._bar_index_counter: int = 0  # Running bar index for bracket resolution

    def _next_id(self) -> str:
        self._signal_counter += 1
        return f"TS-{int(time.time())}-{self._signal_counter}"

    def process_bar(self, bar_time: int, price: float, action: str,
                    confluence: float, atr: float,
                    indicator_data: dict,
                    bar_high: float = 0, bar_low: float = 0,
                    bar_open: float = 0) -> Optional[dict]:
        """
        Process a new bar from the scalping engine.
        Returns a dict event if state changed, None otherwise.

        bar_high/bar_low/bar_open: OHLC data for bracket resolution.
        If not provided, falls back to price-only (legacy mode).
        """
        direction = 'long' if action == 'BUY' else ('short' if action == 'SELL' else '')
        has_signal = action in ('BUY', 'SELL')

        # Use price as fallback for OHLC if not provided
        if bar_high == 0:
            bar_high = price
        if bar_low == 0:
            bar_low = price
        if bar_open == 0:
            bar_open = price

        # ── Update pending outcome signals (legacy) ──
        events = []
        for sig in self.pending_outcomes:
            sig.bars_since_entry += 1
            self._update_outcomes(sig, bar_time, price, atr)

        # Remove fully resolved (legacy)
        resolved = [s for s in self.pending_outcomes if s.resolved]
        for s in resolved:
            events.append(self._make_event(s, 'TRADE_RESOLVED'))
        self.pending_outcomes = [s for s in self.pending_outcomes if not s.resolved]

        # ── State machine (instant confirmation for fleeting signals) ──
        if has_signal:
            if (bar_time - self._last_confirmed_time) >= MIN_SIGNAL_GAP_SECONDS:
                sig = ActiveSignal(
                    signal_id=self._next_id(),
                    source='trade_signal',
                    state=SignalState.CONFIRMED,
                    direction=direction,
                    detected_at=bar_time,
                    confirmed_at=bar_time,
                    confirmation_bars=1,
                    detection_bars=1,
                    confluence_score=confluence,
                    indicator_data=indicator_data,
                    entry_price=price,
                    entry_bar_time=bar_time,
                )
                events.append(self._make_event(sig, 'SIGNAL_CONFIRMED'))

                # Move to active and start legacy outcome tracking
                sig.state = SignalState.ACTIVE
                self.pending_outcomes.append(sig)
                self._last_confirmed_time = bar_time

                # Create bracket resolution signal
                stop_mult = indicator_data.get('stopATR', 1.5) or 1.5
                target_mult = indicator_data.get('targetATR', 2.0) or 2.0
                if atr > 0:
                    bracket_sig = self._create_bracket_signal(
                        signal_id=sig.signal_id,
                        direction=direction,
                        entry_price=price,
                        entry_time=bar_time,
                        atr=atr,
                        confluence=confluence,
                        stop_mult=stop_mult,
                        target_mult=target_mult,
                        indicator_data=indicator_data,
                    )
                    if bracket_sig:
                        events.append({
                            'event_type': 'BRACKET_SIGNAL_CREATED',
                            'signal_id': sig.signal_id,
                            'source': 'trade_signal',
                            'direction': direction,
                            'entry_price': price,
                            'stop_price': bracket_sig.bracket.stop_price,
                            'target_price': bracket_sig.bracket.target_price,
                            'atr': atr,
                            'initial_risk': bracket_sig.bracket.initial_risk,
                            'reward_risk_ratio': bracket_sig.bracket.reward_risk_ratio,
                            'bar_time': bar_time,
                        })

                logger.info(f"Signal confirmed: {direction} @ {price:.2f} "
                           f"(conf={confluence:.3f}, ATR={atr:.2f}, "
                           f"stop_mult={stop_mult}, target_mult={target_mult})")

        return events if events else None

    def _create_bracket_signal(self, signal_id: str, direction: str,
                                entry_price: float, entry_time: int,
                                atr: float, confluence: float,
                                stop_mult: float, target_mult: float,
                                indicator_data: dict) -> Optional[SignalResolution]:
        """Create a bracket resolution signal from trade signal parameters."""
        dir_enum = Direction.LONG if direction == 'long' else Direction.SHORT
        bracket = BracketParams.from_atr(
            entry_price, atr, dir_enum, stop_mult, target_mult
        )

        sig = SignalResolution(
            signal_id=signal_id,
            symbol='',  # Will be set by StatsManager
            direction=dir_enum,
            confluence_score=confluence,
            bracket=bracket,
            entry_time=entry_time,
            entry_bar_index=self._bar_index_counter,
            max_bars=15,
            indicator_data=indicator_data,
        )
        sig.trailing.initialize(bracket)

        self.active_bracket_signals[signal_id] = sig
        return sig

    def update_brackets(self, bar_time: int, bar_open: float,
                        bar_high: float, bar_low: float,
                        bar_close: float) -> List[SignalResolution]:
        """
        Process a new bar through all active bracket signals.
        Returns list of resolved signals.
        """
        self._bar_index_counter += 1
        bar = BarData(
            timestamp=bar_time,
            bar_index=self._bar_index_counter,
            open=bar_open,
            high=bar_high,
            low=bar_low,
            close=bar_close,
        )

        resolved = []
        for sig_id in list(self.active_bracket_signals.keys()):
            sig = self.active_bracket_signals[sig_id]
            if resolve_signal_on_bar(sig, bar):
                resolved.append(sig)
                del self.active_bracket_signals[sig_id]
                logger.info(
                    f"Bracket resolved: {sig.signal_id} -> {sig.outcome.value} "
                    f"(R={sig.r_multiple:.2f}, bars={sig.bars_held})"
                )

        return resolved

    def update(self, bar_time: int, price: float, atr: float = 0) -> list:
        """
        Update pending outcomes with a new bar price WITHOUT processing a new signal.
        Called from backend on every 1-min bar to resolve pending signals at +1m/+5m/+15m.
        """
        events = []
        for sig in self.pending_outcomes:
            sig.bars_since_entry += 1
            self._update_outcomes(sig, bar_time, price, atr)

        resolved = [s for s in self.pending_outcomes if s.resolved]
        for s in resolved:
            events.append(self._make_event(s, 'TRADE_RESOLVED'))
        self.pending_outcomes = [s for s in self.pending_outcomes if not s.resolved]
        return events

    def _update_outcomes(self, sig: ActiveSignal, bar_time: int, price: float, atr: float):
        """Track outcomes at each horizon and MAE/MFE (legacy)."""
        minutes_elapsed = (bar_time - sig.entry_bar_time) / 60.0
        is_long = sig.direction == 'long'

        # Update MAE/MFE
        move = price - sig.entry_price if is_long else sig.entry_price - price
        if move > sig.peak_favorable:
            sig.peak_favorable = move
        if -move > sig.peak_adverse:
            sig.peak_adverse = -move

        # Record horizon outcomes
        for horizon in OUTCOME_HORIZONS:
            key = f'{horizon}m'
            if key not in sig.outcome_prices and minutes_elapsed >= horizon:
                sig.outcome_prices[key] = price

        # Resolve after all horizons measured (15 min max)
        if minutes_elapsed >= max(OUTCOME_HORIZONS) + 1:
            sig.resolved = True
            sig.resolution_type = 'horizons_complete'

    def _make_event(self, sig: ActiveSignal, event_type: str) -> dict:
        """Create event dict for logging."""
        return {
            'event_type': event_type,
            'signal_id': sig.signal_id,
            'source': sig.source,
            'state': sig.state.value,
            'direction': sig.direction,
            'confluence_score': sig.confluence_score,
            'entry_price': sig.entry_price,
            'detected_at': sig.detected_at,
            'confirmed_at': sig.confirmed_at,
            'confirmation_bars': sig.confirmation_bars,
            'indicator_data': sig.indicator_data,
            'outcome_prices': sig.outcome_prices,
            'peak_favorable': sig.peak_favorable,
            'peak_adverse': sig.peak_adverse,
            'resolution_type': sig.resolution_type,
            'bar_time': sig.entry_bar_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PATTERN SNAPSHOT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class PatternSnapshotManager:
    """
    Captures snapshots of pattern predictions on meaningful changes.
    Snapshots on: session open, direction flip, consensus change,
    large projected price change (>0.5%).
    """

    def __init__(self, session_type: str):
        self.session_type = session_type  # 'rth' or 'overnight'
        self.last_snapshot: Optional[dict] = None
        self._snapshot_counter = 0

    def _next_id(self) -> str:
        self._snapshot_counter += 1
        return f"PAT-{self.session_type[:3].upper()}-{int(time.time())}-{self._snapshot_counter}"

    def process_update(self, pattern_result: dict, bar_time: int,
                       current_price: float = 0) -> Optional[dict]:
        """
        Process a pattern update.
        Always returns a prediction_log entry.
        Also returns a snapshot dict if meaningful change detected.
        Returns tuple: (prediction_entry, snapshot_or_None)
        """
        if not pattern_result or not pattern_result.get('forecast'):
            return None, None

        forecast = pattern_result['forecast']
        projection = pattern_result.get('projection', {})

        direction = 'bullish' if forecast.get('direction_probability', 0.5) > 0.5 else 'bearish'
        consensus = forecast.get('consensus', '')
        eod_price = projection.get('end_of_day_price') or projection.get('end_of_session_price') or 0
        peak_price = projection.get('peak_move_price') or eod_price

        current = {
            'direction': direction,
            'consensus': consensus,
            'avg_correlation': forecast.get('avg_correlation', 0),
            'mean_move': forecast.get('mean_move', 0),
            'eod_projected_price': eod_price,
            'peak_projected_price': peak_price,
            'match_count': pattern_result.get('match_count', 0),
            'signal_strength': forecast.get('confluence_score', 0),
        }

        # Always log the prediction
        prediction = {
            'timestamp': bar_time,
            'session_date': get_trading_date(bar_time),
            'session_type': self.session_type,
            'current_price': current_price,
            **current,
        }

        # Snapshot only on meaningful changes
        should_snap, trigger = self._should_snapshot(current)
        snapshot = None
        if should_snap:
            snapshot = {
                'snapshot_id': self._next_id(),
                'source': f'{self.session_type}_pattern',
                'timestamp': bar_time,
                'session_date': get_trading_date(bar_time),
                'session_type': self.session_type,
                'trigger': trigger,
                **current,
            }
            self.last_snapshot = current

        return prediction, snapshot

    def _should_snapshot(self, current: dict) -> tuple:
        if self.last_snapshot is None:
            return True, 'initial'

        last = self.last_snapshot

        # Direction flip
        if current['direction'] != last['direction']:
            return True, 'direction_flip'

        # Consensus change
        if current['consensus'] != last['consensus']:
            return True, 'consensus_change'

        # Large projected price shift (>0.5%)
        if last['eod_projected_price'] and last['eod_projected_price'] > 0:
            change_pct = abs(current['eod_projected_price'] - last['eod_projected_price']) / last['eod_projected_price']
            if change_pct > 0.005:
                return True, 'significant_price_change'

        return False, None


# ═══════════════════════════════════════════════════════════════════════════
# ANCHORED SESSION TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class AnchoredSessionTracker:
    """
    Aggregates per-minute pattern predictions into one session-level prediction.

    Instead of treating each minute as a separate prediction, this:
    - Locks an anchor on the first prediction of a session
    - Tallies bullish/bearish votes every minute
    - Tracks regime changes (direction flips)
    - Computes direction_stability (% of votes for majority direction)
    - At session end, evaluates ONE verdict against actual outcome

    This solves the "60 predictions per hour" problem by producing
    ONE evaluable prediction per session with a confidence score.
    """

    def __init__(self):
        # Key: (session_date, session_type, symbol) -> live session state
        self._active_sessions: Dict[tuple, dict] = {}

    def _session_key(self, session_date: str, session_type: str, symbol: str) -> tuple:
        return (session_date, session_type, symbol)

    def process_prediction(self, prediction: dict, symbol: str) -> Optional[dict]:
        """
        Feed a per-minute prediction into the session anchor.
        Returns a session_update dict with current aggregated state.
        """
        if not prediction:
            return None

        session_date = prediction.get('session_date', '')
        session_type = prediction.get('session_type', '')
        direction = prediction.get('direction', '')
        current_price = prediction.get('current_price', 0)
        eod_proj = prediction.get('eod_projected_price', 0)
        peak_proj = prediction.get('peak_projected_price', 0)
        correlation = prediction.get('avg_correlation', 0)
        timestamp = prediction.get('timestamp', int(time.time()))

        key = self._session_key(session_date, session_type, symbol)

        if key not in self._active_sessions:
            # First prediction of session — this is the anchor
            self._active_sessions[key] = {
                'session_date': session_date,
                'session_type': session_type,
                'symbol': symbol,
                'anchor_direction': direction,
                'anchor_price': current_price,
                'anchor_eod_proj': eod_proj,
                'anchor_peak_proj': peak_proj,
                'anchor_time': timestamp,
                'total_votes': 0,
                'bullish_votes': 0,
                'bearish_votes': 0,
                'eod_proj_sum': 0.0,
                'peak_proj_sum': 0.0,
                'corr_sum': 0.0,
                'regime_changes': 0,
                'regime_change_log': [],
                'last_direction': None,
                'last_price': current_price,
                'last_update_time': timestamp,
            }

        state = self._active_sessions[key]

        # Tally vote
        state['total_votes'] += 1
        if direction == 'bullish':
            state['bullish_votes'] += 1
        else:
            state['bearish_votes'] += 1

        # Running sums for averages
        state['eod_proj_sum'] += eod_proj
        state['peak_proj_sum'] += peak_proj
        state['corr_sum'] += correlation

        # Detect regime change
        if state['last_direction'] and direction != state['last_direction']:
            state['regime_changes'] += 1
            state['regime_change_log'].append({
                'time': timestamp,
                'from': state['last_direction'],
                'to': direction,
                'price': current_price,
            })
            logger.info(f"[{symbol}] Regime change #{state['regime_changes']}: "
                       f"{state['last_direction']} -> {direction} @ {current_price:.1f}")

        state['last_direction'] = direction
        state['last_price'] = current_price
        state['last_update_time'] = timestamp

        # Compute current aggregated state
        total = state['total_votes']
        bull = state['bullish_votes']
        bear = state['bearish_votes']
        majority_dir = 'bullish' if bull >= bear else 'bearish'
        majority_pct = max(bull, bear) / total if total > 0 else 0

        return {
            'session_date': session_date,
            'session_type': session_type,
            'symbol': symbol,
            'anchor_direction': state['anchor_direction'],
            'anchor_price': state['anchor_price'],
            'anchor_eod_proj': state['anchor_eod_proj'],
            'anchor_peak_proj': state['anchor_peak_proj'],
            'anchor_time': state['anchor_time'],
            'total_votes': total,
            'bullish_votes': bull,
            'bearish_votes': bear,
            'direction_stability': round(majority_pct, 3),
            'avg_eod_projected': round(state['eod_proj_sum'] / total, 2) if total else 0,
            'avg_peak_projected': round(state['peak_proj_sum'] / total, 2) if total else 0,
            'avg_correlation': round(state['corr_sum'] / total, 4) if total else 0,
            'final_direction': majority_dir,
            'final_confidence': round(majority_pct, 3),
            'regime_changes': state['regime_changes'],
            'regime_change_log': json.dumps(state['regime_change_log']),
            'last_price': current_price,
            'last_direction': direction,
            'last_update_time': timestamp,
        }

    def get_active_sessions(self) -> list:
        """Return current state of all active session predictions."""
        results = []
        for key, state in self._active_sessions.items():
            total = state['total_votes']
            bull = state['bullish_votes']
            bear = state['bearish_votes']
            majority_dir = 'bullish' if bull >= bear else 'bearish'
            majority_pct = max(bull, bear) / total if total > 0 else 0
            results.append({
                'session_date': state['session_date'],
                'session_type': state['session_type'],
                'symbol': state['symbol'],
                'anchor_direction': state['anchor_direction'],
                'anchor_price': state['anchor_price'],
                'total_votes': total,
                'bullish_votes': bull,
                'bearish_votes': bear,
                'direction_stability': round(majority_pct, 3),
                'final_direction': majority_dir,
                'regime_changes': state['regime_changes'],
                'avg_eod_projected': round(state['eod_proj_sum'] / total, 2) if total else 0,
                'avg_peak_projected': round(state['peak_proj_sum'] / total, 2) if total else 0,
                'last_price': state['last_price'],
                'last_update_time': state['last_update_time'],
            })
        return results

    def clear_session(self, session_date: str, session_type: str, symbol: str):
        """Remove a closed session from active tracking."""
        key = self._session_key(session_date, session_type, symbol)
        self._active_sessions.pop(key, None)


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE (SQLite + WAL + Buffered Writes)
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Main signal log
CREATE TABLE IF NOT EXISTS signal_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc INTEGER NOT NULL,
    session_date TEXT NOT NULL,
    session_type TEXT NOT NULL,
    signal_source TEXT NOT NULL,
    signal_id TEXT NOT NULL,
    state TEXT NOT NULL,
    event_type TEXT NOT NULL,
    direction TEXT,
    confluence_score REAL,
    entry_price REAL,
    signal_data TEXT,
    outcome_1m_price REAL,
    outcome_5m_price REAL,
    outcome_15m_price REAL,
    peak_favorable REAL,
    peak_adverse REAL,
    is_resolved INTEGER DEFAULT 0,
    resolution_type TEXT,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_log(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_signal_session ON signal_log(session_date, signal_source);
CREATE INDEX IF NOT EXISTS idx_signal_id ON signal_log(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_resolved ON signal_log(is_resolved, signal_source);

-- Pattern snapshots
CREATE TABLE IF NOT EXISTS pattern_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc INTEGER NOT NULL,
    session_date TEXT NOT NULL,
    session_type TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    trigger TEXT NOT NULL,
    direction TEXT,
    consensus TEXT,
    avg_correlation REAL,
    mean_move REAL,
    eod_projected_price REAL,
    peak_projected_price REAL,
    match_count INTEGER,
    signal_strength REAL,
    actual_session_close REAL,
    actual_peak_price REAL,
    direction_correct INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_pattern_session ON pattern_snapshots(session_date, session_type);
CREATE INDEX IF NOT EXISTS idx_pattern_timestamp ON pattern_snapshots(timestamp_utc);

-- Every pattern prediction (logged every cycle for accuracy tracking)
CREATE TABLE IF NOT EXISTS prediction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc INTEGER NOT NULL,
    session_date TEXT NOT NULL,
    session_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT,
    consensus TEXT,
    avg_correlation REAL,
    current_price REAL,
    eod_projected_price REAL,
    peak_projected_price REAL,
    mean_move REAL,
    match_count INTEGER,
    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_prediction_session ON prediction_log(session_date, session_type);
CREATE INDEX IF NOT EXISTS idx_prediction_timestamp ON prediction_log(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_prediction_symbol ON prediction_log(symbol);

-- Daily aggregated stats
CREATE TABLE IF NOT EXISTS daily_stats (
    session_date TEXT NOT NULL,
    signal_source TEXT NOT NULL,
    total_signals INTEGER DEFAULT 0,
    confirmed_signals INTEGER DEFAULT 0,
    wins_1m INTEGER DEFAULT 0,
    wins_5m INTEGER DEFAULT 0,
    wins_15m INTEGER DEFAULT 0,
    total_pnl_points REAL DEFAULT 0,
    avg_mfe REAL DEFAULT 0,
    avg_mae REAL DEFAULT 0,
    updated_at INTEGER,
    PRIMARY KEY (session_date, signal_source)
);

-- Anchored session predictions (one per session per symbol per session_type)
-- Aggregates all per-minute predictions into a single evaluable verdict
CREATE TABLE IF NOT EXISTS session_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_date TEXT NOT NULL,
    session_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    -- Anchor (first prediction of session)
    anchor_direction TEXT,
    anchor_price REAL,
    anchor_eod_proj REAL,
    anchor_peak_proj REAL,
    anchor_time INTEGER,
    -- Aggregated consensus over entire session
    total_votes INTEGER DEFAULT 0,
    bullish_votes INTEGER DEFAULT 0,
    bearish_votes INTEGER DEFAULT 0,
    direction_stability REAL DEFAULT 0,
    avg_eod_projected REAL DEFAULT 0,
    avg_peak_projected REAL DEFAULT 0,
    avg_correlation REAL DEFAULT 0,
    -- Final verdict (majority vote direction)
    final_direction TEXT,
    final_confidence REAL,
    -- Regime changes (direction flips during session)
    regime_changes INTEGER DEFAULT 0,
    regime_change_log TEXT,
    -- Last update
    last_price REAL,
    last_direction TEXT,
    last_update_time INTEGER,
    -- Outcome (filled at session end)
    actual_close_price REAL,
    actual_peak_price REAL,
    actual_low_price REAL,
    direction_correct INTEGER,
    eod_error_points REAL,
    peak_error_points REAL,
    -- Status
    is_closed INTEGER DEFAULT 0,
    created_at INTEGER DEFAULT (strftime('%s', 'now')),
    UNIQUE(session_date, session_type, symbol)
);

CREATE INDEX IF NOT EXISTS idx_session_pred_date ON session_predictions(session_date, session_type);
CREATE INDEX IF NOT EXISTS idx_session_pred_symbol ON session_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_session_pred_closed ON session_predictions(is_closed);

-- Rolling stats cache
CREATE TABLE IF NOT EXISTS rolling_stats (
    signal_source TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    total_signals INTEGER DEFAULT 0,
    win_rate_1m REAL,
    win_rate_5m REAL,
    win_rate_15m REAL,
    avg_pnl_points REAL,
    avg_mfe REAL,
    avg_mae REAL,
    computed_at INTEGER,
    PRIMARY KEY (signal_source, window_days)
);

-- ═══════════════════════════════════════════════════════════════════════
-- BRACKET RESOLUTION TABLES (new system)
-- ═══════════════════════════════════════════════════════════════════════

-- Bracket signals: ATR-based bracket parameters computed at signal time
CREATE TABLE IF NOT EXISTS bracket_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('LONG', 'SHORT')),
    confluence_score REAL DEFAULT 0,

    -- Entry details
    entry_price REAL NOT NULL,
    entry_time INTEGER NOT NULL,
    entry_bar_index INTEGER NOT NULL,

    -- ATR-based bracket parameters
    atr_at_entry REAL NOT NULL,
    stop_atr_mult REAL NOT NULL DEFAULT 1.5,
    target_atr_mult REAL NOT NULL DEFAULT 2.0,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    initial_risk REAL NOT NULL,
    reward_risk_ratio REAL NOT NULL,

    -- Configuration
    max_bars INTEGER NOT NULL DEFAULT 15,

    -- Session info
    session_date TEXT NOT NULL,
    session_type TEXT NOT NULL,

    -- Indicator snapshot at entry (JSON)
    indicator_data TEXT,

    created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_bracket_sig_symbol ON bracket_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_bracket_sig_time ON bracket_signals(entry_time);
CREATE INDEX IF NOT EXISTS idx_bracket_sig_session ON bracket_signals(session_date);

-- Bracket resolutions: primary bracket + secondary time snapshots
CREATE TABLE IF NOT EXISTS bracket_resolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL REFERENCES bracket_signals(signal_id),

    -- Resolution type
    resolution_type TEXT NOT NULL CHECK (resolution_type IN (
        'BRACKET', 'SNAPSHOT_1M', 'SNAPSHOT_5M', 'SNAPSHOT_15M'
    )),

    -- Outcome (for BRACKET type)
    outcome TEXT CHECK (outcome IN (
        'TARGET_HIT', 'STOP_HIT', 'TRAILING_STOP', 'BREAKEVEN_STOP',
        'TIMEOUT_PROFIT', 'TIMEOUT_LOSS', 'TIMEOUT_SCRATCH'
    )),

    -- Exit details
    exit_price REAL NOT NULL,
    exit_time INTEGER,
    exit_bar_index INTEGER,
    bars_held INTEGER,

    -- P&L metrics
    pnl_absolute REAL NOT NULL,
    pnl_percent REAL NOT NULL,
    r_multiple REAL NOT NULL,

    -- MFE/MAE tracking
    mfe_price REAL,
    mfe_r REAL,
    mfe_bar_index INTEGER,
    mae_price REAL,
    mae_r REAL,
    mae_bar_index INTEGER,

    -- End-trade drawdown from MFE
    etd_from_mfe_r REAL,

    -- Trailing stop state at resolution
    trailing_stop_state TEXT CHECK (trailing_stop_state IN (
        'INITIAL', 'BREAKEVEN', 'TRAIL_1R', 'TRAIL_2R'
    )),
    final_stop_price REAL,

    -- Quality score (0-100)
    quality_score REAL,

    created_at INTEGER DEFAULT (strftime('%s', 'now')),

    UNIQUE(signal_id, resolution_type)
);

CREATE INDEX IF NOT EXISTS idx_bracket_res_signal ON bracket_resolutions(signal_id);
CREATE INDEX IF NOT EXISTS idx_bracket_res_type ON bracket_resolutions(resolution_type);
CREATE INDEX IF NOT EXISTS idx_bracket_res_outcome ON bracket_resolutions(outcome);

-- Bracket stats summary view
CREATE VIEW IF NOT EXISTS v_bracket_stats AS
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN outcome = 'TARGET_HIT' THEN 1 ELSE 0 END) as target_hits,
    SUM(CASE WHEN outcome = 'STOP_HIT' THEN 1 ELSE 0 END) as stop_hits,
    SUM(CASE WHEN outcome IN ('TRAILING_STOP', 'BREAKEVEN_STOP') THEN 1 ELSE 0 END) as trailing_exits,
    SUM(CASE WHEN outcome LIKE 'TIMEOUT%' THEN 1 ELSE 0 END) as timeouts,
    SUM(CASE WHEN outcome = 'TIMEOUT_SCRATCH' THEN 1 ELSE 0 END) as scratches,
    ROUND(
        CAST(SUM(CASE WHEN outcome = 'TARGET_HIT' THEN 1 ELSE 0 END) AS REAL) /
        NULLIF(SUM(CASE WHEN outcome IN ('TARGET_HIT', 'STOP_HIT') THEN 1 ELSE 0 END), 0),
        4
    ) as bracket_win_rate,
    ROUND(
        CAST(SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*),
        4
    ) as profitable_rate,
    ROUND(AVG(r_multiple), 4) as avg_r,
    ROUND(SUM(r_multiple), 2) as total_r,
    ROUND(AVG(mfe_r), 4) as avg_mfe_r,
    ROUND(AVG(mae_r), 4) as avg_mae_r,
    ROUND(AVG(etd_from_mfe_r), 4) as avg_etd_r,
    ROUND(AVG(bars_held), 1) as avg_bars_held,
    ROUND(AVG(quality_score), 1) as avg_quality
FROM bracket_resolutions
WHERE resolution_type = 'BRACKET';

-- Per-symbol bracket stats view
CREATE VIEW IF NOT EXISTS v_symbol_bracket_stats AS
SELECT
    s.symbol,
    COUNT(*) as trades,
    ROUND(AVG(r.r_multiple), 4) as avg_r,
    ROUND(SUM(r.r_multiple), 2) as total_r,
    ROUND(
        CAST(SUM(CASE WHEN r.outcome = 'TARGET_HIT' THEN 1 ELSE 0 END) AS REAL) /
        NULLIF(SUM(CASE WHEN r.outcome IN ('TARGET_HIT', 'STOP_HIT') THEN 1 ELSE 0 END), 0),
        4
    ) as win_rate,
    ROUND(AVG(r.mfe_r), 4) as avg_mfe_r,
    ROUND(AVG(r.mae_r), 4) as avg_mae_r
FROM bracket_signals s
JOIN bracket_resolutions r ON s.signal_id = r.signal_id
WHERE r.resolution_type = 'BRACKET'
GROUP BY s.symbol;
"""


class TradingStatsDB:
    """
    SQLite database with WAL mode and buffered writes.
    Thread-safe for concurrent reads during writes.
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Write buffer
        self._buffer: deque = deque()
        self._buffer_lock = threading.Lock()
        self._last_flush = time.time()

        # Initialize DB
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._configure()
        self._create_schema()

        # Background flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._background_flush, daemon=True)
        self._flush_thread.start()

        logger.info(f"Stats DB initialized at {self.db_path}")

    def _configure(self):
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-64000;
            PRAGMA temp_store=MEMORY;
        """)

    def _create_schema(self):
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    # ── Write operations ──

    def log_signal_event(self, event: dict):
        """Buffer a signal event for batch writing."""
        row = {
            'timestamp_utc': event.get('bar_time') or event.get('detected_at') or int(time.time()),
            'session_date': get_trading_date(event.get('bar_time') or int(time.time())),
            'session_type': get_session_type(event.get('bar_time') or int(time.time())),
            'signal_source': event.get('source', 'trade_signal'),
            'signal_id': event.get('signal_id', ''),
            'state': event.get('state', ''),
            'event_type': event.get('event_type', ''),
            'direction': event.get('direction', ''),
            'confluence_score': event.get('confluence_score', 0),
            'entry_price': event.get('entry_price', 0),
            'signal_data': json.dumps({
                k: v for k, v in event.get('indicator_data', {}).items()
            }) if event.get('indicator_data') else '{}',
            'outcome_1m_price': event.get('outcome_prices', {}).get('1m'),
            'outcome_5m_price': event.get('outcome_prices', {}).get('5m'),
            'outcome_15m_price': event.get('outcome_prices', {}).get('15m'),
            'peak_favorable': event.get('peak_favorable', 0),
            'peak_adverse': event.get('peak_adverse', 0),
            'is_resolved': 1 if event.get('event_type') == 'TRADE_RESOLVED' else 0,
            'resolution_type': event.get('resolution_type', ''),
        }

        with self._buffer_lock:
            self._buffer.append(('signal', row))
            if len(self._buffer) >= BUFFER_SIZE:
                self._flush_now()

    def log_pattern_snapshot(self, snapshot: dict):
        """Buffer a pattern snapshot for batch writing."""
        row = {
            'timestamp_utc': snapshot.get('timestamp', int(time.time())),
            'session_date': snapshot.get('session_date', ''),
            'session_type': snapshot.get('session_type', ''),
            'snapshot_id': snapshot.get('snapshot_id', ''),
            'trigger': snapshot.get('trigger', ''),
            'direction': snapshot.get('direction', ''),
            'consensus': snapshot.get('consensus', ''),
            'avg_correlation': snapshot.get('avg_correlation', 0),
            'mean_move': snapshot.get('mean_move', 0),
            'eod_projected_price': snapshot.get('eod_projected_price', 0),
            'peak_projected_price': snapshot.get('peak_projected_price', 0),
            'match_count': snapshot.get('match_count', 0),
            'signal_strength': snapshot.get('signal_strength', 0),
        }

        with self._buffer_lock:
            self._buffer.append(('pattern', row))
            if len(self._buffer) >= BUFFER_SIZE:
                self._flush_now()

    def log_prediction(self, prediction: dict, symbol: str):
        """Buffer a prediction entry (every pattern cycle)."""
        row = {
            'timestamp_utc': prediction.get('timestamp', int(time.time())),
            'session_date': prediction.get('session_date', ''),
            'session_type': prediction.get('session_type', ''),
            'symbol': symbol,
            'direction': prediction.get('direction', ''),
            'consensus': prediction.get('consensus', ''),
            'avg_correlation': prediction.get('avg_correlation', 0),
            'current_price': prediction.get('current_price', 0),
            'eod_projected_price': prediction.get('eod_projected_price', 0),
            'peak_projected_price': prediction.get('peak_projected_price', 0),
            'mean_move': prediction.get('mean_move', 0),
            'match_count': prediction.get('match_count', 0),
        }

        with self._buffer_lock:
            self._buffer.append(('prediction', row))
            if len(self._buffer) >= BUFFER_SIZE:
                self._flush_now()

    def upsert_session_prediction(self, data: dict):
        """Insert or update a session-level anchored prediction (called every minute)."""
        with self._buffer_lock:
            self._buffer.append(('session_pred', data))
            if len(self._buffer) >= BUFFER_SIZE:
                self._flush_now()

    def _background_flush(self):
        while self._running:
            time.sleep(1.0)
            with self._buffer_lock:
                if self._buffer and (time.time() - self._last_flush) >= FLUSH_INTERVAL:
                    self._flush_now()

    def _flush_now(self):
        """Execute batch insert (called with lock held)."""
        if not self._buffer:
            return

        items = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.time()

        signal_rows = [row for typ, row in items if typ == 'signal']
        pattern_rows = [row for typ, row in items if typ == 'pattern']
        prediction_rows = [row for typ, row in items if typ == 'prediction']
        session_pred_rows = [row for typ, row in items if typ == 'session_pred']

        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")

            if signal_rows:
                cursor.executemany("""
                    INSERT INTO signal_log (
                        timestamp_utc, session_date, session_type, signal_source,
                        signal_id, state, event_type, direction, confluence_score,
                        entry_price, signal_data,
                        outcome_1m_price, outcome_5m_price, outcome_15m_price,
                        peak_favorable, peak_adverse, is_resolved, resolution_type
                    ) VALUES (
                        :timestamp_utc, :session_date, :session_type, :signal_source,
                        :signal_id, :state, :event_type, :direction, :confluence_score,
                        :entry_price, :signal_data,
                        :outcome_1m_price, :outcome_5m_price, :outcome_15m_price,
                        :peak_favorable, :peak_adverse, :is_resolved, :resolution_type
                    )
                """, signal_rows)

            if pattern_rows:
                cursor.executemany("""
                    INSERT INTO pattern_snapshots (
                        timestamp_utc, session_date, session_type, snapshot_id,
                        trigger, direction, consensus, avg_correlation, mean_move,
                        eod_projected_price, peak_projected_price, match_count,
                        signal_strength
                    ) VALUES (
                        :timestamp_utc, :session_date, :session_type, :snapshot_id,
                        :trigger, :direction, :consensus, :avg_correlation, :mean_move,
                        :eod_projected_price, :peak_projected_price, :match_count,
                        :signal_strength
                    )
                """, pattern_rows)

            if prediction_rows:
                cursor.executemany("""
                    INSERT INTO prediction_log (
                        timestamp_utc, session_date, session_type, symbol,
                        direction, consensus, avg_correlation, current_price,
                        eod_projected_price, peak_projected_price, mean_move,
                        match_count
                    ) VALUES (
                        :timestamp_utc, :session_date, :session_type, :symbol,
                        :direction, :consensus, :avg_correlation, :current_price,
                        :eod_projected_price, :peak_projected_price, :mean_move,
                        :match_count
                    )
                """, prediction_rows)

            if session_pred_rows:
                for sp in session_pred_rows:
                    cursor.execute("""
                        INSERT INTO session_predictions (
                            session_date, session_type, symbol,
                            anchor_direction, anchor_price, anchor_eod_proj, anchor_peak_proj, anchor_time,
                            total_votes, bullish_votes, bearish_votes, direction_stability,
                            avg_eod_projected, avg_peak_projected, avg_correlation,
                            final_direction, final_confidence,
                            regime_changes, regime_change_log,
                            last_price, last_direction, last_update_time
                        ) VALUES (
                            :session_date, :session_type, :symbol,
                            :anchor_direction, :anchor_price, :anchor_eod_proj, :anchor_peak_proj, :anchor_time,
                            :total_votes, :bullish_votes, :bearish_votes, :direction_stability,
                            :avg_eod_projected, :avg_peak_projected, :avg_correlation,
                            :final_direction, :final_confidence,
                            :regime_changes, :regime_change_log,
                            :last_price, :last_direction, :last_update_time
                        )
                        ON CONFLICT(session_date, session_type, symbol) DO UPDATE SET
                            total_votes = excluded.total_votes,
                            bullish_votes = excluded.bullish_votes,
                            bearish_votes = excluded.bearish_votes,
                            direction_stability = excluded.direction_stability,
                            avg_eod_projected = excluded.avg_eod_projected,
                            avg_peak_projected = excluded.avg_peak_projected,
                            avg_correlation = excluded.avg_correlation,
                            final_direction = excluded.final_direction,
                            final_confidence = excluded.final_confidence,
                            regime_changes = excluded.regime_changes,
                            regime_change_log = excluded.regime_change_log,
                            last_price = excluded.last_price,
                            last_direction = excluded.last_direction,
                            last_update_time = excluded.last_update_time
                    """, sp)

            self.conn.commit()
            logger.debug(f"Stats flush: {len(signal_rows)} signals, {len(pattern_rows)} patterns, "
                        f"{len(prediction_rows)} predictions, {len(session_pred_rows)} session preds")
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            logger.error(f"Stats flush error: {e}")
            # Re-queue
            with self._buffer_lock:
                for item in reversed(items):
                    self._buffer.appendleft(item)

    # ── Read operations (for dashboard) ──

    def get_today_stats(self, signal_source: str = None) -> dict:
        """Get today's signal stats for dashboard display."""
        today = get_trading_date(int(time.time()))

        try:
            cursor = self.conn.cursor()

            if signal_source:
                sources = [signal_source]
            else:
                sources = ['trade_signal', 'rth_pattern', 'overnight_pattern']

            stats = {}
            for src in sources:
                # Count signals today
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN event_type = 'SIGNAL_CONFIRMED' THEN 1 ELSE 0 END) as confirmed,
                        SUM(CASE WHEN event_type = 'TRADE_RESOLVED' THEN 1 ELSE 0 END) as resolved
                    FROM signal_log
                    WHERE session_date = ? AND signal_source = ?
                """, (today, src))
                row = cursor.fetchone()

                # Win rates from resolved trades
                cursor.execute("""
                    SELECT
                        COUNT(*) as n,
                        SUM(CASE WHEN
                            (direction = 'long' AND outcome_1m_price > entry_price + 5) OR
                            (direction = 'short' AND outcome_1m_price < entry_price - 5)
                        THEN 1 ELSE 0 END) as wins_1m,
                        SUM(CASE WHEN
                            (direction = 'long' AND outcome_5m_price > entry_price + 10) OR
                            (direction = 'short' AND outcome_5m_price < entry_price - 10)
                        THEN 1 ELSE 0 END) as wins_5m,
                        SUM(CASE WHEN
                            (direction = 'long' AND outcome_15m_price > entry_price + 15) OR
                            (direction = 'short' AND outcome_15m_price < entry_price - 15)
                        THEN 1 ELSE 0 END) as wins_15m,
                        AVG(peak_favorable) as avg_mfe,
                        AVG(peak_adverse) as avg_mae
                    FROM signal_log
                    WHERE session_date = ? AND signal_source = ?
                      AND event_type = 'TRADE_RESOLVED'
                """, (today, src))
                resolved_row = cursor.fetchone()

                n_resolved = resolved_row[0] if resolved_row else 0
                stats[src] = {
                    'total_events': row[0] if row else 0,
                    'confirmed': row[1] if row else 0,
                    'resolved': n_resolved,
                    'wins_1m': resolved_row[1] if resolved_row else 0,
                    'wins_5m': resolved_row[2] if resolved_row else 0,
                    'wins_15m': resolved_row[3] if resolved_row else 0,
                    'win_rate_1m': round(resolved_row[1] / n_resolved, 3) if n_resolved > 0 else 0,
                    'win_rate_5m': round(resolved_row[2] / n_resolved, 3) if n_resolved > 0 else 0,
                    'win_rate_15m': round(resolved_row[3] / n_resolved, 3) if n_resolved > 0 else 0,
                    'avg_mfe': round(resolved_row[4] or 0, 1),
                    'avg_mae': round(resolved_row[5] or 0, 1),
                }

            return {'session_date': today, 'stats': stats}
        except Exception as e:
            logger.error(f"Error reading today stats: {e}")
            return {'session_date': today, 'stats': {}}

    def get_rolling_stats(self, window_days: int = 7) -> dict:
        """Get rolling window statistics."""
        try:
            cursor = self.conn.cursor()
            cutoff = int(time.time()) - (window_days * 86400)

            results = {}
            for src in ['trade_signal', 'rth_pattern', 'overnight_pattern']:
                cursor.execute("""
                    SELECT
                        COUNT(*) as n,
                        SUM(CASE WHEN
                            (direction = 'long' AND outcome_1m_price > entry_price + 5) OR
                            (direction = 'short' AND outcome_1m_price < entry_price - 5)
                        THEN 1 ELSE 0 END) as wins_1m,
                        SUM(CASE WHEN
                            (direction = 'long' AND outcome_5m_price > entry_price + 10) OR
                            (direction = 'short' AND outcome_5m_price < entry_price - 10)
                        THEN 1 ELSE 0 END) as wins_5m,
                        AVG(peak_favorable) as avg_mfe,
                        AVG(peak_adverse) as avg_mae
                    FROM signal_log
                    WHERE timestamp_utc >= ? AND signal_source = ?
                      AND event_type = 'TRADE_RESOLVED'
                """, (cutoff, src))
                row = cursor.fetchone()
                n = row[0] if row else 0
                results[src] = {
                    'window_days': window_days,
                    'total_resolved': n,
                    'win_rate_1m': round(row[1] / n, 3) if n > 0 else 0,
                    'win_rate_5m': round(row[2] / n, 3) if n > 0 else 0,
                    'avg_mfe': round(row[3] or 0, 1),
                    'avg_mae': round(row[4] or 0, 1),
                }

            return results
        except Exception as e:
            logger.error(f"Error reading rolling stats: {e}")
            return {}

    def get_pattern_accuracy(self, session_type: str = None, days: int = 30) -> dict:
        """Get pattern prediction accuracy over recent sessions."""
        try:
            cursor = self.conn.cursor()

            where = "WHERE timestamp_utc >= ?"
            params = [int(time.time()) - days * 86400]

            if session_type:
                where += " AND session_type = ?"
                params.append(session_type)

            cursor.execute(f"""
                SELECT
                    session_type,
                    COUNT(*) as snapshots,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(ABS(eod_projected_price - actual_session_close)) as avg_price_error
                FROM pattern_snapshots
                {where}
                AND actual_session_close IS NOT NULL
                GROUP BY session_type
            """, params)

            results = {}
            for row in cursor.fetchall():
                st = row[0]
                n = row[1]
                results[st] = {
                    'snapshots': n,
                    'direction_correct': row[2],
                    'direction_accuracy': round(row[2] / n, 3) if n > 0 else 0,
                    'avg_price_error': round(row[3] or 0, 2),
                }

            return results
        except Exception as e:
            logger.error(f"Error reading pattern accuracy: {e}")
            return {}

    def get_prediction_history(self, session_type: str = None, symbol: str = None,
                               limit: int = 200) -> list:
        """Get recent prediction log entries."""
        try:
            cursor = self.conn.cursor()
            where_parts = []
            params = []
            if session_type:
                where_parts.append("session_type = ?")
                params.append(session_type)
            if symbol:
                where_parts.append("symbol = ?")
                params.append(symbol)
            where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
            cursor.execute(f"""
                SELECT timestamp_utc, session_date, session_type, symbol,
                       direction, consensus, avg_correlation, current_price,
                       eod_projected_price, peak_projected_price, mean_move, match_count
                FROM prediction_log {where}
                ORDER BY timestamp_utc DESC LIMIT ?
            """, params + [limit])
            cols = ['timestamp', 'session_date', 'session_type', 'symbol',
                    'direction', 'consensus', 'avg_correlation', 'current_price',
                    'eod_projected_price', 'peak_projected_price', 'mean_move', 'match_count']
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error reading prediction history: {e}")
            return []

    def get_session_predictions(self, symbol: str = None, is_closed: int = None,
                                 limit: int = 50) -> list:
        """Get session-level anchored predictions."""
        try:
            cursor = self.conn.cursor()
            where_parts = []
            params = []
            if symbol:
                where_parts.append("symbol = ?")
                params.append(symbol)
            if is_closed is not None:
                where_parts.append("is_closed = ?")
                params.append(is_closed)
            where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
            cursor.execute(f"""
                SELECT session_date, session_type, symbol,
                       anchor_direction, anchor_price, anchor_eod_proj, anchor_peak_proj, anchor_time,
                       total_votes, bullish_votes, bearish_votes, direction_stability,
                       avg_eod_projected, avg_peak_projected, avg_correlation,
                       final_direction, final_confidence,
                       regime_changes, regime_change_log,
                       last_price, last_direction, last_update_time,
                       actual_close_price, direction_correct, eod_error_points,
                       is_closed
                FROM session_predictions {where}
                ORDER BY session_date DESC, session_type DESC LIMIT ?
            """, params + [limit])
            cols = ['session_date', 'session_type', 'symbol',
                    'anchor_direction', 'anchor_price', 'anchor_eod_proj', 'anchor_peak_proj', 'anchor_time',
                    'total_votes', 'bullish_votes', 'bearish_votes', 'direction_stability',
                    'avg_eod_projected', 'avg_peak_projected', 'avg_correlation',
                    'final_direction', 'final_confidence',
                    'regime_changes', 'regime_change_log',
                    'last_price', 'last_direction', 'last_update_time',
                    'actual_close_price', 'direction_correct', 'eod_error_points',
                    'is_closed']
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error reading session predictions: {e}")
            return []

    def get_all_stats_summary(self) -> dict:
        """Get comprehensive stats summary for the stats page."""
        try:
            today = get_trading_date(int(time.time()))
            cursor = self.conn.cursor()

            # Signal stats by source (today)
            signal_stats = self.get_today_stats()

            # Prediction counts
            cursor.execute("""
                SELECT session_type, COUNT(*) as cnt,
                       COUNT(DISTINCT session_date) as days
                FROM prediction_log GROUP BY session_type
            """)
            prediction_counts = {row[0]: {'total_predictions': row[1], 'days_tracked': row[2]}
                                for row in cursor.fetchall()}

            # Recent predictions (last 50)
            recent_predictions = self.get_prediction_history(limit=50)

            # Pattern snapshots summary
            cursor.execute("""
                SELECT session_type, COUNT(*) as snapshots,
                       SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
                       SUM(CASE WHEN direction_correct IS NOT NULL THEN 1 ELSE 0 END) as evaluated
                FROM pattern_snapshots GROUP BY session_type
            """)
            pattern_accuracy = {}
            for row in cursor.fetchall():
                evaluated = row[3] or 0
                pattern_accuracy[row[0]] = {
                    'snapshots': row[1],
                    'evaluated': evaluated,
                    'correct': row[2] or 0,
                    'accuracy': round((row[2] or 0) / evaluated, 3) if evaluated > 0 else None,
                }

            # Session-level anchored predictions
            session_predictions = self.get_session_predictions(limit=20)

            return {
                'session_date': today,
                'signal_stats': signal_stats,
                'prediction_counts': prediction_counts,
                'recent_predictions': recent_predictions,
                'pattern_accuracy': pattern_accuracy,
                'session_predictions': session_predictions,
                'rolling_7d': self.get_rolling_stats(7),
                'rolling_30d': self.get_rolling_stats(30),
            }
        except Exception as e:
            logger.error(f"Error building stats summary: {e}")
            return {}

    # ── Bracket Resolution DB Operations ──

    def insert_bracket_signal(self, sig: SignalResolution, session_date: str,
                               session_type: str) -> None:
        """Persist a new bracket signal to the database."""
        try:
            self.conn.execute("""
                INSERT OR IGNORE INTO bracket_signals (
                    signal_id, symbol, direction, confluence_score,
                    entry_price, entry_time, entry_bar_index,
                    atr_at_entry, stop_atr_mult, target_atr_mult,
                    stop_price, target_price, initial_risk, reward_risk_ratio,
                    max_bars, session_date, session_type, indicator_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig.signal_id, sig.symbol, sig.direction.value, sig.confluence_score,
                sig.bracket.entry_price, sig.entry_time, sig.entry_bar_index,
                sig.bracket.atr_at_entry, sig.bracket.stop_atr_mult, sig.bracket.target_atr_mult,
                sig.bracket.stop_price, sig.bracket.target_price,
                sig.bracket.initial_risk, sig.bracket.reward_risk_ratio,
                sig.max_bars, session_date, session_type,
                json.dumps(sig.indicator_data) if sig.indicator_data else '{}',
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error inserting bracket signal: {e}")

    def save_bracket_resolution(self, sig: SignalResolution) -> None:
        """Save primary bracket resolution and time snapshots."""
        try:
            entry = sig.bracket.entry_price
            exit_p = sig.exit_price
            initial_risk = sig.bracket.initial_risk

            if sig.direction == Direction.LONG:
                pnl_abs = exit_p - entry
            else:
                pnl_abs = entry - exit_p

            pnl_pct = (pnl_abs / entry) * 100 if entry != 0 else 0
            r_mult = pnl_abs / initial_risk if initial_risk != 0 else 0
            etd = sig.mfe_mae.mfe_r - r_mult if sig.mfe_mae.mfe_r > 0 else 0
            quality = score_signal_quality(sig)

            self.conn.execute("""
                INSERT OR IGNORE INTO bracket_resolutions (
                    signal_id, resolution_type, outcome,
                    exit_price, exit_time, exit_bar_index, bars_held,
                    pnl_absolute, pnl_percent, r_multiple,
                    mfe_price, mfe_r, mfe_bar_index,
                    mae_price, mae_r, mae_bar_index,
                    etd_from_mfe_r, trailing_stop_state, final_stop_price,
                    quality_score
                ) VALUES (?, 'BRACKET', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                sig.signal_id, sig.outcome.value,
                exit_p, sig.exit_time, sig.exit_bar_index, sig.bars_held,
                pnl_abs, pnl_pct, r_mult,
                sig.mfe_mae.mfe_price, sig.mfe_mae.mfe_r, sig.mfe_mae.mfe_bar_index,
                sig.mfe_mae.mae_price, sig.mfe_mae.mae_r, sig.mfe_mae.mae_bar_index,
                etd, sig.trailing.state.value, sig.trailing.current_stop,
                quality,
            ))

            # Insert time snapshots as secondary resolutions
            for snap_type, pnl in [
                ('SNAPSHOT_1M', sig.snapshot_1m),
                ('SNAPSHOT_5M', sig.snapshot_5m),
                ('SNAPSHOT_15M', sig.snapshot_15m),
            ]:
                if pnl is not None:
                    snap_r = pnl / initial_risk if initial_risk != 0 else 0
                    if sig.direction == Direction.LONG:
                        snap_price = entry + pnl
                    else:
                        snap_price = entry - pnl
                    self.conn.execute("""
                        INSERT OR IGNORE INTO bracket_resolutions (
                            signal_id, resolution_type,
                            exit_price, exit_time, pnl_absolute, pnl_percent, r_multiple
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        sig.signal_id, snap_type,
                        snap_price, sig.entry_time,
                        pnl, (pnl / entry) * 100 if entry != 0 else 0, snap_r,
                    ))

            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving bracket resolution: {e}")

    def get_bracket_stats(self, session_date: str = None) -> dict:
        """Get bracket resolution statistics, optionally filtered by date."""
        try:
            cursor = self.conn.cursor()

            if session_date:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN r.outcome = 'TARGET_HIT' THEN 1 ELSE 0 END) as target_hits,
                        SUM(CASE WHEN r.outcome = 'STOP_HIT' THEN 1 ELSE 0 END) as stop_hits,
                        SUM(CASE WHEN r.outcome IN ('TRAILING_STOP','BREAKEVEN_STOP') THEN 1 ELSE 0 END) as trailing_exits,
                        SUM(CASE WHEN r.outcome LIKE 'TIMEOUT%' THEN 1 ELSE 0 END) as timeouts,
                        SUM(CASE WHEN r.outcome = 'TIMEOUT_SCRATCH' THEN 1 ELSE 0 END) as scratches,
                        ROUND(AVG(r.r_multiple), 4) as avg_r,
                        ROUND(SUM(r.r_multiple), 2) as total_r,
                        ROUND(AVG(r.mfe_r), 4) as avg_mfe_r,
                        ROUND(AVG(r.mae_r), 4) as avg_mae_r,
                        ROUND(AVG(r.etd_from_mfe_r), 4) as avg_etd_r,
                        ROUND(AVG(r.bars_held), 1) as avg_bars_held,
                        ROUND(AVG(r.quality_score), 1) as avg_quality
                    FROM bracket_signals s
                    JOIN bracket_resolutions r ON s.signal_id = r.signal_id
                    WHERE r.resolution_type = 'BRACKET' AND s.session_date = ?
                """, (session_date,))
            else:
                cursor.execute("SELECT * FROM v_bracket_stats")

            row = cursor.fetchone()
            if not row or row[0] == 0:
                return {}

            total = row[0]
            target_hits = row[1] or 0
            stop_hits = row[2] or 0
            bracket_resolved = target_hits + stop_hits

            result = {
                'total_trades': total,
                'target_hits': target_hits,
                'stop_hits': stop_hits,
                'trailing_exits': row[3] or 0,
                'timeouts': row[4] or 0,
                'scratches': row[5] or 0,
                'bracket_win_rate': round(target_hits / bracket_resolved, 4) if bracket_resolved > 0 else None,
                'profitable_rate': round(sum(1 for _ in [] if True) / total, 4) if total > 0 else None,
                'avg_r': row[6],
                'total_r': row[7],
                'avg_mfe_r': row[8],
                'avg_mae_r': row[9],
                'avg_etd_r': row[10],
                'avg_bars_held': row[11],
                'avg_quality': row[12],
            }

            # Calculate profitable rate properly
            cursor2 = self.conn.cursor()
            if session_date:
                cursor2.execute("""
                    SELECT COUNT(*) FROM bracket_resolutions r
                    JOIN bracket_signals s ON r.signal_id = s.signal_id
                    WHERE r.resolution_type = 'BRACKET' AND r.r_multiple > 0
                      AND s.session_date = ?
                """, (session_date,))
            else:
                cursor2.execute("""
                    SELECT COUNT(*) FROM bracket_resolutions
                    WHERE resolution_type = 'BRACKET' AND r_multiple > 0
                """)
            profitable_count = cursor2.fetchone()[0] or 0
            result['profitable_rate'] = round(profitable_count / total, 4) if total > 0 else None

            # Get R-multiples for SQN
            if session_date:
                cursor2.execute("""
                    SELECT r.r_multiple FROM bracket_resolutions r
                    JOIN bracket_signals s ON r.signal_id = s.signal_id
                    WHERE r.resolution_type = 'BRACKET' AND s.session_date = ?
                """, (session_date,))
            else:
                cursor2.execute("""
                    SELECT r_multiple FROM bracket_resolutions
                    WHERE resolution_type = 'BRACKET'
                """)
            r_multiples = [row[0] for row in cursor2.fetchall() if row[0] is not None]
            if len(r_multiples) >= 2:
                result['sqn'] = round(calculate_sqn(r_multiples), 2)
                result['expectancy'] = round(sum(r_multiples) / len(r_multiples), 4)
            else:
                result['sqn'] = None
                result['expectancy'] = result['avg_r']

            return result
        except Exception as e:
            logger.error(f"Error reading bracket stats: {e}")
            return {}

    def get_bracket_trades(self, session_date: str = None, limit: int = 50) -> list:
        """Get recent bracket trade resolutions."""
        try:
            cursor = self.conn.cursor()
            if session_date:
                cursor.execute("""
                    SELECT s.signal_id, s.symbol, s.direction, s.confluence_score,
                           s.entry_price, s.entry_time,
                           s.atr_at_entry, s.stop_price, s.target_price,
                           s.initial_risk, s.reward_risk_ratio,
                           r.outcome, r.exit_price, r.exit_time, r.bars_held,
                           r.r_multiple, r.mfe_r, r.mae_r, r.etd_from_mfe_r,
                           r.trailing_stop_state, r.quality_score
                    FROM bracket_signals s
                    JOIN bracket_resolutions r ON s.signal_id = r.signal_id
                    WHERE r.resolution_type = 'BRACKET' AND s.session_date = ?
                    ORDER BY s.entry_time DESC LIMIT ?
                """, (session_date, limit))
            else:
                cursor.execute("""
                    SELECT s.signal_id, s.symbol, s.direction, s.confluence_score,
                           s.entry_price, s.entry_time,
                           s.atr_at_entry, s.stop_price, s.target_price,
                           s.initial_risk, s.reward_risk_ratio,
                           r.outcome, r.exit_price, r.exit_time, r.bars_held,
                           r.r_multiple, r.mfe_r, r.mae_r, r.etd_from_mfe_r,
                           r.trailing_stop_state, r.quality_score
                    FROM bracket_signals s
                    JOIN bracket_resolutions r ON s.signal_id = r.signal_id
                    WHERE r.resolution_type = 'BRACKET'
                    ORDER BY s.entry_time DESC LIMIT ?
                """, (limit,))

            cols = [
                'signal_id', 'symbol', 'direction', 'confluence_score',
                'entry_price', 'entry_time',
                'atr_at_entry', 'stop_price', 'target_price',
                'initial_risk', 'reward_risk_ratio',
                'outcome', 'exit_price', 'exit_time', 'bars_held',
                'r_multiple', 'mfe_r', 'mae_r', 'etd_from_mfe_r',
                'trailing_stop_state', 'quality_score',
            ]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error reading bracket trades: {e}")
            return []

    def get_rolling_bracket_stats(self, window_days: int = 7) -> dict:
        """Get rolling bracket stats for a time window."""
        try:
            cursor = self.conn.cursor()
            cutoff = int(time.time()) - (window_days * 86400)

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN r.outcome = 'TARGET_HIT' THEN 1 ELSE 0 END) as target_hits,
                    SUM(CASE WHEN r.outcome = 'STOP_HIT' THEN 1 ELSE 0 END) as stop_hits,
                    SUM(CASE WHEN r.r_multiple > 0 THEN 1 ELSE 0 END) as profitable,
                    ROUND(AVG(r.r_multiple), 4) as avg_r,
                    ROUND(SUM(r.r_multiple), 2) as total_r,
                    ROUND(AVG(r.mfe_r), 4) as avg_mfe_r,
                    ROUND(AVG(r.mae_r), 4) as avg_mae_r,
                    ROUND(AVG(r.quality_score), 1) as avg_quality
                FROM bracket_signals s
                JOIN bracket_resolutions r ON s.signal_id = r.signal_id
                WHERE r.resolution_type = 'BRACKET' AND s.entry_time >= ?
            """, (cutoff,))

            row = cursor.fetchone()
            if not row or row[0] == 0:
                return {'window_days': window_days, 'total_trades': 0}

            total = row[0]
            target_hits = row[1] or 0
            stop_hits = row[2] or 0
            bracket_resolved = target_hits + stop_hits

            return {
                'window_days': window_days,
                'total_trades': total,
                'target_hits': target_hits,
                'stop_hits': stop_hits,
                'bracket_win_rate': round(target_hits / bracket_resolved, 4) if bracket_resolved > 0 else None,
                'profitable_rate': round((row[3] or 0) / total, 4) if total > 0 else None,
                'avg_r': row[4],
                'total_r': row[5],
                'avg_mfe_r': row[6],
                'avg_mae_r': row[7],
                'avg_quality': row[8],
            }
        except Exception as e:
            logger.error(f"Error reading rolling bracket stats: {e}")
            return {'window_days': window_days, 'total_trades': 0}

    # ── Lifecycle ──

    def flush(self):
        """Force flush buffer."""
        with self._buffer_lock:
            self._flush_now()

    def shutdown(self):
        """Graceful shutdown."""
        self._running = False
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        with self._buffer_lock:
            self._flush_now()
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.conn.close()
        except Exception:
            pass
        logger.info("Stats DB shut down")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STATS MANAGER (integrates all components)
# ═══════════════════════════════════════════════════════════════════════════

class StatsManager:
    """
    Top-level manager that integrates:
    - TradeSignalTracker (state machine for scalping signals)
    - PatternSnapshotManager (for daily + overnight patterns)
    - TradingStatsDB (persistence)
    """

    def __init__(self):
        self.db = TradingStatsDB()
        self.trade_trackers: Dict[str, TradeSignalTracker] = {}  # per symbol
        self.rth_snapshots: Dict[str, PatternSnapshotManager] = {}
        self.overnight_snapshots: Dict[str, PatternSnapshotManager] = {}
        self.session_tracker = AnchoredSessionTracker()

        logger.info("StatsManager initialized")

    def get_tracker(self, symbol: str) -> TradeSignalTracker:
        if symbol not in self.trade_trackers:
            self.trade_trackers[symbol] = TradeSignalTracker()
        return self.trade_trackers[symbol]

    def get_rth_snapshot_mgr(self, symbol: str) -> PatternSnapshotManager:
        if symbol not in self.rth_snapshots:
            self.rth_snapshots[symbol] = PatternSnapshotManager('rth')
        return self.rth_snapshots[symbol]

    def get_overnight_snapshot_mgr(self, symbol: str) -> PatternSnapshotManager:
        if symbol not in self.overnight_snapshots:
            self.overnight_snapshots[symbol] = PatternSnapshotManager('overnight')
        return self.overnight_snapshots[symbol]

    # ── Called from frontend via WebSocket ──

    def process_trade_signal(self, symbol: str, bar_time: int, price: float,
                             action: str, confluence: float, atr: float,
                             indicators: dict,
                             bar_high: float = 0, bar_low: float = 0,
                             bar_open: float = 0) -> Optional[List[dict]]:
        """Process a trade signal bar from the scalping engine."""
        tracker = self.get_tracker(symbol)
        events = tracker.process_bar(
            bar_time, price, action, confluence, atr, indicators,
            bar_high=bar_high, bar_low=bar_low, bar_open=bar_open,
        )

        if events:
            for event in events:
                event['symbol'] = symbol
                self.db.log_signal_event(event)

                # Persist bracket signal to DB
                if event.get('event_type') == 'BRACKET_SIGNAL_CREATED':
                    sig_id = event['signal_id']
                    bracket_sig = tracker.active_bracket_signals.get(sig_id)
                    if bracket_sig:
                        bracket_sig.symbol = symbol
                        self.db.insert_bracket_signal(
                            bracket_sig,
                            session_date=get_trading_date(bar_time),
                            session_type=get_session_type(bar_time),
                        )

                if event.get('event_type') in ('SIGNAL_CONFIRMED', 'BRACKET_SIGNAL_CREATED'):
                    logger.info(f"[{symbol}] {event['event_type']}: "
                               f"{event.get('direction')}, conf={event.get('confluence_score', 0):.2f}")

        return events

    # ── Called from backend pattern loop ──

    def process_pattern_update(self, symbol: str, pattern_result: dict,
                               session_type: str = 'rth',
                               current_price: float = 0) -> Optional[dict]:
        """Process a pattern matcher update. Logs every prediction and snapshots on meaningful changes."""
        if session_type == 'rth':
            mgr = self.get_rth_snapshot_mgr(symbol)
        else:
            mgr = self.get_overnight_snapshot_mgr(symbol)

        bar_time = int(time.time())
        prediction, snapshot = mgr.process_update(pattern_result, bar_time, current_price)

        # Log every prediction cycle
        if prediction:
            self.db.log_prediction(prediction, symbol)

            # Feed into anchored session tracker
            session_update = self.session_tracker.process_prediction(prediction, symbol)
            if session_update:
                self.db.upsert_session_prediction(session_update)

        if snapshot:
            snapshot['symbol'] = symbol
            self.db.log_pattern_snapshot(snapshot)
            logger.info(f"[{symbol}] Pattern snapshot ({session_type}): "
                       f"{snapshot['trigger']} - {snapshot['direction']} "
                       f"({snapshot['consensus']})")

        return snapshot

    # ── Outcome resolution (called on every new 1-min bar) ──

    def update_pending_outcomes(self, symbol: str, bar_time: int, price: float,
                               bar_open: float = 0, bar_high: float = 0,
                               bar_low: float = 0):
        """
        Feed a new bar to all pending signals for this symbol.
        Handles both legacy time-snapshot resolution and bracket resolution.
        Called from on_bar_update in app.py on EVERY new bar.
        """
        tracker = self.get_tracker(symbol)

        # Legacy time-snapshot resolution
        if tracker.pending_outcomes:
            events = tracker.update(bar_time=bar_time, price=price, atr=0)
            for event in events:
                self.db.log_signal_event(event)

        # Bracket resolution (OHLC-based)
        if tracker.active_bracket_signals:
            o = bar_open if bar_open > 0 else price
            h = bar_high if bar_high > 0 else price
            l = bar_low if bar_low > 0 else price

            resolved = tracker.update_brackets(bar_time, o, h, l, price)
            for sig in resolved:
                self.db.save_bracket_resolution(sig)
                logger.info(
                    f"[{symbol}] Bracket resolved: {sig.signal_id} -> "
                    f"{sig.outcome.value} (R={sig.r_multiple:.2f})"
                )

    # ── Session close evaluation ──

    def evaluate_closed_sessions(self, symbol: str, current_price: float):
        """
        Check if any active anchored sessions have ended and evaluate them.
        Called every minute from the pattern match loop.

        A session is considered closed when the current time's session type
        no longer matches the session_type of the prediction.
        e.g. if we tracked an 'overnight' prediction and now it's 'rth' time,
        the overnight session is over — evaluate it.
        """
        now = int(time.time())
        current_session = get_session_type(now)

        for key, state in list(self.session_tracker._active_sessions.items()):
            sess_date, sess_type, sym = key
            if sym != symbol:
                continue

            # Session ended if we're in a different session type now
            # (overnight -> rth means overnight ended, rth -> overnight means rth ended)
            if sess_type != current_session and sess_type != 'maintenance':
                total = state['total_votes']
                bull = state['bullish_votes']
                bear = state['bearish_votes']
                majority_dir = 'bullish' if bull >= bear else 'bearish'

                # Did the market move in the predicted direction?
                anchor_price = state['anchor_price']
                actual_move = current_price - anchor_price
                predicted_bullish = majority_dir == 'bullish'
                direction_correct = (actual_move > 0 and predicted_bullish) or \
                                    (actual_move < 0 and not predicted_bullish)

                # Compute avg projected prices
                avg_eod = state['eod_proj_sum'] / total if total > 0 else 0
                eod_error = abs(current_price - avg_eod) if avg_eod else None

                # Update the DB row
                try:
                    cursor = self.db.conn.cursor()
                    cursor.execute("""
                        UPDATE session_predictions SET
                            actual_close_price = ?,
                            direction_correct = ?,
                            eod_error_points = ?,
                            is_closed = 1
                        WHERE session_date = ? AND session_type = ? AND symbol = ?
                    """, (current_price, 1 if direction_correct else 0,
                          eod_error, sess_date, sess_type, sym))
                    self.db.conn.commit()
                    logger.info(f"[{sym}] Session prediction EVALUATED: {sess_type} {sess_date} "
                               f"- predicted {majority_dir}, actual move {actual_move:+.1f}pts "
                               f"-> {'CORRECT' if direction_correct else 'WRONG'}")
                except Exception as e:
                    logger.error(f"Error evaluating session prediction: {e}")

                # Remove from active tracking
                self.session_tracker.clear_session(sess_date, sess_type, sym)

    # ── Dashboard API ──

    def get_dashboard_stats(self) -> dict:
        """Get all stats for dashboard display."""
        today = get_trading_date(int(time.time()))
        return {
            'today': self.db.get_today_stats(),
            'rolling_7d': self.db.get_rolling_stats(7),
            'rolling_30d': self.db.get_rolling_stats(30),
            'bracket_today': self.db.get_bracket_stats(session_date=today),
            'bracket_all': self.db.get_bracket_stats(),
            'bracket_7d': self.db.get_rolling_bracket_stats(7),
            'bracket_30d': self.db.get_rolling_bracket_stats(30),
        }

    def shutdown(self):
        self.db.shutdown()
