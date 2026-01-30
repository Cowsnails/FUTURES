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
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

STATS_DIR = Path(__file__).parent.parent / "trading_stats"
DB_PATH = STATS_DIR / "live" / "signals.db"

# Signal state machine
CONFIRMATION_BARS = 1          # Instant confirmation — signals are fleeting (seconds)
ENTRY_THRESHOLD = 0.30         # |confluence| to trigger detection
EXIT_THRESHOLD = 0.15          # |confluence| to cancel (hysteresis)
DETECTION_TIMEOUT_BARS = 3     # Max updates in DETECTED before expiring
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
    Instant confirmation — signals are fleeting (last seconds to 1 min).
    Cooldown prevents duplicate fires within MIN_SIGNAL_GAP_SECONDS.
    """

    def __init__(self):
        self._signal_counter = 0
        self.current_signal: Optional[ActiveSignal] = None
        self.pending_outcomes: List[ActiveSignal] = []  # Confirmed signals awaiting outcomes
        self._last_confirmed_time: int = 0  # Cooldown tracking

    def _next_id(self) -> str:
        self._signal_counter += 1
        return f"TS-{int(time.time())}-{self._signal_counter}"

    def process_bar(self, bar_time: int, price: float, action: str,
                    confluence: float, atr: float,
                    indicator_data: dict) -> Optional[dict]:
        """
        Process a new bar from the scalping engine.
        Returns a dict event if state changed, None otherwise.
        """
        direction = 'long' if action == 'BUY' else ('short' if action == 'SELL' else '')
        has_signal = action in ('BUY', 'SELL')

        # ── Update pending outcome signals ──
        events = []
        for sig in self.pending_outcomes:
            sig.bars_since_entry += 1
            self._update_outcomes(sig, bar_time, price, atr)

        # Remove fully resolved
        resolved = [s for s in self.pending_outcomes if s.resolved]
        for s in resolved:
            events.append(self._make_event(s, 'TRADE_RESOLVED'))
        self.pending_outcomes = [s for s in self.pending_outcomes if not s.resolved]

        # ── State machine (instant confirmation for fleeting signals) ──
        # Signals last seconds to 1 min max, so we confirm immediately
        # on first BUY/SELL with sufficient confluence. Cooldown prevents
        # duplicate fires within MIN_SIGNAL_GAP_SECONDS.
        if has_signal and abs(confluence) >= ENTRY_THRESHOLD:
            # Check cooldown
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

                # Move to active and start outcome tracking
                sig.state = SignalState.ACTIVE
                self.pending_outcomes.append(sig)
                self._last_confirmed_time = bar_time

                logger.info(f"Signal confirmed: {direction} @ {price:.2f} "
                           f"(conf={confluence:.3f}, cooldown until +{MIN_SIGNAL_GAP_SECONDS}s)")

        return events if events else None

    def _update_outcomes(self, sig: ActiveSignal, bar_time: int, price: float, atr: float):
        """Track outcomes at each horizon and MAE/MFE."""
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

    def process_update(self, pattern_result: dict, bar_time: int) -> Optional[dict]:
        """
        Process a pattern update. Returns a snapshot dict if one should be saved.
        """
        if not pattern_result or not pattern_result.get('forecast'):
            return None

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

        should_snap, trigger = self._should_snapshot(current)
        if not should_snap:
            return None

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
        return snapshot

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

            self.conn.commit()
            logger.debug(f"Stats flush: {len(signal_rows)} signals, {len(pattern_rows)} patterns")
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
                             indicators: dict) -> Optional[List[dict]]:
        """Process a trade signal bar from the scalping engine."""
        tracker = self.get_tracker(symbol)
        events = tracker.process_bar(bar_time, price, action, confluence, atr, indicators)

        if events:
            for event in events:
                event['symbol'] = symbol
                self.db.log_signal_event(event)
                logger.info(f"[{symbol}] Trade signal event: {event['event_type']} "
                           f"({event['direction']}, conf={event['confluence_score']:.2f})")

        return events

    # ── Called from backend pattern loop ──

    def process_pattern_update(self, symbol: str, pattern_result: dict,
                               session_type: str = 'rth') -> Optional[dict]:
        """Process a pattern matcher update. Snapshots on meaningful changes."""
        if session_type == 'rth':
            mgr = self.get_rth_snapshot_mgr(symbol)
        else:
            mgr = self.get_overnight_snapshot_mgr(symbol)

        bar_time = int(time.time())
        snapshot = mgr.process_update(pattern_result, bar_time)

        if snapshot:
            snapshot['symbol'] = symbol
            self.db.log_pattern_snapshot(snapshot)
            logger.info(f"[{symbol}] Pattern snapshot ({session_type}): "
                       f"{snapshot['trigger']} - {snapshot['direction']} "
                       f"({snapshot['consensus']})")

        return snapshot

    # ── Dashboard API ──

    def get_dashboard_stats(self) -> dict:
        """Get all stats for dashboard display."""
        return {
            'today': self.db.get_today_stats(),
            'rolling_7d': self.db.get_rolling_stats(7),
            'rolling_30d': self.db.get_rolling_stats(30),
        }

    def shutdown(self):
        self.db.shutdown()
