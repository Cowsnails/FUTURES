"""
Bracket Order Resolution Engine

Replaces passive time-snapshot resolution with active bracket-order simulation.
Resolves trades immediately when stop or target is hit, using ATR-based dynamic
levels and R-multiple tracking as first-class metrics.

Key features:
- ATR-based bracket parameters (stop/target computed at signal time)
- Bar-by-bar OHLC resolution with pessimistic same-bar handling
- MFE/MAE tracking (Maximum Favorable/Adverse Excursion)
- R-based trailing stop state machine (BE at 1R, trail at 2R, 3R)
- Quality scoring per signal (direction + timing + efficiency)
- SQN (System Quality Number) calculation
- Time snapshots retained as secondary data
"""

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, NamedTuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class ResolutionOutcome(Enum):
    PENDING = "PENDING"
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    TRAILING_STOP = "TRAILING_STOP"
    BREAKEVEN_STOP = "BREAKEVEN_STOP"
    TIMEOUT_PROFIT = "TIMEOUT_PROFIT"
    TIMEOUT_LOSS = "TIMEOUT_LOSS"
    TIMEOUT_SCRATCH = "TIMEOUT_SCRATCH"


class TrailingStopState(Enum):
    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    TRAIL_1R = "TRAIL_1R"
    TRAIL_2R = "TRAIL_2R"


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class BarData(NamedTuple):
    """Single bar OHLC data for resolution processing."""
    timestamp: int       # unix seconds
    bar_index: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class BracketParams:
    """Immutable bracket order parameters calculated at signal time."""
    entry_price: float
    stop_price: float
    target_price: float
    atr_at_entry: float
    stop_atr_mult: float    # e.g., 1.5
    target_atr_mult: float  # e.g., 2.0

    @property
    def initial_risk(self) -> float:
        """R value in price units (always positive)."""
        return abs(self.entry_price - self.stop_price)

    @property
    def reward_risk_ratio(self) -> float:
        if self.initial_risk == 0:
            return 0.0
        return abs(self.target_price - self.entry_price) / self.initial_risk

    @classmethod
    def from_atr(cls, entry: float, atr: float, direction: Direction,
                 stop_mult: float = 1.5, target_mult: float = 2.0) -> 'BracketParams':
        """Factory: create bracket using ATR multipliers."""
        if direction == Direction.LONG:
            stop = entry - (atr * stop_mult)
            target = entry + (atr * target_mult)
        else:
            stop = entry + (atr * stop_mult)
            target = entry - (atr * target_mult)
        return cls(entry, stop, target, atr, stop_mult, target_mult)


@dataclass
class MFEMAETracker:
    """Tracks Maximum Favorable/Adverse Excursion during trade."""
    mfe_price: Optional[float] = None
    mfe_r: float = 0.0
    mfe_bar_index: Optional[int] = None

    mae_price: Optional[float] = None
    mae_r: float = 0.0
    mae_bar_index: Optional[int] = None

    def update(self, high: float, low: float, bar_idx: int,
               entry: float, initial_risk: float, direction: Direction) -> None:
        if initial_risk == 0:
            return

        if direction == Direction.LONG:
            favorable_price, adverse_price = high, low
            favorable_pnl = high - entry
            adverse_pnl = entry - low
        else:
            favorable_price, adverse_price = low, high
            favorable_pnl = entry - low
            adverse_pnl = high - entry

        favorable_r = favorable_pnl / initial_risk
        adverse_r = adverse_pnl / initial_risk

        if favorable_r > self.mfe_r:
            self.mfe_price = favorable_price
            self.mfe_r = favorable_r
            self.mfe_bar_index = bar_idx

        if adverse_r > self.mae_r:
            self.mae_price = adverse_price
            self.mae_r = adverse_r
            self.mae_bar_index = bar_idx


@dataclass
class TrailingStopTracker:
    """R-based trailing stop state machine."""
    state: TrailingStopState = TrailingStopState.INITIAL
    current_stop: float = 0.0

    breakeven_trigger_r: float = 1.0
    trail_1r_trigger_r: float = 2.0
    trail_2r_trigger_r: float = 3.0

    def initialize(self, bracket: BracketParams) -> None:
        self.current_stop = bracket.stop_price
        self.state = TrailingStopState.INITIAL

    def update(self, current_mfe_r: float, entry: float,
               initial_risk: float, direction: Direction) -> float:
        """Update trailing stop based on MFE. Returns current stop price."""
        if self.state == TrailingStopState.INITIAL:
            if current_mfe_r >= self.breakeven_trigger_r:
                self.current_stop = entry
                self.state = TrailingStopState.BREAKEVEN

        elif self.state == TrailingStopState.BREAKEVEN:
            if current_mfe_r >= self.trail_1r_trigger_r:
                if direction == Direction.LONG:
                    self.current_stop = entry + initial_risk
                else:
                    self.current_stop = entry - initial_risk
                self.state = TrailingStopState.TRAIL_1R

        elif self.state == TrailingStopState.TRAIL_1R:
            if current_mfe_r >= self.trail_2r_trigger_r:
                if direction == Direction.LONG:
                    self.current_stop = entry + (2 * initial_risk)
                else:
                    self.current_stop = entry - (2 * initial_risk)
                self.state = TrailingStopState.TRAIL_2R

        return self.current_stop


@dataclass
class SignalResolution:
    """Complete signal with bracket resolution tracking."""
    # Identity
    signal_id: str
    symbol: str
    direction: Direction
    confluence_score: float

    # Bracket params
    bracket: BracketParams

    # Timing
    entry_time: int          # unix timestamp
    entry_bar_index: int
    max_bars: int = 15

    # Setup name (for setup detector signals)
    setup_name: str = ""

    # State tracking
    outcome: ResolutionOutcome = ResolutionOutcome.PENDING
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    exit_bar_index: Optional[int] = None

    # MFE/MAE tracking
    mfe_mae: MFEMAETracker = field(default_factory=MFEMAETracker)

    # Trailing stop
    trailing: TrailingStopTracker = field(default_factory=TrailingStopTracker)

    # Time snapshots (secondary data)
    snapshot_1m: Optional[float] = None
    snapshot_5m: Optional[float] = None
    snapshot_15m: Optional[float] = None

    # Indicator data at entry (for analysis)
    indicator_data: dict = field(default_factory=dict)

    @property
    def r_multiple(self) -> Optional[float]:
        if self.exit_price is None or self.bracket.initial_risk == 0:
            return None
        if self.direction == Direction.LONG:
            pnl = self.exit_price - self.bracket.entry_price
        else:
            pnl = self.bracket.entry_price - self.exit_price
        raw_r = pnl / self.bracket.initial_risk
        # Cap R-multiples at reasonable bounds to prevent extreme values
        # from near-zero initial risk calculations
        MAX_R = 20.0
        return max(-MAX_R, min(MAX_R, raw_r))

    @property
    def is_resolved(self) -> bool:
        return self.outcome != ResolutionOutcome.PENDING

    @property
    def bars_held(self) -> Optional[int]:
        if self.exit_bar_index is None:
            return None
        return self.exit_bar_index - self.entry_bar_index


# ═══════════════════════════════════════════════════════════════════════════
# BAR-BY-BAR RESOLUTION ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════

def resolve_signal_on_bar(signal: SignalResolution, bar: BarData) -> bool:
    """
    Process single bar for bracket resolution.
    Returns True if signal was resolved on this bar.

    Resolution priority (conservative/pessimistic):
    1. Check stop hit first (assume worst case on same-bar)
    2. Then check target hit
    3. Update MFE/MAE regardless
    4. Update trailing stop if still active
    5. Check timeout
    """
    if signal.is_resolved:
        return False

    entry = signal.bracket.entry_price
    initial_risk = signal.bracket.initial_risk

    # Update MFE/MAE for this bar
    signal.mfe_mae.update(
        bar.high, bar.low, bar.bar_index,
        entry, initial_risk, signal.direction
    )

    # Update trailing stop based on new MFE
    current_stop = signal.trailing.update(
        signal.mfe_mae.mfe_r, entry, initial_risk, signal.direction
    )

    # Capture time snapshots (secondary data)
    bars_since_entry = bar.bar_index - signal.entry_bar_index
    _capture_time_snapshot(signal, bars_since_entry, bar.close)

    # === BRACKET RESOLUTION (PRIMARY) ===
    target = signal.bracket.target_price

    if signal.direction == Direction.LONG:
        stop_hit = bar.low <= current_stop
        target_hit = bar.high >= target
    else:
        stop_hit = bar.high >= current_stop
        target_hit = bar.low <= target

    # Same-bar resolution: PESSIMISTIC - assume stop hit first
    if stop_hit and target_hit:
        _resolve_stop_hit(signal, bar, current_stop)
        return True

    if stop_hit:
        _resolve_stop_hit(signal, bar, current_stop)
        return True

    if target_hit:
        signal.outcome = ResolutionOutcome.TARGET_HIT
        signal.exit_price = target
        signal.exit_time = bar.timestamp
        signal.exit_bar_index = bar.bar_index
        return True

    # Check timeout
    if bars_since_entry >= signal.max_bars:
        _resolve_timeout(signal, bar)
        return True

    return False


def _resolve_stop_hit(signal: SignalResolution, bar: BarData,
                      stop_price: float) -> None:
    state = signal.trailing.state

    if state == TrailingStopState.INITIAL:
        signal.outcome = ResolutionOutcome.STOP_HIT
    elif state == TrailingStopState.BREAKEVEN:
        signal.outcome = ResolutionOutcome.BREAKEVEN_STOP
    else:
        signal.outcome = ResolutionOutcome.TRAILING_STOP

    signal.exit_price = stop_price
    signal.exit_time = bar.timestamp
    signal.exit_bar_index = bar.bar_index


def _resolve_timeout(signal: SignalResolution, bar: BarData) -> None:
    entry = signal.bracket.entry_price
    initial_risk = signal.bracket.initial_risk

    signal.exit_price = bar.close
    signal.exit_time = bar.timestamp
    signal.exit_bar_index = bar.bar_index

    if signal.direction == Direction.LONG:
        pnl = bar.close - entry
    else:
        pnl = entry - bar.close

    r_mult = pnl / initial_risk if initial_risk > 0 else 0
    SCRATCH_THRESHOLD = 0.25

    if abs(r_mult) < SCRATCH_THRESHOLD:
        signal.outcome = ResolutionOutcome.TIMEOUT_SCRATCH
    elif r_mult > 0:
        signal.outcome = ResolutionOutcome.TIMEOUT_PROFIT
    else:
        signal.outcome = ResolutionOutcome.TIMEOUT_LOSS


def _capture_time_snapshot(signal: SignalResolution, bars_since: int,
                           close: float) -> None:
    if bars_since == 1 and signal.snapshot_1m is None:
        signal.snapshot_1m = _calc_pnl(signal, close)
    elif bars_since == 5 and signal.snapshot_5m is None:
        signal.snapshot_5m = _calc_pnl(signal, close)
    elif bars_since == 15 and signal.snapshot_15m is None:
        signal.snapshot_15m = _calc_pnl(signal, close)


def _calc_pnl(signal: SignalResolution, price: float) -> float:
    if signal.direction == Direction.LONG:
        return price - signal.bracket.entry_price
    return signal.bracket.entry_price - price


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY SCORING
# ═══════════════════════════════════════════════════════════════════════════

def score_signal_quality(sig: SignalResolution) -> float:
    """
    Score signal quality 0-100 based on:
    - Direction: 40 pts max (R-multiple based)
    - Timing: 30 pts max (faster resolution = better)
    - Efficiency: 30 pts max (how much MFE was captured)
    """
    score = 0.0

    # Direction component (40 pts)
    if sig.r_multiple is not None and sig.r_multiple > 0:
        score += min(40, sig.r_multiple * 20)

    # Timing component (30 pts)
    if sig.bars_held is not None and sig.max_bars > 0:
        time_efficiency = 1 - (sig.bars_held / sig.max_bars)
        if sig.outcome == ResolutionOutcome.TARGET_HIT:
            score += time_efficiency * 30

    # Efficiency component (30 pts)
    if sig.mfe_mae.mfe_r > 0 and sig.r_multiple is not None:
        if sig.r_multiple > 0:
            capture_ratio = sig.r_multiple / sig.mfe_mae.mfe_r
            score += capture_ratio * 30

    return min(100, score)


def calculate_sqn(r_multiples: List[float]) -> float:
    """System Quality Number (Van Tharp formula)."""
    if len(r_multiples) < 2:
        return 0.0
    mean_r = statistics.mean(r_multiples)
    std_r = statistics.stdev(r_multiples)
    if std_r == 0:
        return 0.0
    n = min(len(r_multiples), 100)
    return (mean_r / std_r) * (n ** 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# WIN RATE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WinRateStats:
    """Comprehensive win rate statistics."""
    total_trades: int = 0
    target_hits: int = 0
    stop_hits: int = 0
    trailing_stops: int = 0
    breakeven_stops: int = 0
    timeout_profit: int = 0
    timeout_loss: int = 0
    scratches: int = 0

    @property
    def bracket_resolved(self) -> int:
        return self.target_hits + self.stop_hits

    @property
    def primary_win_rate(self) -> float:
        """Target hits / Bracket-resolved trades (core signal quality metric)."""
        if self.bracket_resolved == 0:
            return 0.0
        return self.target_hits / self.bracket_resolved

    @property
    def overall_profitable_rate(self) -> float:
        """All profitable outcomes / Total trades."""
        if self.total_trades == 0:
            return 0.0
        profitable = self.target_hits + self.trailing_stops + self.timeout_profit
        return profitable / self.total_trades

    @property
    def full_loss_rate(self) -> float:
        """Full stop hits / Non-scratch trades."""
        non_scratch = self.total_trades - self.scratches
        if non_scratch == 0:
            return 0.0
        return self.stop_hits / non_scratch

    @property
    def timeout_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.timeout_profit + self.timeout_loss + self.scratches) / self.total_trades
