"""
Setup Detectors Framework

Base classes and manager for running multiple intraday setup detectors
simultaneously. Each detector analyzes incoming bars and emits signals
that feed into the existing bracket resolution system for tracking.

All detection runs server-side. Results appear on the stats page only.

Includes self-contained indicator computation so detectors don't depend
on external indicator sources — everything is computed from raw bar history.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Eastern timezone offset helpers
ET_OFFSET = timedelta(hours=-5)  # EST (no DST handling — good enough for RTH)
RTH_OPEN_MINUTES = 9 * 60 + 30   # 9:30 ET in minutes-from-midnight
RTH_CLOSE_MINUTES = 16 * 60      # 16:00 ET


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SetupSignal:
    """A signal emitted by a setup detector."""
    setup_name: str         # e.g. "orb_breakout"
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    stop_atr_mult: float
    target_atr_mult: float
    max_bars: int           # timeout in bars
    confidence: float       # 0-1
    reason: str             # human-readable reason
    bar_time: int           # entry bar timestamp
    indicator_snapshot: dict = field(default_factory=dict)


@dataclass
class BarInput:
    """Standardized bar data passed to all detectors."""
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IndicatorState:
    """Pre-computed indicators shared across all detectors (compute once)."""
    # Core
    ema9: Optional[float] = None
    ema20: Optional[float] = None
    rsi14: Optional[float] = None
    rsi14_prev: Optional[float] = None
    adx14: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    atr14: Optional[float] = None
    volume_sma20: Optional[float] = None

    # Bollinger Bands (20, 2)
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None

    # Keltner Channels (20, 1.5)
    kc_upper: Optional[float] = None
    kc_middle: Optional[float] = None
    kc_lower: Optional[float] = None

    # TTM Squeeze
    squeeze_on: bool = False  # True when BB inside KC
    squeeze_momentum: Optional[float] = None

    # VWAP (session-anchored)
    vwap: Optional[float] = None
    vwap_std: Optional[float] = None
    vwap_upper2: Optional[float] = None
    vwap_lower2: Optional[float] = None

    # Levels
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    pdc: Optional[float] = None
    onh: Optional[float] = None
    onl: Optional[float] = None
    orh: Optional[float] = None  # opening range high (first 15 min)
    orl: Optional[float] = None  # opening range low (first 15 min)
    or_complete: bool = False

    # Session info
    session: str = ""
    session_minutes: int = 0  # minutes since RTH open
    is_rth: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR MATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ema(values: List[float], period: int) -> Optional[float]:
    """Compute EMA of last N values. Returns None if not enough data."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema_val = values[0]
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


def _sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _stdev(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    subset = values[-period:]
    mean = sum(subset) / period
    variance = sum((v - mean) ** 2 for v in subset) / period
    return math.sqrt(variance) if variance > 0 else 0.0


def _true_range(bars: List[BarInput], i: int) -> float:
    """True range for bar at index i."""
    b = bars[i]
    if i == 0:
        return b.high - b.low
    prev_c = bars[i - 1].close
    return max(b.high - b.low, abs(b.high - prev_c), abs(b.low - prev_c))


def _compute_atr(bars: List[BarInput], period: int) -> Optional[float]:
    """ATR using Wilder smoothing over last bars."""
    n = len(bars)
    if n < period + 1:
        return None
    # Initial ATR = simple average of first `period` TRs
    tr_sum = sum(_true_range(bars, i) for i in range(n - period, n))
    atr = tr_sum / period
    return atr


def _compute_rsi(closes: List[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(len(closes) - period, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_adx(bars: List[BarInput], period: int = 14):
    """Returns (adx, plus_di, minus_di) or (None, None, None)."""
    n = len(bars)
    if n < period * 2:
        return None, None, None

    plus_dms = []
    minus_dms = []
    trs = []

    for i in range(1, n):
        h = bars[i].high
        l = bars[i].low
        ph = bars[i - 1].high
        pl = bars[i - 1].low
        pc = bars[i - 1].close

        plus_dm = max(h - ph, 0) if (h - ph) > (pl - l) else 0
        minus_dm = max(pl - l, 0) if (pl - l) > (h - ph) else 0
        tr = max(h - l, abs(h - pc), abs(l - pc))

        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)
        trs.append(tr)

    if len(trs) < period:
        return None, None, None

    # Wilder smoothing
    smoothed_plus = sum(plus_dms[:period])
    smoothed_minus = sum(minus_dms[:period])
    smoothed_tr = sum(trs[:period])

    dx_values = []

    for i in range(period, len(trs)):
        smoothed_plus = smoothed_plus - (smoothed_plus / period) + plus_dms[i]
        smoothed_minus = smoothed_minus - (smoothed_minus / period) + minus_dms[i]
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + trs[i]

        if smoothed_tr == 0:
            continue
        plus_di = 100 * smoothed_plus / smoothed_tr
        minus_di = 100 * smoothed_minus / smoothed_tr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0)
        else:
            dx_values.append(100 * abs(plus_di - minus_di) / di_sum)

    if len(dx_values) < period:
        return None, None, None

    adx = sum(dx_values[-period:]) / period

    # Return last plus_di and minus_di
    if smoothed_tr == 0:
        return adx, 0, 0
    final_plus_di = 100 * smoothed_plus / smoothed_tr
    final_minus_di = 100 * smoothed_minus / smoothed_tr
    return adx, final_plus_di, final_minus_di


def _bar_to_et_minutes(bar_time: int) -> int:
    """Convert unix timestamp to minutes-from-midnight in Eastern time."""
    dt = datetime.fromtimestamp(bar_time, tz=timezone.utc) + ET_OFFSET
    return dt.hour * 60 + dt.minute


def _is_rth(bar_time: int) -> bool:
    """Check if bar is during Regular Trading Hours (9:30-16:00 ET)."""
    m = _bar_to_et_minutes(bar_time)
    return RTH_OPEN_MINUTES <= m < RTH_CLOSE_MINUTES


def _session_label(minutes_since_open: int) -> str:
    if minutes_since_open < 0:
        return "pre_market"
    if minutes_since_open < 15:
        return "rth_open"
    if minutes_since_open < 60:
        return "am_drive"
    if minutes_since_open < 150:
        return "am_session"
    if minutes_since_open < 240:
        return "lunch"
    if minutes_since_open < 360:
        return "pm_session"
    if minutes_since_open < 390:
        return "moc"
    return "post_market"


# ═══════════════════════════════════════════════════════════════════════════
# BASE DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SetupDetector:
    """
    Base class for all setup detectors.

    Subclasses must implement:
      - update(bar, indicators, bars_history) -> Optional[SetupSignal]
    """

    name: str = "base"
    display_name: str = "Base Setup"
    category: str = "unknown"
    hold_time: str = "unknown"
    min_cooldown_seconds: int = 60

    def __init__(self):
        self._last_signal_time: int = 0
        self._signal_counter: int = 0
        self._enabled: bool = True

    def update(self, bar: BarInput, indicators: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        raise NotImplementedError

    def can_signal(self, bar_time: int) -> bool:
        if not self._enabled:
            return False
        return (bar_time - self._last_signal_time) >= self.min_cooldown_seconds

    def record_signal(self, bar_time: int):
        self._last_signal_time = bar_time
        self._signal_counter += 1

    def make_signal(self, direction: str, bar: BarInput,
                    indicators: IndicatorState,
                    stop_price: float, target_price: float,
                    stop_atr_mult: float, target_atr_mult: float,
                    max_bars: int, confidence: float,
                    reason: str) -> SetupSignal:
        atr = indicators.atr14 or 1.0
        return SetupSignal(
            setup_name=self.name,
            direction=direction,
            entry_price=bar.close,
            stop_price=stop_price,
            target_price=target_price,
            atr=atr,
            stop_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            max_bars=max_bars,
            confidence=confidence,
            reason=reason,
            bar_time=bar.time,
            indicator_snapshot={
                'ema9': indicators.ema9,
                'rsi14': indicators.rsi14,
                'adx14': indicators.adx14,
                'atr14': indicators.atr14,
                'vwap': indicators.vwap,
                'session': indicators.session,
            },
        )

    def reset(self):
        self._last_signal_time = 0
        self._signal_counter = 0

    def get_info(self) -> dict:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'category': self.category,
            'hold_time': self.hold_time,
            'enabled': self._enabled,
            'signal_count': self._signal_counter,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 DETECTORS
# ═══════════════════════════════════════════════════════════════════════════

# ── 1.1 Opening Range Breakout (ORB) ──────────────────────────────────────

class ORBBreakoutDetector(SetupDetector):
    """
    Opening Range Breakout — first 15 minutes define the range.
    Entry on close above OR high / below OR low after the OR period.
    Filters: ADX > 20, volume above average.
    """
    name = "orb_breakout"
    display_name = "Opening Range Breakout"
    category = "session"
    hold_time = "15-60min"
    min_cooldown_seconds = 300  # 5 min between ORB signals

    def __init__(self):
        super().__init__()
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_complete: bool = False
        self._or_date: str = ""
        self._breakout_fired_long: bool = False
        self._breakout_fired_short: bool = False

    def reset(self):
        super().reset()
        self._or_high = None
        self._or_low = None
        self._or_complete = False
        self._or_date = ""
        self._breakout_fired_long = False
        self._breakout_fired_short = False

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None

        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        # Reset on new day
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._or_date:
            self._or_date = today
            self._or_high = None
            self._or_low = None
            self._or_complete = False
            self._breakout_fired_long = False
            self._breakout_fired_short = False

        # Build opening range during first 15 minutes
        if ind.session_minutes < 15:
            if self._or_high is None:
                self._or_high = bar.high
                self._or_low = bar.low
            else:
                self._or_high = max(self._or_high, bar.high)
                self._or_low = min(self._or_low, bar.low)
            return None

        # Mark OR complete
        if not self._or_complete:
            self._or_complete = True

        if self._or_high is None or self._or_low is None:
            return None

        or_range = self._or_high - self._or_low
        if or_range < 0.5 * atr or or_range > 3.0 * atr:
            return None  # Range too narrow or too wide

        # Only signal in first 2 hours after open
        if ind.session_minutes > 135:
            return None

        # Long breakout
        if not self._breakout_fired_long and bar.close > self._or_high:
            adx_ok = ind.adx14 is not None and ind.adx14 > 20
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.2
            if adx_ok or vol_ok:
                self._breakout_fired_long = True
                stop = self._or_low - 0.25 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 2.0
                conf = 0.6
                if adx_ok and vol_ok:
                    conf = 0.75
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2.0) / atr,
                    max_bars=30, confidence=conf,
                    reason=f"ORB long: close {bar.close:.1f} > OR high {self._or_high:.1f}"
                )

        # Short breakout
        if not self._breakout_fired_short and bar.close < self._or_low:
            adx_ok = ind.adx14 is not None and ind.adx14 > 20
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.2
            if adx_ok or vol_ok:
                self._breakout_fired_short = True
                stop = self._or_high + 0.25 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 2.0
                conf = 0.6
                if adx_ok and vol_ok:
                    conf = 0.75
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2.0) / atr,
                    max_bars=30, confidence=conf,
                    reason=f"ORB short: close {bar.close:.1f} < OR low {self._or_low:.1f}"
                )

        return None


# ── 1.2 VWAP Mean Reversion from ±2σ ─────────────────────────────────────

class VWAPMeanReversionDetector(SetupDetector):
    """
    VWAP Mean Reversion — enter when price touches ±2σ VWAP band
    and shows rejection (close back inside band).
    Filters: RSI oversold/overbought confirmation, not during strong trend.
    """
    name = "vwap_mr_2sigma"
    display_name = "VWAP Mean Reversion ±2σ"
    category = "vwap"
    hold_time = "5-30min"
    min_cooldown_seconds = 300

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth or ind.session_minutes < 30:
            return None  # Need VWAP to stabilize

        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if ind.vwap is None or ind.vwap_std is None or ind.vwap_std <= 0:
            return None

        upper2 = ind.vwap + 2 * ind.vwap_std
        lower2 = ind.vwap - 2 * ind.vwap_std
        rsi = ind.rsi14

        # Strong trend filter — skip if ADX very high
        if ind.adx14 is not None and ind.adx14 > 40:
            return None

        # Long: bar low touched lower 2σ, close pulled back above it
        if bar.low <= lower2 and bar.close > lower2:
            rsi_ok = rsi is not None and rsi < 35
            wick_rejection = (bar.close - bar.low) > 0.5 * (bar.high - bar.low)
            if rsi_ok or wick_rejection:
                stop = bar.low - 0.5 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = ind.vwap  # revert to VWAP
                conf = 0.55
                if rsi_ok and wick_rejection:
                    conf = 0.70
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr,
                    target_atr_mult=abs(target - bar.close) / atr,
                    max_bars=20, confidence=conf,
                    reason=f"VWAP MR long: low {bar.low:.1f} hit -2σ {lower2:.1f}, RSI={rsi:.0f}" if rsi else f"VWAP MR long: wick rejection at -2σ"
                )

        # Short: bar high touched upper 2σ, close pulled back below it
        if bar.high >= upper2 and bar.close < upper2:
            rsi_ok = rsi is not None and rsi > 65
            wick_rejection = (bar.high - bar.close) > 0.5 * (bar.high - bar.low)
            if rsi_ok or wick_rejection:
                stop = bar.high + 0.5 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = ind.vwap
                conf = 0.55
                if rsi_ok and wick_rejection:
                    conf = 0.70
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr,
                    target_atr_mult=abs(bar.close - target) / atr,
                    max_bars=20, confidence=conf,
                    reason=f"VWAP MR short: high {bar.high:.1f} hit +2σ {upper2:.1f}, RSI={rsi:.0f}" if rsi else f"VWAP MR short: wick rejection at +2σ"
                )

        return None


# ── 1.3 PDH/PDL Breakout Continuation ────────────────────────────────────

class PDHPDLBreakoutDetector(SetupDetector):
    """
    Previous Day High / Low breakout continuation.
    Enter on close above PDH or below PDL with volume confirmation.
    Stop behind the level, target 2R.
    """
    name = "pdh_pdl_breakout"
    display_name = "PDH/PDL Breakout"
    category = "level"
    hold_time = "15-60min"
    min_cooldown_seconds = 600  # 10 min

    def __init__(self):
        super().__init__()
        self._pdh_fired: bool = False
        self._pdl_fired: bool = False
        self._last_date: str = ""

    def reset(self):
        super().reset()
        self._pdh_fired = False
        self._pdl_fired = False
        self._last_date = ""

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None

        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        # Reset flags on new day
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._last_date:
            self._last_date = today
            self._pdh_fired = False
            self._pdl_fired = False

        pdh = ind.pdh
        pdl = ind.pdl
        if pdh is None or pdl is None:
            return None

        # Long: close above PDH
        if not self._pdh_fired and bar.close > pdh and bar.low <= pdh + 0.5 * atr:
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20
            if vol_ok or (ind.adx14 is not None and ind.adx14 > 25):
                self._pdh_fired = True
                stop = pdh - 0.75 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 2.0
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                    max_bars=30, confidence=0.65,
                    reason=f"PDH breakout: close {bar.close:.1f} > PDH {pdh:.1f}"
                )

        # Short: close below PDL
        if not self._pdl_fired and bar.close < pdl and bar.high >= pdl - 0.5 * atr:
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20
            if vol_ok or (ind.adx14 is not None and ind.adx14 > 25):
                self._pdl_fired = True
                stop = pdl + 0.75 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 2.0
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                    max_bars=30, confidence=0.65,
                    reason=f"PDL breakout: close {bar.close:.1f} < PDL {pdl:.1f}"
                )

        return None


# ── 1.4 EMA-9 Pullback in Trend ──────────────────────────────────────────

class EMA9PullbackDetector(SetupDetector):
    """
    Linda Raschke's "Holy Grail" — EMA-9 pullback in a confirmed trend.
    Requires ADX > 25 with directional bias. Entry on bar that touches
    EMA-9 and closes back in trend direction.
    """
    name = "ema9_pullback"
    display_name = "EMA-9 Pullback"
    category = "momentum"
    hold_time = "5-30min"
    min_cooldown_seconds = 300

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None

        atr = ind.atr14
        ema = ind.ema9
        adx = ind.adx14
        plus_di = ind.plus_di
        minus_di = ind.minus_di

        if not all([atr, ema, adx, plus_di is not None, minus_di is not None]):
            return None
        if atr <= 0:
            return None

        # Need strong trend
        if adx < 25:
            return None

        # Uptrend pullback: +DI > -DI, bar low touches EMA-9, close above EMA-9
        if plus_di > minus_di:
            if bar.low <= ema + 0.1 * atr and bar.close > ema:
                # Confirm pullback: previous bar(s) were above EMA
                if len(bars_history) >= 3:
                    prev = bars_history[-2]
                    if prev.low > ema - 0.3 * atr:
                        stop = bar.low - 0.75 * atr
                        risk = bar.close - stop
                        if risk <= 0:
                            return None
                        target = bar.close + risk * 2.0
                        conf = 0.60 + min(0.15, (adx - 25) / 100)
                        return self.make_signal(
                            "LONG", bar, ind,
                            stop_price=stop, target_price=target,
                            stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                            max_bars=20, confidence=conf,
                            reason=f"EMA9 pullback long: ADX={adx:.0f}, low {bar.low:.1f} touched EMA {ema:.1f}"
                        )

        # Downtrend pullback: -DI > +DI, bar high touches EMA-9, close below EMA-9
        if minus_di > plus_di:
            if bar.high >= ema - 0.1 * atr and bar.close < ema:
                if len(bars_history) >= 3:
                    prev = bars_history[-2]
                    if prev.high < ema + 0.3 * atr:
                        stop = bar.high + 0.75 * atr
                        risk = stop - bar.close
                        if risk <= 0:
                            return None
                        target = bar.close - risk * 2.0
                        conf = 0.60 + min(0.15, (adx - 25) / 100)
                        return self.make_signal(
                            "SHORT", bar, ind,
                            stop_price=stop, target_price=target,
                            stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                            max_bars=20, confidence=conf,
                            reason=f"EMA9 pullback short: ADX={adx:.0f}, high {bar.high:.1f} touched EMA {ema:.1f}"
                        )

        return None


# ── 1.5 TTM Squeeze ──────────────────────────────────────────────────────

class TTMSqueezeDetector(SetupDetector):
    """
    TTM Squeeze — Bollinger Bands contract inside Keltner Channels
    (squeeze ON), then fire when squeeze releases with momentum.
    """
    name = "ttm_squeeze"
    display_name = "TTM Squeeze Fire"
    category = "volatility"
    hold_time = "15-60min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._was_squeezing: bool = False
        self._squeeze_bars: int = 0

    def reset(self):
        super().reset()
        self._was_squeezing = False
        self._squeeze_bars = 0

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth or ind.session_minutes < 30:
            return None

        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        squeeze_on = ind.squeeze_on
        momentum = ind.squeeze_momentum

        if squeeze_on:
            self._squeeze_bars += 1
            self._was_squeezing = True
            return None

        # Squeeze just released
        if self._was_squeezing and not squeeze_on and momentum is not None:
            self._was_squeezing = False
            bars_in_squeeze = self._squeeze_bars
            self._squeeze_bars = 0

            # Need at least 6 bars of squeeze for reliability
            if bars_in_squeeze < 6:
                return None

            if momentum > 0:
                # Bullish release
                stop = bar.close - 1.5 * atr
                risk = bar.close - stop
                target = bar.close + risk * 2.0
                conf = 0.60 + min(0.15, bars_in_squeeze / 50)
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.5, target_atr_mult=3.0,
                    max_bars=30, confidence=conf,
                    reason=f"TTM squeeze fire LONG: {bars_in_squeeze} bars squeezed, momentum={momentum:.2f}"
                )
            elif momentum < 0:
                # Bearish release
                stop = bar.close + 1.5 * atr
                risk = stop - bar.close
                target = bar.close - risk * 2.0
                conf = 0.60 + min(0.15, bars_in_squeeze / 50)
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.5, target_atr_mult=3.0,
                    max_bars=30, confidence=conf,
                    reason=f"TTM squeeze fire SHORT: {bars_in_squeeze} bars squeezed, momentum={momentum:.2f}"
                )

        if not squeeze_on:
            self._squeeze_bars = 0
            self._was_squeezing = False

        return None


# ── 1.6 Overnight High/Low Sweep Reversal ────────────────────────────────

class ONHLSweepDetector(SetupDetector):
    """
    Overnight High/Low Sweep Reversal — price sweeps ON level then
    reverses back inside. Classic liquidity grab setup.
    """
    name = "on_hl_sweep"
    display_name = "ON H/L Sweep Reversal"
    category = "level"
    hold_time = "10-45min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._onh_swept: bool = False
        self._onl_swept: bool = False
        self._last_date: str = ""

    def reset(self):
        super().reset()
        self._onh_swept = False
        self._onl_swept = False
        self._last_date = ""

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None

        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        # Reset on new day
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._last_date:
            self._last_date = today
            self._onh_swept = False
            self._onl_swept = False

        onh = ind.onh
        onl = ind.onl
        if onh is None or onl is None:
            return None

        # Only during first 2 hours
        if ind.session_minutes > 120:
            return None

        # Short: sweep above ONH then close back below it
        if not self._onh_swept and bar.high > onh and bar.close < onh:
            wick_above = bar.high - onh
            if 0 < wick_above < 1.5 * atr:
                self._onh_swept = True
                stop = bar.high + 0.5 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 1.5
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=25, confidence=0.65,
                    reason=f"ON high sweep: high {bar.high:.1f} > ONH {onh:.1f}, close back {bar.close:.1f}"
                )

        # Long: sweep below ONL then close back above it
        if not self._onl_swept and bar.low < onl and bar.close > onl:
            wick_below = onl - bar.low
            if 0 < wick_below < 1.5 * atr:
                self._onl_swept = True
                stop = bar.low - 0.5 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 1.5
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=25, confidence=0.65,
                    reason=f"ON low sweep: low {bar.low:.1f} < ONL {onl:.1f}, close back {bar.close:.1f}"
                )

        return None


# ── 1.7 Volume Spike Breakout ────────────────────────────────────────────

class VolumeSpikeBreakoutDetector(SetupDetector):
    """
    Volume Spike Breakout — volume surges > 2x 20-SMA while price
    breaks out of recent 5-bar range. Indicates institutional activity.
    """
    name = "volume_spike_breakout"
    display_name = "Volume Spike Breakout"
    category = "momentum"
    hold_time = "5-20min"
    min_cooldown_seconds = 300

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None

        atr = ind.atr14
        vol_sma = ind.volume_sma20
        if not atr or atr <= 0:
            return None
        if vol_sma is None or vol_sma <= 0:
            return None
        if len(bars_history) < 7:
            return None

        # Volume must be > 2x average
        if bar.volume < vol_sma * 2.0:
            return None

        # 5-bar range (excluding current bar)
        lookback = bars_history[-6:-1]
        range_high = max(b.high for b in lookback)
        range_low = min(b.low for b in lookback)
        range_size = range_high - range_low

        # Range shouldn't be too wide (consolidation, not trending)
        if range_size > 2.5 * atr:
            return None
        if range_size < 0.3 * atr:
            return None

        # Long breakout
        if bar.close > range_high and bar.close > bar.open:
            stop = range_low - 0.25 * atr
            risk = bar.close - stop
            if risk <= 0:
                return None
            target = bar.close + risk * 1.5
            vol_ratio = bar.volume / vol_sma
            conf = min(0.80, 0.55 + (vol_ratio - 2.0) * 0.05)
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=15, confidence=conf,
                reason=f"Vol spike long: vol={bar.volume:.0f} ({vol_ratio:.1f}x avg), broke {range_high:.1f}"
            )

        # Short breakout
        if bar.close < range_low and bar.close < bar.open:
            stop = range_high + 0.25 * atr
            risk = stop - bar.close
            if risk <= 0:
                return None
            target = bar.close - risk * 1.5
            vol_ratio = bar.volume / vol_sma
            conf = min(0.80, 0.55 + (vol_ratio - 2.0) * 0.05)
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=15, confidence=conf,
                reason=f"Vol spike short: vol={bar.volume:.0f} ({vol_ratio:.1f}x avg), broke {range_low:.1f}"
            )

        return None


# ── 2.1 VWAP Breakout and Retest ─────────────────────────────────────────

class VWAPBreakoutRetestDetector(SetupDetector):
    """
    VWAP Breakout & Retest — price crosses VWAP with volume, then
    pulls back to VWAP and gets rejected. Multi-step state machine.
    """
    name = "vwap_breakout_retest"
    display_name = "VWAP Breakout & Retest"
    category = "vwap"
    hold_time = "5-30min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._breakout_dir: Optional[str] = None  # "LONG" or "SHORT"
        self._breakout_price: Optional[float] = None
        self._breakout_time: int = 0
        self._awaiting_retest: bool = False
        self._bars_since_breakout: int = 0

    def reset(self):
        super().reset()
        self._breakout_dir = None
        self._breakout_price = None
        self._awaiting_retest = False
        self._bars_since_breakout = 0

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth or ind.session_minutes < 15:
            return None
        atr = ind.atr14
        if not atr or atr <= 0 or ind.vwap is None:
            return None

        vwap = ind.vwap

        # Timeout stale breakout
        if self._awaiting_retest:
            self._bars_since_breakout += 1
            if self._bars_since_breakout > 15:
                self._awaiting_retest = False
                self._breakout_dir = None

        # Step 1: Detect breakout
        if not self._awaiting_retest and len(bars_history) >= 3:
            prev1 = bars_history[-2]
            prev2 = bars_history[-3]
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.5

            # Bullish breakout: previous bars below VWAP, current above
            if prev1.close < vwap and prev2.close < vwap and bar.close > vwap and vol_ok:
                self._breakout_dir = "LONG"
                self._breakout_price = vwap
                self._awaiting_retest = True
                self._bars_since_breakout = 0
                return None

            # Bearish breakout
            if prev1.close > vwap and prev2.close > vwap and bar.close < vwap and vol_ok:
                self._breakout_dir = "SHORT"
                self._breakout_price = vwap
                self._awaiting_retest = True
                self._bars_since_breakout = 0
                return None

        # Step 2: Detect retest
        if self._awaiting_retest and self._breakout_dir == "LONG":
            # Price pulls back to VWAP, rejection candle
            if bar.low <= vwap + 0.1 * atr and bar.close > vwap and bar.close > bar.open:
                self._awaiting_retest = False
                self._breakout_dir = None
                stop = vwap - 1.0 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 2.0
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.0, target_atr_mult=2.0,
                    max_bars=15, confidence=0.55,
                    reason=f"VWAP retest long: pullback to {vwap:.1f}, rejection close {bar.close:.1f}"
                )

        if self._awaiting_retest and self._breakout_dir == "SHORT":
            if bar.high >= vwap - 0.1 * atr and bar.close < vwap and bar.close < bar.open:
                self._awaiting_retest = False
                self._breakout_dir = None
                stop = vwap + 1.0 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 2.0
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.0, target_atr_mult=2.0,
                    max_bars=15, confidence=0.55,
                    reason=f"VWAP retest short: pullback to {vwap:.1f}, rejection close {bar.close:.1f}"
                )

        return None


# ── 2.2 VWAP Cross with Momentum ─────────────────────────────────────────

class VWAPCrossMomentumDetector(SetupDetector):
    """
    VWAP Cross with EMA-9 + ADX + Volume momentum confirmation.
    """
    name = "vwap_cross_momentum"
    display_name = "VWAP Cross + Momentum"
    category = "vwap"
    hold_time = "5-30min"
    min_cooldown_seconds = 300

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0 or ind.vwap is None or ind.ema9 is None:
            return None
        if len(bars_history) < 3:
            return None

        vwap = ind.vwap
        prev = bars_history[-2]
        adx = ind.adx14
        plus_di = ind.plus_di
        minus_di = ind.minus_di
        rsi = ind.rsi14

        # Session filter: best during AM drive or PM session
        if ind.session_minutes > 120 and ind.session_minutes < 270:
            return None  # Skip lunch

        # Long: cross above VWAP + momentum
        if prev.close < vwap and bar.close > vwap and bar.close > ind.ema9:
            adx_ok = adx is not None and adx > 20 and plus_di is not None and minus_di is not None and plus_di > minus_di
            rsi_ok = rsi is not None and 50 < rsi < 70
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.3
            if adx_ok and vol_ok and rsi_ok:
                stop = bar.close - 1.5 * atr
                risk = bar.close - stop
                target = bar.close + risk * 1.5
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.5, target_atr_mult=2.25,
                    max_bars=20, confidence=0.60,
                    reason=f"VWAP cross long: close {bar.close:.1f} > VWAP {vwap:.1f}, ADX={adx:.0f}"
                )

        # Short: cross below VWAP + momentum
        if prev.close > vwap and bar.close < vwap and bar.close < ind.ema9:
            adx_ok = adx is not None and adx > 20 and plus_di is not None and minus_di is not None and minus_di > plus_di
            rsi_ok = rsi is not None and 30 < rsi < 50
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.3
            if adx_ok and vol_ok and rsi_ok:
                stop = bar.close + 1.5 * atr
                risk = stop - bar.close
                target = bar.close - risk * 1.5
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=1.5, target_atr_mult=2.25,
                    max_bars=20, confidence=0.60,
                    reason=f"VWAP cross short: close {bar.close:.1f} < VWAP {vwap:.1f}, ADX={adx:.0f}"
                )

        return None


# ── 2.3 First VWAP Touch After Gap ───────────────────────────────────────

class FirstVWAPTouchAfterGapDetector(SetupDetector):
    """
    First VWAP Touch After Gap — gap up/down, first touch of VWAP
    within 30-90 minutes acts as support/resistance.
    """
    name = "first_vwap_touch_gap"
    display_name = "First VWAP Touch After Gap"
    category = "vwap"
    hold_time = "15-60min"
    min_cooldown_seconds = 900

    def __init__(self):
        super().__init__()
        self._gap_direction: Optional[str] = None  # "UP" or "DOWN"
        self._gap_size: float = 0.0
        self._vwap_touched: bool = False
        self._day_date: str = ""

    def reset(self):
        super().reset()
        self._gap_direction = None
        self._vwap_touched = False
        self._day_date = ""

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0 or ind.vwap is None or ind.pdc is None:
            return None

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._day_date:
            self._day_date = today
            self._gap_direction = None
            self._vwap_touched = False

        # Detect gap on first RTH bar
        if ind.session_minutes == 0 and self._gap_direction is None:
            gap_pct = (bar.open - ind.pdc) / ind.pdc
            if gap_pct >= 0.005:
                self._gap_direction = "UP"
                self._gap_size = gap_pct
            elif gap_pct <= -0.005:
                self._gap_direction = "DOWN"
                self._gap_size = abs(gap_pct)
            return None

        if self._gap_direction is None or self._vwap_touched:
            return None

        # Only within 30-90 minutes
        if ind.session_minutes < 30 or ind.session_minutes > 90:
            return None

        vwap = ind.vwap

        # Gap up: first VWAP touch from above = long
        if self._gap_direction == "UP":
            if bar.low <= vwap + 0.1 * atr and bar.close > vwap:
                self._vwap_touched = True
                rsi = ind.rsi14
                if rsi is not None and rsi < 35:
                    return None  # Too weak
                stop = bar.low - 0.5 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 1.5
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=30, confidence=0.58,
                    reason=f"First VWAP touch after gap up: touched {vwap:.1f}, gap {self._gap_size*100:.1f}%"
                )

        # Gap down: first VWAP touch from below = short
        if self._gap_direction == "DOWN":
            if bar.high >= vwap - 0.1 * atr and bar.close < vwap:
                self._vwap_touched = True
                rsi = ind.rsi14
                if rsi is not None and rsi > 65:
                    return None
                stop = bar.high + 0.5 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 1.5
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=30, confidence=0.58,
                    reason=f"First VWAP touch after gap down: touched {vwap:.1f}, gap {self._gap_size*100:.1f}%"
                )

        return None


# ── 2.4 ORB Failure Reversal ─────────────────────────────────────────────

class ORBFailureReversalDetector(SetupDetector):
    """
    ORB Failure Reversal — price breaks OR boundary, fails to continue,
    reverses back inside. "When what should happen doesn't happen."
    """
    name = "orb_failure_reversal"
    display_name = "ORB Failure Reversal"
    category = "session"
    hold_time = "15-45min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_complete: bool = False
        self._or_date: str = ""
        self._long_break_bar: int = 0
        self._short_break_bar: int = 0
        self._long_failure_fired: bool = False
        self._short_failure_fired: bool = False

    def reset(self):
        super().reset()
        self._or_high = None
        self._or_low = None
        self._or_complete = False
        self._or_date = ""
        self._long_break_bar = 0
        self._short_break_bar = 0
        self._long_failure_fired = False
        self._short_failure_fired = False

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._or_date:
            self._or_date = today
            self._or_high = None
            self._or_low = None
            self._or_complete = False
            self._long_break_bar = 0
            self._short_break_bar = 0
            self._long_failure_fired = False
            self._short_failure_fired = False

        # Build OR
        if ind.session_minutes < 15:
            if self._or_high is None:
                self._or_high = bar.high
                self._or_low = bar.low
            else:
                self._or_high = max(self._or_high, bar.high)
                self._or_low = min(self._or_low, bar.low)
            return None

        if not self._or_complete:
            self._or_complete = True

        if self._or_high is None or self._or_low is None:
            return None

        bar_idx = len(bars_history)

        # Track breakout above OR high
        if bar.close > self._or_high and self._long_break_bar == 0:
            self._long_break_bar = bar_idx

        # Track breakout below OR low
        if bar.close < self._or_low and self._short_break_bar == 0:
            self._short_break_bar = bar_idx

        # Failure of long breakout: broke above, then close back inside within 5-10 bars
        if (self._long_break_bar > 0 and not self._long_failure_fired
                and 5 <= (bar_idx - self._long_break_bar) <= 10
                and bar.close < self._or_high and bar.close < bar.open):
            self._long_failure_fired = True
            stop = self._or_high + 0.5 * atr
            risk = stop - bar.close
            if risk <= 0:
                return None
            target = self._or_low
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=abs(bar.close - target) / atr,
                max_bars=30, confidence=0.65,
                reason=f"ORB failure reversal short: broke {self._or_high:.1f}, failed back to {bar.close:.1f}"
            )

        # Failure of short breakout
        if (self._short_break_bar > 0 and not self._short_failure_fired
                and 5 <= (bar_idx - self._short_break_bar) <= 10
                and bar.close > self._or_low and bar.close > bar.open):
            self._short_failure_fired = True
            stop = self._or_low - 0.5 * atr
            risk = bar.close - stop
            if risk <= 0:
                return None
            target = self._or_high
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=abs(target - bar.close) / atr,
                max_bars=30, confidence=0.65,
                reason=f"ORB failure reversal long: broke {self._or_low:.1f}, failed back to {bar.close:.1f}"
            )

        return None


# ── 2.5 A-B-C Morning Reversal ───────────────────────────────────────────

class ABCMorningReversalDetector(SetupDetector):
    """
    A-B-C Morning Reversal — Elliott-based 3-point pattern in first
    90 minutes. A=initial extreme, B=retracement 38-62%, C=failed retest.
    """
    name = "abc_morning_reversal"
    display_name = "A-B-C Morning Reversal"
    category = "session"
    hold_time = "15-60min"
    min_cooldown_seconds = 900

    def __init__(self):
        super().__init__()
        self._swing_highs: List[float] = []
        self._swing_lows: List[float] = []
        self._day_date: str = ""
        self._fired: bool = False

    def reset(self):
        super().reset()
        self._swing_highs = []
        self._swing_lows = []
        self._day_date = ""
        self._fired = False

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 10:
            return None

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._day_date:
            self._day_date = today
            self._fired = False

        if self._fired:
            return None

        # Only during 9:45-11:00 ET
        if ind.session_minutes < 15 or ind.session_minutes > 90:
            return None

        # Find swing points in recent 20 bars
        recent = bars_history[-20:]
        if len(recent) < 10:
            return None

        # Find highest high and lowest low as potential A points
        max_h = max(b.high for b in recent[:10])
        min_l = min(b.low for b in recent[:10])

        # Bearish ABC: A=high, B=retrace down, C=lower high
        ab_range = max_h - min_l
        if ab_range < 0.5 * atr:
            return None

        # Check if current area is a "C" point (lower high failing)
        if bar.high < max_h and bar.high > min_l:
            # C is a failed retest - check Fibonacci
            retrace_b = (max_h - min_l)
            c_level = (bar.high - min_l) / retrace_b if retrace_b > 0 else 0
            if 0.382 <= c_level <= 0.786:
                # Bar shows rejection (close near low)
                if bar.close < bar.open and (bar.high - bar.close) > 0.5 * (bar.high - bar.low):
                    self._fired = True
                    stop = max_h + 0.25 * atr
                    risk = stop - bar.close
                    if risk <= 0:
                        return None
                    target = bar.close - ab_range * 0.618
                    return self.make_signal(
                        "SHORT", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=abs(bar.close - target) / atr,
                        max_bars=30, confidence=0.60,
                        reason=f"ABC reversal short: A={max_h:.1f} C={bar.high:.1f} fib={c_level:.2f}"
                    )

        # Bullish ABC: A=low, B=retrace up, C=higher low
        if bar.low > min_l and bar.low < max_h:
            retrace_b = (max_h - min_l)
            c_level = (max_h - bar.low) / retrace_b if retrace_b > 0 else 0
            if 0.382 <= c_level <= 0.786:
                if bar.close > bar.open and (bar.close - bar.low) > 0.5 * (bar.high - bar.low):
                    self._fired = True
                    stop = min_l - 0.25 * atr
                    risk = bar.close - stop
                    if risk <= 0:
                        return None
                    target = bar.close + ab_range * 0.618
                    return self.make_signal(
                        "LONG", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=abs(target - bar.close) / atr,
                        max_bars=30, confidence=0.60,
                        reason=f"ABC reversal long: A={min_l:.1f} C={bar.low:.1f} fib={c_level:.2f}"
                    )

        return None


# ── 2.6 NR7+ORB Combination ──────────────────────────────────────────────

class NR7ORBDetector(SetupDetector):
    """
    NR7 + ORB Combination — when today's daily range so far is narrowest
    of 7, ORB breakout has much higher probability.
    Uses intraday rolling range as proxy for daily NR7.
    """
    name = "nr7_orb"
    display_name = "NR7 + ORB Combo"
    category = "volatility"
    hold_time = "30min-4hr"
    min_cooldown_seconds = 900

    def __init__(self):
        super().__init__()
        self._daily_ranges: List[float] = []  # last 7 days
        self._cur_day_range: float = 0.0
        self._cur_day_high: float = 0.0
        self._cur_day_low: float = float('inf')
        self._cur_day_date: str = ""
        self._nr7_flag: bool = False
        self._fired: bool = False

    def reset(self):
        super().reset()
        self._daily_ranges = []
        self._nr7_flag = False
        self._fired = False

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')

        # New day: store previous day range, check NR7
        if today != self._cur_day_date:
            if self._cur_day_date and self._cur_day_high > 0:
                self._daily_ranges.append(self._cur_day_high - self._cur_day_low)
                if len(self._daily_ranges) > 7:
                    self._daily_ranges = self._daily_ranges[-7:]
            self._cur_day_date = today
            self._cur_day_high = bar.high
            self._cur_day_low = bar.low
            self._fired = False

            # Check NR7
            if len(self._daily_ranges) >= 7:
                yesterday = self._daily_ranges[-1]
                self._nr7_flag = yesterday <= min(self._daily_ranges[-7:])
            else:
                self._nr7_flag = False
        else:
            self._cur_day_high = max(self._cur_day_high, bar.high)
            self._cur_day_low = min(self._cur_day_low, bar.low)

        if not self._nr7_flag or self._fired:
            return None

        # Use OR levels from indicator state
        if not ind.or_complete or ind.orh is None or ind.orl is None:
            return None

        # Only first 2 hours
        if ind.session_minutes > 120:
            return None

        # Breakout
        if bar.close > ind.orh:
            self._fired = True
            stop = ind.orl - 0.5 * atr
            risk = bar.close - stop
            if risk <= 0:
                return None
            target = bar.close + risk * 1.5
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=60, confidence=0.65,
                reason=f"NR7+ORB long: NR7 day, close {bar.close:.1f} > OR high {ind.orh:.1f}"
            )

        if bar.close < ind.orl:
            self._fired = True
            stop = ind.orh + 0.5 * atr
            risk = stop - bar.close
            if risk <= 0:
                return None
            target = bar.close - risk * 1.5
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=60, confidence=0.65,
                reason=f"NR7+ORB short: NR7 day, close {bar.close:.1f} < OR low {ind.orl:.1f}"
            )

        return None


# ── 2.7 PDH/PDL Rejection ────────────────────────────────────────────────

class PDHPDLRejectionDetector(SetupDetector):
    """
    PDH/PDL Rejection — price approaches level but fails to break,
    showing rejection candle pattern. Opposite of breakout.
    """
    name = "pdh_pdl_rejection"
    display_name = "PDH/PDL Rejection"
    category = "level"
    hold_time = "15-60min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._pdh_reject_fired: bool = False
        self._pdl_reject_fired: bool = False
        self._last_date: str = ""

    def reset(self):
        super().reset()
        self._pdh_reject_fired = False
        self._pdl_reject_fired = False
        self._last_date = ""

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._last_date:
            self._last_date = today
            self._pdh_reject_fired = False
            self._pdl_reject_fired = False

        pdh = ind.pdh
        pdl = ind.pdl
        if pdh is None or pdl is None:
            return None

        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None

        # Short rejection at PDH: wick touches/exceeds PDH, close below
        if not self._pdh_reject_fired:
            near_pdh = bar.high >= pdh - 0.15 * atr and bar.high <= pdh + 0.5 * atr
            rejection = bar.close < pdh and (bar.high - bar.close) > 0.6 * bar_range  # upper wick
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20
            if near_pdh and rejection and vol_ok:
                self._pdh_reject_fired = True
                stop = bar.high + 0.5 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 2.0
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                    max_bars=30, confidence=0.60,
                    reason=f"PDH rejection short: high {bar.high:.1f} near PDH {pdh:.1f}, wick rejection"
                )

        # Long rejection at PDL
        if not self._pdl_reject_fired:
            near_pdl = bar.low <= pdl + 0.15 * atr and bar.low >= pdl - 0.5 * atr
            rejection = bar.close > pdl and (bar.close - bar.low) > 0.6 * bar_range
            vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20
            if near_pdl and rejection and vol_ok:
                self._pdl_reject_fired = True
                stop = bar.low - 0.5 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 2.0
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                    max_bars=30, confidence=0.60,
                    reason=f"PDL rejection long: low {bar.low:.1f} near PDL {pdl:.1f}, wick rejection"
                )

        return None


# ── 2.8 Round Number Bounce ──────────────────────────────────────────────

class RoundNumberBounceDetector(SetupDetector):
    """
    Round Number Bounce — psychological S/R at major round numbers.
    ES: 50s/100s, NQ: 250s/500s, GC: $50 increments.
    """
    name = "round_number_bounce"
    display_name = "Round Number Bounce"
    category = "level"
    hold_time = "15-45min"
    min_cooldown_seconds = 600

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None

        price = bar.close

        # Determine round number grid based on price magnitude
        if price > 10000:
            # NQ range — use 250 and 500 increments
            major_step = 500
            minor_step = 250
        elif price > 3000:
            # ES range — use 50 and 100 increments
            major_step = 100
            minor_step = 50
        elif price > 1000:
            # GC range — use 50 increments
            major_step = 50
            minor_step = 25
        else:
            major_step = 10
            minor_step = 5

        # Find nearest round number
        nearest_major = round(price / major_step) * major_step
        nearest_minor = round(price / minor_step) * minor_step

        # Use the closest level
        dist_major = abs(price - nearest_major)
        dist_minor = abs(price - nearest_minor)
        level = nearest_major if dist_major <= dist_minor else nearest_minor
        proximity = abs(price - level)

        # Must be within 0.25 * ATR of round number
        if proximity > 0.25 * atr:
            return None

        # Need rejection pattern
        # Long bounce: low near level, close above
        if bar.low <= level + 0.1 * atr and bar.close > level:
            upper_body = (bar.close - bar.low) / bar_range
            if upper_body > 0.55:  # Close in upper portion = rejection
                stop = level - 0.75 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 1.5
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=20, confidence=0.50,
                    reason=f"Round number bounce long at {level:.0f}"
                )

        # Short bounce: high near level, close below
        if bar.high >= level - 0.1 * atr and bar.close < level:
            lower_body = (bar.high - bar.close) / bar_range
            if lower_body > 0.55:
                stop = level + 0.75 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 1.5
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=20, confidence=0.50,
                    reason=f"Round number bounce short at {level:.0f}"
                )

        return None


# ── 2.9 RSI Divergence Reversal ──────────────────────────────────────────

class RSIDivergenceDetector(SetupDetector):
    """
    RSI Divergence Reversal — price makes new low but RSI makes higher
    low (bullish), or price new high but RSI lower high (bearish).
    """
    name = "rsi_divergence"
    display_name = "RSI Divergence Reversal"
    category = "momentum"
    hold_time = "5-30min"
    min_cooldown_seconds = 600

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        rsi = ind.rsi14
        rsi_prev = ind.rsi14_prev
        if not atr or atr <= 0 or rsi is None or rsi_prev is None:
            return None
        if len(bars_history) < 10:
            return None

        # Strong trend filter — divergence fails in strong trends
        if ind.adx14 is not None and ind.adx14 > 30:
            return None

        # Look back 5-10 bars for price extreme vs RSI extreme
        lookback = bars_history[-10:-1]

        # Bullish divergence: new price low, RSI higher low
        cur_low = bar.low
        prev_low = min(b.low for b in lookback)

        if cur_low <= prev_low and rsi > 25:
            # Estimate previous RSI at the low point — use simple heuristic:
            # if current RSI is higher than it "should be" given new low, divergence
            if rsi > 30 and rsi > rsi_prev:  # RSI rising while price falling
                # Confirmation: bullish candle
                if bar.close > bar.open and (bar.close - bar.low) > 0.5 * (bar.high - bar.low):
                    stop = cur_low - 0.75 * atr
                    risk = bar.close - stop
                    if risk <= 0:
                        return None
                    target = bar.close + risk * 1.5
                    return self.make_signal(
                        "LONG", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                        max_bars=20, confidence=0.58,
                        reason=f"RSI bullish divergence: new low {cur_low:.1f}, RSI={rsi:.0f} rising"
                    )

        # Bearish divergence: new price high, RSI lower high
        cur_high = bar.high
        prev_high = max(b.high for b in lookback)

        if cur_high >= prev_high and rsi < 75:
            if rsi < 70 and rsi < rsi_prev:  # RSI falling while price rising
                if bar.close < bar.open and (bar.high - bar.close) > 0.5 * (bar.high - bar.low):
                    stop = cur_high + 0.75 * atr
                    risk = stop - bar.close
                    if risk <= 0:
                        return None
                    target = bar.close - risk * 1.5
                    return self.make_signal(
                        "SHORT", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                        max_bars=20, confidence=0.58,
                        reason=f"RSI bearish divergence: new high {cur_high:.1f}, RSI={rsi:.0f} falling"
                    )

        return None


# ── 2.10 ADX Thrust ──────────────────────────────────────────────────────

class ADXThrustDetector(SetupDetector):
    """
    ADX Thrust — ADX crossing above 25 from below signals new trend.
    """
    name = "adx_thrust"
    display_name = "ADX Thrust"
    category = "momentum"
    hold_time = "15-60min"
    min_cooldown_seconds = 900

    def __init__(self):
        super().__init__()
        self._prev_adx: Optional[float] = None

    def reset(self):
        super().reset()
        self._prev_adx = None

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            self._prev_adx = ind.adx14
            return None
        atr = ind.atr14
        adx = ind.adx14
        if not atr or atr <= 0 or adx is None:
            self._prev_adx = adx
            return None

        # Detect cross above 25
        crossed = (self._prev_adx is not None and self._prev_adx < 25 and adx >= 25)
        self._prev_adx = adx

        if not crossed:
            return None

        plus_di = ind.plus_di
        minus_di = ind.minus_di
        if plus_di is None or minus_di is None:
            return None

        vol_ok = ind.volume_sma20 is not None and bar.volume >= ind.volume_sma20

        # Long: +DI > -DI
        if plus_di > minus_di and bar.close > (ind.ema9 or 0):
            stop = bar.close - 2.0 * atr
            risk = bar.close - stop
            target = bar.close + risk * 1.5
            conf = 0.55 if vol_ok else 0.48
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=2.0, target_atr_mult=3.0,
                max_bars=30, confidence=conf,
                reason=f"ADX thrust long: ADX crossed 25 ({adx:.0f}), +DI={plus_di:.0f} > -DI={minus_di:.0f}"
            )

        # Short: -DI > +DI
        if minus_di > plus_di and bar.close < (ind.ema9 or float('inf')):
            stop = bar.close + 2.0 * atr
            risk = stop - bar.close
            target = bar.close - risk * 1.5
            conf = 0.55 if vol_ok else 0.48
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=2.0, target_atr_mult=3.0,
                max_bars=30, confidence=conf,
                reason=f"ADX thrust short: ADX crossed 25 ({adx:.0f}), -DI={minus_di:.0f} > +DI={plus_di:.0f}"
            )

        return None


# ── 2.11 ATR Expansion Trade ─────────────────────────────────────────────

class ATRExpansionDetector(SetupDetector):
    """
    ATR Expansion Trade — after sustained ATR compression, breakout
    on channel break with momentum.
    """
    name = "atr_expansion"
    display_name = "ATR Expansion Trade"
    category = "volatility"
    hold_time = "15-60min"
    min_cooldown_seconds = 600

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 50:
            return None

        # Compute ATR SMA-50 for compression detection
        recent_atrs = []
        for i in range(max(0, len(bars_history) - 50), len(bars_history)):
            recent_atrs.append(_true_range(bars_history, i))
        if len(recent_atrs) < 50:
            return None

        atr_sma50 = sum(recent_atrs) / len(recent_atrs)
        if atr_sma50 <= 0:
            return None

        # Compression: current ATR < 70% of 50-bar ATR average
        # But current bar shows expansion
        prev_atr_avg = sum(recent_atrs[-15:-1]) / 14 if len(recent_atrs) >= 15 else atr
        compressed = prev_atr_avg < 0.7 * atr_sma50
        cur_bar_range = bar.high - bar.low
        expanding = cur_bar_range > 1.5 * prev_atr_avg

        if not (compressed and expanding):
            return None

        # 20-bar channel break
        highs_20 = [b.high for b in bars_history[-21:-1]]
        lows_20 = [b.low for b in bars_history[-21:-1]]
        chan_high = max(highs_20)
        chan_low = min(lows_20)

        if bar.close > chan_high and bar.close > bar.open:
            stop = bar.close - 2.0 * atr
            risk = bar.close - stop
            target = bar.close + risk * 1.5
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=2.0, target_atr_mult=3.0,
                max_bars=30, confidence=0.55,
                reason=f"ATR expansion long: compression broke up, range={cur_bar_range:.1f} vs avg={prev_atr_avg:.1f}"
            )

        if bar.close < chan_low and bar.close < bar.open:
            stop = bar.close + 2.0 * atr
            risk = stop - bar.close
            target = bar.close - risk * 1.5
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=2.0, target_atr_mult=3.0,
                max_bars=30, confidence=0.55,
                reason=f"ATR expansion short: compression broke down, range={cur_bar_range:.1f} vs avg={prev_atr_avg:.1f}"
            )

        return None


# ── 2.12 NR4/NR7 Standalone Breakout ─────────────────────────────────────

class NR4NR7BreakoutDetector(SetupDetector):
    """
    NR4/NR7 Standalone Breakout — narrowest range bar of 4 or 7,
    enter on breakout of that bar's range.
    Uses intraday bars (not daily like NR7+ORB combo).
    """
    name = "nr4_nr7_breakout"
    display_name = "NR4/NR7 Bar Breakout"
    category = "volatility"
    hold_time = "5-30min"
    min_cooldown_seconds = 420

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 9:
            return None

        # Check if previous bar was NR7 (narrowest of last 7)
        prev = bars_history[-2]
        prev_range = prev.high - prev.low
        if prev_range <= 0:
            return None

        ranges_7 = [b.high - b.low for b in bars_history[-8:-1]]
        is_nr7 = prev_range <= min(ranges_7)

        ranges_4 = [b.high - b.low for b in bars_history[-5:-1]]
        is_nr4 = prev_range <= min(ranges_4)

        if not (is_nr4 or is_nr7):
            return None

        # Current bar breaks out of NR bar range
        if bar.close > prev.high and bar.close > bar.open:
            stop = prev.low - 0.25 * atr
            risk = bar.close - stop
            if risk <= 0:
                return None
            target = bar.close + risk * 1.5
            label = "NR7" if is_nr7 else "NR4"
            conf = 0.60 if is_nr7 else 0.55
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=20, confidence=conf,
                reason=f"{label} breakout long: close {bar.close:.1f} > NR high {prev.high:.1f}"
            )

        if bar.close < prev.low and bar.close < bar.open:
            stop = prev.high + 0.25 * atr
            risk = stop - bar.close
            if risk <= 0:
                return None
            target = bar.close - risk * 1.5
            label = "NR7" if is_nr7 else "NR4"
            conf = 0.60 if is_nr7 else 0.55
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                max_bars=20, confidence=conf,
                reason=f"{label} breakout short: close {bar.close:.1f} < NR low {prev.low:.1f}"
            )

        return None


# ── 2.13 VCP Intraday ────────────────────────────────────────────────────

class VCPIntradayDetector(SetupDetector):
    """
    Volatility Contraction Pattern — progressively smaller pullbacks
    with declining volume, then breakout on volume expansion.
    """
    name = "vcp_intraday"
    display_name = "VCP Intraday"
    category = "volatility"
    hold_time = "15-60min"
    min_cooldown_seconds = 900

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 30:
            return None

        # Look for 3 contractions in last 30 bars
        # Split into 3 segments of ~10 bars each
        seg1 = bars_history[-30:-20]
        seg2 = bars_history[-20:-10]
        seg3 = bars_history[-10:-1]

        range1 = max(b.high for b in seg1) - min(b.low for b in seg1)
        range2 = max(b.high for b in seg2) - min(b.low for b in seg2)
        range3 = max(b.high for b in seg3) - min(b.low for b in seg3)

        vol1 = sum(b.volume for b in seg1)
        vol2 = sum(b.volume for b in seg2)
        vol3 = sum(b.volume for b in seg3)

        # Progressive contraction: each range smaller
        if not (range1 > range2 > range3):
            return None
        # Volume declining
        if not (vol1 > vol2 > vol3):
            return None
        # Minimum contraction ratio
        if range3 > 0.7 * range1:
            return None

        # Pivot = highest high of consolidation
        pivot_high = max(b.high for b in bars_history[-30:-1])
        pivot_low = min(b.low for b in bars_history[-30:-1])

        # Breakout with volume
        vol_ok = ind.volume_sma20 is not None and bar.volume > ind.volume_sma20 * 1.4

        if bar.close > pivot_high and vol_ok:
            stop = pivot_low
            risk = bar.close - stop
            if risk <= 0 or risk > 3 * atr:
                return None
            target = bar.close + risk * 2.0
            return self.make_signal(
                "LONG", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                max_bars=40, confidence=0.60,
                reason=f"VCP long: 3 contractions ({range1:.1f}→{range2:.1f}→{range3:.1f}), breakout {bar.close:.1f}"
            )

        if bar.close < pivot_low and vol_ok:
            stop = pivot_high
            risk = stop - bar.close
            if risk <= 0 or risk > 3 * atr:
                return None
            target = bar.close - risk * 2.0
            return self.make_signal(
                "SHORT", bar, ind,
                stop_price=stop, target_price=target,
                stop_atr_mult=risk / atr, target_atr_mult=(risk * 2) / atr,
                max_bars=40, confidence=0.60,
                reason=f"VCP short: 3 contractions ({range1:.1f}→{range2:.1f}→{range3:.1f}), breakdown {bar.close:.1f}"
            )

        return None


# ── 2.14 Liquidity Sweep & Reversal ──────────────────────────────────────

class LiquiditySweepDetector(SetupDetector):
    """
    Liquidity Sweep — price sweeps beyond a swing high/low to trigger
    stops, then reverses with displacement candle.
    """
    name = "liquidity_sweep"
    display_name = "Liquidity Sweep & Reversal"
    category = "micro"
    hold_time = "5-30min"
    min_cooldown_seconds = 600

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 15:
            return None

        # Find swing high/low in last 10-20 bars
        lookback = bars_history[-20:-1]
        swing_high = max(b.high for b in lookback)
        swing_low = min(b.low for b in lookback)

        bar_body = abs(bar.close - bar.open)
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return None

        vol_ok = ind.volume_sma20 is not None and bar.volume >= ind.volume_sma20 * 1.5

        # Bullish sweep: wick below swing low, close back above
        penetration_low = swing_low - bar.low
        if penetration_low >= 0.2 * atr and bar.close > swing_low and bar.close > bar.open:
            # Displacement: body >= 70% of range
            if bar_body >= 0.6 * bar_range and vol_ok:
                stop = bar.low - 0.25 * atr
                risk = bar.close - stop
                if risk <= 0:
                    return None
                target = bar.close + risk * 1.5
                return self.make_signal(
                    "LONG", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=20, confidence=0.55,
                    reason=f"Liquidity sweep long: swept {swing_low:.1f} by {penetration_low:.1f}, reversed"
                )

        # Bearish sweep: wick above swing high, close back below
        penetration_high = bar.high - swing_high
        if penetration_high >= 0.2 * atr and bar.close < swing_high and bar.close < bar.open:
            if bar_body >= 0.6 * bar_range and vol_ok:
                stop = bar.high + 0.25 * atr
                risk = stop - bar.close
                if risk <= 0:
                    return None
                target = bar.close - risk * 1.5
                return self.make_signal(
                    "SHORT", bar, ind,
                    stop_price=stop, target_price=target,
                    stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                    max_bars=20, confidence=0.55,
                    reason=f"Liquidity sweep short: swept {swing_high:.1f} by {penetration_high:.1f}, reversed"
                )

        return None


# ── 2.15 Fair Value Gap Fill ──────────────────────────────────────────────

class FVGFillDetector(SetupDetector):
    """
    Fair Value Gap Fill — 3-candle imbalance pattern. When price
    returns to fill the gap zone, enter in the original momentum direction.
    """
    name = "fvg_fill"
    display_name = "Fair Value Gap Fill"
    category = "micro"
    hold_time = "5-30min"
    min_cooldown_seconds = 300

    def __init__(self):
        super().__init__()
        self._active_fvgs: List[dict] = []  # {type, top, bottom, bar_time}
        self._max_fvgs: int = 10

    def reset(self):
        super().reset()
        self._active_fvgs = []

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        if len(bars_history) < 4:
            return None

        c1 = bars_history[-3]  # candle 1
        c2 = bars_history[-2]  # candle 2 (momentum)
        c3 = bar              # candle 3

        # Detect new bullish FVG: candle3.low > candle1.high
        if c3.low > c1.high:
            gap_size = c3.low - c1.high
            body_c2 = abs(c2.close - c2.open)
            avg_body = atr * 0.5  # rough proxy
            if gap_size >= 0.5 * atr and body_c2 > avg_body:
                self._active_fvgs.append({
                    'type': 'bullish', 'top': c3.low, 'bottom': c1.high,
                    'bar_time': bar.time
                })

        # Detect new bearish FVG: candle3.high < candle1.low
        if c3.high < c1.low:
            gap_size = c1.low - c3.high
            body_c2 = abs(c2.close - c2.open)
            avg_body = atr * 0.5
            if gap_size >= 0.5 * atr and body_c2 > avg_body:
                self._active_fvgs.append({
                    'type': 'bearish', 'top': c1.low, 'bottom': c3.high,
                    'bar_time': bar.time
                })

        # Trim old FVGs (> 60 bars old)
        self._active_fvgs = [
            f for f in self._active_fvgs
            if (bar.time - f['bar_time']) < 3600
        ][-self._max_fvgs:]

        # Check if current bar fills any FVG
        for fvg in self._active_fvgs:
            # Skip FVGs created on this bar
            if fvg['bar_time'] >= bars_history[-2].time:
                continue

            if fvg['type'] == 'bullish':
                # Price retraces into bullish FVG zone = long
                if bar.low <= fvg['top'] and bar.close > fvg['bottom']:
                    # Rejection in gap zone
                    if bar.close > bar.open:
                        self._active_fvgs.remove(fvg)
                        stop = fvg['bottom'] - 0.25 * atr
                        risk = bar.close - stop
                        if risk <= 0:
                            return None
                        target = bar.close + risk * 1.5
                        return self.make_signal(
                            "LONG", bar, ind,
                            stop_price=stop, target_price=target,
                            stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                            max_bars=15, confidence=0.55,
                            reason=f"FVG fill long: retrace to gap zone {fvg['bottom']:.1f}-{fvg['top']:.1f}"
                        )

            elif fvg['type'] == 'bearish':
                if bar.high >= fvg['bottom'] and bar.close < fvg['top']:
                    if bar.close < bar.open:
                        self._active_fvgs.remove(fvg)
                        stop = fvg['top'] + 0.25 * atr
                        risk = stop - bar.close
                        if risk <= 0:
                            return None
                        target = bar.close - risk * 1.5
                        return self.make_signal(
                            "SHORT", bar, ind,
                            stop_price=stop, target_price=target,
                            stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                            max_bars=15, confidence=0.55,
                            reason=f"FVG fill short: retrace to gap zone {fvg['bottom']:.1f}-{fvg['top']:.1f}"
                        )

        return None


# ── 2.16 Absorption Proxy ────────────────────────────────────────────────

class AbsorptionProxyDetector(SetupDetector):
    """
    Absorption Proxy — high volume + tiny range + tiny body at key
    level = institutional absorption. Enter on "release" candle.
    """
    name = "absorption_proxy"
    display_name = "Absorption Proxy"
    category = "micro"
    hold_time = "5-20min"
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._absorption_detected: bool = False
        self._absorption_level: float = 0.0
        self._absorption_bar_idx: int = 0

    def reset(self):
        super().reset()
        self._absorption_detected = False

    def update(self, bar: BarInput, ind: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        if not ind.is_rth:
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        bar_range = bar.high - bar.low
        bar_body = abs(bar.close - bar.open)
        bar_idx = len(bars_history)
        vol_sma = ind.volume_sma20

        # Detect absorption: high volume + small range + small body
        if vol_sma is not None and vol_sma > 0:
            high_vol = bar.volume >= vol_sma * 2.0
            small_range = bar_range < 0.5 * atr
            small_body = bar_range > 0 and (bar_body / bar_range) < 0.3

            if high_vol and small_range and small_body:
                self._absorption_detected = True
                self._absorption_level = (bar.high + bar.low) / 2
                self._absorption_bar_idx = bar_idx
                return None

        # Look for release candle within 5 bars of absorption
        if self._absorption_detected and (bar_idx - self._absorption_bar_idx) <= 5:
            # Release = strong directional candle
            strong_body = bar_body > 0.7 * bar_range if bar_range > 0 else False
            big_move = bar_range > 1.0 * atr

            if strong_body and big_move:
                self._absorption_detected = False

                if bar.close > bar.open:
                    # Bullish release
                    stop = self._absorption_level - 0.5 * atr
                    risk = bar.close - stop
                    if risk <= 0:
                        return None
                    target = bar.close + risk * 1.5
                    return self.make_signal(
                        "LONG", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                        max_bars=15, confidence=0.50,
                        reason=f"Absorption release long: absorbed at {self._absorption_level:.1f}, bullish release"
                    )
                else:
                    # Bearish release
                    stop = self._absorption_level + 0.5 * atr
                    risk = stop - bar.close
                    if risk <= 0:
                        return None
                    target = bar.close - risk * 1.5
                    return self.make_signal(
                        "SHORT", bar, ind,
                        stop_price=stop, target_price=target,
                        stop_atr_mult=risk / atr, target_atr_mult=(risk * 1.5) / atr,
                        max_bars=15, confidence=0.50,
                        reason=f"Absorption release short: absorbed at {self._absorption_level:.1f}, bearish release"
                    )

        # Timeout absorption after 5 bars
        if self._absorption_detected and (bar_idx - self._absorption_bar_idx) > 5:
            self._absorption_detected = False

        return None


# ═══════════════════════════════════════════════════════════════════════════
# SETUP MANAGER — with self-contained indicator computation
# ═══════════════════════════════════════════════════════════════════════════

class SetupManager:
    """
    Runs all registered setup detectors on each incoming bar.
    Computes all indicators from raw bar history — no external deps.
    """

    def __init__(self):
        self.detectors: List[SetupDetector] = []
        self._bars_history: List[BarInput] = []
        self._max_history: int = 600
        self._indicator_state = IndicatorState()

        # VWAP session accumulators (reset each RTH day)
        self._vwap_cum_vol: float = 0.0
        self._vwap_cum_pv: float = 0.0
        self._vwap_cum_pv2: float = 0.0
        self._vwap_date: str = ""

        # PDH/PDL tracking
        self._prev_day_high: Optional[float] = None
        self._prev_day_low: Optional[float] = None
        self._prev_day_close: Optional[float] = None
        self._cur_day_high: Optional[float] = None
        self._cur_day_low: Optional[float] = None
        self._cur_day_close: Optional[float] = None
        self._cur_day_date: str = ""

        # ON H/L tracking
        self._on_high: Optional[float] = None
        self._on_low: Optional[float] = None
        self._on_date: str = ""

        # OR tracking
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_complete: bool = False
        self._or_date: str = ""

        # RSI state for prev tracking
        self._prev_rsi: Optional[float] = None

        logger.info("SetupManager initialized")

    def register(self, detector: SetupDetector):
        self.detectors.append(detector)
        logger.info(f"Registered setup detector: {detector.name} ({detector.display_name})")

    def register_all_defaults(self):
        """Register all 23 setup detectors from the research catalog."""
        # Phase 1: Foundation (7)
        self.register(ORBBreakoutDetector())            # Setup 5
        self.register(VWAPMeanReversionDetector())       # Setup 1
        self.register(PDHPDLBreakoutDetector())          # Setup 10
        self.register(EMA9PullbackDetector())            # Setup 13
        self.register(TTMSqueezeDetector())              # Setup 17
        self.register(ONHLSweepDetector())               # Setup 11
        self.register(VolumeSpikeBreakoutDetector())     # Setup 15

        # Phase 2: Advanced (16)
        self.register(VWAPBreakoutRetestDetector())      # Setup 2
        self.register(VWAPCrossMomentumDetector())       # Setup 3
        self.register(FirstVWAPTouchAfterGapDetector())  # Setup 4
        self.register(ORBFailureReversalDetector())      # Setup 6
        self.register(ABCMorningReversalDetector())      # Setup 7
        self.register(NR7ORBDetector())                  # Setup 8
        self.register(PDHPDLRejectionDetector())         # Setup 9
        self.register(RoundNumberBounceDetector())       # Setup 12
        self.register(RSIDivergenceDetector())           # Setup 14
        self.register(ADXThrustDetector())               # Setup 16
        self.register(ATRExpansionDetector())            # Setup 18
        self.register(NR4NR7BreakoutDetector())          # Setup 19
        self.register(VCPIntradayDetector())             # Setup 20
        self.register(LiquiditySweepDetector())          # Setup 21
        self.register(FVGFillDetector())                 # Setup 22
        self.register(AbsorptionProxyDetector())         # Setup 23

        logger.info(f"SetupManager: {len(self.detectors)} detectors registered")

    def process_bar(self, bar_data: dict, indicators: dict = None,
                    levels: dict = None, session: str = "") -> List[SetupSignal]:
        bar = BarInput(
            time=bar_data.get('time', 0),
            open=bar_data.get('open', 0),
            high=bar_data.get('high', 0),
            low=bar_data.get('low', 0),
            close=bar_data.get('close', 0),
            volume=bar_data.get('volume', 0),
        )

        self._bars_history.append(bar)
        if len(self._bars_history) > self._max_history:
            self._bars_history = self._bars_history[-self._max_history:]

        # Compute all indicators from bar history
        self._compute_indicators(bar)

        # Override with externally provided values if available
        if indicators:
            self._merge_external_indicators(indicators)
        if levels:
            self._merge_external_levels(levels)

        # Run all detectors
        signals: List[SetupSignal] = []
        for detector in self.detectors:
            if not detector.can_signal(bar.time):
                continue
            try:
                signal = detector.update(bar, self._indicator_state, self._bars_history)
                if signal:
                    detector.record_signal(bar.time)
                    signals.append(signal)
                    logger.info(
                        f"[{detector.name}] Signal: {signal.direction} @ {signal.entry_price:.2f} "
                        f"(conf={signal.confidence:.2f}, reason={signal.reason})"
                    )
            except Exception as e:
                logger.error(f"Error in detector {detector.name}: {e}", exc_info=True)

        return signals

    # ── Self-contained indicator computation ──

    def _compute_indicators(self, bar: BarInput):
        """Compute all indicators from raw bar history."""
        ind = self._indicator_state
        bars = self._bars_history
        n = len(bars)

        # Session info
        ind.is_rth = _is_rth(bar.time)
        et_min = _bar_to_et_minutes(bar.time)
        ind.session_minutes = et_min - RTH_OPEN_MINUTES if ind.is_rth else -1
        ind.session = _session_label(ind.session_minutes)

        # Need minimum bars for indicators
        if n < 20:
            return

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # EMA-9
        ind.ema9 = _ema(closes[-50:], 9) if n >= 9 else None

        # EMA-20
        ind.ema20 = _ema(closes[-60:], 20) if n >= 20 else None

        # RSI-14
        ind.rsi14_prev = ind.rsi14
        ind.rsi14 = _compute_rsi(closes, 14) if n >= 16 else None

        # ATR-14
        ind.atr14 = _compute_atr(bars, 14) if n >= 15 else None

        # ADX-14
        if n >= 30:
            adx, plus_di, minus_di = _compute_adx(bars, 14)
            ind.adx14 = adx
            ind.plus_di = plus_di
            ind.minus_di = minus_di

        # Volume SMA-20
        ind.volume_sma20 = _sma(volumes, 20) if n >= 20 else None

        # Bollinger Bands (20, 2)
        bb_sma = _sma(closes, 20)
        bb_std = _stdev(closes, 20)
        if bb_sma is not None and bb_std is not None:
            ind.bb_middle = bb_sma
            ind.bb_upper = bb_sma + 2 * bb_std
            ind.bb_lower = bb_sma - 2 * bb_std

        # Keltner Channels (20, 1.5 * ATR)
        kc_mid = _ema(closes[-60:], 20) if n >= 20 else None
        kc_atr = ind.atr14
        if kc_mid is not None and kc_atr is not None:
            ind.kc_middle = kc_mid
            ind.kc_upper = kc_mid + 1.5 * kc_atr
            ind.kc_lower = kc_mid - 1.5 * kc_atr

        # TTM Squeeze: BB inside KC
        if (ind.bb_upper is not None and ind.kc_upper is not None):
            ind.squeeze_on = (ind.bb_upper < ind.kc_upper and ind.bb_lower > ind.kc_lower)
            # Momentum: close - midline of (highest high + lowest low)/2 + SMA20 over 20 bars
            if n >= 20:
                hh = max(b.high for b in bars[-20:])
                ll = min(b.low for b in bars[-20:])
                midline = (hh + ll) / 2
                if bb_sma is not None:
                    avg_ml = (midline + bb_sma) / 2
                    ind.squeeze_momentum = bar.close - avg_ml

        # VWAP (reset each RTH day)
        self._update_vwap(bar)

        # PDH/PDL/PDC
        self._update_daily_levels(bar)

        # Overnight H/L
        self._update_overnight_levels(bar)

        # Opening Range
        self._update_opening_range(bar)

    def _update_vwap(self, bar: BarInput):
        """Session-anchored VWAP computation."""
        ind = self._indicator_state
        if not ind.is_rth:
            return

        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._vwap_date:
            self._vwap_date = today
            self._vwap_cum_vol = 0.0
            self._vwap_cum_pv = 0.0
            self._vwap_cum_pv2 = 0.0

        typical = (bar.high + bar.low + bar.close) / 3.0
        self._vwap_cum_vol += bar.volume
        self._vwap_cum_pv += typical * bar.volume
        self._vwap_cum_pv2 += typical * typical * bar.volume

        if self._vwap_cum_vol > 0:
            ind.vwap = self._vwap_cum_pv / self._vwap_cum_vol
            variance = (self._vwap_cum_pv2 / self._vwap_cum_vol) - (ind.vwap ** 2)
            ind.vwap_std = math.sqrt(max(variance, 0))
            ind.vwap_upper2 = ind.vwap + 2 * ind.vwap_std
            ind.vwap_lower2 = ind.vwap - 2 * ind.vwap_std

    def _update_daily_levels(self, bar: BarInput):
        """Track PDH/PDL/PDC from daily OHLC."""
        ind = self._indicator_state
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')

        if today != self._cur_day_date:
            # New day — previous day becomes PDH/PDL
            if self._cur_day_high is not None:
                self._prev_day_high = self._cur_day_high
                self._prev_day_low = self._cur_day_low
                self._prev_day_close = self._cur_day_close
            self._cur_day_date = today
            self._cur_day_high = bar.high
            self._cur_day_low = bar.low
            self._cur_day_close = bar.close
        else:
            if self._cur_day_high is not None:
                self._cur_day_high = max(self._cur_day_high, bar.high)
                self._cur_day_low = min(self._cur_day_low, bar.low)
                self._cur_day_close = bar.close

        ind.pdh = self._prev_day_high
        ind.pdl = self._prev_day_low
        ind.pdc = self._prev_day_close

    def _update_overnight_levels(self, bar: BarInput):
        """Track overnight session high/low (16:00 - 9:30 ET)."""
        ind = self._indicator_state
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')

        if ind.is_rth:
            # During RTH, expose ON levels
            if today != self._on_date:
                self._on_date = today
            ind.onh = self._on_high
            ind.onl = self._on_low
        else:
            # Overnight session — accumulate
            if today != self._on_date:
                self._on_date = today
                self._on_high = bar.high
                self._on_low = bar.low
            else:
                if self._on_high is not None:
                    self._on_high = max(self._on_high, bar.high)
                    self._on_low = min(self._on_low, bar.low)

    def _update_opening_range(self, bar: BarInput):
        """Track opening range (first 15 min of RTH)."""
        ind = self._indicator_state
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')

        if not ind.is_rth:
            return

        if today != self._or_date:
            self._or_date = today
            self._or_high = None
            self._or_low = None
            self._or_complete = False

        if ind.session_minutes < 15:
            if self._or_high is None:
                self._or_high = bar.high
                self._or_low = bar.low
            else:
                self._or_high = max(self._or_high, bar.high)
                self._or_low = min(self._or_low, bar.low)
        else:
            self._or_complete = True

        ind.orh = self._or_high
        ind.orl = self._or_low
        ind.or_complete = self._or_complete

    def _merge_external_indicators(self, indicators: dict):
        """Override computed values with externally provided ones if available."""
        ind = self._indicator_state
        for key in ['ema9', 'rsi14', 'adx14', 'plus_di', 'minus_di',
                     'atr14', 'volume_sma20', 'vwap']:
            val = indicators.get(key)
            if val is not None:
                setattr(ind, key, val)
        # Also accept frontend naming conventions
        for src, dst in [('atr10', 'atr14'), ('rsi7', 'rsi14'),
                         ('adx10', 'adx14'), ('plusDI', 'plus_di'),
                         ('minusDI', 'minus_di')]:
            val = indicators.get(src)
            if val is not None:
                setattr(ind, dst, val)

    def _merge_external_levels(self, levels: dict):
        ind = self._indicator_state
        for key in ['pdh', 'pdl', 'pdc', 'onh', 'onl', 'orh', 'orl']:
            val = levels.get(key)
            if val is not None:
                setattr(ind, key, val)

    def get_detector_info(self) -> List[dict]:
        return [d.get_info() for d in self.detectors]

    def reset_all(self):
        for d in self.detectors:
            d.reset()
        self._bars_history.clear()
        logger.info("SetupManager: all detectors reset")
