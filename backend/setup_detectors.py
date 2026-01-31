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
        """Register all Phase 1 setup detectors."""
        self.register(ORBBreakoutDetector())
        self.register(VWAPMeanReversionDetector())
        self.register(PDHPDLBreakoutDetector())
        self.register(EMA9PullbackDetector())
        self.register(TTMSqueezeDetector())
        self.register(ONHLSweepDetector())
        self.register(VolumeSpikeBreakoutDetector())
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
