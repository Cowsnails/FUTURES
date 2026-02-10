"""
Setup Detectors Framework — Decision Tree Architecture

Base classes and manager for running multiple intraday setup detectors
simultaneously. Each detector uses a weighted decision tree scoring model
with regime classification, session timing, and evidence-tier confidence caps.

All detection runs server-side. Results appear on the stats page only.

Includes self-contained indicator computation so detectors don't depend
on external indicator sources — everything is computed from raw bar history.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Eastern timezone offset helpers
ET_OFFSET = timedelta(hours=-5)  # EST (no DST handling — good enough for RTH)
RTH_OPEN_MINUTES = 9 * 60 + 30   # 9:30 ET in minutes-from-midnight
RTH_CLOSE_MINUTES = 16 * 60      # 16:00 ET


# ═══════════════════════════════════════════════════════════════════════════
# MARKET REGIME CLASSIFICATION (Wilder 1978, Dreiss 1992)
# ═══════════════════════════════════════════════════════════════════════════

class MarketRegime(Enum):
    TRENDING_STRONG = "trending_strong"       # ADX > 30, CI < 38.2
    TRENDING_MODERATE = "trending_moderate"   # ADX 25-30
    RANGING = "ranging"                       # ADX < 20, CI 50-61.8
    CHOPPY = "choppy"                         # ADX < 20, CI > 61.8
    VOLATILE_EXPANSION = "volatile_expansion" # ATR ratio > 1.5
    QUIET_COMPRESSION = "quiet_compression"   # ATR ratio < 0.75
    UNKNOWN = "unknown"


def classify_regime(adx: Optional[float], chop: Optional[float],
                    atr_ratio: Optional[float]) -> MarketRegime:
    """Classify current market regime from ADX, Choppiness Index, ATR ratio."""
    if adx is None:
        return MarketRegime.UNKNOWN

    # Volatility regime overrides if extreme
    if atr_ratio is not None:
        if atr_ratio > 1.5:
            return MarketRegime.VOLATILE_EXPANSION
        if atr_ratio < 0.75:
            return MarketRegime.QUIET_COMPRESSION

    ci = chop if chop is not None else 50.0  # neutral default

    if adx > 30 and ci < 38.2:
        return MarketRegime.TRENDING_STRONG
    if 25 <= adx <= 30:
        return MarketRegime.TRENDING_MODERATE
    if adx < 20 and ci > 61.8:
        return MarketRegime.CHOPPY
    if adx < 20:
        return MarketRegime.RANGING

    return MarketRegime.TRENDING_MODERATE  # fallback


# ═══════════════════════════════════════════════════════════════════════════
# SESSION TIMING MULTIPLIERS
# ═══════════════════════════════════════════════════════════════════════════

# category -> session -> multiplier
SESSION_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "breakout": {
        "rth_open": 1.0, "am_drive": 0.9, "am_session": 0.7,
        "lunch": 0.4, "pm_session": 0.7, "moc": 0.8,
        "pre_market": 0.3, "post_market": 0.2, "globex": 0.5,
    },
    "mean_reversion": {
        "rth_open": 0.5, "am_drive": 0.7, "am_session": 0.8,
        "lunch": 1.0, "pm_session": 0.8, "moc": 0.6,
        "pre_market": 0.3, "post_market": 0.2, "globex": 0.6,
    },
    "momentum": {
        "rth_open": 1.0, "am_drive": 1.0, "am_session": 0.8,
        "lunch": 0.3, "pm_session": 0.7, "moc": 0.6,
        "pre_market": 0.3, "post_market": 0.2, "globex": 0.4,
    },
    "level": {
        "rth_open": 0.9, "am_drive": 1.0, "am_session": 0.8,
        "lunch": 0.5, "pm_session": 0.8, "moc": 0.7,
        "pre_market": 0.3, "post_market": 0.2, "globex": 0.5,
    },
    "volatility": {
        "rth_open": 1.0, "am_drive": 0.9, "am_session": 0.7,
        "lunch": 0.3, "pm_session": 0.6, "moc": 0.5,
        "pre_market": 0.4, "post_market": 0.3, "globex": 0.4,
    },
    "micro": {
        "rth_open": 0.8, "am_drive": 0.9, "am_session": 1.0,
        "lunch": 0.6, "pm_session": 0.8, "moc": 0.5,
        "pre_market": 0.3, "post_market": 0.2, "globex": 0.5,
    },
    "vwap": {
        "rth_open": 0.7, "am_drive": 0.9, "am_session": 1.0,
        "lunch": 0.8, "pm_session": 0.7, "moc": 0.5,
        "pre_market": 0.2, "post_market": 0.2, "globex": 0.4,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE TIERS & CONFIDENCE CAPS
# ═══════════════════════════════════════════════════════════════════════════

class EvidenceTier(Enum):
    TIER1 = 1  # cap 0.85
    TIER2 = 2  # cap 0.70
    TIER3 = 3  # cap 0.55
    TIER4 = 4  # cap 0.40

TIER_CAPS = {
    EvidenceTier.TIER1: 0.85,
    EvidenceTier.TIER2: 0.70,
    EvidenceTier.TIER3: 0.55,
    EvidenceTier.TIER4: 0.40,
}


def confidence_to_bracket(confidence: float, atr: float) -> Dict[str, Any]:
    """Map confidence score to trade bracket parameters."""
    if confidence < 0.40:
        return {"trade": False, "stop_atr": 0, "rr": 0, "max_bars": 0}
    if confidence < 0.55:
        return {"trade": True, "stop_atr": 2.0, "rr": 1.0, "max_bars": 10}
    if confidence < 0.70:
        return {"trade": True, "stop_atr": 1.5, "rr": 1.5, "max_bars": 20}
    if confidence < 0.85:
        return {"trade": True, "stop_atr": 1.0, "rr": 2.0, "max_bars": 30}
    return {"trade": True, "stop_atr": 0.75, "rr": 2.5, "max_bars": 40}


# ═══════════════════════════════════════════════════════════════════════════
# DECISION TREE WEIGHT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

W_PRICE_ACTION = 0.35
W_VOLUME = 0.20
W_MOMENTUM = 0.15
W_REGIME = 0.20
W_SESSION = 0.10


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
    """Pre-computed indicators shared across all detectors."""
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

    # NEW: Decision tree indicators
    chop14: Optional[float] = None    # Choppiness Index (Dreiss)
    atr5: Optional[float] = None      # Short-term ATR
    atr50: Optional[float] = None     # Long-term ATR
    atr_ratio: Optional[float] = None # atr5/atr50
    regime: MarketRegime = MarketRegime.UNKNOWN

    # Bollinger Bands (20, 2)
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None

    # Keltner Channels (20, 1.5)
    kc_upper: Optional[float] = None
    kc_middle: Optional[float] = None
    kc_lower: Optional[float] = None

    # TTM Squeeze
    squeeze_on: bool = False
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

    # Opening Range
    orh: Optional[float] = None
    orl: Optional[float] = None
    or_complete: bool = False

    # Session
    is_rth: bool = False
    session_minutes: int = -1
    session: str = "pre_market"


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS (self-contained, no external deps)
# ═══════════════════════════════════════════════════════════════════════════

def _sma(values: list, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: list, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for val in values[period:]:
        ema = (val - ema) * multiplier + ema
    return ema


def _stdev(values: list, period: int) -> Optional[float]:
    if len(values) < period:
        return None
    subset = values[-period:]
    mean = sum(subset) / period
    variance = sum((x - mean) ** 2 for x in subset) / period
    return math.sqrt(variance)


def _compute_rsi(closes: list, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        delta = closes[-period - 1 + i] - closes[-period - 1 + i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_atr(bars: list, period: int = 14) -> Optional[float]:
    if len(bars) < period + 1:
        return None
    trs = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def _compute_adx(bars: list, period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(bars) < period * 2 + 1:
        return None, None, None
    plus_dms, minus_dms, trs = [], [], []
    for i in range(1, len(bars)):
        h, l = bars[i].high, bars[i].low
        ph, pl = bars[i - 1].high, bars[i - 1].low
        plus_dm = max(h - ph, 0) if (h - ph) > (pl - l) else 0
        minus_dm = max(pl - l, 0) if (pl - l) > (h - ph) else 0
        tr = max(h - l, abs(h - bars[i - 1].close), abs(l - bars[i - 1].close))
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)
        trs.append(tr)
    if len(trs) < period:
        return None, None, None
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
    if smoothed_tr == 0:
        return adx, 0, 0
    final_plus_di = 100 * smoothed_plus / smoothed_tr
    final_minus_di = 100 * smoothed_minus / smoothed_tr
    return adx, final_plus_di, final_minus_di


def _compute_choppiness(bars: list, period: int = 14) -> Optional[float]:
    """Choppiness Index (Dreiss 1992): 100 * LOG10(sum_atr / (HH-LL)) / LOG10(period)."""
    if len(bars) < period + 1:
        return None
    recent = bars[-(period + 1):]
    atr_sum = 0.0
    for i in range(1, len(recent)):
        h, l, pc = recent[i].high, recent[i].low, recent[i - 1].close
        atr_sum += max(h - l, abs(h - pc), abs(l - pc))
    hh = max(b.high for b in recent[1:])
    ll = min(b.low for b in recent[1:])
    hl_range = hh - ll
    if hl_range <= 0:
        return 50.0
    return 100 * math.log10(atr_sum / hl_range) / math.log10(period)


def _bar_to_et_minutes(bar_time: int) -> int:
    dt = datetime.fromtimestamp(bar_time, tz=timezone.utc) + ET_OFFSET
    return dt.hour * 60 + dt.minute


def _is_rth(bar_time: int) -> bool:
    m = _bar_to_et_minutes(bar_time)
    return RTH_OPEN_MINUTES <= m < RTH_CLOSE_MINUTES


def _session_label(minutes_since_open: int, is_rth: bool = True) -> str:
    """Label session for futures 23/5 trading.
    Covers both RTH (9:30-16:00 ET) and globex/overnight."""
    if is_rth:
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
    # Globex / overnight session
    return "globex"


# ═══════════════════════════════════════════════════════════════════════════
# BASE DETECTOR CLASS — Decision Tree Scoring
# ═══════════════════════════════════════════════════════════════════════════

class SetupDetector:
    """
    Base class with weighted decision tree scoring.

    Subclasses implement:
      - score_price_action(bar, ind, bars) -> float 0-1
      - score_volume(bar, ind, bars) -> float 0-1
      - score_momentum(bar, ind, bars) -> float 0-1
      - regime_scores() -> dict[MarketRegime, float]
      - disabled_regimes() -> set of MarketRegime where confidence=0
      - detect_direction(bar, ind, bars) -> Optional[str] "LONG"/"SHORT" or None
    """

    name: str = "base"
    display_name: str = "Base Setup"
    category: str = "unknown"
    hold_time: str = "unknown"
    evidence_tier: EvidenceTier = EvidenceTier.TIER3
    min_cooldown_seconds: int = 60

    def __init__(self):
        self._last_signal_time: int = 0
        self._signal_counter: int = 0
        self._enabled: bool = True

    def can_signal(self, now: int) -> bool:
        if not self._enabled:
            return False
        return (now - self._last_signal_time) >= self.min_cooldown_seconds

    def record_signal(self, now: int):
        self._last_signal_time = now
        self._signal_counter += 1

    def reset(self):
        self._last_signal_time = 0
        self._signal_counter = 0

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "hold_time": self.hold_time,
            "evidence_tier": self.evidence_tier.value,
            "enabled": self._enabled,
            "signal_count": self._signal_counter,
        }

    # ── Decision tree scoring ──

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        return 0.0

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        return 0.5  # neutral default

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        return 0.5

    def regime_scores(self) -> Dict[MarketRegime, float]:
        """Return 0-1 score for each regime. Higher = better fit."""
        return {r: 0.5 for r in MarketRegime}

    def disabled_regimes(self) -> set:
        """Regimes where this setup should NOT fire (confidence forced to 0)."""
        return set()

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        """Return 'LONG', 'SHORT', or None if no setup present."""
        return None

    def compute_confidence(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        """Weighted decision tree confidence scoring."""
        # Check regime disable
        if ind.regime in self.disabled_regimes():
            return 0.0

        pa = self.score_price_action(bar, ind, bars)
        vol = self.score_volume(bar, ind, bars)
        mom = self.score_momentum(bar, ind, bars)

        # Regime score
        r_scores = self.regime_scores()
        regime_score = r_scores.get(ind.regime, 0.5)

        # Session multiplier
        sess_mults = SESSION_MULTIPLIERS.get(self.category, SESSION_MULTIPLIERS["momentum"])
        session_score = sess_mults.get(ind.session, 0.3)

        # Weighted sum
        raw = (pa * W_PRICE_ACTION + vol * W_VOLUME + mom * W_MOMENTUM +
               regime_score * W_REGIME + session_score * W_SESSION)

        # Cap by evidence tier
        cap = TIER_CAPS[self.evidence_tier]
        return min(raw, cap)

    def make_signal_from_confidence(self, direction: str, bar: BarInput,
                                     ind: IndicatorState, confidence: float,
                                     reason: str) -> Optional[SetupSignal]:
        """Build a SetupSignal using confidence-to-bracket mapping."""
        atr = ind.atr14
        if not atr or atr <= 0:
            return None

        bracket = confidence_to_bracket(confidence, atr)
        if not bracket["trade"]:
            return None

        stop_dist = bracket["stop_atr"] * atr
        target_dist = stop_dist * bracket["rr"]

        if direction == "LONG":
            stop = bar.close - stop_dist
            target = bar.close + target_dist
        else:
            stop = bar.close + stop_dist
            target = bar.close - target_dist

        return SetupSignal(
            setup_name=self.name,
            direction=direction,
            entry_price=bar.close,
            stop_price=stop,
            target_price=target,
            atr=atr,
            stop_atr_mult=bracket["stop_atr"],
            target_atr_mult=bracket["stop_atr"] * bracket["rr"],
            max_bars=bracket["max_bars"],
            confidence=confidence,
            reason=reason,
            bar_time=bar.time,
            indicator_snapshot={
                "regime": ind.regime.value,
                "session": ind.session,
                "adx": ind.adx14,
                "chop": ind.chop14,
                "atr_ratio": ind.atr_ratio,
            }
        )

    # Legacy make_signal for compatibility
    def make_signal(self, direction, bar, ind, stop_price, target_price,
                    stop_atr_mult, target_atr_mult, max_bars, confidence, reason):
        atr = ind.atr14 or 1.0
        return SetupSignal(
            setup_name=self.name, direction=direction,
            entry_price=bar.close, stop_price=stop_price,
            target_price=target_price, atr=atr,
            stop_atr_mult=stop_atr_mult, target_atr_mult=target_atr_mult,
            max_bars=max_bars, confidence=confidence,
            reason=reason, bar_time=bar.time,
        )

    def readiness_score(self, bar: BarInput, ind: IndicatorState,
                        bars: List[BarInput]) -> float:
        """
        Confluence meter: 0-100 showing how close this setup is to triggering.
        Computes weighted confidence even when direction isn't confirmed yet.
        """
        if ind.regime in self.disabled_regimes():
            return 0.0

        pa = self.score_price_action(bar, ind, bars)
        vol = self.score_volume(bar, ind, bars)
        mom = self.score_momentum(bar, ind, bars)

        r_scores = self.regime_scores()
        regime_score = r_scores.get(ind.regime, 0.5)

        sess_mults = SESSION_MULTIPLIERS.get(self.category, SESSION_MULTIPLIERS["momentum"])
        session_score = sess_mults.get(ind.session, 0.3)

        raw = (pa * W_PRICE_ACTION + vol * W_VOLUME + mom * W_MOMENTUM +
               regime_score * W_REGIME + session_score * W_SESSION)

        return min(100.0, raw * 100.0)

    def update(self, bar: BarInput, indicators: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        """Main entry: detect direction, score confidence, emit signal."""
        direction = self.detect_direction(bar, indicators, bars_history)
        if direction is None:
            return None

        confidence = self.compute_confidence(bar, indicators, bars_history)
        if confidence < 0.40:
            return None

        reason = f"{self.display_name} {direction} (conf={confidence:.2f}, regime={indicators.regime.value}, session={indicators.session})"
        return self.make_signal_from_confidence(direction, bar, indicators, confidence, reason)


# ═══════════════════════════════════════════════════════════════════════════
# DETECTORS — Each one will be added incrementally below
# ═══════════════════════════════════════════════════════════════════════════

# ── Setup 1: ORB Breakout (Tier 1) ──────────────────────────────────────

class ORBBreakoutDetector(SetupDetector):
    """Opening Range Breakout — Tier 1 evidence. Best in trending regimes during AM."""
    name = "orb_breakout"
    display_name = "ORB Breakout"
    category = "breakout"
    hold_time = "15-60min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_STRONG: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.VOLATILE_EXPANSION: 0.7,
            MarketRegime.RANGING: 0.4,
            MarketRegime.QUIET_COMPRESSION: 0.6,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.or_complete or ind.orh is None or ind.orl is None:
            return None
        # During RTH, only trade first 2 hours after open; globex = always allowed
        if ind.is_rth and (ind.session_minutes < 15 or ind.session_minutes > 120):
            return None
        atr = ind.atr14
        if not atr or atr <= 0:
            return None
        or_range = ind.orh - ind.orl
        if or_range < 0.3 * atr or or_range > 3.0 * atr:
            return None
        if bar.close > ind.orh and bar.close > bar.open:
            return "LONG"
        if bar.close < ind.orl and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if ind.orh is None or ind.orl is None or not ind.atr14:
            return 0.0
        or_range = ind.orh - ind.orl
        # Clean break: close well beyond OR level
        if bar.close > ind.orh:
            penetration = (bar.close - ind.orh) / ind.atr14
        else:
            penetration = (ind.orl - bar.close) / ind.atr14
        # Strong candle body vs wick
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        # Score: penetration quality + candle quality
        pen_score = min(penetration / 0.5, 1.0)  # 0.5 ATR penetration = max
        return min(1.0, pen_score * 0.6 + body_ratio * 0.4)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        if ratio >= 2.0:
            return 1.0
        if ratio >= 1.5:
            return 0.8
        if ratio >= 1.0:
            return 0.6
        return 0.3

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        # EMA alignment
        if ind.ema9 and ind.ema20:
            if ind.orh is not None and bar.close > ind.orh and ind.ema9 > ind.ema20:
                score += 0.2
            elif ind.orl is not None and bar.close < ind.orl and ind.ema9 < ind.ema20:
                score += 0.2
        # ADX confirmation
        if ind.adx14 and ind.adx14 > 25:
            score += 0.15
        # RSI not extreme against direction
        if ind.rsi14:
            if bar.close > (ind.orh or 0) and ind.rsi14 < 75:
                score += 0.1
            elif bar.close < (ind.orl or 0) and ind.rsi14 > 25:
                score += 0.1
        return min(1.0, score)


# ── Setup 2: VWAP Mean Reversion (Tier 3) ──────────────────────────────

class VWAPMeanReversionDetector(SetupDetector):
    """VWAP Mean Reversion — fade extended moves back to VWAP. Best in ranging/lunch."""
    name = "vwap_mean_reversion"
    display_name = "VWAP Mean Reversion"
    category = "mean_reversion"
    hold_time = "5-30min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 1.0,
            MarketRegime.CHOPPY: 0.6,
            MarketRegime.TRENDING_MODERATE: 0.4,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.QUIET_COMPRESSION: 0.7,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.vwap is None or ind.vwap_std is None or not ind.atr14:
            return None
        if ind.is_rth and ind.session_minutes < 30:
            return None
        if ind.vwap_std <= 0:
            return None
        dist = bar.close - ind.vwap
        band = ind.vwap_std * 2
        # Extended above upper band, reversal candle
        if dist > band and bar.close < bar.open:
            return "SHORT"
        if dist < -band and bar.close > bar.open:
            return "LONG"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.vwap_std or ind.vwap_std <= 0 or not ind.vwap:
            return 0.0
        dist = abs(bar.close - ind.vwap) / ind.vwap_std
        # 2-3 std = good, 3+ = excellent
        if dist >= 3.0:
            ext_score = 1.0
        elif dist >= 2.0:
            ext_score = 0.6 + (dist - 2.0) * 0.4
        else:
            ext_score = 0.3
        # Reversal candle quality
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, ext_score * 0.6 + body_ratio * 0.4)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        # For MR, declining volume on extension is good, spike on reversal is good
        if ratio >= 1.5:
            return 0.8
        if ratio >= 1.0:
            return 0.6
        return 0.4

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.rsi14:
            if bar.close < ind.vwap and ind.rsi14 < 30:
                score += 0.3  # Oversold for long
            elif bar.close > ind.vwap and ind.rsi14 > 70:
                score += 0.3  # Overbought for short
        if ind.bb_upper and ind.bb_lower:
            if bar.close > ind.bb_upper or bar.close < ind.bb_lower:
                score += 0.15  # Outside BB
        return min(1.0, score)


# ── Setup 3: PDH/PDL Breakout (Tier 2) ─────────────────────────────────

class PDHPDLBreakoutDetector(SetupDetector):
    """Previous Day High/Low Breakout — Tier 2. Best in trending AM sessions."""
    name = "pdh_pdl_breakout"
    display_name = "PDH/PDL Breakout"
    category = "breakout"
    hold_time = "15-60min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY, MarketRegime.RANGING}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_STRONG: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.VOLATILE_EXPANSION: 0.6,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.RANGING: 0.0,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or ind.atr14 <= 0:
            return None
        if ind.pdh is None or ind.pdl is None:
            return None
        if len(bars) < 3:
            return None
        prev = bars[-2]
        # Long: close breaks above PDH, prev bar was below
        if bar.close > ind.pdh and prev.close <= ind.pdh and bar.close > bar.open:
            return "LONG"
        # Short: close breaks below PDL
        if bar.close < ind.pdl and prev.close >= ind.pdl and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or ind.pdh is None or ind.pdl is None:
            return 0.0
        # How far through the level
        if bar.close > ind.pdh:
            pen = (bar.close - ind.pdh) / ind.atr14
        else:
            pen = (ind.pdl - bar.close) / ind.atr14
        pen_score = min(pen / 0.4, 1.0)
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, pen_score * 0.5 + body_ratio * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        if ratio >= 2.0:
            return 1.0
        if ratio >= 1.3:
            return 0.7
        return 0.4

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.ema9 and ind.ema20:
            if ind.pdh is not None and bar.close > ind.pdh and ind.ema9 > ind.ema20:
                score += 0.2
            elif ind.pdl is not None and bar.close < ind.pdl and ind.ema9 < ind.ema20:
                score += 0.2
        if ind.adx14 and ind.adx14 > 25:
            score += 0.15
        if ind.rsi14:
            if ind.pdh is not None and bar.close > ind.pdh and 50 < ind.rsi14 < 75:
                score += 0.1
            elif ind.pdl is not None and bar.close < ind.pdl and 25 < ind.rsi14 < 50:
                score += 0.1
        return min(1.0, score)


# ── Setup 4: EMA Pullback / Holy Grail (Tier 1) ────────────────────────

class EMA20PullbackDetector(SetupDetector):
    """Holy Grail (Raschke) — EMA-20 pullback in strong trend. ADX > 30, EMA-20 per original specs."""
    name = "ema20_pullback"
    display_name = "Holy Grail (EMA-20)"
    category = "momentum"
    hold_time = "10-40min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY, MarketRegime.RANGING}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_STRONG: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.VOLATILE_EXPANSION: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.3,
            MarketRegime.RANGING: 0.0,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.4,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.ema20 or not ind.adx14 or not ind.atr14:
            return None
        # Raschke original: ADX > 30
        if ind.adx14 < 30:
            return None
        if len(bars) < 5:
            return None
        # Long: uptrend (price above EMA-20 recently), pullback touches EMA-20, bounce
        above_count = sum(1 for b in bars[-10:-1] if b.close > ind.ema20)
        if above_count >= 7:
            # Pullback: low touched EMA-20
            if bar.low <= ind.ema20 * 1.002 and bar.close > ind.ema20 and bar.close > bar.open:
                return "LONG"
        below_count = sum(1 for b in bars[-10:-1] if b.close < ind.ema20)
        if below_count >= 7:
            if bar.high >= ind.ema20 * 0.998 and bar.close < ind.ema20 and bar.close < bar.open:
                return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.ema20 or not ind.atr14:
            return 0.0
        # How precisely did price touch EMA-20?
        touch_dist = abs(bar.low - ind.ema20) / ind.atr14 if bar.close > ind.ema20 else abs(bar.high - ind.ema20) / ind.atr14
        precision = max(0, 1.0 - touch_dist * 3)  # Closer = better
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, precision * 0.5 + body_ratio * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        # Volume drying up on pullback then expanding on bounce = ideal
        if ratio >= 1.3:
            return 0.8
        if ratio >= 0.8:
            return 0.6
        return 0.4

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.adx14:
            if ind.adx14 > 35:
                score += 0.25
            elif ind.adx14 > 30:
                score += 0.15
        if ind.plus_di and ind.minus_di:
            if bar.close > ind.ema20 and ind.plus_di > ind.minus_di:
                score += 0.15
            elif bar.close < ind.ema20 and ind.minus_di > ind.plus_di:
                score += 0.15
        return min(1.0, score)


# ── Setup 5: TTM Squeeze (Tier 3) ──────────────────────────────────────

class TTMSqueezeDetector(SetupDetector):
    """TTM Squeeze — BB inside KC then fires with momentum. Volatility breakout."""
    name = "ttm_squeeze"
    display_name = "TTM Squeeze"
    category = "volatility"
    hold_time = "10-40min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._prev_squeeze_on: bool = False

    def reset(self):
        super().reset()
        self._prev_squeeze_on = False

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.QUIET_COMPRESSION: 1.0,
            MarketRegime.RANGING: 0.7,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        # Squeeze must have just fired (was on, now off)
        was_on = self._prev_squeeze_on
        self._prev_squeeze_on = ind.squeeze_on
        if not was_on or ind.squeeze_on:
            return None
        if ind.squeeze_momentum is None:
            return None
        if ind.squeeze_momentum > 0:
            return "LONG"
        if ind.squeeze_momentum < 0:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if ind.squeeze_momentum is None or not ind.atr14:
            return 0.0
        # Momentum magnitude relative to ATR
        mom_mag = abs(ind.squeeze_momentum) / ind.atr14
        mom_score = min(mom_mag / 1.0, 1.0)
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, mom_score * 0.6 + body_ratio * 0.4)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        if ratio >= 1.5:
            return 0.9
        if ratio >= 1.0:
            return 0.6
        return 0.3

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.ema9 and ind.ema20:
            if ind.squeeze_momentum and ind.squeeze_momentum > 0 and ind.ema9 > ind.ema20:
                score += 0.2
            elif ind.squeeze_momentum and ind.squeeze_momentum < 0 and ind.ema9 < ind.ema20:
                score += 0.2
        if ind.adx14 and ind.adx14 > 20:
            score += 0.15
        return min(1.0, score)


# ── Setup 6: ON H/L Sweep (Tier 2) ─────────────────────────────────────

class ONHLSweepDetector(SetupDetector):
    """Overnight High/Low Sweep — sweep ON level then reverse. Tier 2."""
    name = "on_hl_sweep"
    display_name = "ON H/L Sweep"
    category = "level"
    hold_time = "10-40min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.VOLATILE_EXPANSION: 0.6,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.onh is None or ind.onl is None or not ind.atr14:
            return None
        if ind.is_rth and (ind.session_minutes < 5 or ind.session_minutes > 90):
            return None
        atr = ind.atr14
        # Sweep ONH then reject (close back below)
        if bar.high > ind.onh and bar.close < ind.onh - 0.1 * atr and bar.close < bar.open:
            return "SHORT"
        # Sweep ONL then reject (close back above)
        if bar.low < ind.onl and bar.close > ind.onl + 0.1 * atr and bar.close > bar.open:
            return "LONG"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or ind.onh is None or ind.onl is None:
            return 0.0
        # Wick through level = sweep quality
        if bar.high > ind.onh:
            wick = bar.high - ind.onh
            rejection = ind.onh - bar.close
        elif bar.low < ind.onl:
            wick = ind.onl - bar.low
            rejection = bar.close - ind.onl
        else:
            # Price between ONH and ONL, no sweep
            return 0.0
        # Clamp to 0-1 range
        wick_score = max(0.0, min(wick / (0.5 * ind.atr14), 1.0))
        rej_score = max(0.0, min(rejection / (0.3 * ind.atr14), 1.0))
        return min(1.0, wick_score * 0.4 + rej_score * 0.4 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        if ratio >= 1.5:
            return 0.9
        if ratio >= 1.0:
            return 0.6
        return 0.4

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.rsi14:
            if ind.onl is not None and bar.close > ind.onl and ind.rsi14 < 40:
                score += 0.2
            elif ind.onh is not None and bar.close < ind.onh and ind.rsi14 > 60:
                score += 0.2
        return min(1.0, score)


# ── Setup 7: Volume Spike Breakout (Tier 3) ────────────────────────────

class VolumeSpikeBreakoutDetector(SetupDetector):
    """Volume Spike Breakout — 2x+ volume with range breakout. Tier 3."""
    name = "volume_spike_breakout"
    display_name = "Volume Spike Breakout"
    category = "momentum"
    hold_time = "5-20min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_STRONG: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.VOLATILE_EXPANSION: 0.7,
            MarketRegime.QUIET_COMPRESSION: 0.6,
            MarketRegime.RANGING: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or ind.atr14 <= 0:
            return None
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return None
        if bar.volume < ind.volume_sma20 * 2.0:
            return None
        if len(bars) < 7:
            return None
        lookback = bars[-6:-1]
        rh = max(b.high for b in lookback)
        rl = min(b.low for b in lookback)
        rng = rh - rl
        if rng > 2.5 * ind.atr14 or rng < 0.3 * ind.atr14:
            return None
        if bar.close > rh and bar.close > bar.open:
            return "LONG"
        if bar.close < rl and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or len(bars) < 7:
            return 0.0
        lookback = bars[-6:-1]
        rh = max(b.high for b in lookback)
        rl = min(b.low for b in lookback)
        if bar.close > rh:
            pen = (bar.close - rh) / ind.atr14
        else:
            pen = (rl - bar.close) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, min(pen / 0.4, 1.0) * 0.5 + body_ratio * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        if ratio >= 3.0:
            return 1.0
        if ratio >= 2.0:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.ema9 and ind.ema20:
            if bar.close > bar.open and ind.ema9 > ind.ema20:
                score += 0.2
            elif bar.close < bar.open and ind.ema9 < ind.ema20:
                score += 0.2
        if ind.adx14 and ind.adx14 > 20:
            score += 0.15
        return min(1.0, score)


# ── Setup 8: VWAP Breakout & Retest (Tier 4) ───────────────────────────

class VWAPBreakoutRetestDetector(SetupDetector):
    """VWAP Breakout & Retest — cross VWAP with volume, pullback, rejection. Tier 4."""
    name = "vwap_breakout_retest"
    display_name = "VWAP Breakout & Retest"
    category = "vwap"
    hold_time = "5-30min"
    evidence_tier = EvidenceTier.TIER4
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._breakout_dir: Optional[str] = None
        self._awaiting_retest: bool = False
        self._bars_since_breakout: int = 0

    def reset(self):
        super().reset()
        self._breakout_dir = None
        self._awaiting_retest = False
        self._bars_since_breakout = 0

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.TRENDING_STRONG: 0.6,
            MarketRegime.RANGING: 0.7,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.VOLATILE_EXPANSION: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.vwap is None or not ind.atr14:
            return None
        if ind.is_rth and ind.session_minutes < 15:
            return None
        vwap = ind.vwap
        atr = ind.atr14
        # Timeout
        if self._awaiting_retest:
            self._bars_since_breakout += 1
            if self._bars_since_breakout > 15:
                self._awaiting_retest = False
                self._breakout_dir = None
        # Step 1: detect breakout
        if not self._awaiting_retest and len(bars) >= 3:
            prev1, prev2 = bars[-2], bars[-3]
            vol_ok = ind.volume_sma20 and bar.volume > ind.volume_sma20 * 1.5
            if prev1.close < vwap and prev2.close < vwap and bar.close > vwap and vol_ok:
                self._breakout_dir = "LONG"
                self._awaiting_retest = True
                self._bars_since_breakout = 0
                return None
            if prev1.close > vwap and prev2.close > vwap and bar.close < vwap and vol_ok:
                self._breakout_dir = "SHORT"
                self._awaiting_retest = True
                self._bars_since_breakout = 0
                return None
        # Step 2: detect retest
        if self._awaiting_retest and self._breakout_dir == "LONG":
            if bar.low <= vwap + 0.1 * atr and bar.close > vwap and bar.close > bar.open:
                self._awaiting_retest = False
                self._breakout_dir = None
                return "LONG"
        if self._awaiting_retest and self._breakout_dir == "SHORT":
            if bar.high >= vwap - 0.1 * atr and bar.close < vwap and bar.close < bar.open:
                self._awaiting_retest = False
                self._breakout_dir = None
                return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.vwap or not ind.atr14:
            return 0.0
        # Rejection quality from VWAP
        dist = abs(bar.close - ind.vwap) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        body_ratio = body / full if full > 0 else 0
        return min(1.0, min(dist / 0.3, 1.0) * 0.4 + body_ratio * 0.4 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        ratio = bar.volume / ind.volume_sma20
        return min(1.0, 0.3 + ratio * 0.3)

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        score = 0.5
        if ind.ema9 and ind.vwap:
            if bar.close > ind.vwap and ind.ema9 > ind.vwap:
                score += 0.2
            elif bar.close < ind.vwap and ind.ema9 < ind.vwap:
                score += 0.2
        return min(1.0, score)


# ── Setup 9: VWAP Cross with Momentum (Tier 3) ─────────────────────────

class VWAPCrossMomentumDetector(SetupDetector):
    """VWAP Cross with EMA + ADX + Volume confirmation. Tier 3."""
    name = "vwap_cross_momentum"
    display_name = "VWAP Cross + Momentum"
    category = "vwap"
    hold_time = "5-30min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_MODERATE: 0.9,
            MarketRegime.TRENDING_STRONG: 0.7,
            MarketRegime.RANGING: 0.5,
            MarketRegime.VOLATILE_EXPANSION: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.vwap is None or not ind.ema9 or not ind.adx14 or not ind.atr14:
            return None
        if len(bars) < 3:
            return None
        prev = bars[-2]
        # Bullish cross: prev below VWAP, current above, EMA9 confirms
        if prev.close < ind.vwap and bar.close > ind.vwap and ind.ema9 > ind.vwap:
            if ind.adx14 > 20 and bar.close > bar.open:
                return "LONG"
        if prev.close > ind.vwap and bar.close < ind.vwap and ind.ema9 < ind.vwap:
            if ind.adx14 > 20 and bar.close < bar.open:
                return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or not ind.vwap:
            return 0.0
        pen = abs(bar.close - ind.vwap) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, min(pen / 0.3, 1.0) * 0.5 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.9
        if r >= 1.0:
            return 0.6
        return 0.3

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.adx14 and ind.adx14 > 25:
            s += 0.2
        if ind.rsi14:
            if bar.close > ind.vwap and 50 < ind.rsi14 < 70:
                s += 0.15
            elif bar.close < ind.vwap and 30 < ind.rsi14 < 50:
                s += 0.15
        return min(1.0, s)


# ── Setup 10: First VWAP Touch After Gap (Tier 2) ──────────────────────

class FirstVWAPTouchAfterGapDetector(SetupDetector):
    """First VWAP Touch After Gap — gap day, first approach to VWAP. Tier 2."""
    name = "first_vwap_touch"
    display_name = "First VWAP Touch"
    category = "vwap"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 900

    def __init__(self):
        super().__init__()
        self._touched_today: bool = False
        self._touch_date: str = ""

    def reset(self):
        super().reset()
        self._touched_today = False

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_MODERATE: 0.9,
            MarketRegime.TRENDING_STRONG: 0.7,
            MarketRegime.RANGING: 0.6,
            MarketRegime.VOLATILE_EXPANSION: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.vwap is None or not ind.atr14 or not ind.pdc:
            return None
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._touch_date:
            self._touch_date = today
            self._touched_today = False
        if self._touched_today:
            return None
        if ind.is_rth and (ind.session_minutes < 15 or ind.session_minutes > 120):
            return None
        atr = ind.atr14
        gap = abs(bar.open - ind.pdc)
        if gap < 0.5 * atr:
            return None  # Need meaningful gap
        vwap = ind.vwap
        # Gap up, price pulls back to VWAP from above
        if bar.open > ind.pdc and bar.low <= vwap + 0.15 * atr and bar.close > vwap and bar.close > bar.open:
            self._touched_today = True
            return "LONG"
        # Gap down, price rallies to VWAP from below
        if bar.open < ind.pdc and bar.high >= vwap - 0.15 * atr and bar.close < vwap and bar.close < bar.open:
            self._touched_today = True
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or not ind.vwap:
            return 0.0
        touch_precision = 1.0 - min(abs(bar.low - ind.vwap) / (0.3 * ind.atr14), 1.0)
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, touch_precision * 0.5 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        return min(1.0, 0.3 + r * 0.3)

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if bar.close > ind.vwap and ind.rsi14 < 60:
                s += 0.2
            elif bar.close < ind.vwap and ind.rsi14 > 40:
                s += 0.2
        return min(1.0, s)


# ── Setup 11: ORB Failure Reversal (Tier 3) ────────────────────────────

class ORBFailureReversalDetector(SetupDetector):
    """ORB Failure — false breakout of OR that reverses. Tier 3."""
    name = "orb_failure_reversal"
    display_name = "ORB Failure Reversal"
    category = "mean_reversion"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 1.0,
            MarketRegime.CHOPPY: 0.6,
            MarketRegime.TRENDING_MODERATE: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.or_complete or ind.orh is None or ind.orl is None or not ind.atr14:
            return None
        if ind.is_rth and (ind.session_minutes < 20 or ind.session_minutes > 120):
            return None
        if len(bars) < 3:
            return None
        prev = bars[-2]
        atr = ind.atr14
        # Failed long breakout: prev broke above ORH, current closes back inside
        if prev.high > ind.orh and prev.close > ind.orh and bar.close < ind.orh and bar.close < bar.open:
            return "SHORT"
        # Failed short breakout
        if prev.low < ind.orl and prev.close < ind.orl and bar.close > ind.orl and bar.close > bar.open:
            return "LONG"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or ind.orh is None or ind.orl is None:
            return 0.0
        or_mid = (ind.orh + ind.orl) / 2
        # How far back inside OR did it close?
        if bar.close < ind.orh:
            inside = (ind.orh - bar.close) / ind.atr14
        else:
            inside = (bar.close - ind.orl) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, min(inside / 0.3, 1.0) * 0.5 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if ind.orh is not None and bar.close < ind.orh and ind.rsi14 > 60:
                s += 0.2  # Overbought rejection
            elif ind.orl is not None and bar.close > ind.orl and ind.rsi14 < 40:
                s += 0.2
        return min(1.0, s)


# ── Setup 12: ABC Morning Reversal (Tier 4) ────────────────────────────

class ABCMorningReversalDetector(SetupDetector):
    """ABC Morning Reversal — 3-wave morning pattern. Tier 4."""
    name = "abc_morning_reversal"
    display_name = "ABC Morning Reversal"
    category = "mean_reversion"
    hold_time = "15-45min"
    evidence_tier = EvidenceTier.TIER4
    min_cooldown_seconds = 900

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG, MarketRegime.VOLATILE_EXPANSION}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.VOLATILE_EXPANSION: 0.0,
            MarketRegime.UNKNOWN: 0.4,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14:
            return None
        if ind.is_rth and (ind.session_minutes < 30 or ind.session_minutes > 120):
            return None
        if len(bars) < 15:
            return None
        atr = ind.atr14
        recent = bars[-12:]
        highs = [b.high for b in recent]
        lows = [b.low for b in recent]
        # Find A (first swing), B (retrace), C (second swing lower than A)
        a_idx = highs.index(max(highs))
        # Bearish ABC: high at A, retrace to B, lower high at C
        if a_idx < 4 and a_idx > 0:
            a_high = highs[a_idx]
            b_low = min(lows[a_idx:a_idx + 4]) if a_idx + 4 <= len(lows) else min(lows[a_idx:])
            c_highs = highs[a_idx + 2:]
            if c_highs:
                c_high = max(c_highs)
                if c_high < a_high and bar.close < b_low and bar.close < bar.open:
                    return "SHORT"
        # Bullish ABC
        a_idx_l = lows.index(min(lows))
        if a_idx_l < 4 and a_idx_l > 0:
            a_low = lows[a_idx_l]
            b_high = max(highs[a_idx_l:a_idx_l + 4]) if a_idx_l + 4 <= len(highs) else max(highs[a_idx_l:])
            c_lows = lows[a_idx_l + 2:]
            if c_lows:
                c_low = min(c_lows)
                if c_low > a_low and bar.close > b_high and bar.close > bar.open:
                    return "LONG"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, br * 0.5 + 0.3)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        return min(1.0, 0.3 + r * 0.25)

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if bar.close > bar.open and ind.rsi14 < 45:
                s += 0.2
            elif bar.close < bar.open and ind.rsi14 > 55:
                s += 0.2
        return min(1.0, s)


# ── Setup 13: NR7 + ORB (Tier 1) ───────────────────────────────────────

class NR7ORBDetector(SetupDetector):
    """NR7 day + Opening Range Breakout — narrowest range in 7 days predicts expansion. Tier 1."""
    name = "nr7_orb"
    display_name = "NR7 + ORB"
    category = "volatility"
    hold_time = "15-60min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.VOLATILE_EXPANSION}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.QUIET_COMPRESSION: 1.0,
            MarketRegime.RANGING: 0.8,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.CHOPPY: 0.3,
            MarketRegime.VOLATILE_EXPANSION: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.or_complete or ind.orh is None or ind.orl is None or not ind.atr14:
            return None
        if ind.is_rth and (ind.session_minutes < 15 or ind.session_minutes > 90):
            return None
        if len(bars) < 50:
            return None
        # Check NR7: today's OR range is narrowest in 7 "sessions" (use recent bars as proxy)
        or_range = ind.orh - ind.orl
        # Compare with ranges of last 7 groups of 15 bars
        ranges = []
        for i in range(1, 8):
            start = max(0, len(bars) - i * 15 - 15)
            end = max(0, len(bars) - i * 15)
            if end <= start:
                break
            chunk = bars[start:end]
            if chunk:
                r = max(b.high for b in chunk) - min(b.low for b in chunk)
                ranges.append(r)
        if len(ranges) < 5:
            return None
        if or_range > min(ranges):
            return None  # Not NR7
        # Breakout direction
        if bar.close > ind.orh and bar.close > bar.open:
            return "LONG"
        if bar.close < ind.orl and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or ind.orh is None or ind.orl is None:
            return 0.0
        if bar.close > ind.orh:
            pen = (bar.close - ind.orh) / ind.atr14
        else:
            pen = (ind.orl - bar.close) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, min(pen / 0.4, 1.0) * 0.5 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.9
        if r >= 1.0:
            return 0.6
        return 0.3

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.squeeze_on:
            s += 0.2  # Squeeze confirms compression
        if ind.adx14 and ind.adx14 < 20:
            s += 0.15  # Low ADX = coiled
        return min(1.0, s)


# ── Setup 14: PDH/PDL Rejection (Tier 4) ───────────────────────────────

class PDHPDLRejectionDetector(SetupDetector):
    """PDH/PDL Rejection — touch level and reject. Tier 4."""
    name = "pdh_pdl_rejection"
    display_name = "PDH/PDL Rejection"
    category = "level"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER4
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if ind.pdh is None or ind.pdl is None or not ind.atr14:
            return None
        atr = ind.atr14
        # Rejection at PDH: wick above, close below
        if bar.high >= ind.pdh - 0.1 * atr and bar.close < ind.pdh - 0.1 * atr and bar.close < bar.open:
            return "SHORT"
        if bar.low <= ind.pdl + 0.1 * atr and bar.close > ind.pdl + 0.1 * atr and bar.close > bar.open:
            return "LONG"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14:
            return 0.0
        # Wick quality
        if bar.close < bar.open:  # Bearish
            wick = bar.high - max(bar.close, bar.open)
        else:
            wick = min(bar.close, bar.open) - bar.low
        wick_ratio = wick / (bar.high - bar.low) if (bar.high - bar.low) > 0 else 0
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, wick_ratio * 0.4 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        return min(1.0, 0.3 + r * 0.25)

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if ind.pdh is not None and bar.close < ind.pdh and ind.rsi14 > 65:
                s += 0.2
            elif ind.pdl is not None and bar.close > ind.pdl and ind.rsi14 < 35:
                s += 0.2
        return min(1.0, s)


# ── Setup 15: Round Number Bounce (Tier 1) ─────────────────────────────

class RoundNumberBounceDetector(SetupDetector):
    """Round Number Bounce — institutional magnet levels. Tier 1."""
    name = "round_number_bounce"
    display_name = "Round Number Bounce"
    category = "level"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 600

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.6,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.UNKNOWN: 0.5,
        }

    def _nearest_round(self, price: float) -> float:
        """Find nearest round number (multiples of 50 for ES, 25 for NQ-like)."""
        # Use 25-point intervals
        return round(price / 25) * 25

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or ind.atr14 <= 0:
            return None
        atr = ind.atr14
        rn = self._nearest_round(bar.close)
        dist = abs(bar.close - rn)
        if dist > 0.3 * atr:
            return None  # Not near enough
        # Bounce off round number
        if bar.low <= rn + 0.1 * atr and bar.close > rn and bar.close > bar.open:
            return "LONG"
        if bar.high >= rn - 0.1 * atr and bar.close < rn and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14:
            return 0.0
        rn = self._nearest_round(bar.close)
        precision = 1.0 - min(abs(bar.close - rn) / (0.3 * ind.atr14), 1.0)
        # Prior touches (confluence)
        touches = 0
        for b in bars[-20:]:
            if abs(b.low - rn) < 0.15 * ind.atr14 or abs(b.high - rn) < 0.15 * ind.atr14:
                touches += 1
        touch_score = min(touches / 3, 1.0)
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, precision * 0.3 + touch_score * 0.3 + br * 0.2 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.9
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if bar.close > bar.open and ind.rsi14 < 45:
                s += 0.2
            elif bar.close < bar.open and ind.rsi14 > 55:
                s += 0.2
        return min(1.0, s)


# ── Setup 16: RSI Divergence (Tier 2) ──────────────────────────────────

class RSIDivergenceDetector(SetupDetector):
    """RSI Divergence — price makes new extreme but RSI doesn't. Tier 2."""
    name = "rsi_divergence"
    display_name = "RSI Divergence"
    category = "mean_reversion"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.rsi14 or not ind.rsi14_prev or not ind.atr14:
            return None
        if len(bars) < 10:
            return None
        # Bullish divergence: price new low, RSI higher low
        recent_lows = [b.low for b in bars[-10:]]
        if bar.low <= min(recent_lows[:-1]) and ind.rsi14 > ind.rsi14_prev and ind.rsi14 < 40:
            if bar.close > bar.open:
                return "LONG"
        # Bearish divergence
        recent_highs = [b.high for b in bars[-10:]]
        if bar.high >= max(recent_highs[:-1]) and ind.rsi14 < ind.rsi14_prev and ind.rsi14 > 60:
            if bar.close < bar.open:
                return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, br * 0.5 + 0.3)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        return min(1.0, 0.3 + r * 0.3)

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14 and ind.rsi14_prev:
            div_strength = abs(ind.rsi14 - ind.rsi14_prev)
            s += min(div_strength / 10, 0.3)
        return min(1.0, s)


# ── Setup 17: ADX Thrust (Tier 1) ──────────────────────────────────────

class ADXThrustDetector(SetupDetector):
    """ADX Thrust — ADX rises sharply from low base. Trend initiation. Tier 1."""
    name = "adx_thrust"
    display_name = "ADX Thrust"
    category = "momentum"
    hold_time = "15-60min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 600

    def __init__(self):
        super().__init__()
        self._prev_adx: Optional[float] = None

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_STRONG: 1.0,
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.QUIET_COMPRESSION: 0.7,
            MarketRegime.VOLATILE_EXPANSION: 0.5,
            MarketRegime.RANGING: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.adx14 or not ind.plus_di or not ind.minus_di:
            return None
        prev_adx = self._prev_adx
        self._prev_adx = ind.adx14
        if prev_adx is None:
            return None
        # ADX thrust: was below 20, now above 25 (rapid rise)
        if prev_adx < 20 and ind.adx14 > 25:
            if ind.plus_di > ind.minus_di:
                return "LONG"
            else:
                return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        # Directional candle
        if bar.close > bar.open and ind.plus_di and ind.minus_di and ind.plus_di > ind.minus_di:
            dir_score = 0.3
        elif bar.close < bar.open and ind.minus_di and ind.plus_di and ind.minus_di > ind.plus_di:
            dir_score = 0.3
        else:
            dir_score = 0.1
        return min(1.0, br * 0.4 + dir_score + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.9
        if r >= 1.0:
            return 0.6
        return 0.4

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.adx14 and self._prev_adx:
            thrust = ind.adx14 - self._prev_adx
            s += min(thrust / 10, 0.3)
        if ind.ema9 and ind.ema20:
            if bar.close > bar.open and ind.ema9 > ind.ema20:
                s += 0.1
            elif bar.close < bar.open and ind.ema9 < ind.ema20:
                s += 0.1
        return min(1.0, s)


# ── Setup 18: ATR Expansion (Tier 4) ───────────────────────────────────

class ATRExpansionDetector(SetupDetector):
    """ATR Expansion — volatility expanding from low base. Tier 4."""
    name = "atr_expansion"
    display_name = "ATR Expansion"
    category = "volatility"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER4
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.VOLATILE_EXPANSION: 0.9,
            MarketRegime.QUIET_COMPRESSION: 0.8,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.RANGING: 0.4,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr_ratio or not ind.atr14:
            return None
        # ATR expanding: short-term ATR much larger than long-term
        if ind.atr_ratio < 1.3:
            return None
        # Direction from candle
        body = abs(bar.close - bar.open)
        if body < 0.3 * ind.atr14:
            return None  # Need directional bar
        if bar.close > bar.open:
            return "LONG"
        return "SHORT"

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14:
            return 0.0
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        body_atr = body / ind.atr14
        return min(1.0, min(body_atr / 0.5, 1.0) * 0.4 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.atr_ratio and ind.atr_ratio > 1.5:
            s += 0.2
        if ind.adx14 and ind.adx14 > 20:
            s += 0.15
        return min(1.0, s)


# ── Setup 19: NR4/NR7 Breakout (Tier 1) ────────────────────────────────

class NR4NR7BreakoutDetector(SetupDetector):
    """NR4 inside bar breakout — narrowest range in 4 bars. Tier 1."""
    name = "nr4_nr7_breakout"
    display_name = "NR4/NR7 Breakout"
    category = "volatility"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER1
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.VOLATILE_EXPANSION}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.QUIET_COMPRESSION: 1.0,
            MarketRegime.RANGING: 0.7,
            MarketRegime.TRENDING_MODERATE: 0.6,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.CHOPPY: 0.3,
            MarketRegime.VOLATILE_EXPANSION: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or len(bars) < 6:
            return None
        # Check NR4: current bar has narrowest range in last 4
        ranges = [b.high - b.low for b in bars[-5:-1]]
        cur_range = bar.high - bar.low
        if cur_range > min(ranges):
            return None
        # Inside bar check
        prev = bars[-2]
        if bar.high > prev.high or bar.low < prev.low:
            return None  # Not inside bar
        # Direction on breakout of prev bar
        if bar.close > prev.high * 0.999 and bar.close > bar.open:
            return "LONG"
        if bar.close < prev.low * 1.001 and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, br * 0.5 + 0.3)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.9
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.squeeze_on:
            s += 0.25
        if ind.adx14 and ind.adx14 < 20:
            s += 0.15
        return min(1.0, s)


# ── Setup 20: VCP Intraday (Tier 2) ────────────────────────────────────

class VCPIntradayDetector(SetupDetector):
    """Volatility Contraction Pattern — decreasing range bars. Tier 2."""
    name = "vcp_intraday"
    display_name = "VCP Intraday"
    category = "volatility"
    hold_time = "10-30min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 600

    def disabled_regimes(self) -> set:
        return {MarketRegime.VOLATILE_EXPANSION, MarketRegime.CHOPPY}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.QUIET_COMPRESSION: 1.0,
            MarketRegime.RANGING: 0.7,
            MarketRegime.TRENDING_MODERATE: 0.8,
            MarketRegime.TRENDING_STRONG: 0.5,
            MarketRegime.VOLATILE_EXPANSION: 0.0,
            MarketRegime.CHOPPY: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or len(bars) < 10:
            return None
        # Check 3 contracting ranges
        ranges = [b.high - b.low for b in bars[-5:-1]]
        if len(ranges) < 4:
            return None
        if not (ranges[-1] < ranges[-2] < ranges[-3]):
            return None  # Not contracting
        # Breakout of the contraction
        pivot_high = max(b.high for b in bars[-5:-1])
        pivot_low = min(b.low for b in bars[-5:-1])
        if bar.close > pivot_high and bar.close > bar.open:
            return "LONG"
        if bar.close < pivot_low and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or len(bars) < 5:
            return 0.0
        pivot_high = max(b.high for b in bars[-5:-1])
        pivot_low = min(b.low for b in bars[-5:-1])
        if bar.close > pivot_high:
            pen = (bar.close - pivot_high) / ind.atr14
        else:
            pen = (pivot_low - bar.close) / ind.atr14
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, min(pen / 0.3, 1.0) * 0.4 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        # Volume should expand on breakout
        if r >= 1.5:
            return 0.9
        if r >= 1.0:
            return 0.6
        return 0.3

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.ema9 and ind.ema20:
            if bar.close > bar.open and ind.ema9 > ind.ema20:
                s += 0.2
            elif bar.close < bar.open and ind.ema9 < ind.ema20:
                s += 0.2
        return min(1.0, s)


# ── Setup 21: Liquidity Sweep (Tier 3) ─────────────────────────────────

class LiquiditySweepDetector(SetupDetector):
    """Liquidity Sweep — sweep recent swing then reverse. Tier 3."""
    name = "liquidity_sweep"
    display_name = "Liquidity Sweep"
    category = "micro"
    hold_time = "5-20min"
    evidence_tier = EvidenceTier.TIER3
    min_cooldown_seconds = 300

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.VOLATILE_EXPANSION: 0.6,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.TRENDING_STRONG: 0.4,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or len(bars) < 15:
            return None
        atr = ind.atr14
        lookback = bars[-15:-1]
        swing_low = min(b.low for b in lookback)
        swing_high = max(b.high for b in lookback)
        # Sweep below swing low, close back above
        if bar.low < swing_low and bar.close > swing_low + 0.1 * atr and bar.close > bar.open:
            return "LONG"
        if bar.high > swing_high and bar.close < swing_high - 0.1 * atr and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14 or len(bars) < 15:
            return 0.0
        lookback = bars[-15:-1]
        swing_low = min(b.low for b in lookback)
        swing_high = max(b.high for b in lookback)
        if bar.low < swing_low:
            sweep_depth = (swing_low - bar.low) / ind.atr14
            rejection = (bar.close - swing_low) / ind.atr14
        elif bar.high > swing_high:
            sweep_depth = (bar.high - swing_high) / ind.atr14
            rejection = (swing_high - bar.close) / ind.atr14
        else:
            # No sweep occurred
            return 0.0
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        # Clamp all scores to 0-1
        sweep_score = max(0.0, min(sweep_depth / 0.3, 1.0))
        rej_score = max(0.0, min(rejection / 0.2, 1.0))
        return min(1.0, sweep_score * 0.3 + rej_score * 0.3 + br * 0.2 + 0.1)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 2.0:
            return 1.0
        if r >= 1.5:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if bar.close > bar.open and ind.rsi14 < 35:
                s += 0.25
            elif bar.close < bar.open and ind.rsi14 > 65:
                s += 0.25
        return min(1.0, s)


# ── Setup 22: FVG Fill (Tier 2) ────────────────────────────────────────

class FVGFillDetector(SetupDetector):
    """Fair Value Gap Fill — price returns to fill imbalance. Tier 2."""
    name = "fvg_fill"
    display_name = "FVG Fill"
    category = "micro"
    hold_time = "5-20min"
    evidence_tier = EvidenceTier.TIER2
    min_cooldown_seconds = 300

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.TRENDING_MODERATE: 0.9,
            MarketRegime.TRENDING_STRONG: 0.7,
            MarketRegime.RANGING: 0.6,
            MarketRegime.VOLATILE_EXPANSION: 0.5,
            MarketRegime.QUIET_COMPRESSION: 0.4,
            MarketRegime.CHOPPY: 0.3,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or len(bars) < 5:
            return None
        atr = ind.atr14
        # Scan recent bars for FVG (3-bar pattern where bar2 doesn't overlap bar0)
        for i in range(len(bars) - 4, max(len(bars) - 15, 0), -1):
            if i < 0 or i + 2 >= len(bars) - 1:
                continue
            b0, b1, b2 = bars[i], bars[i + 1], bars[i + 2]
            # Bullish FVG: gap between b0.high and b2.low
            if b2.low > b0.high and (b2.low - b0.high) > 0.2 * atr:
                gap_top = b2.low
                gap_bot = b0.high
                # Current bar fills into the gap
                if bar.low <= gap_top and bar.close > gap_bot and bar.close > bar.open:
                    return "LONG"
            # Bearish FVG
            if b0.low > b2.high and (b0.low - b2.high) > 0.2 * atr:
                gap_top = b0.low
                gap_bot = b2.high
                if bar.high >= gap_bot and bar.close < gap_top and bar.close < bar.open:
                    return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, br * 0.5 + 0.3)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 1.5:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.ema9 and ind.ema20:
            if bar.close > bar.open and ind.ema9 > ind.ema20:
                s += 0.2
            elif bar.close < bar.open and ind.ema9 < ind.ema20:
                s += 0.2
        return min(1.0, s)


# ── Setup 23: Absorption Proxy (Tier 4) ────────────────────────────────

class AbsorptionProxyDetector(SetupDetector):
    """Absorption — high volume with no price progress = institutional absorption. Tier 4."""
    name = "absorption_proxy"
    display_name = "Absorption Proxy"
    category = "micro"
    hold_time = "5-20min"
    evidence_tier = EvidenceTier.TIER4
    min_cooldown_seconds = 300

    def disabled_regimes(self) -> set:
        return {MarketRegime.TRENDING_STRONG}

    def regime_scores(self) -> Dict[MarketRegime, float]:
        return {
            MarketRegime.RANGING: 0.9,
            MarketRegime.TRENDING_MODERATE: 0.7,
            MarketRegime.QUIET_COMPRESSION: 0.5,
            MarketRegime.CHOPPY: 0.4,
            MarketRegime.VOLATILE_EXPANSION: 0.3,
            MarketRegime.TRENDING_STRONG: 0.0,
            MarketRegime.UNKNOWN: 0.5,
        }

    def detect_direction(self, bar: BarInput, ind: IndicatorState,
                         bars: List[BarInput]) -> Optional[str]:
        if not ind.atr14 or not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return None
        if len(bars) < 5:
            return None
        # High volume, small range = absorption
        if bar.volume < ind.volume_sma20 * 2.0:
            return None
        bar_range = bar.high - bar.low
        if bar_range > 0.5 * ind.atr14:
            return None  # Too much movement, not absorption
        # Direction: where is price relative to recent range?
        recent = bars[-10:-1]
        mid = (max(b.high for b in recent) + min(b.low for b in recent)) / 2
        # Absorption at lows = bullish, at highs = bearish
        if bar.close < mid and bar.close > bar.open:
            return "LONG"
        if bar.close > mid and bar.close < bar.open:
            return "SHORT"
        return None

    def score_price_action(self, bar: BarInput, ind: IndicatorState,
                           bars: List[BarInput]) -> float:
        if not ind.atr14:
            return 0.0
        # Smaller range = more absorption
        bar_range = bar.high - bar.low
        compression = 1.0 - min(bar_range / (0.5 * ind.atr14), 1.0)
        body = abs(bar.close - bar.open)
        full = bar.high - bar.low
        br = body / full if full > 0 else 0
        return min(1.0, compression * 0.4 + br * 0.3 + 0.2)

    def score_volume(self, bar: BarInput, ind: IndicatorState,
                     bars: List[BarInput]) -> float:
        if not ind.volume_sma20 or ind.volume_sma20 <= 0:
            return 0.5
        r = bar.volume / ind.volume_sma20
        if r >= 3.0:
            return 1.0
        if r >= 2.0:
            return 0.8
        return 0.5

    def score_momentum(self, bar: BarInput, ind: IndicatorState,
                       bars: List[BarInput]) -> float:
        s = 0.5
        if ind.rsi14:
            if bar.close > bar.open and ind.rsi14 < 40:
                s += 0.2
            elif bar.close < bar.open and ind.rsi14 > 60:
                s += 0.2
        return min(1.0, s)


# ═══════════════════════════════════════════════════════════════════════════
# SETUP MANAGER (placeholder — will be added after all detectors)
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

        # VWAP session accumulators
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

        self._prev_rsi: Optional[float] = None
        logger.info("SetupManager initialized")

    def register(self, detector: SetupDetector):
        self.detectors.append(detector)
        logger.info(f"Registered setup detector: {detector.name} ({detector.display_name})")

    def register_all_defaults(self):
        """Register all 23 setup detectors with decision tree scoring."""
        # Tier 1 (cap 0.85)
        self.register(ORBBreakoutDetector())            # 1
        self.register(NR7ORBDetector())                 # 13
        self.register(NR4NR7BreakoutDetector())         # 19
        self.register(RoundNumberBounceDetector())      # 15
        self.register(EMA20PullbackDetector())          # 4 (Holy Grail)
        self.register(ADXThrustDetector())              # 17

        # Tier 2 (cap 0.70)
        self.register(FirstVWAPTouchAfterGapDetector()) # 10
        self.register(PDHPDLBreakoutDetector())         # 3
        self.register(ONHLSweepDetector())              # 6
        self.register(RSIDivergenceDetector())          # 16
        self.register(FVGFillDetector())                # 22
        self.register(VCPIntradayDetector())            # 20

        # Tier 3 (cap 0.55)
        self.register(VWAPMeanReversionDetector())      # 2
        self.register(VWAPCrossMomentumDetector())      # 9
        self.register(ORBFailureReversalDetector())     # 11
        self.register(VolumeSpikeBreakoutDetector())    # 7
        self.register(TTMSqueezeDetector())             # 5
        self.register(LiquiditySweepDetector())         # 21

        # Tier 4 (cap 0.40)
        self.register(VWAPBreakoutRetestDetector())     # 8
        self.register(ABCMorningReversalDetector())     # 12
        self.register(PDHPDLRejectionDetector())        # 14
        self.register(ATRExpansionDetector())           # 18
        self.register(AbsorptionProxyDetector())        # 23

        logger.info(f"SetupManager: {len(self.detectors)} detectors registered")

    def seed_history(self, bars: List[dict]):
        """Pre-seed bar history from historical data (no signal emission).
        Call this ONCE after registration to warm up indicators."""
        if not bars:
            logger.warning("seed_history called with empty bars list")
            return
        logger.info(f"Seeding SetupManager with {len(bars)} historical bars...")
        for bar_data in bars:
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
            self._compute_indicators(bar)
        logger.info(f"SetupManager seeded: {len(self._bars_history)} bars in history")

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

        self._compute_indicators(bar)

        if indicators:
            self._merge_external_indicators(indicators)
        if levels:
            self._merge_external_levels(levels)

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

    def get_all_readiness(self) -> List[dict]:
        """Return confluence meter (0-100%) for all registered detectors."""
        if not self._bars_history:
            logger.debug(f"get_all_readiness: no bars in history yet")
            return [{"name": d.name, "display_name": d.display_name,
                     "category": d.category, "tier": d.evidence_tier.value,
                     "readiness": 0.0} for d in self.detectors]
        bar = self._bars_history[-1]
        ind = self._indicator_state
        results = []
        for d in self.detectors:
            try:
                score = d.readiness_score(bar, ind, self._bars_history)
            except Exception as e:
                logger.error(f"readiness_score error for {d.name}: {e}", exc_info=True)
                score = 0.0
            results.append({
                "name": d.name,
                "display_name": d.display_name,
                "category": d.category,
                "tier": d.evidence_tier.value,
                "readiness": round(score, 1),
            })
        return results

    def _compute_indicators(self, bar: BarInput):
        ind = self._indicator_state
        bars = self._bars_history
        n = len(bars)

        # Session info
        ind.is_rth = _is_rth(bar.time)
        et_min = _bar_to_et_minutes(bar.time)
        ind.session_minutes = et_min - RTH_OPEN_MINUTES if ind.is_rth else -1
        ind.session = _session_label(ind.session_minutes, ind.is_rth)

        if n < 20:
            return

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        ind.ema9 = _ema(closes[-50:], 9) if n >= 9 else None
        ind.ema20 = _ema(closes[-60:], 20) if n >= 20 else None
        ind.rsi14_prev = ind.rsi14
        ind.rsi14 = _compute_rsi(closes, 14) if n >= 16 else None
        ind.atr14 = _compute_atr(bars, 14) if n >= 15 else None

        # NEW: ATR-5 and ATR-50 for regime classification
        ind.atr5 = _compute_atr(bars, 5) if n >= 6 else None
        ind.atr50 = _compute_atr(bars, 50) if n >= 51 else None
        if ind.atr5 and ind.atr50 and ind.atr50 > 0:
            ind.atr_ratio = ind.atr5 / ind.atr50
        else:
            ind.atr_ratio = None

        # NEW: Choppiness Index
        ind.chop14 = _compute_choppiness(bars, 14) if n >= 15 else None

        # ADX
        if n >= 30:
            adx, plus_di, minus_di = _compute_adx(bars, 14)
            ind.adx14 = adx
            ind.plus_di = plus_di
            ind.minus_di = minus_di

        # NEW: Regime classification
        ind.regime = classify_regime(ind.adx14, ind.chop14, ind.atr_ratio)

        ind.volume_sma20 = _sma(volumes, 20) if n >= 20 else None

        # Bollinger Bands
        bb_sma = _sma(closes, 20)
        bb_std = _stdev(closes, 20)
        if bb_sma is not None and bb_std is not None:
            ind.bb_middle = bb_sma
            ind.bb_upper = bb_sma + 2 * bb_std
            ind.bb_lower = bb_sma - 2 * bb_std

        # Keltner Channels
        kc_mid = _ema(closes[-60:], 20) if n >= 20 else None
        kc_atr = ind.atr14
        if kc_mid is not None and kc_atr is not None:
            ind.kc_middle = kc_mid
            ind.kc_upper = kc_mid + 1.5 * kc_atr
            ind.kc_lower = kc_mid - 1.5 * kc_atr

        # TTM Squeeze
        if ind.bb_upper is not None and ind.kc_upper is not None:
            ind.squeeze_on = (ind.bb_upper < ind.kc_upper and ind.bb_lower > ind.kc_lower)
            if n >= 20:
                hh = max(b.high for b in bars[-20:])
                ll = min(b.low for b in bars[-20:])
                midline = (hh + ll) / 2
                if bb_sma is not None:
                    avg_ml = (midline + bb_sma) / 2
                    ind.squeeze_momentum = bar.close - avg_ml

        self._update_vwap(bar)
        self._update_daily_levels(bar)
        self._update_overnight_levels(bar)
        self._update_opening_range(bar)

    def _update_vwap(self, bar: BarInput):
        """Compute VWAP anchored to daily session (resets each calendar day)."""
        ind = self._indicator_state
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
        ind = self._indicator_state
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if today != self._cur_day_date:
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
        ind = self._indicator_state
        today = datetime.fromtimestamp(bar.time, tz=timezone.utc).strftime('%Y-%m-%d')
        if ind.is_rth:
            if today != self._on_date:
                self._on_date = today
            ind.onh = self._on_high
            ind.onl = self._on_low
        else:
            if today != self._on_date:
                self._on_date = today
                self._on_high = bar.high
                self._on_low = bar.low
            else:
                if self._on_high is not None:
                    self._on_high = max(self._on_high, bar.high)
                    self._on_low = min(self._on_low, bar.low)

    def _update_opening_range(self, bar: BarInput):
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
        ind = self._indicator_state
        for key in ['ema9', 'rsi14', 'adx14', 'plus_di', 'minus_di',
                     'atr14', 'volume_sma20', 'vwap']:
            val = indicators.get(key)
            if val is not None:
                setattr(ind, key, val)
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
