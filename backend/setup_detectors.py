"""
Setup Detectors Framework

Base classes and manager for running multiple intraday setup detectors
simultaneously. Each detector analyzes incoming bars and emits signals
that feed into the existing bracket resolution system for tracking.

All detection runs server-side. Results appear on the stats page only.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


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
    # Core indicators (from scalping engine or computed here)
    ema9: Optional[float] = None
    rsi7: Optional[float] = None
    rsi7_prev: Optional[float] = None
    adx10: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    atr10: Optional[float] = None
    atr_sma20: Optional[float] = None
    atr_percentile: Optional[float] = None
    chop14: Optional[float] = None
    volume_sma20: Optional[float] = None

    # VWAP
    vwap: Optional[float] = None
    vwap_upper1: Optional[float] = None
    vwap_lower1: Optional[float] = None
    vwap_upper2: Optional[float] = None
    vwap_lower2: Optional[float] = None
    vwap_std: Optional[float] = None

    # Levels
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    pdc: Optional[float] = None
    onh: Optional[float] = None
    onl: Optional[float] = None
    orh: Optional[float] = None
    orl: Optional[float] = None

    # Session
    session: str = ""       # e.g. "rth_open", "am_session", "lunch", "pm_session"
    session_minutes: int = 0  # minutes since RTH open (9:30 ET)


# ═══════════════════════════════════════════════════════════════════════════
# BASE DETECTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SetupDetector:
    """
    Base class for all setup detectors.

    Subclasses must implement:
      - update(bar, indicators, bars_history) -> Optional[SetupSignal]

    Each detector:
      - Maintains its own internal state
      - Has its own cooldown tracking (independent of other detectors)
      - Emits SetupSignal objects when entry conditions are met
    """

    name: str = "base"
    display_name: str = "Base Setup"
    category: str = "unknown"   # vwap, session, level, momentum, volatility, micro
    hold_time: str = "unknown"  # "1-5min", "5-30min", "15-60min", "30min-4hr"
    min_cooldown_seconds: int = 60  # minimum time between signals from this detector

    def __init__(self):
        self._last_signal_time: int = 0
        self._signal_counter: int = 0
        self._enabled: bool = True

    def update(self, bar: BarInput, indicators: IndicatorState,
               bars_history: List[BarInput]) -> Optional[SetupSignal]:
        """
        Process a new bar. Return a SetupSignal if entry conditions are met.

        Args:
            bar: Current bar OHLCV
            indicators: Pre-computed indicator values
            bars_history: Recent bar history (last ~500 bars)

        Returns:
            SetupSignal if conditions met, None otherwise
        """
        raise NotImplementedError

    def can_signal(self, bar_time: int) -> bool:
        """Check cooldown."""
        if not self._enabled:
            return False
        return (bar_time - self._last_signal_time) >= self.min_cooldown_seconds

    def record_signal(self, bar_time: int):
        """Mark that a signal was emitted."""
        self._last_signal_time = bar_time
        self._signal_counter += 1

    def make_signal(self, direction: str, bar: BarInput,
                    indicators: IndicatorState,
                    stop_price: float, target_price: float,
                    stop_atr_mult: float, target_atr_mult: float,
                    max_bars: int, confidence: float,
                    reason: str) -> SetupSignal:
        """Helper to construct a signal with common fields."""
        atr = indicators.atr10 or 1.0
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
                'rsi7': indicators.rsi7,
                'adx10': indicators.adx10,
                'atr10': indicators.atr10,
                'vwap': indicators.vwap,
                'chop14': indicators.chop14,
                'session': indicators.session,
            },
        )

    def reset(self):
        """Reset detector state (e.g. on symbol/timeframe change)."""
        self._last_signal_time = 0
        self._signal_counter = 0

    def get_info(self) -> dict:
        """Return detector metadata for the stats page."""
        return {
            'name': self.name,
            'display_name': self.display_name,
            'category': self.category,
            'hold_time': self.hold_time,
            'enabled': self._enabled,
            'signal_count': self._signal_counter,
        }


# ═══════════════════════════════════════════════════════════════════════════
# SETUP MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class SetupManager:
    """
    Runs all registered setup detectors on each incoming bar.

    - Computes shared indicators once
    - Feeds every detector
    - Collects emitted signals
    - Manages per-detector cooldowns
    """

    def __init__(self):
        self.detectors: List[SetupDetector] = []
        self._bars_history: List[BarInput] = []
        self._max_history: int = 600  # keep last 600 bars (10 hours of 1-min)
        self._indicator_state = IndicatorState()
        logger.info("SetupManager initialized")

    def register(self, detector: SetupDetector):
        """Register a setup detector."""
        self.detectors.append(detector)
        logger.info(f"Registered setup detector: {detector.name} ({detector.display_name})")

    def register_all_defaults(self):
        """Register all available setup detectors. Called at startup."""
        # Phase 1 detectors will be added here as they're implemented
        # For now, list is empty — this is the hook point
        count = len(self.detectors)
        logger.info(f"SetupManager: {count} detectors registered")

    def process_bar(self, bar_data: dict, indicators: dict = None,
                    levels: dict = None, session: str = "") -> List[SetupSignal]:
        """
        Process a new bar through all detectors.

        Args:
            bar_data: dict with time, open, high, low, close, volume
            indicators: dict with pre-computed indicator values (optional)
            levels: dict with PDH/PDL/ONH/ONL etc (optional)
            session: current session type string

        Returns:
            List of SetupSignal objects emitted by detectors
        """
        # Build bar
        bar = BarInput(
            time=bar_data.get('time', 0),
            open=bar_data.get('open', 0),
            high=bar_data.get('high', 0),
            low=bar_data.get('low', 0),
            close=bar_data.get('close', 0),
            volume=bar_data.get('volume', 0),
        )

        # Update history
        self._bars_history.append(bar)
        if len(self._bars_history) > self._max_history:
            self._bars_history = self._bars_history[-self._max_history:]

        # Update shared indicator state
        self._update_indicators(indicators, levels, session)

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

    def _update_indicators(self, indicators: dict = None,
                           levels: dict = None, session: str = ""):
        """Update the shared indicator state from provided dicts."""
        ind = self._indicator_state

        if indicators:
            ind.ema9 = indicators.get('ema9')
            ind.rsi7 = indicators.get('rsi7')
            ind.rsi7_prev = indicators.get('rsi7_prev')
            ind.adx10 = indicators.get('adx10')
            ind.plus_di = indicators.get('plus_di') or indicators.get('plusDI')
            ind.minus_di = indicators.get('minus_di') or indicators.get('minusDI')
            ind.atr10 = indicators.get('atr10')
            ind.atr_sma20 = indicators.get('atr_sma20') or indicators.get('atrSMA20')
            ind.atr_percentile = indicators.get('atr_percentile') or indicators.get('atrPercentile')
            ind.chop14 = indicators.get('chop14')
            ind.volume_sma20 = indicators.get('volume_sma20') or indicators.get('volSMA20')
            ind.vwap = indicators.get('vwap')
            ind.vwap_upper1 = indicators.get('vwap_upper1') or indicators.get('vwapUpper1')
            ind.vwap_lower1 = indicators.get('vwap_lower1') or indicators.get('vwapLower1')
            ind.vwap_upper2 = indicators.get('vwap_upper2') or indicators.get('vwapUpper2')
            ind.vwap_lower2 = indicators.get('vwap_lower2') or indicators.get('vwapLower2')
            ind.vwap_std = indicators.get('vwap_std') or indicators.get('vwapStdDev')

        if levels:
            ind.pdh = levels.get('pdh')
            ind.pdl = levels.get('pdl')
            ind.pdc = levels.get('pdc')
            ind.onh = levels.get('onh')
            ind.onl = levels.get('onl')
            ind.orh = levels.get('orh')
            ind.orl = levels.get('orl')

        ind.session = session

    def get_detector_info(self) -> List[dict]:
        """Return metadata for all registered detectors."""
        return [d.get_info() for d in self.detectors]

    def reset_all(self):
        """Reset all detector states."""
        for d in self.detectors:
            d.reset()
        self._bars_history.clear()
        logger.info("SetupManager: all detectors reset")
