"""
Multi-Factor Decision Tree Engine
==================================

Combines multiple market factors to generate trading signals:
1. Gamma Regime (GEX Ratio)
2. Gravity Position (price magnet)
3. Price vs VWAP
4. Delta Flow

Decision Tree Logic:
├── 1) What is the Gamma Regime?
│   ├── SHORT GAMMA (GEX < 0.35) ──► Trends allowed
│   │   ├── 2) Where is Gravity?
│   │   │   ├── Gravity ABOVE price
│   │   │   │   ├── 3) Price vs VWAP?
│   │   │   │   │   ├── Above VWAP ──► BUY CALLS (strong bullish)
│   │   │   │   │   └── Below VWAP ──► BUY CALLS (pullback opportunity)
│   │   │   └── Gravity BELOW price
│   │   │       ├── 3) Price vs VWAP?
│   │   │       │   ├── Below VWAP ──► BUY PUTS (strong bearish)
│   │   │       │   └── Above VWAP ──► BUY PUTS (bull trap)
│   │
│   └── LONG GAMMA (GEX >= 0.35) ──► Pinning/chop expected
│       ├── 2) Where is Gravity?
│       │   ├── Gravity ABOVE price
│       │   │   ├── 3) Price vs VWAP?
│       │   │   │   ├── Below VWAP ──► BUY CALLS (snap up to gravity)
│       │   │   │   └── Above VWAP ──► NEUTRAL (overextended)
│       │   └── Gravity BELOW price
│       │       ├── 3) Price vs VWAP?
│       │       │   ├── Above VWAP ──► BUY PUTS (snap down to gravity)
│       │       │   └── Below VWAP ──► NEUTRAL (overextended)
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY_CALLS = "BUY_CALLS"
    BUY_PUTS = "BUY_PUTS"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    """Signal confidence levels"""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


@dataclass
class TradingSignal:
    """Complete trading signal with reasoning"""
    timestamp: any

    # Primary Signal
    signal: SignalType
    strength: SignalStrength
    confidence: float  # 0-100

    # Decision Tree Path
    gamma_regime: str  # "SHORT_GAMMA", "LONG_GAMMA", "MIXED"
    gravity_position: str  # "ABOVE", "BELOW", "AT"
    price_vs_vwap: str  # "ABOVE", "BELOW", "AT"

    # Supporting Factors
    delta_flow_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    delta_confirmation: bool  # Does delta support the signal?

    # Reasoning
    decision_path: str
    risk_factors: list
    confluence_score: float  # 0-100, how many factors agree

    # Raw Metrics
    gex_ratio: float
    gravity_price: float
    current_price: float
    vwap_price: float
    delta_value: float


class DecisionTreeEngine:
    """
    Multi-Factor Trading Decision Engine

    Evaluates market conditions across multiple dimensions:
    - Gamma Regime (trending vs pinning)
    - Gravity (institutional interest zones)
    - Price vs VWAP (relative value)
    - Delta Flow (order flow pressure)

    Generates clear trading signals with detailed reasoning.
    """

    def __init__(self):
        """Initialize decision tree engine"""
        self.signal_history = []

    def evaluate(self,
                 timestamp: any,
                 current_price: float,
                 gex_ratio: float,
                 gamma_regime: str,
                 gravity_price: float,
                 gravity_above: bool,
                 gravity_below: bool,
                 vwap_price: float,
                 delta_flow: Optional[float] = None) -> TradingSignal:
        """
        Evaluate all factors and generate trading signal

        Args:
            timestamp: Current timestamp
            current_price: Current market price
            gex_ratio: GEX ratio value (0.0-1.0+)
            gamma_regime: "SHORT_GAMMA", "LONG_GAMMA", or "MIXED"
            gravity_price: Gravity level price
            gravity_above: Is gravity above current price?
            gravity_below: Is gravity below current price?
            vwap_price: Current VWAP price
            delta_flow: Current delta flow (optional)

        Returns:
            TradingSignal with complete analysis
        """
        signal = TradingSignal(
            timestamp=timestamp,
            signal=SignalType.NEUTRAL,
            strength=SignalStrength.WEAK,
            confidence=0.0,
            gamma_regime=gamma_regime,
            gravity_position="AT",
            price_vs_vwap="AT",
            delta_flow_bias="NEUTRAL",
            delta_confirmation=False,
            decision_path="",
            risk_factors=[],
            confluence_score=0.0,
            gex_ratio=gex_ratio,
            gravity_price=gravity_price,
            current_price=current_price,
            vwap_price=vwap_price,
            delta_value=delta_flow if delta_flow is not None else 0.0
        )

        # Determine gravity position
        gravity_threshold = current_price * 0.002  # 0.2%
        if gravity_above:
            signal.gravity_position = "ABOVE"
        elif gravity_below:
            signal.gravity_position = "BELOW"
        else:
            signal.gravity_position = "AT"

        # Determine price vs VWAP
        vwap_threshold = current_price * 0.001  # 0.1%
        price_vwap_diff = current_price - vwap_price

        if abs(price_vwap_diff) <= vwap_threshold:
            signal.price_vs_vwap = "AT"
        elif price_vwap_diff > 0:
            signal.price_vs_vwap = "ABOVE"
        else:
            signal.price_vs_vwap = "BELOW"

        # Determine delta flow bias
        if delta_flow is not None:
            if delta_flow > 0:
                signal.delta_flow_bias = "BULLISH"
            elif delta_flow < 0:
                signal.delta_flow_bias = "BEARISH"
            else:
                signal.delta_flow_bias = "NEUTRAL"

        # ============================================
        # DECISION TREE EVALUATION
        # ============================================

        # Level 1: Gamma Regime
        if gex_ratio < 0.35:
            # SHORT GAMMA REGIME - Trends allowed
            signal = self._evaluate_short_gamma(signal)

        elif gex_ratio >= 0.35 and gex_ratio < 0.65:
            # MIXED REGIME - Use both logics with reduced confidence
            signal_short = self._evaluate_short_gamma(signal)
            signal_long = self._evaluate_long_gamma(signal)

            # Blend signals based on proximity to thresholds
            # Closer to 0.35 = more short gamma
            # Closer to 0.65 = more long gamma
            weight_short = (0.65 - gex_ratio) / 0.30
            weight_long = (gex_ratio - 0.35) / 0.30

            # Take stronger signal but reduce confidence
            if signal_short.confidence > signal_long.confidence:
                signal = signal_short
                signal.confidence *= weight_short
                signal.decision_path = f"MIXED REGIME (leaning SHORT GAMMA): {signal.decision_path}"
            else:
                signal = signal_long
                signal.confidence *= weight_long
                signal.decision_path = f"MIXED REGIME (leaning LONG GAMMA): {signal.decision_path}"

            signal.risk_factors.append("Mixed gamma regime - lower confidence")

        else:
            # LONG GAMMA REGIME - Pinning/chop expected
            signal = self._evaluate_long_gamma(signal)

        # Calculate confluence score
        signal.confluence_score = self._calculate_confluence(signal)

        # Adjust confidence based on confluence
        signal.confidence *= (signal.confluence_score / 100.0)

        # Classify strength based on final confidence
        if signal.confidence >= 70:
            signal.strength = SignalStrength.STRONG
        elif signal.confidence >= 40:
            signal.strength = SignalStrength.MODERATE
        else:
            signal.strength = SignalStrength.WEAK

        # Store in history
        self.signal_history.append(signal)

        return signal

    def _evaluate_short_gamma(self, signal: TradingSignal) -> TradingSignal:
        """
        Evaluate SHORT GAMMA regime (GEX < 0.35)
        Trends are allowed, dealer hedging adds to momentum
        """
        signal.decision_path = "SHORT GAMMA REGIME → "

        # Level 2: Gravity Position
        if signal.gravity_position == "ABOVE":
            # Gravity above = bullish bias
            signal.decision_path += "Gravity ABOVE price → "

            # Level 3: Price vs VWAP
            if signal.price_vs_vwap == "ABOVE":
                # Strong bullish: Above VWAP + Gravity above + Short gamma
                signal.signal = SignalType.BUY_CALLS
                signal.confidence = 85.0
                signal.decision_path += "Price ABOVE VWAP → BUY CALLS (strong bullish trend)"

            elif signal.price_vs_vwap == "BELOW":
                # Bullish pullback: Below VWAP but gravity pulling up
                signal.signal = SignalType.BUY_CALLS
                signal.confidence = 70.0
                signal.decision_path += "Price BELOW VWAP → BUY CALLS (pullback to VWAP, gravity pulling up)"
                signal.risk_factors.append("Price below VWAP - wait for bounce confirmation")

            else:  # AT
                signal.signal = SignalType.BUY_CALLS
                signal.confidence = 60.0
                signal.decision_path += "Price AT VWAP → BUY CALLS (at fair value, gravity above)"
                signal.risk_factors.append("At equilibrium - watch for direction")

        elif signal.gravity_position == "BELOW":
            # Gravity below = bearish bias
            signal.decision_path += "Gravity BELOW price → "

            # Level 3: Price vs VWAP
            if signal.price_vs_vwap == "BELOW":
                # Strong bearish: Below VWAP + Gravity below + Short gamma
                signal.signal = SignalType.BUY_PUTS
                signal.confidence = 85.0
                signal.decision_path += "Price BELOW VWAP → BUY PUTS (strong bearish trend)"

            elif signal.price_vs_vwap == "ABOVE":
                # Bearish reversal: Above VWAP but gravity pulling down
                signal.signal = SignalType.BUY_PUTS
                signal.confidence = 70.0
                signal.decision_path += "Price ABOVE VWAP → BUY PUTS (bull trap, gravity pulling down)"
                signal.risk_factors.append("Price above VWAP - potential bull trap")

            else:  # AT
                signal.signal = SignalType.BUY_PUTS
                signal.confidence = 60.0
                signal.decision_path += "Price AT VWAP → BUY PUTS (at fair value, gravity below)"
                signal.risk_factors.append("At equilibrium - watch for direction")

        else:  # Gravity AT price
            signal.decision_path += "Gravity AT price → "
            # Neutral - no clear bias
            signal.signal = SignalType.NEUTRAL
            signal.confidence = 30.0
            signal.decision_path += "NEUTRAL (gravity at price, no clear bias)"
            signal.risk_factors.append("No clear gravity bias")

        return signal

    def _evaluate_long_gamma(self, signal: TradingSignal) -> TradingSignal:
        """
        Evaluate LONG GAMMA regime (GEX >= 0.35)
        Pinning/chop expected, dealer hedging dampens moves
        Look for mean reversion to gravity
        """
        signal.decision_path = "LONG GAMMA REGIME (pinning/chop) → "

        # Level 2: Gravity Position
        if signal.gravity_position == "ABOVE":
            # Gravity above = expect snap up
            signal.decision_path += "Gravity ABOVE price → "

            # Level 3: Price vs VWAP
            if signal.price_vs_vwap == "BELOW":
                # Below VWAP, gravity above = snap up expected
                signal.signal = SignalType.BUY_CALLS
                signal.confidence = 75.0
                signal.decision_path += "Price BELOW VWAP → BUY CALLS (snap up to gravity)"

            elif signal.price_vs_vwap == "ABOVE":
                # Above VWAP, gravity above = overextended
                signal.signal = SignalType.NEUTRAL
                signal.confidence = 40.0
                signal.decision_path += "Price ABOVE VWAP → NEUTRAL (overextended, gravity already above)"
                signal.risk_factors.append("Overextended - pinning likely")

            else:  # AT
                signal.signal = SignalType.BUY_CALLS
                signal.confidence = 50.0
                signal.decision_path += "Price AT VWAP → BUY CALLS (at fair value, expect pull to gravity)"
                signal.risk_factors.append("Long gamma regime - expect chop")

        elif signal.gravity_position == "BELOW":
            # Gravity below = expect snap down
            signal.decision_path += "Gravity BELOW price → "

            # Level 3: Price vs VWAP
            if signal.price_vs_vwap == "ABOVE":
                # Above VWAP, gravity below = snap down expected
                signal.signal = SignalType.BUY_PUTS
                signal.confidence = 75.0
                signal.decision_path += "Price ABOVE VWAP → BUY PUTS (snap down to gravity)"

            elif signal.price_vs_vwap == "BELOW":
                # Below VWAP, gravity below = overextended
                signal.signal = SignalType.NEUTRAL
                signal.confidence = 40.0
                signal.decision_path += "Price BELOW VWAP → NEUTRAL (overextended, gravity already below)"
                signal.risk_factors.append("Overextended - pinning likely")

            else:  # AT
                signal.signal = SignalType.BUY_PUTS
                signal.confidence = 50.0
                signal.decision_path += "Price AT VWAP → BUY PUTS (at fair value, expect pull to gravity)"
                signal.risk_factors.append("Long gamma regime - expect chop")

        else:  # Gravity AT price
            signal.decision_path += "Gravity AT price → "
            # Neutral - price at equilibrium in pinning regime
            signal.signal = SignalType.NEUTRAL
            signal.confidence = 20.0
            signal.decision_path += "NEUTRAL (price at gravity, pinning regime)"
            signal.risk_factors.append("Price at gravity in pinning regime - expect chop")

        return signal

    def _calculate_confluence(self, signal: TradingSignal) -> float:
        """
        Calculate confluence score (0-100)
        How many factors agree with the signal?
        """
        factors_in_agreement = 0
        total_factors = 0

        # Factor 1: Gamma Regime
        total_factors += 1
        if signal.signal == SignalType.BUY_CALLS and signal.gamma_regime == "SHORT_GAMMA":
            # Short gamma supports trending (good for calls if other factors bullish)
            if signal.gravity_position == "ABOVE":
                factors_in_agreement += 1
        elif signal.signal == SignalType.BUY_PUTS and signal.gamma_regime == "SHORT_GAMMA":
            if signal.gravity_position == "BELOW":
                factors_in_agreement += 1
        elif signal.gamma_regime == "LONG_GAMMA":
            # Long gamma supports mean reversion
            factors_in_agreement += 0.5  # Partial credit for reversion regime

        # Factor 2: Gravity
        total_factors += 1
        if signal.signal == SignalType.BUY_CALLS and signal.gravity_position == "ABOVE":
            factors_in_agreement += 1
        elif signal.signal == SignalType.BUY_PUTS and signal.gravity_position == "BELOW":
            factors_in_agreement += 1

        # Factor 3: VWAP
        total_factors += 1
        if signal.signal == SignalType.BUY_CALLS and signal.price_vs_vwap in ["BELOW", "AT"]:
            # Calls when below/at VWAP = good value
            factors_in_agreement += 1
        elif signal.signal == SignalType.BUY_PUTS and signal.price_vs_vwap in ["ABOVE", "AT"]:
            # Puts when above/at VWAP = good value
            factors_in_agreement += 1

        # Factor 4: Delta Flow (if available)
        if signal.delta_value != 0.0:
            total_factors += 1
            if signal.signal == SignalType.BUY_CALLS and signal.delta_flow_bias == "BULLISH":
                factors_in_agreement += 1
                signal.delta_confirmation = True
            elif signal.signal == SignalType.BUY_PUTS and signal.delta_flow_bias == "BEARISH":
                factors_in_agreement += 1
                signal.delta_confirmation = True

        # Calculate percentage
        if total_factors > 0:
            return (factors_in_agreement / total_factors) * 100
        return 50.0

    def get_signal_summary(self, num_signals: int = 10) -> dict:
        """
        Get summary of recent signals

        Args:
            num_signals: Number of recent signals to analyze

        Returns:
            Dictionary with signal statistics
        """
        if not self.signal_history:
            return {}

        recent = self.signal_history[-num_signals:]

        buy_calls = sum(1 for s in recent if s.signal == SignalType.BUY_CALLS)
        buy_puts = sum(1 for s in recent if s.signal == SignalType.BUY_PUTS)
        neutral = sum(1 for s in recent if s.signal == SignalType.NEUTRAL)

        avg_confidence = sum(s.confidence for s in recent) / len(recent)
        avg_confluence = sum(s.confluence_score for s in recent) / len(recent)

        current_signal = recent[-1]

        return {
            "current_signal": current_signal.signal.value,
            "current_strength": current_signal.strength.value,
            "current_confidence": round(current_signal.confidence, 1),
            "recent_signals": {
                "buy_calls": buy_calls,
                "buy_puts": buy_puts,
                "neutral": neutral
            },
            "avg_confidence": round(avg_confidence, 1),
            "avg_confluence": round(avg_confluence, 1),
            "delta_confirmation": current_signal.delta_confirmation
        }
