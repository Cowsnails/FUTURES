"""
Integrated Multi-Factor Analysis System
========================================

Combines all analysis modules into a unified system:
- Delta Flow Analysis
- GEX Ratio Calculation
- Gravity Detection
- Decision Tree Signals- Real-time integration with price dataProvides comprehensive "cockpit view" of market conditions.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .delta_flow import DeltaFlowAnalyzer, DeltaMetrics
from .gex_analyzer import GEXAnalyzer, GEXMetrics
from .gravity import GravityAnalyzer, GravityMetrics
from .decision_tree import DecisionTreeEngine, TradingSignal

logger = logging.getLogger(__name__)


class IntegratedAnalyzer:
    """
    Unified analysis system that combines all factors.

    Processes each bar and tick to generate comprehensive market analysis.
    """

    def __init__(self):
        """Initialize all analysis modules"""
        self.delta_analyzer = DeltaFlowAnalyzer(lookback_bars=50)
        self.gex_analyzer = GEXAnalyzer(lookback_period=50)
        self.gravity_analyzer = GravityAnalyzer(lookback_period=100)
        self.decision_engine = DecisionTreeEngine()

        logger.info("IntegratedAnalyzer initialized with all modules")

    def process_tick(self, timestamp: datetime, price: float, size: int):
        """
        Process individual tick for delta analysis

        Args:
            timestamp: Tick timestamp
            price: Trade price
            size: Trade size
        """
        self.delta_analyzer.process_tick(timestamp, price, size)

    def process_bar(self,
                   timestamp: datetime,
                   open_price: float,
                   high: float,
                   low: float,
                   close: float,
                   volume: int,
                   is_new_bar: bool) -> Dict[str, Any]:
        """
        Process completed or updated bar and generate full analysis

        Args:
            timestamp: Bar timestamp
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            is_new_bar: Whether this is a new bar or update

        Returns:
            Dictionary with all analysis results
        """
        # Only finalize delta on new bar
        if is_new_bar:
            delta_metrics = self.delta_analyzer.finalize_bar(
                timestamp, open_price, close
            )
        else:
            delta_metrics = self.delta_analyzer.get_current_metrics()

        # Update GEX analysis
        gex_metrics = self.gex_analyzer.update(
            timestamp, open_price, high, low, close, volume
        )

        # Update Gravity analysis
        gravity_metrics = self.gravity_analyzer.update(
            timestamp, open_price, high, low, close, volume
        )

        # Generate trading signal from decision tree
        signal = self.decision_engine.evaluate(
            timestamp=timestamp,
            current_price=close,
            gex_ratio=gex_metrics.gex_ratio,
            gamma_regime=gex_metrics.regime,
            gravity_price=gravity_metrics.gravity_price,
            gravity_above=gravity_metrics.gravity_above_price,
            gravity_below=gravity_metrics.gravity_below_price,
            vwap_price=gravity_metrics.vwap_level,
            delta_flow=delta_metrics.raw_delta if delta_metrics else None
        )

        # Compile comprehensive analysis
        analysis = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,

            # Price Data
            'price': {
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            },

            # Delta Flow Analysis
            'delta': self._format_delta_metrics(delta_metrics) if delta_metrics else None,

            # GEX Analysis
            'gex': self._format_gex_metrics(gex_metrics),

            # Gravity Analysis
            'gravity': self._format_gravity_metrics(gravity_metrics),

            # Trading Signal
            'signal': self._format_signal(signal),

            # Market State Summary
            'market_state': {
                'regime': gex_metrics.regime,
                'trend': self._determine_trend(delta_metrics, gravity_metrics),
                'bias': gravity_metrics.bias,
                'volatility': gex_metrics.current_volatility
            }
        }

        return analysis

    def _format_delta_metrics(self, metrics: DeltaMetrics) -> Dict[str, Any]:
        """Format delta metrics for output"""
        return {
            'buy_volume': metrics.buy_volume,
            'sell_volume': metrics.sell_volume,
            'total_volume': metrics.total_volume,
            'raw_delta': metrics.raw_delta,
            'delta_percent': round(metrics.delta_percent, 2),
            'cumulative_delta': metrics.cumulative_delta,
            'delta_momentum': metrics.delta_momentum,
            'delta_divergence': metrics.delta_divergence,
            'large_trade_delta': metrics.large_trade_delta,
            'absorption_score': round(metrics.absorption_score, 2),
            'exhaustion_level': round(metrics.exhaustion_level, 3),
            'tick_count': metrics.tick_count,
            'avg_trade_size': round(metrics.avg_trade_size, 1)
        }

    def _format_gex_metrics(self, metrics: GEXMetrics) -> Dict[str, Any]:
        """Format GEX metrics for output"""
        return {
            'gex_ratio': round(metrics.gex_ratio, 3),
            'regime': metrics.regime,
            'regime_strength': round(metrics.regime_strength, 1),
            'trending_regime': metrics.trending_regime,
            'pinning_regime': metrics.pinning_regime,
            'volatility_factor': round(metrics.volatility_factor, 3),
            'volume_concentration': round(metrics.volume_concentration, 3),
            'reversion_tendency': round(metrics.reversion_tendency, 3),
            'round_number_pin': round(metrics.round_number_pin, 3),
            'current_volatility': round(metrics.current_volatility, 2),
            'price_at_strike': metrics.price_at_strike
        }

    def _format_gravity_metrics(self, metrics: GravityMetrics) -> Dict[str, Any]:
        """Format gravity metrics for output"""
        return {
            'gravity_price': round(metrics.gravity_price, 2),
            'gravity_strength': round(metrics.gravity_strength, 1),
            'gravity_above_price': metrics.gravity_above_price,
            'gravity_below_price': metrics.gravity_below_price,
            'gravity_at_price': metrics.gravity_at_price,
            'distance_to_gravity': round(metrics.distance_to_gravity, 2),
            'distance_percent': round(metrics.distance_percent, 3),
            'poc_level': round(metrics.poc_level, 2) if metrics.poc_level > 0 else None,
            'vwap_level': round(metrics.vwap_level, 2) if metrics.vwap_level > 0 else None,
            'swing_high': round(metrics.swing_high, 2) if metrics.swing_high > 0 else None,
            'swing_low': round(metrics.swing_low, 2) if metrics.swing_low > 0 else None,
            'bias': metrics.bias,
            'bias_strength': round(metrics.bias_strength, 1)
        }

    def _format_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Format trading signal for output"""
        return {
            'signal': signal.signal.value,
            'strength': signal.strength.value,
            'confidence': round(signal.confidence, 1),
            'decision_path': signal.decision_path,
            'risk_factors': signal.risk_factors,
            'confluence_score': round(signal.confluence_score, 1),
            'delta_confirmation': signal.delta_confirmation,

            # Decision factors
            'factors': {
                'gamma_regime': signal.gamma_regime,
                'gravity_position': signal.gravity_position,
                'price_vs_vwap': signal.price_vs_vwap,
                'delta_bias': signal.delta_flow_bias
            }
        }

    def _determine_trend(self,
                        delta_metrics: Optional[DeltaMetrics],
                        gravity_metrics: GravityMetrics) -> str:
        """
        Determine overall trend direction

        Returns:
            "UPTREND", "DOWNTREND", or "SIDEWAYS"
        """
        factors = []

        # Factor 1: Gravity position
        if gravity_metrics.gravity_above_price:
            factors.append(1)
        elif gravity_metrics.gravity_below_price:
            factors.append(-1)
        else:
            factors.append(0)

        # Factor 2: Delta flow
        if delta_metrics and delta_metrics.raw_delta != 0:
            if delta_metrics.raw_delta > 0:
                factors.append(1)
            else:
                factors.append(-1)

        # Factor 3: Price vs VWAP
        if gravity_metrics.vwap_level > 0:
            # This would need current price, skipping for now
            pass

        # Calculate trend
        if not factors:
            return "SIDEWAYS"

        avg = sum(factors) / len(factors)
        if avg > 0.3:
            return "UPTREND"
        elif avg < -0.3:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"

    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all analysis

        Returns:
            Dictionary with high-level summary
        """
        delta_profile = self.delta_analyzer.get_delta_profile(10)
        signal_summary = self.decision_engine.get_signal_summary(10)

        current_gex = self.gex_analyzer.get_current_metrics()
        current_gravity = self.gravity_analyzer.get_current_metrics()
        current_delta = self.delta_analyzer.get_current_metrics()

        return {
            'delta_profile': delta_profile,
            'signal_summary': signal_summary,
            'current_state': {
                'gex_regime': current_gex.regime if current_gex else None,
                'gravity_bias': current_gravity.bias if current_gravity else None,
                'cumulative_delta': current_delta.cumulative_delta if current_delta else 0
            }
        }

    def reset(self):
        """Reset all analyzers (for new day/session)"""
        self.delta_analyzer.reset()
        self.gex_analyzer = GEXAnalyzer(lookback_period=50)
        self.gravity_analyzer.reset_vwap()
        logger.info("IntegratedAnalyzer reset for new session")
