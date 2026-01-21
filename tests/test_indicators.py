"""
Tests for Indicator System

Tests indicator calculations and manager functionality.
"""

import pytest
import pandas as pd
import numpy as np
from backend.indicators import (
    SMA, EMA, RSI, MACD, BollingerBands,
    IndicatorManager, create_indicator
)


# Sample data fixture
@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n = 100

    # Generate realistic price data
    close = np.cumsum(np.random.randn(n) * 0.01) + 100
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.3

    return pd.DataFrame({
        'time': range(n),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })


class TestSMA:
    """Test Simple Moving Average"""

    def test_sma_calculation(self, sample_data):
        """Test SMA calculates correctly"""
        sma = SMA(period=20)
        result = sma.calculate(sample_data)

        assert 'time' in result.columns
        assert 'value' in result.columns
        assert len(result) <= len(sample_data)
        assert not result['value'].isna().all()

    def test_sma_period_validation(self):
        """Test SMA period validation"""
        with pytest.raises(ValueError):
            SMA(period=1)  # Too small

    def test_sma_plot_config(self):
        """Test SMA plot configuration"""
        sma = SMA(period=20, color='#FF0000')
        config = sma.plot_config

        assert config['type'] == 'line'
        assert config['pane'] == 'main'
        assert '#FF0000' in str(config)


class TestEMA:
    """Test Exponential Moving Average"""

    def test_ema_calculation(self, sample_data):
        """Test EMA calculates correctly"""
        ema = EMA(period=20)
        result = ema.calculate(sample_data)

        assert len(result) <= len(sample_data)
        assert 'value' in result.columns

    def test_ema_different_from_sma(self, sample_data):
        """Test EMA differs from SMA (more recent weight)"""
        sma = SMA(period=20)
        ema = EMA(period=20)

        sma_result = sma.calculate(sample_data)
        ema_result = ema.calculate(sample_data)

        # EMA should differ from SMA (even if slightly)
        # They're close but not identical due to different weighting
        assert not np.array_equal(
            sma_result['value'].values[-10:],
            ema_result['value'].values[-10:]
        )


class TestRSI:
    """Test Relative Strength Index"""

    def test_rsi_calculation(self, sample_data):
        """Test RSI calculates correctly"""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_data)

        assert 'value' in result.columns
        assert len(result) > 0

    def test_rsi_range(self, sample_data):
        """Test RSI stays within 0-100 range"""
        rsi = RSI(period=14)
        result = rsi.calculate(sample_data)

        values = result['value'].dropna()
        assert values.min() >= 0
        assert values.max() <= 100

    def test_rsi_plot_config(self):
        """Test RSI renders in separate pane"""
        rsi = RSI(period=14)
        config = rsi.plot_config

        assert config['pane'] == 'separate'
        assert 'levels' in config
        assert 30 in config['levels']
        assert 70 in config['levels']


class TestMACD:
    """Test Moving Average Convergence Divergence"""

    def test_macd_calculation(self, sample_data):
        """Test MACD calculates all three series"""
        macd = MACD(fast=12, slow=26, signal=9)
        result = macd.calculate(sample_data)

        assert 'macd' in result.columns
        assert 'signal' in result.columns
        assert 'histogram' in result.columns
        assert len(result) > 0

    def test_macd_histogram(self, sample_data):
        """Test MACD histogram is difference of MACD and signal"""
        macd = MACD()
        result = macd.calculate(sample_data)

        # Check histogram = macd - signal (where not NaN)
        valid = result.dropna()
        if len(valid) > 0:
            calculated_hist = valid['macd'] - valid['signal']
            assert np.allclose(calculated_hist, valid['histogram'], rtol=1e-5)

    def test_macd_plot_config(self):
        """Test MACD special plot configuration"""
        macd = MACD()
        config = macd.plot_config

        assert config['type'] == 'macd'
        assert 'macdColor' in config
        assert 'signalColor' in config
        assert 'histogramUpColor' in config


class TestBollingerBands:
    """Test Bollinger Bands"""

    def test_bb_calculation(self, sample_data):
        """Test Bollinger Bands calculates three bands"""
        bb = BollingerBands(period=20, std_dev=2.0)
        result = bb.calculate(sample_data)

        assert 'middle' in result.columns
        assert 'upper' in result.columns
        assert 'lower' in result.columns

    def test_bb_band_ordering(self, sample_data):
        """Test upper > middle > lower"""
        bb = BollingerBands(period=20)
        result = bb.calculate(sample_data)

        valid = result.dropna()
        if len(valid) > 0:
            assert (valid['upper'] >= valid['middle']).all()
            assert (valid['middle'] >= valid['lower']).all()


class TestIndicatorManager:
    """Test Indicator Manager"""

    def test_manager_init(self):
        """Test manager initialization"""
        manager = IndicatorManager()
        assert len(manager.active_indicators) == 0

    def test_add_indicator(self):
        """Test adding indicators"""
        manager = IndicatorManager()

        sma = manager.add_indicator('sma', {'period': 20})
        assert sma is not None
        assert len(manager.active_indicators) == 1

        ema = manager.add_indicator('ema', {'period': 50})
        assert ema is not None
        assert len(manager.active_indicators) == 2

    def test_add_invalid_indicator(self):
        """Test adding invalid indicator type"""
        manager = IndicatorManager()

        result = manager.add_indicator('invalid_indicator')
        assert result is None
        assert len(manager.active_indicators) == 0

    def test_remove_indicator(self):
        """Test removing indicators"""
        manager = IndicatorManager()

        sma = manager.add_indicator('sma', {'period': 20})
        ind_id = sma.id

        assert len(manager.active_indicators) == 1

        success = manager.remove_indicator(ind_id)
        assert success
        assert len(manager.active_indicators) == 0

    def test_calculate_all(self, sample_data):
        """Test calculating all active indicators"""
        manager = IndicatorManager()

        manager.add_indicator('sma', {'period': 20})
        manager.add_indicator('rsi', {'period': 14})

        results = manager.calculate_all(sample_data)

        assert len(results) == 2
        for ind_id, result in results.items():
            assert 'data' in result
            assert 'config' in result
            assert 'metadata' in result

    def test_clear_all(self):
        """Test clearing all indicators"""
        manager = IndicatorManager()

        manager.add_indicator('sma')
        manager.add_indicator('ema')
        manager.add_indicator('rsi')

        assert len(manager.active_indicators) == 3

        manager.clear_all_indicators()
        assert len(manager.active_indicators) == 0

    def test_get_statistics(self):
        """Test getting manager statistics"""
        manager = IndicatorManager()

        manager.add_indicator('sma')
        manager.add_indicator('rsi')

        stats = manager.get_statistics()

        assert stats['active_indicators'] == 2
        assert len(stats['indicators']) == 2


class TestIndicatorFactory:
    """Test indicator factory function"""

    def test_create_indicator(self):
        """Test create_indicator factory"""
        sma = create_indicator('sma', period=20)
        assert sma is not None
        assert isinstance(sma, SMA)

        rsi = create_indicator('rsi', period=14)
        assert rsi is not None
        assert isinstance(rsi, RSI)

    def test_create_invalid(self):
        """Test creating invalid indicator"""
        result = create_indicator('invalid_type')
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
