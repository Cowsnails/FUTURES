#!/usr/bin/env python3
"""
Memory Profiler for Futures Charting Application

This script profiles memory usage of key components:
- Historical data loading and caching
- Indicator calculations
- WebSocket connections
- Data structure memory footprint

Usage:
    python tests/memory_profiler.py --profile all
    python tests/memory_profiler.py --profile indicators
    python tests/memory_profiler.py --profile cache
    python tests/memory_profiler.py --leak-test
"""

import argparse
import sys
import os
import time
import gc
from datetime import datetime, timedelta
import tracemalloc
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.indicators import IndicatorManager, SMA, EMA, RSI, MACD, BollingerBands
from backend.cache import DataCache
from backend.contracts import get_current_contract


def format_bytes(bytes_val):
    """Format bytes as human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def profile_decorator(func):
    """Decorator to profile memory usage of a function"""
    def wrapper(*args, **kwargs):
        # Start tracing
        tracemalloc.start()
        gc.collect()

        snapshot_before = tracemalloc.take_snapshot()
        mem_before = tracemalloc.get_traced_memory()[0]

        print(f"\n{'='*60}")
        print(f"Profiling: {func.__name__}")
        print(f"{'='*60}")
        print(f"Memory before: {format_bytes(mem_before)}")

        start_time = time.time()

        # Run function
        result = func(*args, **kwargs)

        elapsed = time.time() - start_time

        # Get memory after
        mem_after = tracemalloc.get_traced_memory()[0]
        snapshot_after = tracemalloc.take_snapshot()

        print(f"Memory after:  {format_bytes(mem_after)}")
        print(f"Delta:         {format_bytes(mem_after - mem_before)}")
        print(f"Time:          {elapsed:.3f}s")

        # Show top memory allocations
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        print(f"\nTop 5 Memory Allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        tracemalloc.stop()

        print(f"{'='*60}\n")

        return result

    return wrapper


def create_sample_data(num_bars: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    base_price = 100.0
    close_prices = []
    current = base_price

    for _ in range(num_bars):
        change = np.random.randn() * 0.02  # 2% volatility
        current *= (1 + change)
        close_prices.append(current)

    close = np.array(close_prices)
    high = close + np.abs(np.random.randn(num_bars) * 0.5)
    low = close - np.abs(np.random.randn(num_bars) * 0.5)
    open_price = close + np.random.randn(num_bars) * 0.3

    return pd.DataFrame({
        'time': range(num_bars),
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, num_bars)
    })


@profile_decorator
def profile_indicator_calculations(num_bars: int = 50000):
    """Profile indicator calculations with large dataset"""
    print(f"Creating dataset with {num_bars} bars...")

    data = create_sample_data(num_bars)

    print(f"Dataset size: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Create indicator manager
    manager = IndicatorManager()

    # Add multiple indicators
    indicators = [
        ('sma', {'period': 20}),
        ('sma', {'period': 50}),
        ('ema', {'period': 20}),
        ('ema', {'period': 50}),
        ('rsi', {'period': 14}),
        ('macd', {}),
        ('bb', {'period': 20}),
    ]

    print(f"\nAdding {len(indicators)} indicators...")

    for ind_type, params in indicators:
        manager.add_indicator(ind_type, params)

    print(f"Calculating all indicators on {num_bars} bars...")

    # Calculate all indicators
    results = manager.calculate_all(data)

    print(f"\nCalculated {len(results)} indicators")

    # Calculate total size of results
    total_result_size = 0
    for ind_id, ind_data in results.items():
        if 'data' in ind_data:
            df = pd.DataFrame(ind_data['data'])
            size = df.memory_usage(deep=True).sum()
            total_result_size += size
            print(f"  {ind_id}: {len(df)} points, {format_bytes(size)}")

    print(f"\nTotal results size: {format_bytes(total_result_size)}")

    return results


@profile_decorator
def profile_cache_operations():
    """Profile cache read/write operations"""
    cache = DataCache()

    # Create sample data
    num_bars = 100000
    print(f"Creating {num_bars} bars of data...")

    data = create_sample_data(num_bars)

    print(f"DataFrame size: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Cache the data
    symbol = "TEST_MNQ"
    bar_size = "1 min"

    print(f"\nCaching data...")
    cache.save(symbol, bar_size, data)

    # Read from cache
    print(f"\nReading from cache...")
    cached_data = cache.load(symbol, bar_size)

    if cached_data is not None:
        print(f"Loaded {len(cached_data)} bars from cache")
        print(f"Cached data size: {cached_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    else:
        print("Failed to load from cache")

    # Get cache info
    info = cache.get_info(symbol, bar_size)
    if info:
        print(f"\nCache Info:")
        print(f"  File size: {format_bytes(info.get('file_size', 0))}")
        print(f"  Bar count: {info.get('bar_count', 0)}")
        print(f"  Compression: {info.get('file_size', 0) / (data.memory_usage(deep=True).sum()) * 100:.1f}%")

    return cached_data


@profile_decorator
def profile_data_structures():
    """Profile memory usage of common data structures"""
    print("Creating various data structures...")

    # Test DataFrame sizes
    sizes = [1000, 10000, 100000]

    for size in sizes:
        df = create_sample_data(size)
        mem_usage = df.memory_usage(deep=True).sum()
        print(f"\nDataFrame with {size} rows:")
        print(f"  Memory: {format_bytes(mem_usage)}")
        print(f"  Per row: {format_bytes(mem_usage / size)}")

    # Test array vs list
    print("\n\nArray vs List comparison (1M elements):")

    # NumPy array
    arr = np.random.randn(1000000)
    arr_size = arr.nbytes
    print(f"  NumPy array: {format_bytes(arr_size)}")

    # Python list
    lst = list(arr)
    lst_size = sys.getsizeof(lst) + sum(sys.getsizeof(x) for x in lst[:100])  # Sample
    print(f"  Python list (estimated): {format_bytes(lst_size * 10000)}")

    return True


def run_leak_test(iterations: int = 100):
    """Test for memory leaks by running operations repeatedly"""
    print(f"\n{'='*60}")
    print(f"Memory Leak Test ({iterations} iterations)")
    print(f"{'='*60}\n")

    tracemalloc.start()

    initial_memory = tracemalloc.get_traced_memory()[0]
    print(f"Initial memory: {format_bytes(initial_memory)}")

    manager = IndicatorManager()
    data = create_sample_data(1000)

    memory_samples = []

    for i in range(iterations):
        # Add and remove indicators repeatedly
        ind_id = manager.add_indicator('sma', {'period': 20})
        results = manager.calculate_all(data)
        manager.remove_indicator(ind_id)

        if i % 10 == 0:
            gc.collect()
            current_memory = tracemalloc.get_traced_memory()[0]
            memory_samples.append(current_memory)
            print(f"  Iteration {i:3d}: {format_bytes(current_memory)} "
                  f"(+{format_bytes(current_memory - initial_memory)})")

    final_memory = tracemalloc.get_traced_memory()[0]

    print(f"\nFinal memory:   {format_bytes(final_memory)}")
    print(f"Memory growth:  {format_bytes(final_memory - initial_memory)}")
    print(f"Growth per iteration: {format_bytes((final_memory - initial_memory) / iterations)}")

    # Analyze trend
    if len(memory_samples) > 1:
        growth_rate = (memory_samples[-1] - memory_samples[0]) / len(memory_samples)
        print(f"Average growth rate: {format_bytes(growth_rate)} per 10 iterations")

        if growth_rate < 10000:  # Less than 10KB per 10 iterations
            print(f"\n✅ No significant memory leak detected")
        elif growth_rate < 100000:  # Less than 100KB per 10 iterations
            print(f"\n⚠️  Possible minor memory leak")
        else:
            print(f"\n❌ Significant memory leak detected!")

    tracemalloc.stop()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Profile memory usage of the application')
    parser.add_argument('--profile', type=str, default='all',
                      choices=['all', 'indicators', 'cache', 'structures'],
                      help='What to profile (default: all)')
    parser.add_argument('--leak-test', action='store_true',
                      help='Run memory leak test')
    parser.add_argument('--bars', type=int, default=50000,
                      help='Number of bars for indicator profiling (default: 50000)')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of iterations for leak test (default: 100)')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# Memory Profiler for Futures Charting Application")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")

    try:
        if args.leak_test:
            run_leak_test(args.iterations)
        elif args.profile == 'all':
            profile_data_structures()
            profile_indicator_calculations(args.bars)
            profile_cache_operations()
            print("\n✅ All profiling completed successfully\n")
        elif args.profile == 'indicators':
            profile_indicator_calculations(args.bars)
        elif args.profile == 'cache':
            profile_cache_operations()
        elif args.profile == 'structures':
            profile_data_structures()

    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
    except Exception as e:
        print(f"\n\nProfiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
