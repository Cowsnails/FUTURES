#!/usr/bin/env python3
"""
Load Testing Script for Futures Charting Application

This script simulates multiple concurrent WebSocket connections to test:
- Server capacity under load
- Memory usage with multiple clients
- Response time degradation
- WebSocket connection stability
- Indicator calculation performance

Usage:
    python tests/load_test.py --clients 10 --duration 60
    python tests/load_test.py --clients 50 --duration 300 --profile
"""

import asyncio
import websockets
import json
import time
import argparse
import statistics
from datetime import datetime
from typing import List, Dict
import psutil
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class LoadTestClient:
    """Simulates a single WebSocket client"""

    def __init__(self, client_id: int, url: str = "ws://localhost:8000/ws"):
        self.client_id = client_id
        self.url = url
        self.messages_received = 0
        self.errors = 0
        self.response_times = []
        self.connected = False
        self.start_time = None

    async def connect_and_subscribe(self, symbol: str = "MNQ", duration: int = 60):
        """Connect to WebSocket and subscribe to a symbol"""
        try:
            async with websockets.connect(self.url) as websocket:
                self.connected = True
                self.start_time = time.time()

                # Subscribe to symbol
                subscribe_msg = {
                    "action": "subscribe",
                    "symbol": symbol,
                    "bar_size": "1 min"
                }

                send_time = time.time()
                await websocket.send(json.dumps(subscribe_msg))

                # Receive messages for the specified duration
                end_time = time.time() + duration

                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        receive_time = time.time()

                        self.messages_received += 1

                        # Parse message
                        data = json.loads(message)

                        # Track response time for historical data
                        if data.get('type') == 'historical':
                            response_time = receive_time - send_time
                            self.response_times.append(response_time)

                    except asyncio.TimeoutError:
                        # No message received in timeout period (normal for real-time)
                        continue
                    except json.JSONDecodeError as e:
                        self.errors += 1
                    except Exception as e:
                        self.errors += 1

        except Exception as e:
            self.errors += 1
            print(f"Client {self.client_id} error: {e}")
        finally:
            self.connected = False

    def get_stats(self) -> Dict:
        """Get client statistics"""
        return {
            'client_id': self.client_id,
            'messages_received': self.messages_received,
            'errors': self.errors,
            'avg_response_time': statistics.mean(self.response_times) if self.response_times else 0,
            'max_response_time': max(self.response_times) if self.response_times else 0,
            'min_response_time': min(self.response_times) if self.response_times else 0,
        }


class LoadTester:
    """Orchestrates load testing with multiple clients"""

    def __init__(self, num_clients: int, duration: int, url: str = "ws://localhost:8000/ws"):
        self.num_clients = num_clients
        self.duration = duration
        self.url = url
        self.clients: List[LoadTestClient] = []
        self.start_time = None
        self.end_time = None
        self.process = psutil.Process()

    async def run(self):
        """Run load test"""
        print(f"\n{'='*60}")
        print(f"Load Test Configuration")
        print(f"{'='*60}")
        print(f"Concurrent Clients: {self.num_clients}")
        print(f"Duration: {self.duration}s")
        print(f"WebSocket URL: {self.url}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # Create clients
        self.clients = [LoadTestClient(i, self.url) for i in range(self.num_clients)]

        # Get initial memory usage
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        print(f"Initial Memory Usage: {initial_memory:.2f} MB")
        print(f"\nStarting {self.num_clients} concurrent connections...\n")

        self.start_time = time.time()

        # Run all clients concurrently
        symbols = ["MNQ", "MES", "MGC"]
        tasks = []
        for i, client in enumerate(self.clients):
            # Distribute clients across different symbols
            symbol = symbols[i % len(symbols)]
            task = asyncio.create_task(client.connect_and_subscribe(symbol, self.duration))
            tasks.append(task)

        # Monitor progress
        monitor_task = asyncio.create_task(self.monitor_progress())

        # Wait for all clients to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Stop monitoring
        monitor_task.cancel()

        self.end_time = time.time()

        # Get final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Print results
        self.print_results(initial_memory, final_memory)

    async def monitor_progress(self):
        """Monitor and print progress during test"""
        try:
            while True:
                await asyncio.sleep(10)

                active_clients = sum(1 for c in self.clients if c.connected)
                total_messages = sum(c.messages_received for c in self.clients)
                total_errors = sum(c.errors for c in self.clients)

                elapsed = time.time() - self.start_time
                memory = self.process.memory_info().rss / 1024 / 1024

                print(f"[{elapsed:.0f}s] Active: {active_clients}/{self.num_clients} | "
                      f"Messages: {total_messages} | Errors: {total_errors} | "
                      f"Memory: {memory:.2f} MB")

        except asyncio.CancelledError:
            pass

    def print_results(self, initial_memory: float, final_memory: float):
        """Print comprehensive test results"""
        print(f"\n{'='*60}")
        print(f"Load Test Results")
        print(f"{'='*60}")

        elapsed = self.end_time - self.start_time

        # Aggregate statistics
        total_messages = sum(c.messages_received for c in self.clients)
        total_errors = sum(c.errors for c in self.clients)
        all_response_times = []
        for c in self.clients:
            all_response_times.extend(c.response_times)

        print(f"\n📊 Overall Statistics:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Total Messages Received: {total_messages}")
        print(f"  Total Errors: {total_errors}")
        print(f"  Messages/Second: {total_messages / elapsed:.2f}")
        print(f"  Error Rate: {(total_errors / max(total_messages, 1)) * 100:.2f}%")

        if all_response_times:
            print(f"\n⏱️  Response Times (Historical Data):")
            print(f"  Average: {statistics.mean(all_response_times):.3f}s")
            print(f"  Median: {statistics.median(all_response_times):.3f}s")
            print(f"  Min: {min(all_response_times):.3f}s")
            print(f"  Max: {max(all_response_times):.3f}s")
            if len(all_response_times) > 1:
                print(f"  Std Dev: {statistics.stdev(all_response_times):.3f}s")

        print(f"\n💾 Memory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Delta: {final_memory - initial_memory:.2f} MB")
        print(f"  Per Client: {(final_memory - initial_memory) / self.num_clients:.2f} MB")

        # Per-client statistics
        print(f"\n👥 Per-Client Statistics:")
        print(f"  {'Client':<10} {'Messages':<12} {'Errors':<10} {'Avg Response':<15}")
        print(f"  {'-'*50}")

        for client in self.clients[:10]:  # Show first 10 clients
            stats = client.get_stats()
            print(f"  {stats['client_id']:<10} "
                  f"{stats['messages_received']:<12} "
                  f"{stats['errors']:<10} "
                  f"{stats['avg_response_time']:.3f}s")

        if self.num_clients > 10:
            print(f"  ... and {self.num_clients - 10} more clients")

        # Performance assessment
        print(f"\n✅ Performance Assessment:")

        avg_response = statistics.mean(all_response_times) if all_response_times else 0

        if avg_response < 2.0:
            print(f"  ✅ Excellent: Average response time < 2s")
        elif avg_response < 5.0:
            print(f"  ⚠️  Good: Average response time < 5s")
        else:
            print(f"  ❌ Poor: Average response time >= 5s")

        error_rate = (total_errors / max(total_messages, 1)) * 100
        if error_rate < 1.0:
            print(f"  ✅ Excellent: Error rate < 1%")
        elif error_rate < 5.0:
            print(f"  ⚠️  Acceptable: Error rate < 5%")
        else:
            print(f"  ❌ High: Error rate >= 5%")

        memory_per_client = (final_memory - initial_memory) / self.num_clients
        if memory_per_client < 10:
            print(f"  ✅ Excellent: Memory usage < 10 MB per client")
        elif memory_per_client < 50:
            print(f"  ⚠️  Acceptable: Memory usage < 50 MB per client")
        else:
            print(f"  ❌ High: Memory usage >= 50 MB per client")

        print(f"\n{'='*60}\n")

        # Recommendations
        print(f"📝 Recommendations:")

        if avg_response > 3.0:
            print(f"  - Consider caching historical data")
            print(f"  - Optimize indicator calculations")
            print(f"  - Use message batching for real-time updates")

        if error_rate > 2.0:
            print(f"  - Review error logs for connection issues")
            print(f"  - Check server resource limits")
            print(f"  - Consider rate limiting")

        if memory_per_client > 20:
            print(f"  - Profile memory usage per connection")
            print(f"  - Consider connection pooling")
            print(f"  - Review data structures for leaks")

        if not any([avg_response > 3.0, error_rate > 2.0, memory_per_client > 20]):
            print(f"  ✅ System performing well under load!")

        print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Load test the futures charting application')
    parser.add_argument('--clients', type=int, default=10,
                      help='Number of concurrent clients (default: 10)')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds (default: 60)')
    parser.add_argument('--url', type=str, default='ws://localhost:8000/ws',
                      help='WebSocket URL (default: ws://localhost:8000/ws)')

    args = parser.parse_args()

    # Validate arguments
    if args.clients < 1:
        print("Error: --clients must be at least 1")
        sys.exit(1)

    if args.duration < 10:
        print("Error: --duration must be at least 10 seconds")
        sys.exit(1)

    # Run load test
    tester = LoadTester(args.clients, args.duration, args.url)

    try:
        asyncio.run(tester.run())
    except KeyboardInterrupt:
        print("\n\nLoad test interrupted by user")
    except Exception as e:
        print(f"\n\nLoad test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
