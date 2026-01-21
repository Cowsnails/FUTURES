#!/usr/bin/env python
"""
Futures Charting Application - Single Command Startup

This script starts the FastAPI server and optionally opens a browser.
"""

import uvicorn
import webbrowser
import threading
import time
import argparse
import sys
import yaml
from pathlib import Path


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / 'config.yaml'

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def open_browser(host: str, port: int, delay: float = 2.0):
    """Open browser after a delay"""
    time.sleep(delay)
    url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    print(f"\n🌐 Opening browser at {url}")
    webbrowser.open(url)


def print_startup_banner(config):
    """Print startup information"""
    print("\n" + "=" * 70)
    print("  📈 Futures Charting Application")
    print("=" * 70)
    print(f"  IB Gateway: {config['ib_gateway']['host']}:{config['ib_gateway']['port']}")
    print(f"  Server: http://localhost:{config['server']['port']}")
    print(f"  Cache: {config['data']['cache_dir']}")
    print("=" * 70)
    print("\n⚙️  Starting server...\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Start the Futures Charting application'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development'
    )
    parser.add_argument(
        '--host',
        type=str,
        help='Override host from config'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='Override port from config'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Override log level from config'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Get server settings
    host = args.host or config['server']['host']
    port = args.port or config['server']['port']
    log_level = args.log_level or config['server']['log_level']
    reload = args.reload or config['server']['reload']

    # Print startup banner
    print_startup_banner(config)

    # Open browser in background thread (unless --no-browser)
    if not args.no_browser:
        browser_thread = threading.Thread(
            target=open_browser,
            args=(host, port, 2.0),
            daemon=True
        )
        browser_thread.start()

    # Start server
    try:
        uvicorn.run(
            "backend.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
