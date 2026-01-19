# Futures Charting Application

Professional-grade real-time futures charting webapp using IB Gateway and TradingView Lightweight Charts.

## Features

- **Real-Time Streaming**: Sub-second updates for MNQ, MES, MGC futures
- **Historical Data**: 1 year of 1-minute bars with intelligent caching
- **TradingView Quality**: Exact color scheme and visual fidelity
- **Production Ready**: Comprehensive error handling and auto-recovery
- **Performance Optimized**: Handles 60K+ bars smoothly

## Quick Start

### Prerequisites

1. **IB Gateway** installed and running
2. **Market Data Subscriptions**:
   - CME Real-Time (Level 1) for MNQ, MES
   - COMEX for MGC (included in CME bundle)
3. **API Acknowledgement Form** completed in Client Portal
4. **Python 3.8+** installed

### Installation

```bash
# Clone repository
git clone <repository-url>
cd FUTURES

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your IB Gateway settings
```

### Configuration

Edit `config.yaml` to configure:
- IB Gateway connection (host, port, client ID)
- Server settings (host, port)
- Cache settings (directory, max age)
- Contract expiration dates (update quarterly)

### First Run - Test Connection

```bash
# Verify IB Gateway connection and market data
python test_connection.py
```

This will test:
- ✓ IB Gateway connection
- ✓ Contract qualification (MNQ, MES, MGC)
- ✓ Market data subscription
- ✓ Historical data access

### Start Application

```bash
# Start server (auto-opens browser)
python run.py

# Or without browser
python run.py --no-browser

# Development mode with auto-reload
python run.py --reload
```

The application will be available at `http://localhost:8000`

## Architecture

```
┌─────────────────┐     ┌──────────────────────────┐     ┌─────────────────────────┐
│   IB Gateway    │────▶│   Python Backend         │────▶│   JavaScript Frontend   │
│   (Port 4001)   │     │   (FastAPI + WebSocket)  │     │   (Lightweight Charts)  │
└─────────────────┘     └──────────────────────────┘     └─────────────────────────┘
        │                         │                               │
   Tick/Bar Data          Candle Aggregation              Real-time Rendering
   (keepUpToDate)          & WebSocket Server              (series.update())
```

### Backend Components

- **IB Connection Service** (`backend/ib_service.py`)
  - Connection management with auto-recovery
  - Error code handling (1100, 1101, 1102, etc.)
  - Health monitoring

- **Historical Data Fetcher** (`backend/historical_data.py`)
  - Fetches 1 year of data in daily chunks
  - Rate limiting compliance (IB pacing rules)
  - Intelligent caching system

- **Real-Time Streamer** (`backend/realtime.py`)
  - Two modes: tick-by-tick (50-300ms) or keepUpToDate (5s)
  - Live candlestick aggregation
  - Automatic bar finalization

- **FastAPI WebSocket Server** (`backend/app.py`)
  - Multi-client support
  - Broadcasts updates to all connected clients
  - REST endpoints for health and statistics

### Frontend

- **TradingView Lightweight Charts** - Official library
- **WebSocket Client** - Real-time updates
- **Symbol Switcher** - MNQ, MES, MGC
- **Responsive Design** - Desktop and mobile

## Data Flow

1. **Startup**:
   - Connect to IB Gateway
   - Load historical data from cache (or fetch from IB)
   - Display chart with 1 year of data

2. **Real-Time**:
   - Stream live bar updates via WebSocket
   - Update current bar as it forms
   - Create new bar on minute boundary

3. **Symbol Switch**:
   - Clear current chart
   - Load new symbol's historical data
   - Start new real-time stream

## API Endpoints

### REST Endpoints

- `GET /` - Main application page
- `GET /health` - Health check with IB Gateway status
- `GET /api/contracts` - List available contracts
- `GET /api/cache/{symbol}` - Cache metadata for symbol
- `GET /api/statistics` - Application statistics

### WebSocket

- `WS /ws/{symbol}` - Real-time data stream

**Messages from server**:
```javascript
// Historical data (on connect)
{
  type: 'historical',
  data: [{time, open, high, low, close, volume}, ...],
  symbol: 'MNQ',
  bar_count: 375000
}

// Real-time bar update
{
  type: 'bar_update',
  data: {time, open, high, low, close, volume},
  is_new_bar: true/false,
  symbol: 'MNQ'
}

// Error
{
  type: 'error',
  message: 'Error description'
}
```

## Project Structure

```
futures-chart/
├── README.md
├── ROADMAP.md                  # Complete implementation roadmap
├── run.py                      # Single-command startup
├── test_connection.py          # Connection test script
├── requirements.txt
├── config.yaml
├── .env.example
├── backend/
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── ib_service.py           # IB Gateway connection
│   ├── contracts.py            # Contract definitions
│   ├── historical_data.py      # Historical data fetching
│   ├── realtime.py             # Real-time streaming
│   ├── pacing.py               # Rate limiting
│   ├── cache.py                # Data caching
│   └── indicators/             # Indicator system
│       ├── __init__.py
│       ├── base.py
│       ├── moving_averages.py
│       └── oscillators.py
├── frontend/
│   ├── templates/
│   │   └── index.html          # Main application page
│   └── static/
│       ├── css/
│       └── js/
├── data/
│   └── cache/                  # Historical data cache
└── tests/
    └── test_contracts.py
```

## Contract Rolling

Futures contracts expire quarterly. Update contract months in `config.yaml`:

```yaml
contracts:
  MNQ:
    last_trade_date: "202503"  # Update to 202506 in March
  MES:
    last_trade_date: "202503"  # Update to 202506 in March
  MGC:
    last_trade_date: "202502"  # Update monthly
```

**Roll Schedule 2025**:
- H25 (Mar): Roll by Mar 11-12
- M25 (Jun): Roll by Jun 10-11
- U25 (Sep): Roll by Sep 9-10
- Z25 (Dec): Roll by Dec 9-10

## Troubleshooting

### Connection Issues

**Error: "Connection refused"**
- Ensure IB Gateway is running
- Check port number (4002 for paper, 4001 for live)
- Enable API in Gateway settings

**Error 354: "Not subscribed"**
- Subscribe to CME Real-Time in Client Portal
- Complete API Acknowledgement form
- Enable paper account data sharing (24h wait)

**Error 162: "Pacing violation"**
- Normal during initial data fetch
- Automatic retry with 15s delay
- Reduce concurrent requests if persistent

### Data Issues

**No historical data**
- Check market data subscriptions
- Verify contract expiration date
- Try different contract month

**Real-time not updating**
- Check IB Gateway connection
- Verify markets are open
- Check error logs in browser console

### Performance

**Slow chart rendering**
- Reduce cached data age (fetch less history)
- Clear browser cache
- Check network connection

**High memory usage**
- Reduce `max_bars_in_memory` in config
- Clear old cache files

## Development

### Running Tests

```bash
# Run contract tests
python -m pytest tests/test_contracts.py -v

# Test IB Gateway connection
python test_connection.py
```

### Development Mode

```bash
# Auto-reload on code changes
python run.py --reload --log-level debug
```

### Adding New Indicators

See `backend/indicators/base.py` for the indicator interface.

Example:
```python
from backend.indicators.base import Indicator

class CustomIndicator(Indicator):
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Your calculation here
        return pd.DataFrame({
            'time': df['time'],
            'value': your_calculation
        })

    @property
    def plot_config(self):
        return {
            'type': 'line',
            'color': '#2962FF',
            'pane': 'main'  # or 'separate'
        }
```

## Production Deployment

### Important Settings

1. **Auto-restart IB Gateway** at 11:45 PM ET (before midnight reset)
2. **Monitor error codes**: 1100, 1101, 1102, 162, 354
3. **Cache management**: Clear old cache files periodically
4. **Health checks**: Monitor `/health` endpoint
5. **Logging**: Configure log rotation (100MB per file)

### Environment Variables

See `.env.example` for all available settings.

## Performance Specifications

- **Initial load**: <2 seconds (from cache)
- **Historical fetch**: 30-45 minutes (365 days, first time)
- **Real-time latency**: 50-300ms (tick-by-tick) or 5s (keepUpToDate)
- **Max bars (desktop)**: 60,000 bars smooth
- **Max bars (mobile)**: 20,000 bars recommended
- **Memory usage**: ~500MB typical

## License

[Your License Here]

## Support

For issues and questions:
- Check ROADMAP.md for implementation details
- Review troubleshooting section above
- Check IB Gateway logs
- Test connection with `test_connection.py`

## Credits

- **TradingView Lightweight Charts** - Chart rendering
- **ib_insync** - IB Gateway connectivity
- **FastAPI** - WebSocket server
- Built following the comprehensive roadmap in ROADMAP.md
