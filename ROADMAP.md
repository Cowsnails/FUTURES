# Complete Implementation Roadmap: TradingView-Quality Futures Charting with IB Gateway

**Project Goal**: Build a professional futures charting webapp matching TradingView's visual quality and real-time performance using IB Gateway, Python FastAPI, and Lightweight Charts.

**Target Features**:
- Real-time MNQ, MES, MGC futures charting with sub-second updates
- TradingView-exact color scheme and visual fidelity
- 1 year of historical 1-minute data
- Custom indicator system (SMA, EMA, RSI, MACD, etc.)
- Professional volume bars with proper coloring
- Production-ready performance handling 60K+ candles

---

## Phase 1: Foundation and Environment Setup

### 1.1 IB Gateway Configuration and Connectivity
**Priority**: CRITICAL | **Duration**: 1-2 days

**Objectives**:
- Set up IB Gateway with proper API configuration
- Establish reliable connection patterns
- Configure market data subscriptions
- Verify contract specifications

**Tasks**:

- [ ] **Install and configure IB Gateway**
  - Download latest stable IB Gateway version
  - Set up paper trading account (4002) for development
  - Configure TWS API settings: File → Global Configuration → API → Settings
  - Enable "ActiveX and Socket Clients"
  - Set socket port: 4002 (paper) / 4001 (live)
  - Configure auto-restart at 11:45 PM ET for production

- [ ] **Complete market data subscription setup**
  - Navigate to Client Portal → Settings → Market Data Subscriptions
  - Subscribe to CME Real-Time (Level 1) for MES, MNQ
  - Subscribe to COMEX for MGC (included in CME bundle)
  - **CRITICAL**: Complete Market Data API Acknowledgement form at Client Portal → Settings → User Settings
  - Enable "Share real-time market data" for paper trading account
  - Wait 24 hours for propagation (note the date/time)

- [ ] **Test basic IB Gateway connection**
  - Create `test_connection.py` script
  - Test connection to port 4002 with client ID 1
  - Verify connection success
  - Test contract qualification for MNQ, MES, MGC

**Deliverables**:
- Working IB Gateway installation
- Verified market data subscriptions
- Basic connection test script confirming API access

**Critical References**:
- Missing Pieces document: "IB Gateway configuration essentials" (lines 7-66)
- Missing Pieces document: "Market data subscription verification" (lines 47-58)

**Common Issues & Solutions**:
- Error 354 ("not subscribed"): Check API Acknowledgement form completion
- Paper account no data: Verify data sharing enabled and wait 24 hours
- Connection refused: Verify socket port and "Enable ActiveX" setting

---

### 1.2 Contract Specifications and Validation
**Priority**: CRITICAL | **Duration**: 0.5 days

**Objectives**:
- Validate exact futures contract specifications
- Understand contract rolling requirements
- Test contract qualification

**Tasks**:

- [ ] **Create contract definition module**
  - Create `backend/contracts.py`
  - Define MNQ contract (CME, lastTradeDateOrContractMonth='202503')
  - Define MES contract (CME, lastTradeDateOrContractMonth='202503')
  - Define MGC contract (COMEX, lastTradeDateOrContractMonth='202502')
  - Document tick sizes and multipliers in comments

- [ ] **Test contract qualification**
  - Create `test_contracts.py`
  - Use `ib.qualifyContracts()` for each contract
  - Verify conId population
  - Log contract details (exchange, symbol, lastTradeDate)

- [ ] **Document contract rolling schedule**
  - Create roll schedule table for 2025-2026
  - Implement `get_contracts_for_rolling()` function
  - Implement `should_roll()` function (8 days before expiry)
  - Plan for quarterly rolls: H25 (Mar 21), M25 (Jun 20), U25 (Sep 19), Z25 (Dec 19)

**Deliverables**:
- `backend/contracts.py` with all contract definitions
- `test_contracts.py` validation script
- Contract rolling logic and schedule

**Critical References**:
- Main Guide: "IB Gateway connection and futures contract specifications" (lines 31-98)
- Missing Pieces: "Futures contract rolling and continuous data" (lines 179-305)

---

### 1.3 Development Environment Setup
**Priority**: HIGH | **Duration**: 0.5 days

**Objectives**:
- Set up Python backend environment
- Set up frontend structure
- Configure version control

**Tasks**:

- [ ] **Create project structure**
```
futures-chart/
├── run.py
├── requirements.txt
├── config.yaml
├── .env
├── .gitignore
├── backend/
│   ├── __init__.py
│   ├── app.py
│   ├── contracts.py
│   ├── ib_service.py
│   └── indicators/
│       ├── __init__.py
│       ├── base.py
│       ├── moving_averages.py
│       └── oscillators.py
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── indicators.js
└── data/
    └── cache/
```

- [ ] **Install Python dependencies**
  - Create `requirements.txt`:
    ```
    ib_insync>=0.9.86
    fastapi>=0.109.0
    uvicorn[standard]>=0.27.0
    websockets>=12.0
    pandas>=2.0.0
    pyarrow>=14.0.0
    pyyaml>=6.0
    python-dotenv>=1.0.0
    ```
  - Create virtual environment: `python -m venv venv`
  - Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
  - Install: `pip install -r requirements.txt`

- [ ] **Configure environment variables**
  - Create `.env` file:
    ```env
    IB_HOST=127.0.0.1
    IB_PORT=4002
    IB_CLIENT_ID=1
    SERVER_HOST=0.0.0.0
    SERVER_PORT=8000
    MAX_BARS_IN_MEMORY=10000
    ```

- [ ] **Initialize git repository**
  - `git init` (if not already done)
  - Create `.gitignore`:
    ```
    venv/
    __pycache__/
    *.pyc
    .env
    data/cache/*.parquet
    *.log
    ```

**Deliverables**:
- Complete project structure
- Working Python environment
- Version control setup

---

## Phase 2: Backend Development - Data Pipeline

### 2.1 IB Gateway Connection Service
**Priority**: CRITICAL | **Duration**: 1-2 days

**Objectives**:
- Implement robust connection management
- Handle connection errors and recovery
- Implement connection state machine

**Tasks**:

- [ ] **Create basic IB service class**
  - Create `backend/ib_service.py`
  - Implement `IBConnectionManager` class
  - Add connection state enum (DISCONNECTED, CONNECTING, CONNECTED, READY, etc.)
  - Implement `connect()` async method
  - Implement `disconnect()` method

- [ ] **Implement error handling**
  - Add error callback `_on_error()`
  - Handle error codes: 1100 (connection lost), 1101 (restored, data lost), 1102 (restored, data maintained)
  - Handle error codes: 2103, 2105 (farm disconnected)
  - Distinguish informational codes (2104, 2106, 2158) from actionable errors
  - Create `RETRYABLE_ERRORS` and `FATAL_ERRORS` sets

- [ ] **Implement connection recovery**
  - Create exponential backoff with jitter: `base_delay * (2 ** attempt)`
  - Max delay: 60 seconds
  - Add random jitter: `delay *= (0.75 + random() * 0.5)`
  - Special handling for pacing violations (error 162): min 15-second delay
  - Implement re-subscription logic for 1101 events

- [ ] **Add connection health monitoring**
  - Track last successful data timestamp
  - Implement heartbeat check
  - Add connection metrics (uptime, reconnection count)

**Deliverables**:
- `backend/ib_service.py` with full connection management
- Error handling for all critical error codes
- Automatic reconnection with exponential backoff

**Critical References**:
- Missing Pieces: "Error handling and recovery patterns" (lines 70-173)
- Missing Pieces: "Connection recovery state machine" (lines 145-174)

---

### 2.2 Historical Data Fetching
**Priority**: CRITICAL | **Duration**: 2-3 days

**Objectives**:
- Fetch 1 year of 1-minute historical data
- Implement proper rate limiting
- Cache data locally for fast reloading

**Tasks**:

- [ ] **Implement historical data request with rate limiting**
  - Create `backend/historical_data.py`
  - Implement `download_year_of_minute_data()` function
  - Request 1-day chunks: `durationStr='1 D'`, `barSizeSetting='1 min'`
  - Set `useRTH=0` for extended hours (critical for futures)
  - Set `whatToShow='TRADES'`
  - Set `keepUpToDate=False` (historical only)

- [ ] **Implement pacing compliance**
  - Create `PacingManager` class
  - Rule 1: 15-second minimum between identical requests
  - Rule 2: Max 6 requests per 2 seconds per contract
  - Rule 3: Max 60 requests per 10-minute rolling window
  - BID_ASK requests count double
  - Add request queue with automatic pacing

- [ ] **Add retry logic for historical requests**
  - Wrap requests with `ib_request_with_retry()`
  - Max 5 retries with exponential backoff
  - Handle pacing violations (162) with 15+ second pause
  - Handle connection errors with reconnection
  - Stop retrying on fatal errors (200, 354, 502)

- [ ] **Implement data caching**
  - Create `backend/cache.py`
  - Save to Parquet format: `mnq_1min_1year.parquet`
  - Include metadata: fetch date, contract, bar count
  - Implement cache validation (age check, data completeness)
  - Load from cache on startup if fresh (<24 hours old)

- [ ] **Data format conversion**
  - Convert IB bar format to standard OHLCV
  - Convert date strings to Unix timestamps
  - Structure: `{'time': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': int}`

**Deliverables**:
- `backend/historical_data.py` with full download logic
- `backend/pacing.py` with rate limit management
- `backend/cache.py` with Parquet caching
- Successfully cached 1 year of data for MNQ, MES, MGC

**Critical References**:
- Main Guide: "Fetching one year of historical 1-minute data" (lines 103-186)
- Missing Pieces: "Historical data pacing rules" (lines 86-142)
- Critical Answers: Rate limits table (lines 853-859)

**Performance Notes**:
- Full year download: ~365 requests
- With pacing: ~30-45 minutes total time
- Subsequent loads from cache: <1 second

---

### 2.3 Real-Time Data Streaming
**Priority**: CRITICAL | **Duration**: 2-3 days

**Objectives**:
- Stream real-time tick data
- Build live candlesticks from ticks
- Handle bar formation and updates

**Tasks**:

- [ ] **Understand keepUpToDate limitations**
  - Note: `keepUpToDate=True` provides 5-second updates, NOT tick-by-tick
  - Note: ~12 updates per minute, not sub-second
  - Decision: Use tick-by-tick for true real-time OR accept 5-second updates

- [ ] **Implement tick-by-tick streaming (OPTION A - TRUE REAL-TIME)**
  - Create `backend/realtime.py`
  - Implement `LiveCandlestickBuilder` class
  - Use `ib.reqTickByTickData(contract, 'AllLast')`
  - Set `numberOfTicks=0` for continuous stream
  - Subscribe to `ticker.updateEvent`

- [ ] **Implement tick aggregation into candles**
  - Create `_on_tick()` callback
  - Initialize bar with first tick (set open)
  - Update high: `max(current_high, tick.price)`
  - Update low: `min(current_low, tick.price)`
  - Update close: `tick.price`
  - Accumulate volume: `volume += tick.size`

- [ ] **Implement bar finalization logic**
  - Check if minute boundary crossed
  - When new minute starts:
    - Finalize current bar (set `is_new_bar=True`)
    - Send final bar to WebSocket
    - Initialize new bar with next tick
  - During minute:
    - Update forming bar (set `is_new_bar=False`)
    - Send updates to WebSocket

- [ ] **OR: Implement keepUpToDate streaming (OPTION B - SIMPLER)**
  - Create `RealtimeBars` class
  - Use `keepUpToDate=True` in historical request
  - Load 1 day of history initially
  - Subscribe to `bars.updateEvent`
  - Handle `hasNewBar` flag to distinguish new vs update

- [ ] **Add data validation**
  - Implement `validate_bar()` function
  - Check: `low <= open <= high`
  - Check: `low <= close <= high`
  - Check: `high >= low`
  - Alert on >5% price jumps (possible bad ticks)

**Deliverables**:
- `backend/realtime.py` with tick-by-tick OR keepUpToDate implementation
- Bar aggregation logic with proper minute boundaries
- Data validation preventing bad ticks

**Critical References**:
- Critical Answers: "IB Gateway's keepUpToDate doesn't stream ticks" (lines 5-51)
- Main Guide: "Real-time streaming with live candlestick formation" (lines 189-242)

**Architecture Decision**:
- **Option A (Tick-by-Tick)**: 50-300ms latency, true real-time, more complex
- **Option B (keepUpToDate)**: 5-second updates, simpler, sufficient for most use cases
- **Recommendation**: Start with Option B, upgrade to Option A if latency critical

---

### 2.4 FastAPI WebSocket Backend
**Priority**: CRITICAL | **Duration**: 2-3 days

**Objectives**:
- Build WebSocket server for real-time data streaming
- Manage multiple client connections
- Bridge IB Gateway to frontend

**Tasks**:

- [ ] **Create FastAPI application**
  - Create `backend/app.py`
  - Initialize FastAPI: `app = FastAPI()`
  - Add CORS middleware for development
  - Create lifespan context manager for IB connection

- [ ] **Implement connection manager**
  - Create `ConnectionManager` class
  - Maintain set of active WebSocket connections
  - Implement `connect()`, `disconnect()`, `broadcast()` methods
  - Handle disconnected clients gracefully

- [ ] **Create WebSocket endpoint**
  - Route: `/ws/{symbol}` (e.g., `/ws/MNQ`)
  - Accept WebSocket connection
  - Qualify contract from symbol
  - Load historical data from cache
  - Start real-time streaming

- [ ] **Implement message protocol**
  - Message type: `historical`
    ```json
    {
      "type": "historical",
      "data": [{"time": 1234567890, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000}, ...]
    }
    ```
  - Message type: `bar_update`
    ```json
    {
      "type": "bar_update",
      "data": {"time": 1234567890, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
      "is_new_bar": false
    }
    ```

- [ ] **Add health check endpoint**
  - Route: `/health`
  - Return JSON with:
    - `ib_connected`: boolean
    - `last_tick_age_seconds`: float
    - `memory_usage_mb`: float
    - `active_subscriptions`: int
  - HTTP 200 if healthy, 503 if unhealthy

- [ ] **Implement graceful shutdown**
  - Register SIGTERM and SIGINT handlers
  - Cancel all market data subscriptions
  - Close WebSocket connections
  - Disconnect from IB Gateway
  - Save state if needed

**Deliverables**:
- `backend/app.py` with complete WebSocket server
- Connection manager handling multiple clients
- Message protocol for historical + real-time data
- Health check endpoint

**Critical References**:
- Main Guide: "FastAPI WebSocket backend" (lines 244-339)
- Missing Pieces: "Health check implementation" (lines 423-444)
- Missing Pieces: "Graceful shutdown handler" (lines 490-507)

---

## Phase 3: Frontend Development

### 3.1 TradingView Lightweight Charts Integration
**Priority**: CRITICAL | **Duration**: 2-3 days

**Objectives**:
- Set up Lightweight Charts with TradingView color scheme
- Render candlestick and volume series
- Implement responsive design

**Tasks**:

- [ ] **Create base HTML structure**
  - Create `frontend/templates/index.html`
  - Import Lightweight Charts from CDN:
    ```html
    <script type="module">
      import { createChart, ColorType } from 'https://unpkg.com/lightweight-charts@5.0.0/dist/lightweight-charts.standalone.production.mjs';
    </script>
    ```
  - Create chart container div: `<div id="chart"></div>`
  - Add status indicator div

- [ ] **Implement TradingView color scheme**
  - Define color constants:
    ```javascript
    const COLORS = {
      UP: '#26a69a',              // TradingView green
      DOWN: '#ef5350',            // TradingView red
      VOLUME_UP: 'rgba(38, 166, 154, 0.5)',
      VOLUME_DOWN: 'rgba(239, 83, 80, 0.5)',
      BACKGROUND: '#131722',
      TEXT: '#d1d4dc',
      GRID: '#1e222d',
      CROSSHAIR: '#758696'
    };
    ```

- [ ] **Initialize chart with proper options**
  - Create chart: `createChart(container, options)`
  - Set layout background: `ColorType.Solid`, `COLORS.BACKGROUND`
  - Set text color: `COLORS.TEXT`
  - Set grid lines: `COLORS.GRID`
  - Set crosshair: `COLORS.CROSSHAIR`
  - Configure time scale: `timeVisible: true`, `secondsVisible: false`
  - Configure price scale: proper margins and border color

- [ ] **Add candlestick series**
  - Create series: `chart.addCandlestickSeries()`
  - Set colors: `upColor`, `downColor`, `wickUpColor`, `wickDownColor`
  - Set `borderVisible: false`
  - Store reference for updates

- [ ] **Add volume series**
  - Create histogram series: `chart.addHistogramSeries()`
  - Set price format: `{type: 'volume'}`
  - Set price scale margins: `top: 0.85, bottom: 0` (renders at bottom)
  - Color bars based on candle: `close >= open ? VOLUME_UP : VOLUME_DOWN`

- [ ] **Implement responsive resize**
  - Add window resize listener
  - Update chart dimensions: `chart.applyOptions({width, height})`

**Deliverables**:
- `frontend/templates/index.html` with complete chart setup
- TradingView-exact color scheme
- Candlestick + volume series configured
- Responsive layout

**Critical References**:
- Main Guide: "Frontend with TradingView Lightweight Charts" (lines 342-514)
- Critical Answers: "Volume bar coloring" (lines 128-156)

**Volume Coloring Logic**:
- Default: Same-candle comparison (`close >= open`)
- Doji candles (`open === close`): GREEN (because `>=`)
- Optional: Previous-close comparison (not default)

---

### 3.2 WebSocket Client Connection
**Priority**: CRITICAL | **Duration**: 1-2 days

**Objectives**:
- Connect to backend WebSocket
- Handle historical data loading
- Process real-time updates
- Implement reconnection logic

**Tasks**:

- [ ] **Implement WebSocket connection**
  - Create WebSocket: `new WebSocket('ws://localhost:8000/ws/MNQ')`
  - Make symbol configurable
  - Handle connection open event
  - Update status indicator on connect

- [ ] **Handle historical data message**
  - Listen for `type: 'historical'`
  - Load data: `candleSeries.setData(msg.data)`
  - Transform volume data:
    ```javascript
    const volumeData = msg.data.map(bar => ({
      time: bar.time,
      value: bar.volume,
      color: bar.close >= bar.open ? COLORS.VOLUME_UP : COLORS.VOLUME_DOWN
    }));
    ```
  - Load volume: `volumeSeries.setData(volumeData)`
  - Fit content: `chart.timeScale().fitContent()`

- [ ] **Handle real-time bar updates**
  - Listen for `type: 'bar_update'`
  - Update candle: `candleSeries.update(msg.data)`
  - Update volume with color:
    ```javascript
    volumeSeries.update({
      time: bar.time,
      value: bar.volume,
      color: bar.close >= bar.open ? COLORS.VOLUME_UP : COLORS.VOLUME_DOWN
    });
    ```

- [ ] **Implement reconnection logic**
  - Handle `onclose` event
  - Exponential backoff: `Math.min(1000 * Math.pow(2, attempts), 30000)`
  - Update status indicator: "Disconnected - Reconnecting..."
  - Reset reconnection count on successful connection

- [ ] **Add error handling**
  - Handle `onerror` event
  - Log errors to console
  - Show error status to user

**Deliverables**:
- Complete WebSocket client implementation
- Historical data loading
- Real-time updates with proper bar updates
- Automatic reconnection

**Critical References**:
- Main Guide: "WebSocket connection" JavaScript code (lines 449-511)

---

### 3.3 User Interface Enhancements
**Priority**: MEDIUM | **Duration**: 1-2 days

**Objectives**:
- Add symbol selector
- Add timeframe controls
- Add status indicators
- Improve UX

**Tasks**:

- [ ] **Add symbol selector**
  - Create dropdown: MNQ, MES, MGC
  - On change: reconnect WebSocket with new symbol
  - Clear existing chart data

- [ ] **Add connection status indicator**
  - Show "Connecting..." (yellow)
  - Show "Connected" (green)
  - Show "Disconnected" (red)
  - Position: top-left overlay

- [ ] **Add loading indicator**
  - Show during historical data fetch
  - Hide when chart rendered

- [ ] **Add basic styling**
  - Create `frontend/static/css/style.css`
  - Style controls
  - Style status indicators
  - Ensure dark theme consistency

- [ ] **Add error messages**
  - Display connection errors to user
  - Display data errors
  - Clear, actionable messages

**Deliverables**:
- Symbol selector with switching capability
- Professional status indicators
- Error messaging system
- Polished UI matching TradingView aesthetic

---

## Phase 4: Indicator System

### 4.1 Indicator Architecture
**Priority**: HIGH | **Duration**: 1-2 days

**Objectives**:
- Design plugin-based indicator system
- Create base indicator class
- Implement indicator registry

**Tasks**:

- [ ] **Create indicator base class**
  - Create `backend/indicators/base.py`
  - Define abstract `Indicator` class
  - Abstract method: `calculate(df) -> DataFrame`
  - Abstract property: `plot_config -> dict`
  - Store indicator parameters in `__init__`

- [ ] **Create indicator registry**
  - Create `backend/indicators/manager.py`
  - Define `INDICATOR_REGISTRY` dict: `{name: class}`
  - Implement `IndicatorManager` class
  - Method: `add_indicator(type, params)`
  - Method: `calculate_all(df)` returning serializable results
  - Method: `update_indicator(indicator, new_bar, df)` for incremental updates

- [ ] **Design indicator data format**
  - Return format:
    ```python
    {
      'data': [{'time': 1234567890, 'value': 100.5}, ...],
      'config': {
        'type': 'line',
        'color': '#2962FF',
        'lineWidth': 2,
        'pane': 'main'  # or 'separate'
      }
    }
    ```

**Deliverables**:
- `backend/indicators/base.py` with abstract base class
- `backend/indicators/manager.py` with registry
- Indicator data format specification

**Critical References**:
- Main Guide: "Custom indicator system with plugin architecture" (lines 518-696)

---

### 4.2 Implement Core Indicators
**Priority**: HIGH | **Duration**: 2-3 days

**Objectives**:
- Implement moving averages (SMA, EMA)
- Implement oscillators (RSI, MACD)
- Test calculations against known values

**Tasks**:

- [ ] **Implement SMA (Simple Moving Average)**
  - Create `backend/indicators/moving_averages.py`
  - Create `SMA` class extending `Indicator`
  - Calculate: `df['close'].rolling(window=period).mean()`
  - Plot config: line overlay on main pane
  - Default period: 20

- [ ] **Implement EMA (Exponential Moving Average)**
  - Create `EMA` class in same file
  - Calculate: `df['close'].ewm(span=period, adjust=False).mean()`
  - Plot config: line overlay on main pane
  - Default period: 20, different color than SMA

- [ ] **Implement RSI (Relative Strength Index)**
  - Create `backend/indicators/oscillators.py`
  - Create `RSI` class
  - Calculate gains: `delta.where(delta > 0, 0).rolling(period).mean()`
  - Calculate losses: `(-delta.where(delta < 0, 0)).rolling(period).mean()`
  - Calculate RSI: `100 - (100 / (1 + rs))`
  - Plot config: separate pane with levels [30, 70]
  - Default period: 14

- [ ] **Implement MACD**
  - Create `MACD` class in oscillators.py
  - Calculate fast EMA (default: 12)
  - Calculate slow EMA (default: 26)
  - Calculate MACD line: `fast_ema - slow_ema`
  - Calculate signal line: `macd_line.ewm(span=9).mean()`
  - Calculate histogram: `macd_line - signal_line`
  - Plot config: separate pane with 3 series (MACD, signal, histogram)

- [ ] **Add indicators to registry**
  - Register in `INDICATOR_REGISTRY`:
    ```python
    INDICATOR_REGISTRY = {
      'sma': SMA,
      'ema': EMA,
      'rsi': RSI,
      'macd': MACD,
    }
    ```

- [ ] **Create indicator tests**
  - Create `tests/test_indicators.py`
  - Test SMA with known values
  - Test EMA calculation
  - Test RSI boundaries (0-100)
  - Test MACD values
  - Use pandas_ta or TA-Lib as reference

**Deliverables**:
- `backend/indicators/moving_averages.py` with SMA, EMA
- `backend/indicators/oscillators.py` with RSI, MACD
- Unit tests validating calculations
- All indicators registered

**Critical References**:
- Main Guide: "Indicator base class and implementations" (lines 522-646)

---

### 4.3 Frontend Indicator Rendering
**Priority**: HIGH | **Duration**: 2-3 days

**Objectives**:
- Render indicators on frontend
- Support main pane overlays
- Support separate panes for oscillators

**Tasks**:

- [ ] **Create indicator renderer class**
  - Create `frontend/static/js/indicators.js`
  - Create `IndicatorRenderer` class
  - Store chart reference
  - Maintain `series` map: `{indicator_name: series_object}`

- [ ] **Implement main pane indicator rendering**
  - Method: `addIndicator(name, config, data)`
  - For `config.pane === 'main'`:
    - Create line series: `chart.addLineSeries()`
    - Set color, line width from config
    - Set data: `series.setData(data)`
    - Store in `this.series[name]`

- [ ] **Implement separate pane rendering**
  - For `config.pane === 'separate'`:
    - Create line series with separate pane
    - Configure separate price scale
    - Add reference levels for RSI (30, 70)

- [ ] **Implement indicator updates**
  - Method: `updateIndicator(name, newPoint)`
  - Get series: `this.series[name]`
  - Update: `series.update(newPoint)`

- [ ] **Extend WebSocket protocol for indicators**
  - Backend: calculate indicators on historical data
  - Send with historical message:
    ```json
    {
      "type": "historical",
      "data": [...],
      "indicators": {
        "SMA_20": {
          "data": [...],
          "config": {...}
        }
      }
    }
    ```
  - Frontend: render each indicator

- [ ] **Add indicator controls to UI**
  - Add checkboxes for common indicators
  - Send indicator preferences to backend
  - Toggle indicators on/off

**Deliverables**:
- `frontend/static/js/indicators.js` with rendering
- Support for overlay and separate pane indicators
- WebSocket protocol extension for indicators
- UI controls for toggling indicators

**Critical References**:
- Main Guide: "Frontend indicator rendering" (lines 700-752)

---

### 4.4 Simplified Indicator Syntax (Optional)
**Priority**: LOW | **Duration**: 1-2 days

**Objectives**:
- Provide simpler indicator definition syntax
- Pine Script-inspired but Python-based

**Tasks**:

- [ ] **Choose indicator syntax approach**
  - Option A: Expression-based DSL (Pine-like, 1-2 lines)
  - Option B: Minimal decorator pattern (5-10 lines, full Python)
  - Option C: YAML config for non-programmers

- [ ] **If choosing Option A (Expression DSL)**
  - Install `simpleeval` library
  - Create `IndicatorEvaluator` class
  - Expose price series: `close`, `open`, `high`, `low`, `volume`
  - Expose functions: `EMA`, `SMA`, `RSI`, `MACD`
  - Parse and evaluate expressions
  - Never use raw `eval()` - security risk!

- [ ] **If choosing Option B (Decorator)**
  - Create `@indicator` decorator
  - Mark functions as indicators
  - Auto-register decorated functions
  - Keep full Python expressiveness

- [ ] **If choosing Option C (YAML)**
  - Define YAML schema for indicators
  - Parse YAML config
  - Instantiate indicators from config

**Deliverables**:
- (Optional) Simplified indicator syntax system
- Documentation with examples

**Critical References**:
- Critical Answers: "Three concrete alternatives for simpler indicator syntax" (lines 53-127)

---

## Phase 5: Performance Optimization

### 5.1 Data Volume Management
**Priority**: HIGH | **Duration**: 1-2 days

**Objectives**:
- Handle large datasets efficiently
- Implement lazy loading
- Prevent memory issues

**Tasks**:

- [ ] **Analyze dataset size**
  - 1 minute bars × 375 trading days × 23 hours/day × 60 minutes = ~517,500 bars/year
  - Lightweight Charts benchmark: smooth up to ~60K bars
  - Target: Keep <60K bars loaded at once

- [ ] **Implement data decimation**
  - Pre-compute multiple timeframes server-side:
    - 1m: Full resolution
    - 5m: ~103,500 bars/year
    - 15m: ~34,500 bars/year
    - 1H: ~8,625 bars/year
    - 1D: ~375 bars/year
  - Store in cache: `mnq_1min.parquet`, `mnq_5min.parquet`, etc.

- [ ] **Implement lazy loading with visible range**
  - Subscribe to visible range changes:
    ```javascript
    chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
      const barsVisible = range.to - range.from;
      // Choose appropriate timeframe
      // Fetch only visible range + buffer
    });
    ```
  - Load initial: Most recent 5,000-10,000 bars
  - Load more on scroll left (history)
  - Buffer: 50 bars before/after visible range

- [ ] **Enable data conflation**
  - Lightweight Charts v5.1.0+ feature
  - Enable in chart options:
    ```javascript
    timeScale: {
      enableConflation: true,
      conflationThresholdFactor: 1.0,
      precomputeConflationOnInit: true
    }
    ```
  - Provides 10-100x zoom performance improvement

- [ ] **Implement memory-efficient storage**
  - Use `collections.deque` with `maxlen`
  - Use `numpy.float32` instead of `float64` where possible
  - Use typed arrays in JavaScript: `Float64Array`

**Deliverables**:
- Multi-timeframe data pre-computation
- Lazy loading with visible range tracking
- Data conflation enabled
- Memory-efficient storage

**Critical References**:
- Main Guide: "Performance optimization for 375,000 candles" (lines 829-848)
- Critical Answers: "375,000 candles will crash mobile" (lines 159-209)

**Performance Targets**:
- Desktop: Smooth scrolling with <60K bars
- Mobile: Smooth scrolling with <20K bars
- Initial load: <2 seconds
- Real-time updates: <100ms latency

---

### 5.2 WebSocket and Network Optimization
**Priority**: MEDIUM | **Duration**: 1 day

**Objectives**:
- Minimize WebSocket message size
- Batch updates efficiently
- Reduce latency

**Tasks**:

- [ ] **Optimize message serialization**
  - Use compact JSON (no whitespace)
  - Consider MessagePack for binary serialization
  - Compress large historical payloads with gzip

- [ ] **Implement message batching**
  - During high-frequency updates, batch multiple bar updates
  - Send max 20 updates/second to frontend
  - Frontend: process queue and update chart

- [ ] **Add connection keepalive**
  - Send ping/pong messages every 30 seconds
  - Detect dead connections
  - Clean up on timeout

- [ ] **Implement backpressure handling**
  - Detect slow clients
  - Drop updates if client can't keep up
  - Send only latest bar state, not all intermediate updates

**Deliverables**:
- Optimized message format
- Message batching for high-frequency updates
- Connection keepalive
- Backpressure handling

---

### 5.3 Caching and Startup Optimization
**Priority**: MEDIUM | **Duration**: 1 day

**Objectives**:
- Fast application startup
- Efficient cache management
- Minimize IB Gateway requests

**Tasks**:

- [ ] **Implement intelligent cache invalidation**
  - Check cache age on startup
  - If cache <24 hours old: use cache
  - If cache older: fetch only new data since last cache
  - Append new data to cache

- [ ] **Implement incremental cache updates**
  - On shutdown: save current state
  - On startup: load cache, fetch only gaps
  - Fetch from last cached timestamp to now

- [ ] **Pre-compute indicators on historical data**
  - Calculate all indicators once on historical data
  - Cache results
  - Only calculate incrementally for new bars

- [ ] **Implement startup sequence**
  1. Load cached historical data (instant)
  2. Display cached data immediately
  3. Connect to IB Gateway in background
  4. Fetch any missing recent data
  5. Start real-time streaming
  6. Update cache with new data

**Deliverables**:
- Intelligent cache with age validation
- Incremental cache updates
- Fast startup (<2 seconds to initial display)

---

## Phase 6: Production Readiness

### 6.1 Error Handling and Resilience
**Priority**: CRITICAL | **Duration**: 2-3 days

**Objectives**:
- Handle all error scenarios gracefully
- Implement comprehensive logging
- Ensure application stability

**Tasks**:

- [ ] **Implement comprehensive error handling**
  - Try/catch around all IB Gateway calls
  - Handle network timeouts
  - Handle malformed data
  - Handle WebSocket disconnections
  - Log all errors with context

- [ ] **Implement structured logging**
  - Use Python `logging` module
  - Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  - Log format: timestamp, level, module, message
  - Log rotation: max 100MB per file, keep 10 files
  - Log IB Gateway errors with error codes

- [ ] **Add application monitoring**
  - Track key metrics:
    - Connection uptime
    - Data latency (time from market to chart)
    - Memory usage
    - WebSocket connection count
    - Bar processing rate
  - Expose metrics on `/metrics` endpoint

- [ ] **Implement circuit breaker pattern**
  - If IB Gateway repeatedly fails, stop retrying temporarily
  - Backoff period: 5 minutes
  - Alert on circuit open

- [ ] **Add data quality checks**
  - Validate all incoming bars (OHLC relationships)
  - Detect gaps in data
  - Alert on missing data
  - Alert on stale data (no updates for 1 minute during trading hours)

**Deliverables**:
- Comprehensive error handling throughout codebase
- Structured logging with rotation
- Application metrics
- Circuit breaker for IB Gateway failures
- Data quality validation

**Critical References**:
- Missing Pieces: "Error handling and recovery patterns" (lines 70-173)
- Missing Pieces: "Data quality validation" (lines 447-466)

---

### 6.2 Configuration and Deployment
**Priority**: HIGH | **Duration**: 1-2 days

**Objectives**:
- Externalize all configuration
- Prepare for production deployment
- Document deployment process

**Tasks**:

- [ ] **Create configuration system**
  - Create `config.yaml`:
    ```yaml
    ib_gateway:
      host: 127.0.0.1
      port: 4002
      client_id: 1
      auto_restart_time: "11:45 PM"

    server:
      host: 0.0.0.0
      port: 8000
      reload: false

    data:
      cache_dir: ./data/cache
      max_bars_in_memory: 10000
      cache_max_age_hours: 24

    logging:
      level: INFO
      file: ./logs/app.log
      max_bytes: 104857600  # 100MB
      backup_count: 10
    ```
  - Load with PyYAML
  - Override with environment variables

- [ ] **Implement configuration validation**
  - Validate all required fields present
  - Validate value ranges
  - Fail fast on invalid config

- [ ] **Create startup script**
  - Create `run.py`:
    ```python
    import uvicorn
    import webbrowser
    import threading

    if __name__ == '__main__':
        # Open browser after 1.5s delay
        threading.Thread(target=lambda: time.sleep(1.5) or webbrowser.open('http://localhost:8000'), daemon=True).start()

        # Start server
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, log_level="info")
    ```
  - Make executable: `chmod +x run.py`

- [ ] **Create Docker support (optional)**
  - Create `Dockerfile` for application
  - Create `docker-compose.yml` with IB Gateway + app
  - Use `ghcr.io/gnzsnz/ib-gateway:stable` for IB Gateway
  - Configure networking between containers

- [ ] **Create deployment checklist**
  - See "Production deployment checklist" below

**Deliverables**:
- `config.yaml` with all settings
- Configuration validation
- Single-command startup script
- (Optional) Docker deployment
- Deployment checklist

**Critical References**:
- Main Guide: "Complete project structure and startup script" (lines 756-826)
- Missing Pieces: "Docker containers for IB Gateway" (lines 332-354)
- Missing Pieces: "Production deployment checklist" (lines 512-545)

---

### 6.3 Testing Strategy
**Priority**: HIGH | **Duration**: 2-3 days

**Objectives**:
- Unit test critical components
- Integration testing
- Performance testing

**Tasks**:

- [ ] **Create unit tests for indicators**
  - Test each indicator calculation
  - Validate against known values
  - Test edge cases (empty data, single bar, etc.)
  - Use pytest framework

- [ ] **Create unit tests for data processing**
  - Test bar aggregation from ticks
  - Test data validation
  - Test cache operations
  - Test rate limiting logic

- [ ] **Create integration tests**
  - Test IB Gateway connection (requires Gateway running)
  - Test historical data fetch
  - Test WebSocket message flow
  - Mock IB Gateway for CI/CD

- [ ] **Create performance tests**
  - Test chart rendering with 60K bars
  - Test real-time update latency
  - Test memory usage over time
  - Test with multiple concurrent clients

- [ ] **Create manual test plan**
  - Test symbol switching
  - Test indicator toggle
  - Test during market hours
  - Test reconnection after disconnect
  - Test mobile responsiveness

**Deliverables**:
- Unit tests with >80% coverage for critical modules
- Integration test suite
- Performance test results
- Manual test plan document

---

### 6.4 Production Deployment Checklist
**Priority**: CRITICAL

**Complete before production launch**:

**IB Gateway Configuration**:
- [ ] IB Gateway Docker image pinned to specific stable version
- [ ] Auto-restart configured for 11:45 PM ET
- [ ] CME Real-Time subscription active and verified
- [ ] API acknowledgement form completed in Client Portal
- [ ] Paper trading data sharing enabled (if using paper)
- [ ] Unique client IDs assigned per application instance
- [ ] Read-Only API enabled (if not placing orders)

**Application Configuration**:
- [ ] Production config file created
- [ ] All secrets moved to environment variables
- [ ] Logging configured with rotation
- [ ] Log level set to INFO or WARNING
- [ ] Cache directory configured with sufficient disk space
- [ ] Max bars in memory configured appropriately

**Error Handling**:
- [ ] Connection state machine handles 1100/1101/1102
- [ ] Historical data pacing queue implemented
- [ ] Exponential backoff with jitter for retries
- [ ] Data quality validation on incoming bars
- [ ] Graceful shutdown handlers registered

**Monitoring**:
- [ ] Health check endpoint tested
- [ ] Application metrics exposed
- [ ] Logging verified and readable
- [ ] Alert system configured (optional)

**Performance**:
- [ ] Data conflation enabled
- [ ] Lazy loading implemented
- [ ] Memory limits tested
- [ ] Load tested with multiple clients

**Documentation**:
- [ ] README with setup instructions
- [ ] Configuration documentation
- [ ] API documentation
- [ ] Troubleshooting guide

**Critical References**:
- Missing Pieces: "Production deployment checklist" (lines 512-545)

---

## Phase 7: Documentation and Handoff

### 7.1 User Documentation
**Priority**: MEDIUM | **Duration**: 1 day

**Tasks**:

- [ ] **Create README.md**
  - Project description
  - Features list
  - Requirements
  - Installation instructions
  - Quick start guide
  - Configuration guide
  - Troubleshooting section

- [ ] **Create API documentation**
  - WebSocket message protocol
  - REST endpoints (if any)
  - Data formats

- [ ] **Create developer guide**
  - Architecture overview
  - Code structure
  - How to add indicators
  - How to modify UI
  - Testing guide

**Deliverables**:
- Comprehensive README.md
- API documentation
- Developer guide

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| **Phase 1: Foundation** | 2-3 days | None |
| 1.1 IB Gateway Configuration | 1-2 days | - |
| 1.2 Contract Specifications | 0.5 days | 1.1 |
| 1.3 Development Environment | 0.5 days | - |
| **Phase 2: Backend Data Pipeline** | 7-11 days | Phase 1 |
| 2.1 IB Connection Service | 1-2 days | 1.1, 1.3 |
| 2.2 Historical Data Fetching | 2-3 days | 2.1 |
| 2.3 Real-Time Streaming | 2-3 days | 2.1 |
| 2.4 FastAPI WebSocket | 2-3 days | 2.2, 2.3 |
| **Phase 3: Frontend** | 4-7 days | Phase 2 |
| 3.1 Lightweight Charts | 2-3 days | 2.4 |
| 3.2 WebSocket Client | 1-2 days | 3.1 |
| 3.3 UI Enhancements | 1-2 days | 3.2 |
| **Phase 4: Indicators** | 5-8 days | Phase 3 |
| 4.1 Indicator Architecture | 1-2 days | 3.1 |
| 4.2 Core Indicators | 2-3 days | 4.1 |
| 4.3 Frontend Rendering | 2-3 days | 4.2 |
| 4.4 Simplified Syntax (Optional) | 1-2 days | 4.2 |
| **Phase 5: Performance** | 3-4 days | Phase 4 |
| 5.1 Data Volume Management | 1-2 days | Phase 3 |
| 5.2 WebSocket Optimization | 1 day | 2.4 |
| 5.3 Caching Optimization | 1 day | 2.2 |
| **Phase 6: Production** | 6-9 days | Phase 5 |
| 6.1 Error Handling | 2-3 days | All previous |
| 6.2 Configuration & Deployment | 1-2 days | All previous |
| 6.3 Testing Strategy | 2-3 days | All previous |
| 6.4 Deployment Checklist | 1 day | All previous |
| **Phase 7: Documentation** | 1 day | Phase 6 |
| 7.1 User Documentation | 1 day | All previous |
| **TOTAL** | **28-43 days** | |

**Realistic Timeline for Experienced Developer**: 4-6 weeks

**Critical Path**:
1. IB Gateway setup and market data subscriptions (1-2 days + 24-hour wait)
2. Backend data pipeline (7-11 days)
3. Frontend with Lightweight Charts (4-7 days)
4. Production readiness (6-9 days)

---

## Critical Success Factors

### Must-Have Features for MVP
1. ✅ IB Gateway connection with proper error handling
2. ✅ Historical data (1 year, 1-minute bars) with caching
3. ✅ Real-time updates (5-second or tick-by-tick)
4. ✅ TradingView-quality candlestick chart
5. ✅ Volume bars with proper coloring
6. ✅ Basic indicators (SMA, EMA, RSI, MACD)
7. ✅ Symbol switching (MNQ, MES, MGC)
8. ✅ Proper data volume management (<60K bars)

### Nice-to-Have Features for V2
- Multiple chart panes
- Advanced indicators (Bollinger Bands, Stochastic, etc.)
- Drawing tools (trendlines, Fibonacci, etc.)
- Alerts system
- Order placement integration
- Multi-symbol watchlist
- Mobile app version

### Technical Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Market data subscription delays | Start subscription 48+ hours before development |
| IB Gateway connection instability | Implement robust reconnection logic with exponential backoff |
| Performance issues with large datasets | Implement lazy loading and multi-timeframe aggregation from start |
| Real-time latency | Use tick-by-tick if needed, benchmark early |
| Mobile performance | Test on real devices, keep bar count <20K |
| Memory leaks | Implement deque with max length, monitor memory usage |

---

## Key Architectural Decisions

### Real-Time Data: keepUpToDate vs Tick-by-Tick
**Decision**: Start with `keepUpToDate=True` (5-second updates)
**Rationale**:
- Simpler implementation
- Sufficient for most trading use cases
- 95% of users won't notice 5-second vs real-time
- Upgrade to tick-by-tick only if users demand sub-second updates

### Data Volume: Full Load vs Lazy Loading
**Decision**: Lazy loading with multi-timeframe aggregation
**Rationale**:
- 375K+ bars will crash mobile and degrade desktop
- Lazy loading matches TradingView's behavior
- Pre-computing timeframes enables smooth zoom
- Desktop limit: 60K bars, Mobile limit: 20K bars

### Indicator Syntax: Complex vs Simple
**Decision**: Start with decorator pattern, add DSL later if needed
**Rationale**:
- Decorator provides full Python power with minimal boilerplate
- DSL adds complexity and security concerns
- Most users comfortable with basic Python
- Can add simplified syntax in V2 based on feedback

### Deployment: Standalone vs Dockerized
**Decision**: Provide both options
**Rationale**:
- Standalone better for development and simple setups
- Docker better for production and IB Gateway management
- Docker handles IB Gateway auto-restart cleanly

---

## Next Steps

### Immediate Actions (Before Starting Development)
1. **Subscribe to market data** (must wait 24-48 hours)
2. **Enable paper trading data sharing** (must wait 24 hours)
3. **Complete API Acknowledgement form** in Client Portal
4. **Download and test IB Gateway connection**
5. **Clone repository and set up development environment**

### Week 1 Focus
- Complete Phase 1 (Foundation)
- Start Phase 2 (Backend) with connection service and historical data

### Week 2-3 Focus
- Complete Phase 2 (Backend) with real-time streaming
- Complete Phase 3 (Frontend) with full chart rendering

### Week 4-5 Focus
- Complete Phase 4 (Indicators)
- Start Phase 5 (Performance)

### Week 6 Focus
- Complete Phase 6 (Production Readiness)
- Testing and deployment

---

## Resources and References

### Official Documentation
- Interactive Brokers API: https://interactivebrokers.github.io/tws-api/
- ib_insync Documentation: https://ib-insync.readthedocs.io/
- TradingView Lightweight Charts: https://tradingview.github.io/lightweight-charts/
- FastAPI Documentation: https://fastapi.tiangolo.com/

### Critical Document Sections
- Main Guide: Complete implementation (all sections)
- Critical Answers: keepUpToDate limitations, volume coloring, performance reality
- Missing Pieces: Production deployment, error handling, configuration

### GitHub References
- ib_insync: https://github.com/erdewit/ib_insync
- Lightweight Charts: https://github.com/tradingview/lightweight-charts
- IB Gateway Docker: https://github.com/gnzsnz/ib-gateway-docker

---

## Appendix: Quick Reference

### IB Gateway Error Codes
- **1100**: Connection lost - stop new requests
- **1101**: Connection restored, data lost - re-subscribe everything
- **1102**: Connection restored, data maintained - resume
- **162**: Pacing violation - wait 15+ seconds
- **354**: Not subscribed - check subscriptions and API form
- **2103/2105**: Farm disconnected - usually temporary

### Contract Specifications Quick Reference
```python
# MNQ - Micro E-mini Nasdaq-100
Future(symbol='MNQ', exchange='CME', currency='USD', lastTradeDateOrContractMonth='202503')

# MES - Micro E-mini S&P 500
Future(symbol='MES', exchange='CME', currency='USD', lastTradeDateOrContractMonth='202503')

# MGC - Micro Gold
Future(symbol='MGC', exchange='COMEX', currency='USD', lastTradeDateOrContractMonth='202502')
```

### TradingView Color Scheme
```javascript
UP: '#26a69a'
DOWN: '#ef5350'
VOLUME_UP: 'rgba(38, 166, 154, 0.5)'
VOLUME_DOWN: 'rgba(239, 83, 80, 0.5)'
BACKGROUND: '#131722'
TEXT: '#d1d4dc'
GRID: '#1e222d'
CROSSHAIR: '#758696'
```

### Performance Limits
- Desktop max: 60,000 bars
- Mobile max: 20,000 bars
- Initial load target: <2 seconds
- Real-time latency target: <100ms
- WebSocket updates: max 20/second

---

**Ready to begin!** Start with Phase 1.1 (IB Gateway Configuration) and work through the roadmap sequentially. Each phase builds on the previous, so don't skip ahead. Good luck! 🚀
