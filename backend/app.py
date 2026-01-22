"""
FastAPI WebSocket Backend for Futures Charting

Provides WebSocket streaming of real-time and historical market data.
"""

import asyncio
import sys

# CRITICAL: Windows-specific event loop configuration - MUST be first
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nest_asyncio
    nest_asyncio.apply()

import logging
import os
from contextlib import asynccontextmanager
from typing import Set, Dict, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yaml
from pathlib import Path

from .ib_service import IBConnectionManager
from .contracts import get_current_contract, get_contract_info
from .historical_data import HistoricalDataFetcher
from .realtime import RealtimeManager
from .cache import DataCache
from .indicators import IndicatorManager, list_available_indicators
from .security import (
    RateLimiter,
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    validate_symbol,
    validate_bar_size,
    validate_indicator_params
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = Path(__file__).parent.parent / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Global instances
ib_manager: Optional[IBConnectionManager] = None
realtime_manager: Optional[RealtimeManager] = None
historical_fetcher: Optional[HistoricalDataFetcher] = None
cache: Optional[DataCache] = None
indicator_manager: Optional[IndicatorManager] = None


class ConnectionManager:
    """Manages WebSocket connections to clients with message batching"""

    def __init__(self, batch_interval: float = 0.1, max_batch_size: int = 50):
        """
        Initialize connection manager with batching support.

        Args:
            batch_interval: Time in seconds to wait before sending batched messages (default: 0.1s)
            max_batch_size: Maximum number of messages to batch before sending (default: 50)
        """
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.message_queues: Dict[str, Dict[WebSocket, list]] = {}
        self.batch_interval = batch_interval
        self.max_batch_size = max_batch_size
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        self.batching_enabled = True

    async def connect(self, websocket: WebSocket, symbol: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()

        if symbol not in self.active_connections:
            self.active_connections[symbol] = set()
            self.message_queues[symbol] = {}

        self.active_connections[symbol].add(websocket)
        self.message_queues[symbol][websocket] = []

        logger.info(f"Client connected to {symbol} (total: {len(self.active_connections[symbol])})")

        # Start batch processor for this symbol if not already running
        if symbol not in self.batch_tasks and self.batching_enabled:
            self.batch_tasks[symbol] = asyncio.create_task(self._batch_processor(symbol))

    def disconnect(self, websocket: WebSocket, symbol: str):
        """Remove a WebSocket connection"""
        if symbol in self.active_connections:
            self.active_connections[symbol].discard(websocket)

            # Clean up message queue
            if symbol in self.message_queues and websocket in self.message_queues[symbol]:
                del self.message_queues[symbol][websocket]

            logger.info(f"Client disconnected from {symbol} (remaining: {len(self.active_connections[symbol])})")

            # Clean up empty symbol entries
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

                # Stop batch processor
                if symbol in self.batch_tasks:
                    self.batch_tasks[symbol].cancel()
                    del self.batch_tasks[symbol]

                if symbol in self.message_queues:
                    del self.message_queues[symbol]

    async def send_to_client(self, websocket: WebSocket, message: dict, immediate: bool = True):
        """
        Send message to a specific client.

        Args:
            websocket: WebSocket connection
            message: Message to send
            immediate: If True, send immediately. If False, batch the message.
        """
        try:
            if immediate or not self.batching_enabled:
                await websocket.send_json(message)
            else:
                # Find symbol for this websocket
                for symbol, connections in self.active_connections.items():
                    if websocket in connections:
                        if symbol in self.message_queues and websocket in self.message_queues[symbol]:
                            self.message_queues[symbol][websocket].append(message)

                            # If batch is full, send immediately
                            if len(self.message_queues[symbol][websocket]) >= self.max_batch_size:
                                await self._flush_client_queue(symbol, websocket)
                        break
        except Exception as e:
            logger.error(f"Error sending to client: {e}")

    async def broadcast(self, symbol: str, message: dict, immediate: bool = False):
        """
        Broadcast message to all clients subscribed to symbol.

        Args:
            symbol: Symbol to broadcast to
            message: Message to send
            immediate: If True, send immediately. If False, batch the message.
        """
        if symbol not in self.active_connections:
            return

        if immediate or not self.batching_enabled:
            # Send immediately without batching
            disconnected = set()

            for websocket in self.active_connections[symbol]:
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")
                    disconnected.add(websocket)

            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket, symbol)
        else:
            # Add to batch queues
            if symbol in self.message_queues:
                for websocket in self.active_connections[symbol]:
                    if websocket in self.message_queues[symbol]:
                        self.message_queues[symbol][websocket].append(message)

                        # If batch is full, flush immediately
                        if len(self.message_queues[symbol][websocket]) >= self.max_batch_size:
                            await self._flush_client_queue(symbol, websocket)

    async def _flush_client_queue(self, symbol: str, websocket: WebSocket):
        """Flush message queue for a specific client"""
        if symbol not in self.message_queues or websocket not in self.message_queues[symbol]:
            return

        messages = self.message_queues[symbol][websocket]
        if not messages:
            return

        try:
            # Send batched messages
            if len(messages) == 1:
                await websocket.send_json(messages[0])
            else:
                await websocket.send_json({
                    'type': 'batch',
                    'messages': messages,
                    'count': len(messages)
                })

            # Clear queue
            self.message_queues[symbol][websocket] = []

        except Exception as e:
            logger.error(f"Error flushing client queue: {e}")
            self.disconnect(websocket, symbol)

    async def _batch_processor(self, symbol: str):
        """Background task to flush message queues at regular intervals"""
        try:
            while True:
                await asyncio.sleep(self.batch_interval)

                if symbol not in self.message_queues:
                    break

                # Flush all client queues for this symbol
                disconnected = set()
                for websocket in list(self.message_queues[symbol].keys()):
                    try:
                        await self._flush_client_queue(symbol, websocket)
                    except Exception as e:
                        logger.error(f"Error in batch processor: {e}")
                        disconnected.add(websocket)

                # Clean up disconnected clients
                for websocket in disconnected:
                    self.disconnect(websocket, symbol)

        except asyncio.CancelledError:
            # Final flush before shutdown
            if symbol in self.message_queues:
                for websocket in list(self.message_queues[symbol].keys()):
                    try:
                        await self._flush_client_queue(symbol, websocket)
                    except:
                        pass


connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown"""
    global ib_manager, realtime_manager, historical_fetcher, cache, indicator_manager

    logger.info("Starting application...")

    # Initialize cache
    cache = DataCache(cache_dir=config['data']['cache_dir'])

    # Initialize indicator manager
    indicator_manager = IndicatorManager()

    # Connect to IB Gateway
    ib_manager = IBConnectionManager(
        host=config['ib_gateway']['host'],
        port=config['ib_gateway']['port'],
        client_id=config['ib_gateway']['client_id'],
        timeout=config['ib_gateway']['timeout']
    )

    connected = await ib_manager.connect()

    if not connected:
        logger.error("Failed to connect to IB Gateway - application will not function")
        # Continue anyway to allow health checks
    else:
        # Initialize real-time manager (use keepUpToDate by default)
        realtime_manager = RealtimeManager(
            ib=ib_manager.ib,
            use_tick_by_tick=False,  # Can be made configurable
            bar_size_minutes=1
        )

        # Initialize historical data fetcher
        historical_fetcher = HistoricalDataFetcher(
            ib=ib_manager.ib,
            cache=cache
        )

        logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down application...")

    if realtime_manager:
        realtime_manager.stop_all_streams()

    if ib_manager:
        ib_manager.disconnect()

    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Futures Charting API",
    description="Real-time futures charting with IB Gateway",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add rate limiting middleware
rate_limiter = RateLimiter(
    requests_per_minute=os.getenv("RATE_LIMIT_PER_MINUTE", 100),
    requests_per_hour=os.getenv("RATE_LIMIT_PER_HOUR", 1000)
)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Mount static files
static_dir = Path(__file__).parent.parent / 'frontend' / 'static'
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent.parent / 'frontend' / 'templates' / 'index.html'

    if html_path.exists():
        return FileResponse(html_path)
    else:
        return {"message": "Futures Charting API", "status": "running"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns application health status including IB Gateway connection.
    """
    if not ib_manager:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "IB manager not initialized"
            }
        )

    health = ib_manager.get_health_status()

    is_healthy = health.get('connected', False)

    # Check data staleness if we have subscriptions
    if 'data_stale' in health and health['data_stale']:
        is_healthy = False

    status_code = 200 if is_healthy else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "ib_gateway": health,
            "realtime_streams": (
                len(realtime_manager.streamers) if realtime_manager else 0
            ),
            "cache_info": cache.get_cache_size() if cache else {}
        }
    )


@app.get("/api/contracts")
async def list_contracts():
    """List available contracts"""
    return {
        "contracts": [
            {
                "symbol": symbol,
                **get_contract_info(symbol)
            }
            for symbol in ['MNQ', 'MES', 'MGC']
        ]
    }


@app.get("/api/cache/{symbol}")
async def get_cache_info(symbol: str):
    """Get cache information for a symbol"""
    if not cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")

    metadata = cache.get_metadata(symbol, '1min')

    if metadata:
        return metadata
    else:
        raise HTTPException(status_code=404, detail=f"No cache found for {symbol}")


@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time market data.

    Protocol:
    - Sends historical data on connection
    - Streams real-time bar updates
    - Messages: {type: 'historical'|'bar_update', data: {...}, is_new_bar: bool}
    """
    logger.info(f"WebSocket connection request for {symbol}")

    # Validate symbol
    if symbol not in ['MNQ', 'MES', 'MGC']:
        logger.error(f"Invalid symbol: {symbol}")
        await websocket.close(code=1003, reason=f"Invalid symbol: {symbol}")
        return

    # Check if IB Gateway is connected
    if not ib_manager or not ib_manager.is_connected():
        logger.error("IB Gateway not connected - closing WebSocket")
        await websocket.close(code=1011, reason="IB Gateway not connected")
        return

    # Accept connection
    logger.info(f"Accepting WebSocket connection for {symbol}")
    await connection_manager.connect(websocket, symbol)

    try:
        # Get contract
        contract = get_current_contract(symbol)

        logger.info(f"Starting data stream for {symbol} (contract: {contract.localSymbol})")

        # Step 1: Send cached or fetch historical data (5 days for fast loading)
        if historical_fetcher:
            try:
                logger.info(f"Fetching historical data for {symbol}...")
                historical_data = await historical_fetcher.fetch_recent(
                    contract,
                    duration='5 D',  # 5 days of data for fast loading
                    bar_size='1 min'
                )

                if historical_data is not None and len(historical_data) > 0:
                    # Convert DataFrame to list of dicts
                    data_list = historical_data.to_dict('records')

                    # Calculate indicators if any are active
                    indicators_data = {}
                    if indicator_manager and len(indicator_manager.active_indicators) > 0:
                        try:
                            indicators_data = indicator_manager.calculate_all(historical_data)
                            logger.info(f"Calculated {len(indicators_data)} indicators for {symbol}")
                        except Exception as e:
                            logger.error(f"Error calculating indicators: {e}")

                    # Send historical data with indicators
                    await connection_manager.send_to_client(websocket, {
                        'type': 'historical',
                        'data': data_list,
                        'symbol': symbol,
                        'bar_count': len(data_list),
                        'indicators': indicators_data
                    })

                    logger.info(f"Sent {len(data_list)} historical bars to client")
                else:
                    logger.warning(f"No historical data available for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                await connection_manager.send_to_client(websocket, {
                    'type': 'error',
                    'message': f"Failed to fetch historical data: {str(e)}"
                })

        # Step 2: Start real-time streaming
        if realtime_manager:
            # Callback for bar updates
            async def on_bar_update(bar_data: dict, is_new_bar: bool):
                """Called on every bar update"""
                await connection_manager.broadcast(symbol, {
                    'type': 'bar_update',
                    'data': bar_data,
                    'is_new_bar': is_new_bar,
                    'symbol': symbol
                })

            # Start streaming
            success = await realtime_manager.start_stream(contract, on_bar_update)

            if success:
                logger.info(f"Real-time stream started for {symbol}")
            else:
                logger.error(f"Failed to start real-time stream for {symbol}")
                await connection_manager.send_to_client(websocket, {
                    'type': 'error',
                    'message': "Failed to start real-time stream"
                })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive messages from client (for keepalive or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)

                # Handle client commands here if needed
                # For now, just log
                logger.debug(f"Received from client: {data}")

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({'type': 'ping'})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {symbol}")

    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")

    finally:
        # Clean up
        connection_manager.disconnect(websocket, symbol)

        # Stop streaming if no more clients for this symbol
        if realtime_manager and symbol not in connection_manager.active_connections:
            realtime_manager.stop_stream(symbol)
            logger.info(f"Stopped real-time stream for {symbol} (no clients)")


@app.get("/api/statistics")
async def get_statistics():
    """Get application statistics"""
    stats = {
        "ib_gateway": ib_manager.get_health_status() if ib_manager else {},
        "realtime_streams": (
            realtime_manager.get_all_statistics() if realtime_manager else {}
        ),
        "historical_fetcher": (
            historical_fetcher.get_statistics() if historical_fetcher else {}
        ),
        "cache": {
            "size": cache.get_cache_size() if cache else {},
            "symbols": cache.list_cached_symbols() if cache else []
        },
        "active_connections": {
            symbol: len(connections)
            for symbol, connections in connection_manager.active_connections.items()
        }
    }

    return stats


@app.get("/api/indicators")
async def get_available_indicators():
    """Get list of available indicators"""
    return {
        "indicators": list_available_indicators()
    }


@app.get("/api/indicators/active")
async def get_active_indicators():
    """Get list of active indicators"""
    if not indicator_manager:
        raise HTTPException(status_code=503, detail="Indicator manager not initialized")

    return {
        "indicators": indicator_manager.list_active_indicators()
    }


@app.post("/api/indicators/{indicator_type}")
async def add_indicator(
    indicator_type: str,
    params: Optional[Dict[str, Any]] = None
):
    """
    Add an indicator.

    Example:
        POST /api/indicators/sma
        Body: {"period": 20, "color": "#2962FF"}
    """
    if not indicator_manager:
        raise HTTPException(status_code=503, detail="Indicator manager not initialized")

    indicator = indicator_manager.add_indicator(indicator_type, params)

    if indicator:
        return {
            "success": True,
            "indicator": indicator.to_dict()
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to add indicator: {indicator_type}"
        )


@app.delete("/api/indicators/{indicator_id}")
async def remove_indicator(indicator_id: str):
    """Remove an indicator by ID"""
    if not indicator_manager:
        raise HTTPException(status_code=503, detail="Indicator manager not initialized")

    success = indicator_manager.remove_indicator(indicator_id)

    if success:
        return {"success": True, "message": f"Indicator {indicator_id} removed"}
    else:
        raise HTTPException(status_code=404, detail=f"Indicator not found: {indicator_id}")


@app.get("/api/indicators/calculate/{symbol}")
async def calculate_indicators(symbol: str):
    """Calculate all active indicators for a symbol"""
    if not indicator_manager:
        raise HTTPException(status_code=503, detail="Indicator manager not initialized")

    if not cache:
        raise HTTPException(status_code=503, detail="Cache not initialized")

    # Load cached data
    cached_data = cache.load(symbol, bar_size='1min', max_age_hours=24)

    if cached_data is None or len(cached_data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data available for {symbol}"
        )

    # Calculate all indicators
    results = indicator_manager.calculate_all(cached_data)

    return {
        "symbol": symbol,
        "indicators": results,
        "data_points": len(cached_data)
    }


@app.get("/ready")
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes/container orchestration.

    Returns 200 if the application is ready to serve requests,
    503 if it's still starting up or has critical issues.
    """
    checks = {
        "ib_gateway": False,
        "cache": False,
        "indicators": False,
    }

    # Check IB Gateway connection
    if ib_manager and ib_manager.is_connected():
        checks["ib_gateway"] = True

    # Check cache
    if cache:
        checks["cache"] = True

    # Check indicator manager
    if indicator_manager:
        checks["indicators"] = True

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "ready": all_ready,
            "checks": checks
        }
    )


@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    import time
    import psutil
    import os

    # Get process info
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    # Build metrics
    metrics_text = []

    # Help and type declarations
    metrics_text.append("# HELP futures_charting_info Application information")
    metrics_text.append("# TYPE futures_charting_info gauge")
    metrics_text.append('futures_charting_info{version="0.1.0"} 1')

    # Memory metrics
    metrics_text.append("# HELP process_memory_bytes Process memory usage in bytes")
    metrics_text.append("# TYPE process_memory_bytes gauge")
    metrics_text.append(f"process_memory_bytes {{type=\"rss\"}} {memory_info.rss}")
    metrics_text.append(f"process_memory_bytes {{type=\"vms\"}} {memory_info.vms}")

    # CPU metrics
    metrics_text.append("# HELP process_cpu_percent Process CPU usage percentage")
    metrics_text.append("# TYPE process_cpu_percent gauge")
    metrics_text.append(f"process_cpu_percent {process.cpu_percent()}")

    # IB Gateway connection
    if ib_manager:
        connected = 1 if ib_manager.is_connected() else 0
        metrics_text.append("# HELP ib_gateway_connected IB Gateway connection status")
        metrics_text.append("# TYPE ib_gateway_connected gauge")
        metrics_text.append(f"ib_gateway_connected {connected}")

    # Active WebSocket connections
    total_connections = sum(
        len(connections)
        for connections in connection_manager.active_connections.values()
    )
    metrics_text.append("# HELP websocket_connections_total Total active WebSocket connections")
    metrics_text.append("# TYPE websocket_connections_total gauge")
    metrics_text.append(f"websocket_connections_total {total_connections}")

    # Per-symbol connections
    for symbol, connections in connection_manager.active_connections.items():
        metrics_text.append(f'websocket_connections{{symbol="{symbol}"}} {len(connections)}')

    # Realtime streams
    if realtime_manager:
        stream_count = len(realtime_manager.streamers)
        metrics_text.append("# HELP realtime_streams_active Active real-time data streams")
        metrics_text.append("# TYPE realtime_streams_active gauge")
        metrics_text.append(f"realtime_streams_active {stream_count}")

    # Cache statistics
    if cache:
        cache_stats = cache.get_cache_size()
        if 'total_size' in cache_stats:
            metrics_text.append("# HELP cache_size_bytes Total cache size in bytes")
            metrics_text.append("# TYPE cache_size_bytes gauge")
            metrics_text.append(f"cache_size_bytes {cache_stats['total_size']}")

        if 'total_bars' in cache_stats:
            metrics_text.append("# HELP cache_bars_total Total bars cached")
            metrics_text.append("# TYPE cache_bars_total gauge")
            metrics_text.append(f"cache_bars_total {cache_stats['total_bars']}")

    # Active indicators
    if indicator_manager:
        indicator_count = len(indicator_manager.active_indicators)
        metrics_text.append("# HELP indicators_active_total Active indicators")
        metrics_text.append("# TYPE indicators_active_total gauge")
        metrics_text.append(f"indicators_active_total {indicator_count}")

    return Response(
        content="\n".join(metrics_text) + "\n",
        media_type="text/plain; version=0.0.4"
    )


@app.get("/api/rate-limit-info")
async def rate_limit_info():
    """Get rate limiting information for the current client"""
    # This would need access to the client IP from the request
    # For now, return general rate limit configuration
    return {
        "limits": {
            "per_minute": int(os.getenv("RATE_LIMIT_PER_MINUTE", 100)),
            "per_hour": int(os.getenv("RATE_LIMIT_PER_HOUR", 1000))
        },
        "note": "Rate limits are per IP address"
    }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host=config['server']['host'],
        port=config['server']['port'],
        log_level=config['server']['log_level']
    )
