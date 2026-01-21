# Startup Guide - Step by Step

This guide will walk you through starting the Futures Charting application from scratch.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10, 3.11, or 3.12 installed
- [ ] IB Gateway or TWS installed (or Docker)
- [ ] Interactive Brokers account (paper or live)
- [ ] Market data subscriptions for CME futures
- [ ] At least 2GB RAM available
- [ ] 10GB disk space for cache

---

## Step 1: Verify Python Installation

```bash
# Check Python version (should be 3.10+)
python --version
# OR
python3 --version

# If you don't have Python 3.10+, install it:
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y
```

**Expected Output:**
```
Python 3.11.x
```

---

## Step 2: Set Up IB Gateway

### Option A: Install IB Gateway Directly (Recommended for First Time)

1. **Download IB Gateway:**
   ```bash
   # Linux
   wget https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh

   # macOS
   # Download from: https://www.interactivebrokers.com/en/trading/ibgateway-latest.php
   ```

2. **Install:**
   ```bash
   # Linux
   chmod +x ibgateway-latest-standalone-linux-x64.sh
   ./ibgateway-latest-standalone-linux-x64.sh

   # Follow the installation wizard
   # Default install location: ~/Jts
   ```

3. **Configure IB Gateway for API:**
   - Launch IB Gateway (GUI will open)
   - Login with your credentials
   - Go to **Configure → Settings → API → Settings**
   - Check these boxes:
     - ✅ Enable ActiveX and Socket Clients
     - ✅ Allow connections from localhost
     - ✅ Read-Only API = **NO** (unchecked)
   - Set **Socket Port**:
     - `7497` for live trading
     - `7496` for paper trading
   - Click **OK**

4. **Important Market Data Setup:**
   - Go to **Configure → Settings → Market Data Subscriptions**
   - Ensure you have:
     - ✅ CME Real-Time (for MES, MNQ)
     - ✅ COMEX (for MGC)
   - Go to **Configure → Settings → User Settings**
   - Complete **Market Data API Acknowledgement Form**
   - Wait 24 hours for activation (new accounts)

5. **Keep IB Gateway Running:**
   - Leave the IB Gateway window open
   - You'll see "Logged in" status

### Option B: Use Docker (Advanced)

```bash
# Pull IB Gateway Docker image
docker pull ghcr.io/unusualcode/ibgateway-docker:latest

# Run IB Gateway in Docker
docker run -d \
  --name ibgateway \
  -p 7497:7497 \
  -e TWS_USERID=your_username \
  -e TWS_PASSWORD=your_password \
  -e TRADING_MODE=paper \
  ghcr.io/unusualcode/ibgateway-docker:latest

# Check logs
docker logs -f ibgateway
```

---

## Step 3: Prepare the Application

### 3.1: Navigate to Project Directory

```bash
cd /home/user/FUTURES
```

### 3.2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate

# Windows:
# venv\Scripts\activate

# You should see (venv) in your prompt
```

**Expected Output:**
```
(venv) user@machine:~/FUTURES$
```

### 3.3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This will install:
# - ib_insync (IB Gateway connection)
# - fastapi, uvicorn (web server)
# - pandas, numpy (data processing)
# - websockets (real-time communication)
# - and all other dependencies
```

**Expected Output:**
```
Successfully installed fastapi-0.109.0 uvicorn-0.27.0 pandas-2.0.0 ...
```

**If you get errors:**
```bash
# Try installing system dependencies first (Ubuntu/Debian):
sudo apt install python3-dev build-essential -y

# Then retry:
pip install -r requirements.txt
```

---

## Step 4: Configure Environment

### 4.1: Create .env File

```bash
# Copy example to .env
cp .env.example .env

# Edit .env file
nano .env
# Or use your preferred editor: vim, code, etc.
```

### 4.2: Configure .env Settings

**For Paper Trading (Recommended First):**
```bash
# IB Gateway Connection
IB_HOST=127.0.0.1
IB_PORT=7496  # Paper trading port
IB_CLIENT_ID=1
IB_TIMEOUT=30

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info

# Cache
CACHE_MAX_AGE_HOURS=24
CACHE_DIR=data/cache

# Rate Limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

**For Live Trading:**
```bash
IB_PORT=7497  # Change to live port
# Everything else stays the same
```

**Save and exit** (Ctrl+X, then Y, then Enter for nano)

### 4.3: Verify Configuration File

```bash
# Check config.yaml exists
cat config.yaml

# Should show:
# ib_gateway:
#   host: 127.0.0.1
#   port: 7497
#   client_id: 1
#   ...
```

---

## Step 5: Test IB Gateway Connection

**BEFORE starting the full application, test the connection:**

```bash
# Make sure IB Gateway is running first!
# Then run:
python test_connection.py
```

**Expected Output (Success):**
```
Connecting to IB Gateway at 127.0.0.1:7496...
✓ Connected to IB Gateway
Testing contract qualification...
✓ MNQ contract qualified: Contract(secType='FUT', conId=595414933, symbol='MNQ', ...)
✓ MES contract qualified: Contract(secType='FUT', conId=525280036, symbol='MES', ...)
✓ MGC contract qualified: Contract(secType='FUT', conId=491493606, symbol='MGC', ...)

All tests passed! IB Gateway is ready.
```

**If you see errors:**

| Error | Solution |
|-------|----------|
| `Connection refused` | IB Gateway is not running. Start it first. |
| `TimeoutError` | Check firewall, verify port is 7496/7497 |
| `Error 354: not subscribed` | Complete Market Data API form, wait 24 hours |
| `Error 162: HMDS data farm connection is broken` | Normal during off-hours, retry during market hours |

---

## Step 6: Start the Application

### Method 1: Direct Python (Easiest for Testing)

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start the application
python run.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:backend.app:Starting application...
INFO:backend.app:Connecting to IB Gateway at 127.0.0.1:7496...
INFO:backend.app:✓ Connected to IB Gateway
INFO:backend.app:Application started successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Method 2: Using Deployment Script

```bash
# Make script executable
chmod +x deployment/deploy.sh

# Run deployment
./deployment/deploy.sh dev

# This will:
# - Check requirements
# - Create backup
# - Install dependencies
# - Run tests
# - Start application
```

### Method 3: Using Systemd (Production)

```bash
# Install service
sudo cp deployment/futures-charting.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start futures-charting

# Check status
sudo systemctl status futures-charting

# View logs
sudo journalctl -u futures-charting -f
```

---

## Step 7: Verify Application is Running

### 7.1: Check Health Endpoint

```bash
# In a new terminal:
curl http://localhost:8000/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "ib_gateway": {
    "connected": true,
    "connection_time": "2026-01-21T10:30:00",
    "data_stale": false
  },
  "realtime_streams": 0,
  "cache_info": {}
}
```

### 7.2: Check Readiness

```bash
curl http://localhost:8000/ready
```

**Expected Output:**
```json
{
  "ready": true,
  "checks": {
    "ib_gateway": true,
    "cache": true,
    "indicators": true
  }
}
```

### 7.3: Check Available Contracts

```bash
curl http://localhost:8000/api/contracts
```

**Expected Output:**
```json
{
  "contracts": [
    {
      "symbol": "MNQ",
      "name": "Micro E-mini Nasdaq-100",
      "exchange": "CME",
      "currency": "USD",
      "multiplier": 2,
      "tick_size": 0.25
    },
    ...
  ]
}
```

---

## Step 8: Open the Web Interface

1. **Open your web browser**
2. **Navigate to:** `http://localhost:8000`
3. **You should see:**
   - Chart interface
   - Symbol selector (MNQ, MES, MGC)
   - Indicator buttons (SMA, EMA, RSI, MACD, BB)

4. **Select a symbol** (e.g., MNQ)
5. **Wait for data to load:**
   - First time: 30-45 minutes (downloading from IB)
   - Subsequent loads: <2 seconds (from cache)

---

## Step 9: Monitor the Application

### View Logs

```bash
# If running with python run.py:
# Logs are in the terminal

# If running with systemd:
sudo journalctl -u futures-charting -f

# Application logs:
tail -f logs/app.log

# Error logs:
tail -f logs/error.log
```

### Check Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Statistics
curl http://localhost:8000/api/statistics
```

---

## Troubleshooting Common Issues

### Issue 1: "Cannot connect to IB Gateway"

**Solution:**
```bash
# 1. Check IB Gateway is running
ps aux | grep ibgateway

# 2. Check the port
# In IB Gateway GUI: Configure → Settings → API → Settings
# Verify Socket Port matches your .env (7496 or 7497)

# 3. Test connection
telnet localhost 7496
# Should connect, type Ctrl+] then 'quit' to exit

# 4. Check firewall
sudo ufw status
# Make sure port 7496/7497 is allowed

# 5. Try restarting IB Gateway
```

### Issue 2: "ModuleNotFoundError"

**Solution:**
```bash
# 1. Ensure virtual environment is activated
source venv/bin/activate

# 2. Reinstall dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import fastapi; import pandas; print('OK')"
```

### Issue 3: "Error 354: not subscribed"

**Solution:**
1. Login to Client Portal: https://www.interactivebrokers.com
2. Go to Settings → User Settings
3. Complete "Market Data API Acknowledgement Form"
4. Wait 24 hours for activation
5. For paper trading, go to Settings → Paper Trading Account
6. Enable "Share real-time market data"

### Issue 4: "Port 8000 already in use"

**Solution:**
```bash
# Find what's using port 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# OR change the port in .env:
SERVER_PORT=8001
```

### Issue 5: Application starts but no data loads

**Solution:**
```bash
# 1. Check if market is open
# CME futures: Sunday 6 PM - Friday 5 PM ET (with breaks)

# 2. Check cache
ls -lh data/cache/
# If empty, first load takes 30-45 minutes

# 3. Check IB Gateway market data subscriptions
# Must have CME Real-Time and COMEX subscriptions

# 4. Wait for market hours
# Historical data loads faster during market hours
```

---

## Complete Startup Checklist

Use this checklist every time you start the application:

- [ ] **IB Gateway Running**
  ```bash
  # Check: ps aux | grep ibgateway
  ```

- [ ] **Virtual Environment Activated**
  ```bash
  # Check: which python (should show venv path)
  ```

- [ ] **Dependencies Installed**
  ```bash
  # Check: pip list | grep fastapi
  ```

- [ ] **Environment Configured**
  ```bash
  # Check: cat .env (verify IB_PORT matches Gateway)
  ```

- [ ] **Connection Test Passed**
  ```bash
  # Run: python test_connection.py
  ```

- [ ] **Application Started**
  ```bash
  # Run: python run.py
  ```

- [ ] **Health Check OK**
  ```bash
  # Run: curl http://localhost:8000/health
  ```

- [ ] **Web Interface Accessible**
  ```bash
  # Open: http://localhost:8000
  ```

---

## Quick Start Commands (Summary)

```bash
# 1. Start IB Gateway (GUI or Docker)
# GUI: Click IB Gateway icon, login
# Docker: docker start ibgateway

# 2. Navigate to project
cd /home/user/FUTURES

# 3. Activate virtual environment
source venv/bin/activate

# 4. Test connection (first time or after IB Gateway restart)
python test_connection.py

# 5. Start application
python run.py

# 6. Open browser
# Go to: http://localhost:8000

# 7. Monitor logs (in another terminal)
tail -f logs/app.log
```

---

## Stopping the Application

### If running with `python run.py`:
```bash
# Press Ctrl+C in the terminal
# Application will shutdown gracefully
```

### If running with systemd:
```bash
sudo systemctl stop futures-charting
```

### If running with Docker:
```bash
docker stop futures-charting
```

---

## Next Steps After Successful Startup

1. **Test the features:**
   - Select different symbols (MNQ, MES, MGC)
   - Add indicators (click SMA, EMA, RSI buttons)
   - Watch real-time updates (during market hours)

2. **Check the cache:**
   ```bash
   ls -lh data/cache/
   # You should see .parquet files for each symbol
   ```

3. **Monitor performance:**
   ```bash
   # Run load test
   python tests/load_test.py --clients 5 --duration 30

   # Run memory profiler
   python tests/memory_profiler.py --profile all
   ```

4. **Read the documentation:**
   - `README.md` - Overview and features
   - `TESTING.md` - Testing guide
   - `DEPLOYMENT_GUIDE.md` - Production deployment
   - `VALIDATION_REPORT.md` - System validation

---

## Getting Help

If you encounter issues not covered here:

1. **Check the logs:**
   ```bash
   tail -100 logs/app.log
   ```

2. **Check IB Gateway logs:**
   ```bash
   tail -100 ~/Jts/log/ibgateway.*.log
   ```

3. **Test individual components:**
   ```bash
   # Test imports
   python -c "from backend.app import app; print('OK')"

   # Test contracts
   python -c "from backend.contracts import get_current_contract; print(get_current_contract('MNQ'))"

   # Test cache
   python -c "from backend.cache import DataCache; c=DataCache(); print('OK')"
   ```

4. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

---

## Important Notes

⚠️ **Market Hours**: Historical data downloads are faster during market hours (Sun 6 PM - Fri 5 PM ET)

⚠️ **First Load**: The first time you request a symbol, it downloads 1 year of 1-minute bars from IB Gateway. This takes 30-45 minutes. Subsequent loads are <2 seconds from cache.

⚠️ **Market Data Subscriptions**: You MUST have CME Real-Time market data subscriptions and complete the API acknowledgement form.

⚠️ **Connection Limits**: IB Gateway allows limited API connections. If you restart frequently, you may hit rate limits (wait 15 minutes).

✅ **Cache is Good**: Once data is cached, you don't need to re-download unless cache expires (24 hours) or you delete it.

✅ **Real-time Works**: Real-time updates happen ~every 5 seconds during market hours using IB's keepUpToDate mode.

---

**Ready to start? Follow the steps above and you'll be up and running!**

If everything is working, you should see the TradingView-style chart interface with real-time futures data. 🚀
