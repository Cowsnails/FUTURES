# Quick Start Reference

**Ultra-condensed startup instructions for experienced users.**

---

## Prerequisites

- Python 3.10+ installed
- IB Gateway running on port 7496 (paper) or 7497 (live)
- Market data subscriptions active

---

## 5-Minute Startup

```bash
# 1. Clone/navigate to project
cd /home/user/FUTURES

# 2. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env: Set IB_PORT=7496 (paper) or 7497 (live)

# 5. Test IB Gateway connection
python test_connection.py

# 6. Start application
python run.py

# 7. Open browser
# http://localhost:8000
```

---

## Essential Commands

### Start Application
```bash
# Development
python run.py

# Production (systemd)
sudo systemctl start futures-charting
```

### Check Status
```bash
# Health check
curl http://localhost:8000/health

# Metrics
curl http://localhost:8000/metrics

# Logs
tail -f logs/app.log
```

### Run Tests
```bash
# Unit tests
python -m pytest tests/test_*.py -v

# Load test
python tests/load_test.py --clients 10 --duration 60

# Memory profile
python tests/memory_profiler.py --profile all
```

---

## Configuration Quick Reference

### .env File (Minimal)
```bash
IB_HOST=127.0.0.1
IB_PORT=7496              # Paper: 7496, Live: 7497
IB_CLIENT_ID=1
SERVER_PORT=8000
LOG_LEVEL=info
```

### IB Gateway Settings
- API → Settings → Enable ActiveX and Socket Clients: ✅
- Socket Port: 7496 (paper) or 7497 (live)
- Read-Only API: ❌ (unchecked)

---

## Common Issues

| Issue | Fix |
|-------|-----|
| Can't connect | Check IB Gateway running: `ps aux \| grep ibgateway` |
| Port in use | Change `SERVER_PORT` in .env or kill process: `lsof -i :8000` |
| No module | Activate venv: `source venv/bin/activate` |
| Error 354 | Complete Market Data API form, wait 24h |
| No data | First load takes 30-45 min, check cache: `ls data/cache/` |

---

## Endpoints

- **Web UI**: http://localhost:8000
- **Health**: http://localhost:8000/health
- **Ready**: http://localhost:8000/ready
- **Metrics**: http://localhost:8000/metrics
- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)

---

## Monitoring

```bash
# Real-time metrics
watch -n 1 'curl -s http://localhost:8000/metrics | grep -E "(memory|connections|gateway)"'

# Logs
tail -f logs/app.log

# System status
curl http://localhost:8000/api/statistics | python -m json.tool
```

---

## Stopping

```bash
# Ctrl+C (if running python run.py)

# OR
sudo systemctl stop futures-charting
```

---

## Key Files

- `README.md` - Full documentation
- `STARTUP_GUIDE.md` - Detailed startup instructions ⭐
- `DEPLOYMENT_GUIDE.md` - Production deployment
- `TESTING.md` - Testing procedures
- `config.yaml` - Main configuration
- `.env` - Environment variables
- `test_connection.py` - IB Gateway connection test

---

**Need more help?** → See `STARTUP_GUIDE.md` for detailed instructions.
