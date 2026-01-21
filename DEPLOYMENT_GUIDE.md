# Deployment Guide

Complete guide for deploying the Futures Charting application to production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [IB Gateway Configuration](#ib-gateway-configuration)
4. [Application Deployment](#application-deployment)
5. [Security Configuration](#security-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or RHEL 8+)
- **Python**: 3.10, 3.11, or 3.12
- **RAM**: Minimum 2GB, recommended 4GB+
- **Disk Space**: Minimum 10GB (for caching historical data)
- **Network**: Stable internet connection

### Required Software

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3-pip -y

# Install system dependencies
sudo apt install git nginx supervisor -y

# Optional: Install monitoring tools
sudo apt install prometheus node-exporter -y
```

### IB Gateway Requirements

- Interactive Brokers account (paper or live)
- TWS API installed
- Market data subscriptions for CME futures

---

## Environment Setup

### 1. Create Application User

```bash
# Create dedicated user
sudo useradd -r -s /bin/bash -d /opt/futures-charting futures

# Create application directory
sudo mkdir -p /opt/futures-charting
sudo chown futures:futures /opt/futures-charting
```

### 2. Clone Repository

```bash
# Switch to futures user
sudo su - futures

# Clone repository
cd /opt/futures-charting
git clone <repository-url> .

# Or upload deployment tarball
# tar -xzf futures-charting-<version>.tar.gz
```

### 3. Create Virtual Environment

```bash
# Create venv
python3.11 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy example .env
cp .env.example .env

# Edit .env with your settings
nano .env
```

**.env Configuration:**

```bash
# IB Gateway Connection
IB_HOST=127.0.0.1
IB_PORT=7497  # 7497 for live, 7496 for paper
IB_CLIENT_ID=1
IB_TIMEOUT=30

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info

# Security
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
ALLOWED_ORIGINS=https://yourdomain.com

# Cache
CACHE_MAX_AGE_HOURS=24
CACHE_DIR=/opt/futures-charting/data/cache

# Optional: Secrets
SECRET_KEY=<generate-random-secret-key>
```

**Generate Secret Key:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 5. Create Required Directories

```bash
mkdir -p data/cache logs backups
chmod 755 data logs backups
chmod 700 data/cache  # Restrict cache access
```

---

## IB Gateway Configuration

### 1. Install IB Gateway

```bash
# Download IB Gateway (headless version)
wget https://download2.interactivebrokers.com/installers/ibgateway/latest-standalone/ibgateway-latest-standalone-linux-x64.sh

# Install
chmod +x ibgateway-latest-standalone-linux-x64.sh
./ibgateway-latest-standalone-linux-x64.sh -q

# Or use Docker
docker pull ghcr.io/unusualcode/ibgateway-docker:latest
```

### 2. Configure API Settings

Edit `~/Jts/jts.ini`:

```ini
[IBGateway]
ApiPort=7497
ReadOnlyApi=no
AcceptIncomingConnectionAction=accept
AllowBlindConcurrentOrders=yes
```

### 3. Set Up Auto-Login (Optional)

Create `~/ibc/config.ini`:

```ini
IbLoginId=your_username
IbPassword=your_password
TradingMode=paper  # or 'live'
AcceptIncomingConnectionAction=accept
```

### 4. Start IB Gateway

```bash
# Manual start
~/Jts/ibgateway &

# Or with IBC (auto-login)
~/ibc/scripts/IBCStart.sh &

# Or with Docker
docker run -d --name ibgateway \
  -p 7497:7497 \
  -e TWS_USERID=your_username \
  -e TWS_PASSWORD=your_password \
  -e TRADING_MODE=paper \
  ghcr.io/unusualcode/ibgateway-docker:latest
```

### 5. Verify Connection

```bash
# Test connection
python test_connection.py

# Expected output:
# ✓ Connected to IB Gateway
# ✓ MNQ contract qualified
# ✓ MES contract qualified
# ✓ MGC contract qualified
```

---

## Application Deployment

### Option 1: Manual Deployment

```bash
# Switch to futures user
sudo su - futures
cd /opt/futures-charting

# Activate venv
source venv/bin/activate

# Run deployment script
./deployment/deploy.sh production

# The script will:
# - Create backup
# - Install dependencies
# - Run tests
# - Start the application
```

### Option 2: Systemd Service

```bash
# Install service file
sudo cp deployment/futures-charting.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/futures-charting.service

# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable futures-charting

# Start service
sudo systemctl start futures-charting

# Check status
sudo systemctl status futures-charting

# View logs
sudo journalctl -u futures-charting -f
```

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 futures && \\
    chown -R futures:futures /app

USER futures

EXPOSE 8000

CMD ["python", "run.py"]
```

Build and run:

```bash
# Build image
docker build -t futures-charting:latest .

# Run container
docker run -d \\
  --name futures-charting \\
  -p 8000:8000 \\
  -v $(pwd)/data:/app/data \\
  -v $(pwd)/logs:/app/logs \\
  --env-file .env \\
  futures-charting:latest

# Check logs
docker logs -f futures-charting
```

---

## Security Configuration

### 1. Configure Firewall

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS (if using Nginx)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct access to app port
sudo ufw deny 8000/tcp

# Enable firewall
sudo ufw enable
```

### 2. Set Up Nginx Reverse Proxy

Create `/etc/nginx/sites-available/futures-charting`:

```nginx
server {
    listen 80;
    server_name futures-charting.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name futures-charting.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;

    # Proxy settings
    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for WebSocket
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    location /api/ {
        limit_req zone=api burst=20;
        proxy_pass http://localhost:8000;
    }
}
```

Enable and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/futures-charting /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 3. Obtain SSL Certificate

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d futures-charting.yourdomain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### 4. Configure Application Security

See `deployment/security.conf` for comprehensive security guidelines.

---

## Monitoring Setup

### 1. Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Readiness check (for Kubernetes)
curl http://localhost:8000/ready

# Rate limit info
curl http://localhost:8000/api/rate-limit-info
```

### 2. Prometheus Metrics

```bash
# View metrics
curl http://localhost:8000/metrics

# Configure Prometheus scraping
# Add to /etc/prometheus/prometheus.yml:
# scrape_configs:
#   - job_name: 'futures-charting'
#     static_configs:
#       - targets: ['localhost:8000']
```

### 3. Set Up Monitoring Alerts

Create alert rules in Prometheus:

```yaml
groups:
  - name: futures_charting
    rules:
      - alert: IBGatewayDown
        expr: ib_gateway_connected == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "IB Gateway connection lost"

      - alert: HighMemoryUsage
        expr: process_memory_bytes{type="rss"} > 2000000000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Memory usage above 2GB"

      - alert: NoActiveConnections
        expr: websocket_connections_total == 0
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "No active WebSocket connections for 30 minutes"
```

### 4. Log Management

Configure log rotation (`/etc/logrotate.d/futures-charting`):

```
/opt/futures-charting/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 futures futures
}
```

---

## Troubleshooting

### Application Won't Start

```bash
# Check logs
sudo journalctl -u futures-charting -n 100

# Check if port is in use
sudo netstat -tulpn | grep 8000

# Test configuration
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Verify environment
source venv/bin/activate
python -c "import backend.app; print('OK')"
```

### IB Gateway Connection Issues

```bash
# Check if IB Gateway is running
ps aux | grep ibgateway

# Check firewall
sudo ufw status

# Test connection
telnet localhost 7497

# Check IB Gateway logs
tail -f ~/Jts/log/ibgateway.*.log
```

### Performance Issues

```bash
# Check memory usage
python tests/memory_profiler.py --profile all

# Run load test
python tests/load_test.py --clients 10 --duration 60

# Monitor real-time metrics
watch -n 1 'curl -s http://localhost:8000/metrics | grep process_'
```

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor application logs
- Check IB Gateway connection status
- Verify cache size

**Weekly:**
- Review performance metrics
- Check for dependency updates
- Test backups

**Monthly:**
- Update dependencies
- Review security logs
- Performance optimization review

### Backup Strategy

```bash
# Manual backup
./deployment/deploy.sh production --no-backup
cd /opt/futures-charting
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/

# Automated backup (add to crontab)
0 2 * * * cd /opt/futures-charting && tar -czf backups/backup-$(date +\%Y\%m\%d).tar.gz data/ logs/ && find backups/ -mtime +30 -delete
```

### Updating the Application

```bash
# Pull latest changes
cd /opt/futures-charting
git pull origin main

# Run deployment script
./deployment/deploy.sh production

# Or restart service
sudo systemctl restart futures-charting
```

---

## Production Checklist

Before going live, ensure:

- [ ] IB Gateway configured and tested
- [ ] SSL certificates installed and valid
- [ ] Firewall rules configured
- [ ] Environment variables set correctly
- [ ] All tests passing
- [ ] Monitoring and alerting configured
- [ ] Backup strategy implemented
- [ ] Security hardening applied
- [ ] Rate limiting configured
- [ ] Log rotation set up
- [ ] Health checks accessible
- [ ] Documentation reviewed

---

## Support

For issues and questions:

1. Check logs: `sudo journalctl -u futures-charting -n 100`
2. Review documentation: `README.md`, `TESTING.md`
3. Run diagnostics: `python test_connection.py`
4. Check health: `curl http://localhost:8000/health`

---

**Last Updated**: 2026-01-21
**Version**: 0.1.0
