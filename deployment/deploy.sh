#!/usr/bin/env bash
#
# Deployment Script for Futures Charting Application
#
# This script handles deployment of the application to various environments.
#
# Usage:
#   ./deploy.sh [dev|staging|production] [options]
#
# Options:
#   --skip-tests        Skip running tests before deployment
#   --no-backup         Don't create a backup before deployment
#   --restart-only      Only restart the service without redeploying
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-dev}"
SKIP_TESTS=false
NO_BACKUP=false
RESTART_ONLY=false

# Parse options
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        --restart-only)
            RESTART_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check pip
    if ! command -v pip &> /dev/null; then
        log_error "pip is not installed"
        exit 1
    fi

    # Check virtual environment
    if [ ! -d "$PROJECT_ROOT/venv" ] && [ ! -d "$PROJECT_ROOT/.venv" ]; then
        log_warning "No virtual environment found. Creating one..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi

    log_success "Requirements check passed"
}

activate_venv() {
    if [ -d "$PROJECT_ROOT/venv" ]; then
        source "$PROJECT_ROOT/venv/bin/activate"
    elif [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        log_error "Virtual environment not found"
        exit 1
    fi
}

install_dependencies() {
    log_info "Installing dependencies..."
    cd "$PROJECT_ROOT"

    activate_venv

    pip install --upgrade pip
    pip install -r requirements.txt

    log_success "Dependencies installed"
}

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warning "Skipping tests (--skip-tests flag set)"
        return
    fi

    log_info "Running tests..."
    cd "$PROJECT_ROOT"

    activate_venv

    python -m pytest tests/ -v

    if [ $? -ne 0 ]; then
        log_error "Tests failed. Aborting deployment."
        exit 1
    fi

    log_success "All tests passed"
}

create_backup() {
    if [ "$NO_BACKUP" = true ]; then
        log_warning "Skipping backup (--no-backup flag set)"
        return
    fi

    log_info "Creating backup..."

    BACKUP_DIR="$PROJECT_ROOT/backups"
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"

    mkdir -p "$BACKUP_DIR"

    # Backup data directory
    if [ -d "$PROJECT_ROOT/data" ]; then
        tar -czf "$BACKUP_DIR/${BACKUP_NAME}_data.tar.gz" -C "$PROJECT_ROOT" data/
        log_success "Data backup created: ${BACKUP_NAME}_data.tar.gz"
    fi

    # Backup logs directory
    if [ -d "$PROJECT_ROOT/logs" ]; then
        tar -czf "$BACKUP_DIR/${BACKUP_NAME}_logs.tar.gz" -C "$PROJECT_ROOT" logs/
        log_success "Logs backup created: ${BACKUP_NAME}_logs.tar.gz"
    fi

    # Keep only last 5 backups
    cd "$BACKUP_DIR"
    ls -t backup_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm
}

setup_environment() {
    log_info "Setting up environment for: $ENVIRONMENT"

    cd "$PROJECT_ROOT"

    # Create .env if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            log_info "Creating .env from .env.example"
            cp .env.example .env
            log_warning "Please configure .env file with your settings"
        else
            log_warning ".env file not found. You may need to create it manually."
        fi
    fi

    # Create necessary directories
    mkdir -p data/cache
    mkdir -p logs

    # Set permissions
    chmod 755 data logs
    chmod 755 data/cache

    log_success "Environment setup complete"
}

stop_service() {
    log_info "Stopping service..."

    # Try systemd first
    if systemctl is-active --quiet futures-charting 2>/dev/null; then
        sudo systemctl stop futures-charting
        log_success "Service stopped via systemd"
        return
    fi

    # Try finding and killing the process
    if pgrep -f "run.py" > /dev/null; then
        pkill -f "run.py"
        sleep 2
        log_success "Service stopped via process kill"
        return
    fi

    log_info "Service was not running"
}

start_service() {
    log_info "Starting service..."

    cd "$PROJECT_ROOT"

    # Try systemd first
    if systemctl list-unit-files | grep -q futures-charting; then
        sudo systemctl start futures-charting
        sleep 3

        if systemctl is-active --quiet futures-charting; then
            log_success "Service started via systemd"
            sudo systemctl status futures-charting --no-pager
            return
        fi
    fi

    # Start manually in the background
    log_info "Starting service manually..."
    activate_venv

    nohup python run.py > logs/app.log 2>&1 &
    APP_PID=$!

    echo $APP_PID > /tmp/futures-charting.pid

    sleep 3

    if ps -p $APP_PID > /dev/null; then
        log_success "Service started with PID: $APP_PID"
        log_info "Logs: tail -f $PROJECT_ROOT/logs/app.log"
    else
        log_error "Failed to start service"
        exit 1
    fi
}

restart_service() {
    stop_service
    start_service
}

check_health() {
    log_info "Checking application health..."

    # Wait for application to start
    sleep 5

    # Check if the application is responding
    HEALTH_URL="http://localhost:8000/health"

    if command -v curl &> /dev/null; then
        RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

        if [ "$RESPONSE" = "200" ] || [ "$RESPONSE" = "503" ]; then
            log_success "Application is responding (HTTP $RESPONSE)"

            # Get detailed health info
            curl -s $HEALTH_URL | python3 -m json.tool || true
        else
            log_warning "Application health check returned: HTTP $RESPONSE"
        fi
    else
        log_warning "curl not found. Skipping health check."
    fi
}

deploy() {
    echo ""
    echo "============================================"
    echo " Deploying Futures Charting Application"
    echo " Environment: $ENVIRONMENT"
    echo "============================================"
    echo ""

    if [ "$RESTART_ONLY" = true ]; then
        log_info "Restart-only mode"
        restart_service
        check_health
        log_success "Service restarted successfully"
        return
    fi

    # Full deployment
    check_requirements
    create_backup
    stop_service
    install_dependencies
    setup_environment
    run_tests
    start_service
    check_health

    echo ""
    log_success "Deployment completed successfully!"
    echo ""
    echo "Application is running at: http://localhost:8000"
    echo ""
}

# Main execution
deploy
