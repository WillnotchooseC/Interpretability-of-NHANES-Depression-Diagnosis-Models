#!/bin/bash
# Keep app.py + SGLang + cloudflared alive. Restart the whole stack if any
# component is missing or unhealthy. URL will change on every restart — see
# current_url_history.log for the running history.

set -u
cd "$(dirname "$0")"

LOG=watchdog.log
APP_LOG=app_run.log
URL_HISTORY=current_url_history.log
CHECK_INTERVAL=30
STARTUP_TIMEOUT_S=900
GRACE_AFTER_START_S=10

ts() { date '+%F %T'; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

check_all() {
    pgrep -f "python app.py" > /dev/null || return 1
    pgrep -f "sglang.launch_server" > /dev/null || return 1
    pgrep -f "./cloudflared tunnel" > /dev/null || return 1
    curl -sf --max-time 3 http://localhost:30000/health > /dev/null 2>&1 || return 2
    curl -sf --max-time 3 http://localhost:7860 > /dev/null 2>&1 || return 3
    return 0
}

start_stack() {
    log "==> Restarting stack"
    pkill -9 -f "python app.py" 2>/dev/null
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    pkill -9 -f "sglang::" 2>/dev/null   # orphaned scheduler/detokenizer children hold GPU memory
    pkill -9 -f "./cloudflared tunnel" 2>/dev/null
    sleep 3
    nohup python app.py > "$APP_LOG" 2>&1 &
    disown
    local pid=$(pgrep -f "python app.py" | head -1)
    log "    app.py PID=$pid, waiting up to ${STARTUP_TIMEOUT_S}s for ready..."

    local waited=0
    while [ $waited -lt $STARTUP_TIMEOUT_S ]; do
        sleep 5
        waited=$((waited + 5))
        if check_all; then
            # Named Tunnel: URL is static, regardless of cloudflared restart
            local url="https://nhanes-depression-qa.com"
            log "==> READY in ${waited}s · URL: $url"
            echo "[$(ts)] $url" >> "$URL_HISTORY"
            return 0
        fi
    done
    log "==> TIMEOUT after ${STARTUP_TIMEOUT_S}s — will retry next cycle"
    return 1
}

log "Watchdog started (interval=${CHECK_INTERVAL}s)"
# Initial start
start_stack
sleep "$GRACE_AFTER_START_S"

while true; do
    sleep "$CHECK_INTERVAL"
    check_all
    rc=$?
    if [ "$rc" -ne 0 ]; then
        case "$rc" in
            1) log "process(es) missing" ;;
            2) log "SGLang health failing" ;;
            3) log "Gradio not reachable" ;;
        esac
        start_stack
        sleep "$GRACE_AFTER_START_S"
    fi
done
