#!/bin/bash
# One-command launcher: starts the watchdog which keeps the whole stack alive.
# Usage:
#   ./start.sh         # start everything
#   ./start.sh stop    # stop everything

set -u
cd "$(dirname "$0")"

cmd="${1:-start}"

kill_all() {
    pkill -f "watchdog.sh" 2>/dev/null
    pkill -9 -f "python app.py" 2>/dev/null
    pkill -9 -f "sglang.launch_server" 2>/dev/null
    pkill -9 -f "./cloudflared tunnel" 2>/dev/null
    sleep 2
    echo "Stopped."
}

case "$cmd" in
    stop)
        kill_all
        exit 0
        ;;
    restart)
        kill_all
        ;;
    start)
        # Make sure nothing is running already
        if pgrep -f "watchdog.sh" > /dev/null; then
            echo "Watchdog already running. Use './start.sh restart' to reset."
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 [start|stop|restart]"
        exit 1
        ;;
esac

echo "Launching watchdog..."
nohup ./watchdog.sh > /dev/null 2>&1 &
disown

echo "Watchdog PID: $(pgrep -f watchdog.sh | head -1)"
echo "Waiting for stack to be ready (SGLang load takes ~50s)..."

for i in $(seq 1 120); do
    sleep 3
    if curl -sf --max-time 2 http://localhost:7860 > /dev/null 2>&1; then
        # Named Tunnel: URL is static
        url="https://nhanes-depression-qa.com"
        echo ""
        echo "========================================================================"
        printf "  PUBLIC URL: %s\n" "$url"
        echo "========================================================================"
        echo ""
        echo "Watchdog will restart the stack if anything crashes."
        echo "URL is a Named Tunnel — it stays the same across all restarts."
        echo "  - History of URLs:   current_url_history.log"
        echo "  - Watchdog log:      watchdog.log"
        echo "  - App log:           app_run.log"
        echo "  - Cloudflared log:   cloudflared.log"
        echo ""
        echo "Commands:"
        echo "  ./start.sh stop      # stop everything"
        echo "  ./start.sh restart   # full reset"
        echo "  tail -f watchdog.log # watch for restarts"
        exit 0
    fi
done

echo ""
echo "TIMEOUT — stack didn't come up. Check app_run.log and watchdog.log"
exit 1
