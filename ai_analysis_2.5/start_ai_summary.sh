#!/usr/bin/env bash
# ──────────────────────────────────────────────
# start_ai_summary.sh — run the AI summary pipeline in the background.
#
# Starts ai_analysis_summary_v3.1.1.13_rag.py with --ai_summary only,
# as a nohup background process with timestamped log output and a pid file.
#
# Usage:
#   ./start_ai_summary.sh start     # start the pipeline (default)
#   ./start_ai_summary.sh stop      # graceful stop (SIGTERM)
#   ./start_ai_summary.sh status    # show whether it is running
#   ./start_ai_summary.sh restart   # stop then start
#
# Logs:   logs/prod_ai_summary_<timestamp>.log   (one file per start)
# Pid:    logs/prod_ai_summary.pid
# ──────────────────────────────────────────────
set -u

SCRIPT="ai_analysis_summary_v3.1.1.13_rag.py"
PID_FILE="logs/prod_ai_summary.pid"
LOG_DIR="logs"
# LLM context window for the configured model (gemma2:9b = 8192).
# Override: LLM_CONTEXT_TOKENS=32768 ./start_ai_summary.sh start
LLM_CONTEXT_TOKENS="${LLM_CONTEXT_TOKENS:-8192}"
# Extra poll interval arg for the pipeline (matches script default).
POLL_INTERVAL="${AI_SUMMARY_POLL_INTERVAL:-30}"

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

is_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

start() {
    if is_running; then
        echo "[START] Already running (pid $(cat "$PID_FILE")). Nothing to do."
        exit 0
    fi
    local log_file="$LOG_DIR/prod_ai_summary_$(date +%Y%m%d_%H%M%S).log"
    echo "[START] Launching $SCRIPT --ai_summary (llm_context_tokens=$LLM_CONTEXT_TOKENS)"
    nohup python3 -u "$SCRIPT" \
        --ai_summary \
        --llm_context_tokens "$LLM_CONTEXT_TOKENS" \
        > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "[START] Running: pid=$pid  log=$log_file"
        echo "[START] Stop with: $0 stop"
    else
        echo "[START] FAILED — process exited immediately. Check log:"
        tail -20 "$log_file"
        rm -f "$PID_FILE"
        exit 1
    fi
}

stop() {
    if ! is_running; then
        echo "[STOP] Not running (no live pid file)."
        rm -f "$PID_FILE"
        exit 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    echo "[STOP] Sending SIGTERM to pid $pid ..."
    kill "$pid"
    for _ in $(seq 1 15); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "[STOP] Still alive after 15s — sending SIGKILL."
        kill -9 "$pid"
    fi
    rm -f "$PID_FILE"
    echo "[STOP] Stopped."
}

status() {
    if is_running; then
        local pid
        pid="$(cat "$PID_FILE")"
        echo "[STATUS] RUNNING (pid $pid)"
        # Newest log line, so a quick status also shows health.
        local latest_log
        latest_log="$(ls -1t "$LOG_DIR"/prod_ai_summary_*.log 2>/dev/null | head -1)"
        [ -n "$latest_log" ] && echo "[STATUS] Log: $latest_log" && tail -3 "$latest_log"
    else
        echo "[STATUS] NOT running."
        rm -f "$PID_FILE"
        exit 1
    fi
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    restart) stop; sleep 2; start ;;
    *)       echo "Usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
