#!/bin/bash
# One collection session on Modal: deploy, warm up, run the given domains, ALWAYS stop the app.
# Usage (Linux/WSL): setsid nohup scripts/modal_session.sh retail [airline | path/to/cfg.yaml ...] > /dev/null 2>&1 < /dev/null &
# Usage (Windows):   Start-Process "C:\Program Files\Git\usr\bin\bash.exe" -ArgumentList
#                    "scripts/modal_session.sh","airline","retail" -WindowStyle Hidden
# The trap also fires on a graceful OS shutdown (SIGTERM), so the GPU never outlives the session.
#
# Off-laptop backup: if ATL_HF_DATASET is set in .env (e.g. "user/atl-data", a private HF dataset)
# and HF_TOKEN is set, data/ (minus the LLM cache and smoke runs) is uploaded every 30 min and once
# more on exit. A laptop crash then loses at most 30 min of traces, never the whole dataset.
set -u
cd "$(dirname "$0")/.."
BIN=.venv/bin; [ -d .venv/Scripts ] && BIN=.venv/Scripts  # Linux vs Windows venv layout
LOG=data/logs/modal_session.log
mkdir -p data/logs
set -a; [ -f .env ] && . ./.env; set +a
export PYTHONUTF8=1

backup() {
  [ -n "${ATL_HF_DATASET:-}" ] || return 0
  echo "$(date -u +%T) backup -> $ATL_HF_DATASET" >> $LOG
  $BIN/hf upload "$ATL_HF_DATASET" data . --repo-type dataset --private \
    --exclude "llm_cache/*" --exclude "smoke/*" \
    --commit-message "session backup $(date -u +%FT%TZ)" >> $LOG 2>&1 \
    || echo "$(date -u +%T) BACKUP FAILED" >> $LOG
}
BACKUP_PID=
stop_app() {
  echo "$(date -u +%T) stopping app" >> $LOG
  $BIN/modal app stop atl-serving --yes >> $LOG 2>&1
  [ -n "$BACKUP_PID" ] && kill "$BACKUP_PID" 2>/dev/null
  backup
  echo "$(date -u +%T) SESSION DONE" >> $LOG
}
trap stop_app EXIT

echo "$(date -u +%T) session start: $*" >> $LOG
$BIN/modal deploy src/atl/serving/modal_app.py >> $LOG 2>&1 || exit 1
$BIN/python - >> $LOG 2>&1 <<'PY' || exit 1
import time, urllib.request
from dotenv import dotenv_values
url = dotenv_values(".env")["ATL_SERVING_URL"]; t = time.time()
urllib.request.urlopen(f"{url}/health", timeout=1500); print(f"WARM after {time.time()-t:.0f}s", flush=True)
PY
( while sleep 1800; do backup; done ) &
BACKUP_PID=$!
for cfg in "$@"; do
  # A domain name ("retail") means configs/agent/modal_collect_<domain>.yaml; a path is used as is.
  case $cfg in *.yaml) path=$cfg ;; *) path=configs/agent/modal_collect_$cfg.yaml ;; esac
  name=$(basename "$path" .yaml)
  echo "$(date -u +%T) run $path" >> $LOG
  $BIN/atl-agent run --config "$path" >> "data/logs/$name.log" 2>&1
  echo "$(date -u +%T) $name exit=$?" >> $LOG
done
