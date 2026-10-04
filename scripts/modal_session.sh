#!/bin/bash
# One collection session on Modal: deploy, warm up, run the given domains, ALWAYS stop the app.
# Usage: setsid nohup scripts/modal_session.sh retail [airline ...] > /dev/null 2>&1 < /dev/null &
# The trap also fires on a graceful OS shutdown (SIGTERM), so the GPU never outlives the session.
set -u
cd "$(dirname "$0")/.."
LOG=data/logs/modal_session.log
mkdir -p data/logs
stop_app() { echo "$(date -u +%T) stopping app" >> $LOG; .venv/bin/modal app stop atl-serving --yes >> $LOG 2>&1; echo "$(date -u +%T) SESSION DONE" >> $LOG; }
trap stop_app EXIT
echo "$(date -u +%T) session start: $*" >> $LOG
.venv/bin/modal deploy src/atl/serving/modal_app.py >> $LOG 2>&1 || exit 1
.venv/bin/python - >> $LOG 2>&1 <<'PY' || exit 1
import time, urllib.request
from dotenv import dotenv_values
url = dotenv_values(".env")["ATL_SERVING_URL"]; t = time.time()
urllib.request.urlopen(f"{url}/health", timeout=1500); print(f"WARM after {time.time()-t:.0f}s", flush=True)
PY
for cfg in "$@"; do
  echo "$(date -u +%T) run $cfg" >> $LOG
  .venv/bin/atl-agent run --config configs/agent/modal_collect_$cfg.yaml >> data/logs/modal_collect_$cfg.log 2>&1
  echo "$(date -u +%T) $cfg exit=$?" >> $LOG
done
