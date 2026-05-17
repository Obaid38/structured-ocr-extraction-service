#!/usr/bin/env bash
set -euo pipefail

# One-step launcher for LMDeploy (23333) -> Pixtral (3203) -> API (8080)
# Usage: bash run_all.sh [--tp N] [--venv /path/to/venv]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TP=${1:-}
if [[ "$TP" == "--tp" ]]; then
  TP_VAL=${2:-1}
  shift 2
else
  TP_VAL=${TP:-1}
fi

VENV_PATH=${1:-}
if [[ "$VENV_PATH" == "--venv" ]]; then
  VENV=${2:-"$SCRIPT_DIR/venv"}
  shift 2
else
  VENV="$SCRIPT_DIR/venv"
fi

ROOT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
cd "$ROOT_DIR"

if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
fi
# Check if core dependencies are installed, only install if missing
if ! python -c "import torch, fastapi, lmdeploy" 2>/dev/null; then
  echo "Dependencies missing, installing..."
  pip install -r requirements.txt >/dev/null
else
  echo "Dependencies already installed, skipping..."
fi

echo "Starting LMDeploy (tp=$TP_VAL) on :23333..."
# Adjust tp by editing lmdeploy_app.py dynamically
sed -i -E "s/tp=\s*[0-9]+/tp=${TP_VAL}/" lmdeploy_app.py || true

python lmdeploy_app.py > logs_lmdeploy.txt 2>&1 &
LM_PID=$!

echo -n "Waiting for LMDeploy to be ready"
for i in {1..120}; do
  if curl -s http://localhost:23333/v1/models >/dev/null; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 2
done

if ! curl -s http://localhost:23333/v1/models >/dev/null; then
  echo -e "\nERROR: LMDeploy not responding on :23333. Check logs_lmdeploy.txt"
  exit 1
fi

echo "Starting Pixtral on :3203..."
python pixtral.py > logs_pixtral.txt 2>&1 &
PX_PID=$!

echo -n "Waiting for Pixtral to be ready"
for i in {1..60}; do
  if curl -s http://localhost:3203/docs >/dev/null; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -s http://localhost:3203/docs >/dev/null; then
  echo -e "\nERROR: Pixtral not responding on :3203. Check logs_pixtral.txt"
  kill $LM_PID || true
  exit 1
fi

echo "Starting API on :8080..."
python app.py > logs_api.txt 2>&1 &
API_PID=$!

echo -n "Waiting for API to be ready"
for i in {1..60}; do
  if curl -s http://localhost:8080/docs >/dev/null; then
    echo " - OK"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -s http://localhost:8080/docs >/dev/null; then
  echo -e "\nERROR: API not responding on :8080. Check logs_api.txt"
  kill $PX_PID $LM_PID || true
  exit 1
fi

echo "All services up. PIDs: LMDeploy=$LM_PID Pixtral=$PX_PID API=$API_PID"
echo "Logs: logs_lmdeploy.txt, logs_pixtral.txt, logs_api.txt"
wait
