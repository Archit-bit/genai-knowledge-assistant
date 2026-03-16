#!/usr/bin/env bash
set -euo pipefail

PORT_TO_USE="${PORT:-8000}"

exec python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${PORT_TO_USE}" \
  --browser.gatherUsageStats false
