#!/usr/bin/env bash
set -euo pipefail
# Run from the omniui/ project root.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
