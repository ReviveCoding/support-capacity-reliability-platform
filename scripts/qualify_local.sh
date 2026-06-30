#!/usr/bin/env bash
set -euo pipefail
PROFILE="${1:-CORE}"
python scripts/qualify_local.py --profile "$PROFILE"
