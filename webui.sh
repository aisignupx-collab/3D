#!/usr/bin/env sh
# Opens the asset pipeline web UI in your browser. Uses .venv if it exists.
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then exec .venv/bin/python webui.py "$@"; fi
exec python3 webui.py "$@"
