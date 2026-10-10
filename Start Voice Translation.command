#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print 'The project environment is missing. Follow the setup instructions in README.md.'
  read -r '?Press Enter to close.'
  exit 1
fi
if .venv/bin/python - <<'PY'
import json,sys,urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:7860/config', timeout=2) as response:
        config=json.load(response)
    sys.exit(0 if config.get('title') == 'Real-Time Voice Translation' else 1)
except Exception:
    sys.exit(1)
PY
then
  open 'http://127.0.0.1:7860'
  exit 0
fi
export VOICE_OPEN_BROWSER=1
export VOICE_TORCH_THREADS="${VOICE_TORCH_THREADS:-2}"
print 'Starting Voice Translation. Keep this Terminal window open while using the app.'
exec .venv/bin/python app.py
