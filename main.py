"""
Thin entry shim. All logic lives in src/main.py.

Preferred invocation:
    python -m src.main publish --cadence daily --type trending
    python -m src.scheduler               # long-lived daemon

Legacy shim (this file) kept so older cron/PM2 configs still work:
    python main.py daily
    python main.py weekly
    python main.py maintenance 20
"""
import sys
from src.main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
