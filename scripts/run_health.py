#!/usr/bin/env python3
"""Run the health endpoint on 127.0.0.1:8765."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from src.service.health import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
