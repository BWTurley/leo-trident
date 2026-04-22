#!/usr/bin/env python3
"""Run the health + API endpoints on 127.0.0.1:8765."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from fastapi import FastAPI

from src.service.api import app as api_app
from src.service.health import app as health_app

app = FastAPI(title="Leo Trident", docs_url=None, redoc_url=None)

for route in health_app.routes:
    app.router.routes.append(route)
for route in api_app.routes:
    app.router.routes.append(route)

for handler_type, handler in api_app.exception_handlers.items():
    app.add_exception_handler(handler_type, handler)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
