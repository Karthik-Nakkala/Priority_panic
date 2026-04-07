# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Priority Panic Environment.

This module creates an HTTP server that exposes the PriorityPanicEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys

# Standardize pathing so imports work whether running from root or server/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    ) from e

# Cleaned up imports to avoid redundancy
try:
    from models import PriorityPanicAction, PriorityPanicObservation
    from server.priority_panic_environment import PriorityPanicEnvironment
except ImportError:
    from ..models import PriorityPanicAction, PriorityPanicObservation
    from .priority_panic_environment import PriorityPanicEnvironment

# Create the app instance
app = create_app(
    PriorityPanicEnvironment,
    PriorityPanicAction,
    PriorityPanicObservation,
    env_name="priority_panic",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution. 
    Note: Port 7860 is required for Hugging Face Spaces compatibility.
    """
    import uvicorn
    # Use string reference to allow for better worker management
    uvicorn.run("server.app:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Defaulting to 7860 for HF/OpenEnv compatibility
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    main(host=args.host, port=args.port)