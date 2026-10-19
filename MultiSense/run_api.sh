#!/bin/bash
# MultiSense API Launch Script
echo "================================================================="
echo "  LAUNCHING MULTISENSE ASD STIMMING DETECTION API SERVER"
echo "  - Interactive Swagger Docs: http://127.0.0.1:8000/docs"
echo "  - Interactive Portal:       http://127.0.0.1:8000/dashboard"
echo "================================================================="

cd "$(dirname "$0")/.."

if [ -f "./venv/bin/python3" ]; then
    PYTHON_EXEC="./venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

$PYTHON_EXEC -m uvicorn MultiSense.api.main:app --host 0.0.0.0 --port 8000 --reload
