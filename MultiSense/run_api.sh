#!/bin/bash
# MultiSense API Launch Script
echo "================================================================="
echo "  LAUNCHING MULTISENSE ASD STIMMING DETECTION API SERVER"
echo "  - Interactive Swagger Docs: http://127.0.0.1:8000/docs"
echo "  - Interactive Portal:       http://127.0.0.1:8000/dashboard"
echo "================================================================="

cd "$(dirname "$0")/.."
python3 -m uvicorn MultiSense.api.main:app --host 127.0.0.1 --port 8000 --reload
