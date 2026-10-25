#!/usr/bin/env bash
# Sanity check script — creates venv, installs requirements, and starts the backend server.
# Run from repository root.

set -e

PY=python3
VENV_DIR=venv

echo "Creating virtual environment in ./${VENV_DIR}..."
$PY -m venv ${VENV_DIR}

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source ${VENV_DIR}/bin/activate

echo "Installing requirements (this requires network)..."
pip install --upgrade pip
pip install -r requirements.txt

echo "If MediaPipe protobuf issues occur, run:"
echo "  pip install protobuf==3.20.3 --force-reinstall"

echo "Starting server (backend)..."
cd backend
uvicorn main:app --reload --port 8000
