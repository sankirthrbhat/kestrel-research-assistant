#!/usr/bin/env bash
set -e

echo "Running ingestion (if needed)..."
python -m kestrel.ingest

echo "Starting the assistant..."
python -m kestrel.cli
