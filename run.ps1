$ErrorActionPreference = "Stop"

Write-Host "Running ingestion (if needed)..."
python -m kestrel.ingest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Starting the assistant..."
python -m kestrel.cli
exit $LASTEXITCODE
