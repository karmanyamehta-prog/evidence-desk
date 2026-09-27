$ErrorActionPreference = "Stop"
# Run from the repository root. No admin rights or execution-policy change needed
# when using the equivalent commands in README.md.
py -3.12 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Install Python 3.12 x64 first." }
& .\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed." }
& .\.venv\Scripts\python.exe -m pip install -e .
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
if (!(Test-Path .env)) { Copy-Item .env.example .env }
& .\.venv\Scripts\python.exe -m grounded_qa.cli doctor
