# Helper script to run Lab Day 14 with the portable Python
# Usage: .\run.ps1              → run full benchmark
#        .\run.ps1 gen          → only generate dataset
#        .\run.ps1 check        → only validate artifacts
#        .\run.ps1 test         → run unit tests

$py  = "D:\AI_Thuc_chien\python-embed\python.exe"
$here = $PSScriptRoot

function Invoke-Python {
    param([string]$Code)
    $wrapped = "import sys`nsys.path.insert(0, r'$here')`n$Code"
    $wrapped | & $py 2>&1
}

$cmd = if ($args.Count -gt 0) { $args[0] } else { "all" }

switch ($cmd) {
    "gen" {
        Write-Host "Generating golden dataset..." -ForegroundColor Cyan
        Invoke-Python "exec(open('data/synthetic_gen.py').read())"
    }
    "check" {
        Write-Host "Validating submission artifacts..." -ForegroundColor Cyan
        Invoke-Python "import check_lab, sys; sys.exit(check_lab.validate_lab())"
    }
    "test" {
        Write-Host "Running unit tests..." -ForegroundColor Cyan
        & $py -m pytest tests/ -v --tb=short 2>&1
    }
    default {
        Write-Host "=== Step 1: Generate golden dataset ===" -ForegroundColor Cyan
        Invoke-Python "exec(open('data/synthetic_gen.py').read())"

        Write-Host "`n=== Step 2: Run benchmark ===" -ForegroundColor Cyan
        Invoke-Python "import main, asyncio; asyncio.run(main.main())"

        Write-Host "`n=== Step 3: Validate artifacts ===" -ForegroundColor Cyan
        Invoke-Python "import check_lab, sys; sys.exit(check_lab.validate_lab())"
    }
}
