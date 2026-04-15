param(
    [switch]$Evaluate,
    [switch]$Resume,
    [string]$Device = "cpu",
    [string]$PythonPath = ""
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $PythonPath) {
    $venvPython = Join-Path $root "venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        $Python = $venvPython
    } else {
        $Python = "python"
    }
} else {
    $Python = $PythonPath
}

$datasets = @("cora","citeseer","pubmed","chameleon","squirrel","actor")

Write-Host "Using Python: $Python"
Write-Host "Device: $Device"
if ($Evaluate) { Write-Host "Evaluation mode: ON" } else { Write-Host "Evaluation mode: OFF" }
if ($Resume) { Write-Host "Resume mode: ON" } else { Write-Host "Resume mode: OFF" }

foreach ($ds in $datasets) {
    Write-Host "=== Running dataset: $ds ===" -ForegroundColor Cyan
    $runArgs = @("--dataset", $ds, "--device", $Device)
    if ($Evaluate) { $runArgs += "--evaluate" }
    if ($Resume) { $runArgs += "--resume" }

    & $Python main.py @runArgs
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        Write-Host "Run failed for dataset $ds (exit code $exit). Continuing..." -ForegroundColor Yellow
    } else {
        Write-Host "Finished dataset $ds" -ForegroundColor Green
    }

    Start-Sleep -Seconds 1
}

Write-Host "All datasets processed." 
