# Rebuild the repo venv from the pinned lockfile and run a smoke test.
# Uses py -3.14 (E:\Installs\Python314). Does not request admin / UAC.
# Usage (from E:\Football\football):  .\bootstrap.ps1

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$Lock = Join-Path $PSScriptRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $Lock)) {
    throw "Missing lockfile: $Lock"
}

$VenvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$NeedVenv = $true
if (Test-Path -LiteralPath $VenvPython) {
    try {
        $Ver = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($LASTEXITCODE -eq 0 -and $Ver.Trim() -eq '3.14') {
            $NeedVenv = $false
        } else {
            Write-Host "Existing .venv is Python $($Ver.Trim()), rebuilding with 3.14"
        }
    } catch {
        Write-Host 'Existing .venv is broken, rebuilding with 3.14'
    }
}

if ($NeedVenv) {
    $VenvDir = Join-Path $PSScriptRoot '.venv'
    if (Test-Path -LiteralPath $VenvDir) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
    Write-Host 'Creating .venv with py -3.14'
    py -3.14 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw 'py -3.14 -m venv .venv failed'
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "venv created but $VenvPython is missing"
    }
}

Write-Host "Installing $Lock"
& $VenvPython -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) {
    throw 'pip install -r requirements.lock.txt failed'
}

Write-Host 'Smoke test: python -m unittest discover -s tests'
& $VenvPython -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) {
    throw 'unit tests failed'
}

Write-Host 'bootstrap ok'