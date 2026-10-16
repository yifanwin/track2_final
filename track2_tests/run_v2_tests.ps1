param([string]$Python="python",[string]$Output="replay_v2_windows")
$ErrorActionPreference="Stop"
Set-Location -LiteralPath $PSScriptRoot
& $Python -B check_install.py
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $Python -B tests.py --output "$Output/v1_regression"
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
& $Python -B v2_tests.py --output "$Output/functional"
exit $LASTEXITCODE
