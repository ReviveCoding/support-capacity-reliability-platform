param(
    [ValidateSet("CORE", "STANDARD", "EXTENDED")]
    [string]$Profile = "CORE"
)
$ErrorActionPreference = "Stop"
python scripts/qualify_local.py --profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
