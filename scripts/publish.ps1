param(
    [switch]$SkipInfra
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)
Push-Location ..
try {
    if (-not $SkipInfra) {
        Write-Host "Publishing llmptr-infra..."
        Push-Location llmptr-infra
        cargo publish --registry crates-io
        Pop-Location
    } else {
        Write-Host "Skipping llmptr-infra publish."
    }

    Write-Host "Publishing llmptr..."
    cargo publish --registry crates-io
} finally {
    Pop-Location
    Pop-Location
}
