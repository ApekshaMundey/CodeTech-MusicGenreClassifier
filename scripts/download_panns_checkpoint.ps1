$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$checkpointDirectory = Join-Path $projectRoot 'artifacts\pretrained\panns'
$checkpointPath = Join-Path $checkpointDirectory 'Cnn14_16k_mAP_0.438.pth'
$downloadUrl = 'https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1'
$expectedMd5 = '362fc5ff18f1d6ad2f6d464b45893f2c'

New-Item -ItemType Directory -Path $checkpointDirectory -Force | Out-Null

$needsDownload = -not (Test-Path -LiteralPath $checkpointPath)
if (-not $needsDownload) {
    $existingMd5 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm MD5).Hash.ToLowerInvariant()
    $needsDownload = $existingMd5 -ne $expectedMd5
}

if ($needsDownload) {
    Write-Host 'Downloading the official PANNs CNN14 16 kHz checkpoint (about 343 MiB)...'
    curl.exe --location --fail --output $checkpointPath $downloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Checkpoint download failed with curl exit code $LASTEXITCODE."
    }
}

$actualMd5 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm MD5).Hash.ToLowerInvariant()
if ($actualMd5 -ne $expectedMd5) {
    throw "Checkpoint checksum mismatch. Expected $expectedMd5, received $actualMd5."
}

Write-Host "PANNs checkpoint is ready: $checkpointPath"
