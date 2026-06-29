param(
    [string]$SourceRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$OutputPath = "$PWD\parascale-remote.zip"
)

$ErrorActionPreference = "Stop"

$source = Resolve-Path $SourceRoot
$output = [System.IO.Path]::GetFullPath($OutputPath)
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("parascale-remote-" + [System.Guid]::NewGuid().ToString("N"))

try {
    New-Item -ItemType Directory -Path $tempRoot | Out-Null
    $staging = Join-Path $tempRoot "2-ParaScale-master"
    New-Item -ItemType Directory -Path $staging | Out-Null

    $excludeNames = @(
        ".git",
        ".pytest_cache",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "runs"
    )

    Get-ChildItem -LiteralPath $source -Force | ForEach-Object {
        if ($excludeNames -contains $_.Name) {
            return
        }
        if ($_.Name -like ".codex_*") {
            return
        }
        if ($_.Name -like "parascale-upload*.zip" -or $_.Name -like "parascale-remote*.zip" -or $_.Name -like "*.zip.b64") {
            return
        }
        if ($_.Name -like "pytest-cache-files-*") {
            return
        }
        Copy-Item -LiteralPath $_.FullName -Destination $staging -Recurse -Force
    }

    if (Test-Path -LiteralPath $output) {
        Remove-Item -LiteralPath $output -Force
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $output -Force
    Write-Output $output
}
finally {
    if (Test-Path -LiteralPath $tempRoot) {
        Remove-Item -LiteralPath $tempRoot -Recurse -Force
    }
}
