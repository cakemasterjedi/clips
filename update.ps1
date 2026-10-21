<#
Update clips on Windows: pull the latest code from GitHub and refresh the install.
  .\update.ps1        (or double-click update.bat; the web UI's "Update clips" runs this too)
#>
# Not "Stop": when the web UI runs this with its output piped, Windows PowerShell 5.1 turns
# anything git or pip print on stderr into errors. Exit codes are checked instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
# PowerShell reads the whole script before running it, so `git pull` changing this file is fine.
Set-Location $PSScriptRoot

function Say($msg) { Write-Host "`n==> $msg" }
function Fail($msg) { Write-Host "error: $msg"; exit 1 }
function Run {
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe $($rest -join ' ') failed (exit $LASTEXITCODE)" }
}
function Has-Pkg($py, $name) {
    & $py -c "import importlib.metadata as m, sys; sys.exit(sys.argv[1] not in {(d.metadata['Name'] or '').lower() for d in m.distributions()})" $name
    return $LASTEXITCODE -eq 0
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { Fail "no .venv here; run install.bat first" }
if (-not (Get-Command git -ErrorAction SilentlyContinue) -or -not (Test-Path .git)) {
    Fail "updating needs git and a git checkout; run install.bat again from a fresh download"
}

Say "Pulling the latest version"
$old = (git rev-parse HEAD)
git pull --ff-only --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "update stopped: you have local changes or a different branch."
    Write-Host "See them with 'git status'; 'git stash' sets them aside, then run update.bat again."
    exit 1
}
$new = (git rev-parse HEAD)
if ($old -eq $new) {
    Write-Host "already up to date ($(git log -1 --format='%h %s'))"
} else {
    Write-Host "updated $(git rev-parse --short $old) -> $(git rev-parse --short $new):"
    git log --format='  - %s' --no-merges "$old..$new"
}

Say "Refreshing requirements"
Run $venvPy -m pip install -q -r requirements.txt
# Sites like TikTok change often; keep the downloader (and its browser impersonation) current.
Run $venvPy -m pip install -q -U "yt-dlp[default,curl-cffi]"
Run $venvPy -m pip install -q --no-deps -e .
# Vision extras only when install.ps1 -Vision put them in (DirectML, or the CPU build).
if (Has-Pkg $venvPy "onnxruntime-directml") {
    Run $venvPy -m pip install -q "onnxruntime-directml>=1.17"
} elseif (Has-Pkg $venvPy "onnxruntime") {
    Run $venvPy -m pip install -q -r requirements-vision.txt
}

& (Join-Path $PSScriptRoot ".venv\Scripts\clips.exe") --help | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "clips --help failed after the update" }
Say "Done"
