<#
One-command setup for clips on Windows 10/11 (double-click install.bat, or run in PowerShell).
  .\install.ps1                 core: Python, ffmpeg, .venv, requirements, `clips` command, Start menu
  .\install.ps1 -Vision         + local subject detection (--subject, ONNX Runtime DirectML: any GPU)
  .\install.ps1 -Ollama         + Ollama (local AI for --subject-desc) and its vision model
  .\install.ps1 -Model NAME     Ollama model to pull (default auto: best fit for your GPU and RAM; "none" skips it)
  .\install.ps1 -Yes            no questions: only what the switches say
  .\install.ps1 -Dir PATH       where to download clips when run outside a checkout (default ~\clips)
Without -Vision/-Ollama/-Yes it asks about the optional parts. Uses winget; re-running is safe.
#>
param(
    [switch]$Vision,
    [switch]$Ollama,
    [string]$Model = "auto",
    [switch]$Yes,
    [string]$Dir = (Join-Path $HOME "clips"),
    [switch]$Help
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoUrl = "https://github.com/cakemasterjedi/clips"

if ($Help) {
    if ($PSCommandPath) { (Get-Content $PSCommandPath)[1..8] | ForEach-Object { $_ } }
    exit 0
}

function Say($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "warning: $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

# Run a program and stop on a non-zero exit code (PowerShell does not do that by itself).
function Run {
    $exe = $args[0]; $rest = @($args | Select-Object -Skip 1)
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "$exe $($rest -join ' ') failed (exit $LASTEXITCODE)" }
}

# winget changes PATH in the registry; pick that up without opening a new terminal.
function Update-Path {
    $m = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $u = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = (@($m, $u) | Where-Object { $_ }) -join ";"
}

function Has($cmd) { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Winget-Install($id, [string[]]$extra = @()) {
    if (-not (Has winget)) {
        Warn "winget not found (install 'App Installer' from the Microsoft Store), cannot install $id"
        return $false
    }
    Write-Host "winget install $id"
    & winget install --id $id -e --source winget --accept-package-agreements `
        --accept-source-agreements --disable-interactivity @extra | Out-Host
    $ok = $LASTEXITCODE -eq 0
    Update-Path
    return $ok
}

# Is a pip package installed in the venv? (no stderr: Windows PowerShell 5.1 turns it into errors)
function Has-Pkg($py, $name) {
    & $py -c "import importlib.metadata as m, sys; sys.exit(sys.argv[1] not in {(d.metadata['Name'] or '').lower() for d in m.distributions()})" $name
    return $LASTEXITCODE -eq 0
}

function Ask($question) {
    if ($Yes -or -not [Environment]::UserInteractive) { return $false }
    try { $a = Read-Host "$question [y/N]" } catch { return $false }
    return $a -match '^(y|yes)$'
}

# A Python 3.10+ that really runs (the Microsoft Store "python" alias only opens the Store).
function Find-Python {
    $cands = @(
        @("py", "-3"),
        @("python"),
        @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"),
        @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe")
    )
    foreach ($c in $cands) {
        $exe = $c[0]; $rest = @($c | Select-Object -Skip 1)
        if (-not (Has $exe)) { continue }
        try {
            $v = & $exe @rest -c "import sys; print(sys.version_info >= (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0 -and "$v".Trim() -eq "True") { return , $c }
        } catch { }
    }
    return $null
}

# --- 1. Where clips lives: this checkout, or download it -------------------------------------
$here = $PSScriptRoot
if ($here -and (Test-Path (Join-Path $here "clips\cli.py"))) {
    $Repo = $here
} else {
    Say "Downloading clips to $Dir"
    if (-not (Has git)) {
        Winget-Install "Git.Git" | Out-Null
        if (-not (Has git)) { Fail "git is needed to download clips; install it from https://git-scm.com and run this again" }
    }
    if (Test-Path (Join-Path $Dir ".git")) {
        Run git -C $Dir pull --ff-only
    } else {
        Run git clone $RepoUrl $Dir
    }
    $Repo = $Dir
}
Set-Location $Repo

$askVision = -not ($PSBoundParameters.ContainsKey("Vision") -or $Yes)
$askOllama = -not ($PSBoundParameters.ContainsKey("Ollama") -or $Yes)
if ($askVision) {
    $Vision = Ask "Install subject detection (--subject; uses your GPU through DirectML, ~60 MB)?"
}
if ($askOllama) {
    $Ollama = Ask "Install Ollama, the free local AI that picks a person from a description (--subject-desc; picks a model that fits your GPU, 2-21 GB download)?"
}

# --- 2. Python 3.10+ ---------------------------------------------------------------------------
Say "Checking Python"
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10+ not found, installing Python 3.12 for this user"
    Winget-Install "Python.Python.3.12" @("--scope", "user") | Out-Null
    $py = Find-Python
    if (-not $py) { Fail "Python 3.10+ is needed; install it from https://www.python.org (tick 'Add to PATH') and run this again" }
}
$pyExe = $py[0]; $pyArgs = @($py | Select-Object -Skip 1)
Write-Host "found $(& $pyExe @pyArgs -c 'import sys; print(sys.executable, sys.version.split()[0])')"

# --- 3. ffmpeg (optional: a bundled build is used when there is none) --------------------------
Say "Checking ffmpeg"
if (Has ffmpeg) {
    Write-Host "found $((Get-Command ffmpeg).Source)"
} elseif (-not (Winget-Install "Gyan.FFmpeg") -or -not (Has ffmpeg)) {
    Write-Host "could not install ffmpeg; the bundled imageio-ffmpeg build will be used"
}

# --- 4. venv + requirements --------------------------------------------------------------------
$venvPy = Join-Path $Repo ".venv\Scripts\python.exe"
Say "Creating .venv"
if (-not (Test-Path $venvPy)) { Run $pyExe @pyArgs -m venv .venv }
Say "Installing requirements"
Run $venvPy -m pip install -q --upgrade pip
Run $venvPy -m pip install -q -r requirements.txt
Run $venvPy -m pip install -q --no-deps -e .

if ($Vision) {
    # DirectML runs on any DirectX 12 GPU (AMD Radeon included); it replaces the CPU build.
    Say "Installing vision extras (ONNX Runtime DirectML)"
    if (Has-Pkg $venvPy "onnxruntime") { Run $venvPy -m pip uninstall -q -y onnxruntime }
    Run $venvPy -m pip install -q "onnxruntime-directml>=1.17"
}

# --- 5. `clips` command on PATH, Start menu shortcuts ------------------------------------------
Say "Adding the clips command"
$clipsExe = Join-Path $Repo ".venv\Scripts\clips.exe"
$bin = Join-Path $env:LOCALAPPDATA "clips\bin"
New-Item -ItemType Directory -Force $bin | Out-Null
# a small shim instead of putting .venv\Scripts (and its python.exe) on PATH
Set-Content -Encoding ASCII (Join-Path $bin "clips.cmd") "@`"$clipsExe`" %*"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not (($userPath -split ";") -contains $bin)) {
    $newPath = (@($userPath, $bin) | Where-Object { $_ }) -join ";"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "added $bin to your PATH (new terminals)"
}
Update-Path

try {
    $menu = Join-Path ([Environment]::GetFolderPath("Programs")) "Clips"
    New-Item -ItemType Directory -Force $menu | Out-Null
    $sh = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut((Join-Path $menu "Clips.lnk"))
    $lnk.TargetPath = $clipsExe
    $lnk.Arguments = "ui"
    $lnk.WorkingDirectory = $HOME
    $lnk.Description = "Open the clips web UI (keep this window open while you use it)"
    $lnk.Save()
    $lnk = $sh.CreateShortcut((Join-Path $menu "Update Clips.lnk"))
    $lnk.TargetPath = Join-Path $Repo "update.bat"
    $lnk.WorkingDirectory = $Repo
    $lnk.Save()
    Write-Host "Start menu: Clips, Update Clips"
} catch {
    Warn "could not create Start menu shortcuts: $_"
}

# --- 6. Ollama (optional) ----------------------------------------------------------------------
function Ollama-Up {
    $h = $env:OLLAMA_HOST
    if (-not $h) { $h = "127.0.0.1:11434" }
    if ($h -notmatch "^https?://") { $h = "http://$h" }
    try { Invoke-RestMethod "$h/api/version" -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

# Biggest Ollama vision model that fits: the model plus ~2 GB for the image and context in
# graphics memory, or in RAM (slower, CPU) when the GPU is too small. qwen2.5vl is what
# --subject-desc is tested with; moondream is the low-memory fallback.
function Pick-Model([double]$vramGB, [double]$ramGB) {
    if ($vramGB -ge 23 -and $ramGB -ge 16) { return "qwen2.5vl:32b" }    # 21 GB
    if ($vramGB -ge 8) { return "qwen2.5vl:7b" }                          # 6.0 GB
    if ($vramGB -ge 5) { return "qwen2.5vl:3b" }                          # 3.2 GB
    if ($ramGB -ge 16) { return "qwen2.5vl:3b" }                          # on the CPU
    return "moondream"                                                    # 1.7 GB
}

# Graphics memory of the biggest GPU in GB. Win32_VideoController.AdapterRAM stops at 4 GB,
# so read the driver's 64-bit qwMemorySize from the display adapter class in the registry.
function Get-VramGB {
    $best = 0.0
    $key = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    foreach ($k in (Get-ChildItem $key -ErrorAction SilentlyContinue)) {
        $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
        if (-not $p) { continue }
        $v = $p."HardwareInformation.qwMemorySize"
        if (-not $v) { $v = $p."HardwareInformation.MemorySize" }
        if ($v -is [byte[]]) { $v = [BitConverter]::ToUInt32($v, 0) }
        if ($v) { $best = [Math]::Max($best, [double]$v / 1GB) }
    }
    if ($best -eq 0) {
        try {
            foreach ($g in (Get-CimInstance Win32_VideoController)) {
                if ($g.AdapterRAM) { $best = [Math]::Max($best, [double]$g.AdapterRAM / 1GB) }
            }
        } catch { }
    }
    return [Math]::Round($best, 1)
}

function Get-RamGB {
    try { return [Math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1) }
    catch { return 0 }
}

if ($Ollama) {
    if (-not $Model -or $Model -eq "auto") {
        $gpuName = "GPU"
        try { $gpuName = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ", " } catch { }
        $vram = Get-VramGB
        $ram = Get-RamGB
        $Model = Pick-Model $vram $ram
        Say "Picking a model: $gpuName, $vram GB graphics memory, $ram GB RAM -> $Model"
    }
    Say "Installing Ollama"
    if (Has ollama) {
        Write-Host "found $((Get-Command ollama).Source)"
    } else {
        Winget-Install "Ollama.Ollama" | Out-Null
        if (-not (Has ollama)) {
            $guess = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            if (Test-Path (Join-Path $guess "ollama.exe")) { $env:Path += ";$guess" }
        }
    }
    if (-not (Has ollama)) {
        Warn "Ollama did not install; get it from https://ollama.com/download and run: ollama pull $Model"
    } else {
        if (-not (Ollama-Up)) {
            # the tray app runs the server and starts with Windows; fall back to a hidden `ollama serve`
            $app = Join-Path (Split-Path (Get-Command ollama).Source) "ollama app.exe"
            if (Test-Path $app) { Start-Process $app } else { Start-Process ollama "serve" -WindowStyle Hidden }
            for ($i = 0; $i -lt 30 -and -not (Ollama-Up); $i++) { Start-Sleep 1 }
        }
        if ($Model -and $Model -ne "none") {
            if (Ollama-Up) {
                Say "Downloading the $Model model (one time, several GB)"
                & ollama pull $Model
                if ($LASTEXITCODE -ne 0) { Warn "ollama pull $Model failed; run it again later" }
            } else {
                Warn "Ollama is installed but not running yet; start Ollama, then run: ollama pull $Model"
            }
        }
        if ($Model -and $Model -ne "none") {
            # clips --subject-desc uses this model unless --ollama-model says otherwise
            [Environment]::SetEnvironmentVariable("CLIPS_OLLAMA_MODEL", $Model, "User")
            $env:CLIPS_OLLAMA_MODEL = $Model
            Write-Host "clips will use $Model (CLIPS_OLLAMA_MODEL; new terminals)"
        }
        # AMD Radeon on Windows (RX 6700 XT included) runs through Ollama's Vulkan backend: no setup.
    }
}

# --- 7. Check ----------------------------------------------------------------------------------
Say "Checking the install"
& $clipsExe --help | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "clips --help failed" }
Run $venvPy -c "from clips.ffmpeg import ffmpeg_exe; print('ffmpeg:', ffmpeg_exe())"

Say "Done"
Write-Host "Open 'Clips' from the Start menu for the web UI, or in a new terminal run: clips --help"
if (-not $Vision) { Write-Host "(subject detection later: .\install.ps1 -Vision)" }
if (-not $Ollama) { Write-Host "(Ollama later: .\install.ps1 -Ollama)" }
