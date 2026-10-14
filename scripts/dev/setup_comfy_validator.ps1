# Installs a CPU-only ComfyUI (plus the custom node packs higgsless uses) on a machine
# without an NVIDIA GPU, so workflows can be validated against real node definitions:
#
#   powershell -ExecutionPolicy Bypass -File scripts/dev/setup_comfy_validator.ps1
#   <target>\ComfyUI\.venv\Scripts\python.exe <target>\ComfyUI\main.py --cpu --port 8189
#   higgsless workflows validate --comfy-url http://127.0.0.1:8189
param(
    [string]$Target = "$env:LOCALAPPDATA\higgsless\comfy-validator",
    [string]$Python = "3.12"
)
# Windows PowerShell turns native stderr into errors under "Stop"; check exit codes explicitly instead.
$ErrorActionPreference = "Continue"

function Get-Repo([string]$Url, [string]$Dir) {
    if (Test-Path (Join-Path $Dir ".git")) { return }
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    for ($i = 1; $i -le 3; $i++) {
        & git -c http.sslBackend=openssl clone --depth 1 $Url $Dir 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { return }
        if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
        Start-Sleep -Seconds (5 * $i)
    }
    # Fall back to the branch archive (no git metadata, which validation doesn't need).
    $zip = Join-Path $env:TEMP ("hl-" + [IO.Path]::GetFileName($Dir) + ".zip")
    foreach ($branch in @("main", "master")) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$Url/archive/refs/heads/$branch.zip" -OutFile $zip
            $tmp = "$Dir.extract"
            Expand-Archive -Force $zip $tmp
            Move-Item (Get-ChildItem $tmp | Select-Object -First 1).FullName $Dir
            Remove-Item -Recurse -Force $tmp, $zip
            return
        } catch { }
    }
    throw "could not fetch $Url"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv is not on PATH" }

function Invoke-Checked([scriptblock]$Block, [string]$What) {
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null
$comfy = Join-Path $Target "ComfyUI"
Get-Repo "https://github.com/Comfy-Org/ComfyUI" $comfy

$venv = Join-Path $comfy ".venv"
if (-not (Test-Path $venv)) { Invoke-Checked { uv venv --python $Python $venv } "uv venv" }
$py = Join-Path $venv "Scripts\python.exe"
Invoke-Checked { uv pip install --python $py torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu } "torch install"
Invoke-Checked { uv pip install --python $py -r (Join-Path $comfy "requirements.txt") } "ComfyUI requirements"

$packs = @(
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
    "https://github.com/kijai/ComfyUI-KJNodes",
    "https://github.com/kijai/ComfyUI-WanAnimatePreprocess",
    "https://github.com/Fannovel16/comfyui_controlnet_aux",
    "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
    "https://github.com/1038lab/ComfyUI-FlashVSR",
    "https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper",
    "https://github.com/1038lab/ComfyUI-QwenVL",
    "https://github.com/kijai/ComfyUI-GIMM-VFI",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation"
)
$nodes = Join-Path $comfy "custom_nodes"
foreach ($url in $packs) {
    $name = ($url -split "/")[-1]
    $dir = Join-Path $nodes $name
    try { Get-Repo $url $dir } catch { Write-Warning "skipping $name ($_)"; continue }
    $req = Join-Path $dir "requirements.txt"
    if (Test-Path $req) {
        # Never let a node pack replace the torch build.
        $filtered = Get-Content $req | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|xformers)\b' }
        $tmp = Join-Path $env:TEMP "hl-req-$name.txt"
        $filtered | Set-Content $tmp
        uv pip install --python $py -r $tmp
        if ($LASTEXITCODE -ne 0) { Write-Warning "requirements for $name failed; its nodes may not load" }
    }
}
Write-Host "Done. Start with: $py $comfy\main.py --cpu --listen 127.0.0.1 --port 8189"
