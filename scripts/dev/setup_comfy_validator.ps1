# Installs a CPU-only ComfyUI (plus the custom node packs higgsless uses) on a machine
# without an NVIDIA GPU, so workflows can be validated against real node definitions
# with `higgsless workflows validate --comfy-url http://127.0.0.1:8189`.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts/dev/setup_comfy_validator.ps1 [-Target <dir>]
param(
    [string]$Target = "$env:LOCALAPPDATA\higgsless\comfy-validator",
    [string]$Python = "3.12"
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Set-Location $Target

if (-not (Test-Path "ComfyUI")) {
    git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git ComfyUI
}
Set-Location ComfyUI
if (-not (Test-Path ".venv")) { uv venv --python $Python .venv }
$py = Join-Path (Get-Location) ".venv\Scripts\python.exe"

uv pip install --python $py torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $py -r requirements.txt

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
Set-Location custom_nodes
foreach ($url in $packs) {
    $name = ($url -split "/")[-1]
    if (-not (Test-Path $name)) { git clone --depth 1 $url $name }
    $req = Join-Path $name "requirements.txt"
    if (Test-Path $req) {
        # Never let a node pack replace the torch build.
        $filtered = Get-Content $req | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|xformers)\b' }
        $tmp = Join-Path $env:TEMP "hl-req-$name.txt"
        $filtered | Set-Content $tmp
        uv pip install --python $py -r $tmp
    }
}
Set-Location ..
Write-Host "Done. Start with: $py main.py --cpu --listen 127.0.0.1 --port 8189"
