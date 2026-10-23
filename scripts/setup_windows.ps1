# One-shot environment setup for Windows + NVIDIA GPU (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -CudaTag cu124
#
# Prerequisites: NVIDIA driver, git, Python 3.10/3.11 ("py" launcher), the CUDA
# toolkit matching -CudaTag, and "Desktop development with C++" from the Visual
# Studio Build Tools (needed to compile torchmcubes for TripoSR).
param(
    [string]$CudaTag = "cu128",
    [string]$PythonVersion = "3.11",
    [string]$BlenderVersion = "4.2.23"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force -Path third_party | Out-Null

Write-Host "==> GPU"
try { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv } catch { Write-Warning "nvidia-smi not found" }

Write-Host "==> Python venv"
py -$PythonVersion -m venv .venv
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install --upgrade pip wheel setuptools

Write-Host "==> PyTorch ($CudaTag)"
& $Py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"
& $Py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available:', torch.cuda.is_available())"

Write-Host "==> Pipeline requirements"
& $Py -m pip install -r requirements.txt

Write-Host "==> TripoSR"
if (-not (Test-Path third_party\TripoSR)) {
    git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR third_party\TripoSR
}
& $Py -m pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
Get-Content third_party\TripoSR\requirements.txt |
    Where-Object { $_ -notmatch "torchmcubes" -and $_ -notmatch "^gradio" } |
    Set-Content third_party\triposr-req.txt
& $Py -m pip install -r third_party\triposr-req.txt

Write-Host "==> Blender $BlenderVersion"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender $($BlenderVersion.Substring(0, $BlenderVersion.LastIndexOf('.')))\blender.exe"
if (-not (Test-Path $BlenderExe)) {
    $Series = $BlenderVersion.Substring(0, $BlenderVersion.LastIndexOf('.'))
    $Zip = "third_party\blender.zip"
    Invoke-WebRequest "https://download.blender.org/release/Blender$Series/blender-$BlenderVersion-windows-x64.zip" -OutFile $Zip
    Expand-Archive $Zip -DestinationPath third_party -Force
    Remove-Item $Zip
    $BlenderExe = Join-Path $Root "third_party\blender-$BlenderVersion-windows-x64\blender.exe"
}
& $BlenderExe --version | Select-Object -First 1

Write-Host "==> Local LLMs (Ollama)"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull qwen2.5:14b-instruct
    ollama pull qwen2.5vl:7b
} else {
    Write-Host "Ollama not installed (https://ollama.com) - or point llm/vision at vLLM / LM Studio / Claude."
}

if (-not (Test-Path config.yaml)) {
    $BlenderYaml = $BlenderExe -replace "\\", "/"
    (Get-Content config.example.yaml) -replace "^  executable: null.*", "  executable: `"$BlenderYaml`"" |
        Set-Content config.yaml
    Write-Host "Wrote config.yaml"
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .venv\Scripts\activate"
Write-Host "  python main.py --check --config config.yaml"
Write-Host "  python main.py --config config.yaml --prompt `"A functional MP5 submachine gun`" --enable_rigging"
