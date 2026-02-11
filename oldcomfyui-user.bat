@echo OFF
color 0A
set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"


set "COMMANDLINE_ARGS=--preview-method auto --auto-launch --use-quad-cross-attention --reserve-vram 0.5 --cuda-device 0"

REM set "COMMANDLINE_ARGS=--cpu-vae --auto-launch --use-quad-cross-attention --reserve-vram 0.9 --cuda-device 0 --preview-method auto"

set "ZLUDA_COMGR_LOG_LEVEL=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"


echo [INFO] Launching application via ZLUDA...

@echo.

echo [INFO]
echo [INFO]
echo [INFO]
echo [INFO]

 .\zluda\zluda.exe -- %PYTHON% main.py %COMMANDLINE_ARGS%



REM .\zluda\zluda.exe -- %PYTHON% "D:\ComfyUI-Zluda-master\Wan2GP\wgp.py" --i2v
REM set X
REM %PYTHON% -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
REM %PYTHON% -m pip install -r  "D:\ComfyUI-Zluda-master\requirements.txt"
REM %PYTHON% -m pip install "numpy<2" 

REM %PYTHON% -m ensurepip --default-pip
REM %PYTHON% -m pip install diffusers

REM copy copy "C:\AmuseConverters\zludaworking\*.*" ""D:\ComfyUI-Zluda-master\venv\Lib\site-packages\torch\lib\" /y
set TRITON_OVERRIDE_ARCH=gfx903
set HSA_OVERRIDE_GFX_VERSION=9.0.0
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3
set MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=
set TORCH_CUDA_ARCH_LIST=
set CUDA_VISIBLE_DEVICES=0
set TRITON_PTXAS_PATH=none
set TRITON_LIBDEVICE_PATH=none
set TORCHINDUCTOR_DISABLE=1
set TRITON_DISABLE=1
set nproc_per_node=4


pause

