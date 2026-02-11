@echo off

title ComfyUI-Zluda Installer

set ZLUDA_COMGR_LOG_LEVEL=1
setlocal EnableDelayedExpansion
set "startTime=%time: =0%"

cls
echo -----------------------------------------------------------------------
Echo * COMFYUI-ZLUDA INSTALL (for HIP 6.2.4 / 6.4.2 with MIOPEN and Triton)*
echo -----------------------------------------------------------------------
echo.
echo  ::  %time:~0,8%  ::  - Setting up the virtual enviroment
Set "VIRTUAL_ENV=venv"
If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" (
    python.exe -m venv %VIRTUAL_ENV%
)

If Not Exist "%VIRTUAL_ENV%\Scripts\activate.bat" Exit /B 1

echo  ::  %time:~0,8%  ::  - Virtual enviroment activation
Call "%VIRTUAL_ENV%\Scripts\activate.bat"
echo  ::  %time:~0,8%  ::  - Updating the pip package 
python.exe -m pip install --upgrade pip --quiet
echo.
echo  ::  %time:~0,8%  ::  Beginning installation ...
echo.
echo  ::  %time:~0,8%  ::  - Installing required packages
pip install -r requirements.txt --quiet
echo  ::  %time:~0,8%  ::  - Installing torch for AMD GPUs (First file is 2.7 GB, please be patient)
pip uninstall torch torchvision torchaudio -y --quiet



pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118


echo  ::  %time:~0,8%  ::  - Installing onnxruntime (required by some nodes)
pip install onnxruntime --quiet
echo  ::  %time:~0,8%  ::  - (temporary numpy fix)
pip uninstall numpy -y --quiet
pip install numpy==1.26.4 --quiet

echo  ::  %time:~0,8%  ::  - Detecting Python version and installing appropriate triton package

for /f "tokens=1,2 delims=." %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
    goto :version_detected
)

:version_detected
echo  ::  %time:~0,8%  ::  - Detected Python %PY_MAJOR%.%PY_MINOR%

if "%PY_MINOR%"=="12" (
    echo  ::  %time:~0,8%  ::  - Python 3.12 detected, installing triton for 3.12
    pip install --force-reinstall https://github.com/lshqqytiger/triton/releases/download/a9c80202/triton-3.4.0+gita9c80202-cp312-cp312-win_amd64.whl
) else if "%PY_MINOR%"=="11" (
    echo  ::  %time:~0,8%  ::  - Python 3.11 detected, installing triton for 3.11
    pip install --force-reinstall https://github.com/lshqqytiger/triton/releases/download/a9c80202/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.whl
) else (
    echo  ::  %time:~0,8%  ::  - WARNING: Unsupported Python version 3.%PY_MINOR%, skipping triton installation
    echo  ::  %time:~0,8%  ::  - Full version info:
    python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
)

:: patching triton & torch (from sfinktah ; https://github.com/sfinktah/amd-torch )
pip install --force-reinstall pypatch-url --quiet
pypatch-url apply https://raw.githubusercontent.com/sfinktah/amd-torch/refs/heads/main/patches/triton-3.4.0+gita9c80202-cp311-cp311-win_amd64.patch -p 4 triton
pypatch-url apply https://raw.githubusercontent.com/sfinktah/amd-torch/refs/heads/main/patches/torch-2.7.0+cu118-cp311-cp311-win_amd64.patch -p 4 torch

:: echo  ::  %time:~0,8%  ::  - Installing flash-attention

:: %SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/user-attachments/files/20140536/flash_attn-2.7.4.post1-py3-none-any.zip > fa.zip
:: %SystemRoot%\system32\tar.exe -xf fa.zip
:: pip install flash_attn-2.7.4.post1-py3-none-any.whl --quiet
:: del fa.zip
:: del flash_attn-2.7.4.post1-py3-none-any.whl
:: copy comfy\customzluda\fa\distributed.py %VIRTUAL_ENV%\Lib\site-packages\flash_attn\utils\distributed.py /y >NUL

echo  ::  %time:~0,8%  ::  - Installing and patching sage-attention
pip install sageattention==1.0.6 --quiet
copy comfy\customzluda\sa\quant_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\quant_per_block.py /y >NUL
copy comfy\customzluda\sa\attn_qk_int8_per_block_causal.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py /y >NUL
copy comfy\customzluda\sa\attn_qk_int8_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block.py /y >NUL

echo.

echo. 

echo ..................................................... 
echo *** Installation is completed in %hh:~1%%time:~2,1%%mm:~1%%time:~2,1%%ss:~1%%time:~8,1%%cc:~1% . 
echo *** You can use "comfyui-n.bat" or "comfyui-user.bat" to start the app later. 
echo *** If you want to modify the launcher please use the "comfyui-user.bat" as it is not effected by the updates.
echo *** You can use -- "--use-pytorch-cross-attention" , "--use-quad-cross-attention" , "--use-flash-attention" or "--use-sage-attention" 
echo ..................................................... 
echo.
echo *** Starting the Comfyui-ZLUDA for the first time, please be patient...
echo.
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3
.\zluda\zluda.exe -- python main.py --auto-launch --use-sage-attention --lowvram