@echo off



set ZLUDA_COMGR_LOG_LEVEL=1
setlocal EnableDelayedExpansion
set "PYTHON=%~dp0venv\Scripts\python.exe"
set "GIT="
set "VENV_DIR=.\venv"


:: %SystemRoot%\system32\curl.exe -sL --ssl-no-revoke https://github.com/user-attachments/files/20140536/flash_attn-2.7.4.post1-py3-none-any.zip > fa.zip
:: %SystemRoot%\system32\tar.exe -xf fa.zip
:: pip install flash_attn-2.7.4.post1-py3-none-any.whl --quiet

:: copy comfy\customzluda\fa\distributed.py %VIRTUAL_ENV%\Lib\site-packages\flash_attn\utils\distributed.py /y >NUL

echo  ::  %time:~0,8%  ::  - Installing and patching sage-attention
pip install sageattention==1.0.6 --quiet
copy comfy\customzluda\sa\quant_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\quant_per_block.py /y >NUL
copy comfy\customzluda\sa\attn_qk_int8_per_block_causal.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block_causal.py /y >NUL
copy comfy\customzluda\sa\attn_qk_int8_per_block.py %VIRTUAL_ENV%\Lib\site-packages\sageattention\attn_qk_int8_per_block.py /y >NUL


copy zluda\cublas.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cublas64_11.dll /y >NUL
copy zluda\cusparse.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cusparse64_11.dll /y >NUL
copy %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc_cuda.dll /y >NUL
copy zluda\nvrtc.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\nvrtc64_112_0.dll /y >NUL
:: removed cudnn dll patching , check the results
:: copy zluda\cudnn.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cudnn64_9.dll /y >NUL
copy zluda\cufft.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cufft64_10.dll /y >NUL
copy zluda\cufftw.dll %VIRTUAL_ENV%\Lib\site-packages\torch\lib\cufftw64_10.dll /y >NUL
copy comfy\customzluda\zluda.py comfy\zluda.py /y >NUL

echo. 
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
set MIOPEN_FIND_MODE=2
set MIOPEN_LOG_LEVEL=3
.\zluda\zluda.exe -- python main.py --auto-launch --use-quad-cross-attention