@echo off
REM Manual install for Windows box (bypasses broken install.ps1 here-strings)
REM Uses explicit python path to avoid the Microsoft Store WindowsApps python stub
set "PY=C:\Users\videocard\AppData\Local\Programs\Python\Python310\python.exe"
cd /d C:\Users\videocard\w\Transcriber
del /q step_*.flag install_done.flag install_failed.flag 2>nul

echo ===VENV===
if not exist venv\Scripts\python.exe "%PY%" -m venv venv || goto :fail
echo ok> step_venv.flag
set "VPY=C:\Users\videocard\w\Transcriber\venv\Scripts\python.exe"

echo ===PIP_UPGRADE===
"%VPY%" -m pip install --upgrade pip || goto :fail
echo ok> step_pip.flag

echo ===TORCH===
"%VPY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 || goto :fail
echo ok> step_torch.flag

echo ===REQUIREMENTS===
"%VPY%" -m pip install -r requirements.txt || goto :fail
echo ok> step_reqs.flag

echo ===VIBEVOICE_CLONE===
if not exist VibeVoice git clone https://github.com/microsoft/VibeVoice.git || goto :fail
echo ok> step_vvclone.flag

echo ===VIBEVOICE_INSTALL===
"%VPY%" -m pip install -e VibeVoice || goto :fail
echo ok> step_vvinstall.flag

echo ===VERIFY_TORCH===
"%VPY%" -c "import torch; print('TORCH', torch.__version__, 'CUDA_OK', torch.cuda.is_available(), 'DEV', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')" || goto :fail

echo ===VERIFY_VIBEVOICE===
"%VPY%" -c "from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor; print('VIBEVOICE_IMPORT_OK')" || goto :fail

echo INSTALL_DONE
echo ok> install_done.flag
exit /b 0
:fail
echo INSTALL_FAILED errorlevel=%errorlevel%
echo %errorlevel%> install_failed.flag
exit /b 1
