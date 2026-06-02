# How to Start the Transcriber on Machine 25

This is the step-by-step guide for starting the Meeting Transcriber on the
office PC **"25"** (the machine with the RTX 3090 graphics card, IP address
`192.168.87.25`). No programming knowledge needed — just follow along at the
machine's monitor and keyboard.

## What this is

The Transcriber turns meeting audio (MP3 and similar) into text with speaker
labels and timestamps. It runs as a small web app on machine 25. Once started,
anyone in the office can use it from their own browser.

## Before you start

- You need to be **physically at machine 25** (monitor + keyboard), or
  remoted into it.
- The app must be **started by a person** and stays running only while the
  black command window stays open. If the machine is rebooted or the window is
  closed, someone has to start it again with these steps.

## Start it

1. Click the **Start menu**, type `powershell`, and press **Enter**.
   A blue or black window opens.

2. Click into that window, type the following line exactly, and press **Enter**:

   ```
   cd C:\Users\videocard\w\Transcriber
   ```

3. Type this line and press **Enter**:

   ```
   venv\Scripts\python.exe -m app.main
   ```

4. Wait. After a few seconds you should see a line like:

   ```
   Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
   ```

   That means it is running. **Leave this window open** — closing it stops the app.

## Check that it works

1. On machine 25, open a web browser and go to:

   ```
   http://localhost:7860
   ```

   You should see the Transcriber page.

2. From **any other computer in the office**, open a browser and go to:

   ```
   http://192.168.87.25:7860
   ```

To transcribe: upload an MP3 on the page and click **Transcribe**. The very
first transcription after starting takes a little longer (the AI model loads
into the graphics card); after that it is fast — roughly a 45-second clip in
about 10 seconds.

## Stop it

- Click the command window and press **Ctrl + C**, or simply **close the window**.

## If something goes wrong

- **The page won't open / "can't reach this site":**
  Make sure the command window from the *Start it* steps is still open and shows
  the `Uvicorn running on ...` line. If the window was closed, start again.

- **You see `error while attempting to bind on address ... 7860`:**
  It is already running (another command window has it open). You don't need to
  start it again — just open `http://192.168.87.25:7860` in your browser. If you
  want a fresh start, close all the command windows first, then follow *Start it*.

- **`python` is not recognized, or it opens the Microsoft Store:**
  Make sure you typed the command exactly, including `venv\Scripts\python.exe`
  (not just `python`). The `venv\Scripts\python.exe` part is important.

- **It mentions the GPU / CUDA / "no kernel image":**
  Restart machine 25 and try the *Start it* steps again. If it still fails,
  contact the person who maintains the Transcriber.

## Quick reference

| | |
|---|---|
| Machine | Office PC **25** (RTX 3090) |
| Folder | `C:\Users\videocard\w\Transcriber` |
| Start command | `venv\Scripts\python.exe -m app.main` (run from the folder above) |
| Local address | `http://localhost:7860` |
| Office address | `http://192.168.87.25:7860` |
| Stop | `Ctrl + C` in the command window, or close it |
