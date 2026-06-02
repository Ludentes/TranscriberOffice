# Meeting Recordings: From OBS to Summary

A practical guide for office colleagues who have **Claude Code** and want to turn
a meeting recording into clean, named, summarized notes.

You can do three things with this guide. Each works on its own — jump to the one
you need:

- [**Transcribe** a recording](#transcribe-a-recording) — turn an OBS video into text with speaker labels
- [**Summarize** a meeting](#summarize-a-meeting) — get a structured summary from the transcript
- [**Name the speakers**](#name-the-speakers) — replace "Speaker 0 / Speaker 1" with real people

This guide is meant to be **copy-paste friendly**: most steps are a prompt you
paste into Claude Code, and Claude does the work for you.

## What you need

- **Claude Code** installed on your computer.
- Your **OBS recording** (usually a `.mkv` or `.mp4` file). OBS saves recordings
  to the folder set in *Settings → Output → Recording Path*.
- For transcription, access to the office **Transcriber on machine 25**:
  `http://192.168.87.25:7860`. Someone must have it started — if the page
  doesn't open, see [`runbook-start-transcriber-windows-25.md`](runbook-start-transcriber-windows-25.md).

You do **not** need a powerful computer or a graphics card. The heavy work
(transcription) happens on machine 25; your computer only handles small things
like extracting audio and writing the summary.

---

## Transcribe a recording

There are two ways. **Path A (web page)** is the easy default and works for most
meetings. **Path B (full pipeline)** is for very long recordings (multiple hours)
or when you want everything automated end-to-end.

### Path A — The easy way (web page on machine 25)

**Step 1 — Get the audio out of the video.**
Transcription needs an audio file (MP3). If your OBS file is a video (`.mkv` /
`.mp4`), paste this into Claude Code (replace the filename):

> Extract the audio from `meeting.mkv` to an MP3 file named `meeting.mp3` using
> ffmpeg. Use good quality (`-q:a 2`). If ffmpeg isn't installed, tell me how to
> install it.

If your recording is already an MP3 (or `.wav`, `.m4a`, `.ogg`, `.flac`), skip
this step.

**Step 2 — Upload it.**
1. Open `http://192.168.87.25:7860` in your browser.
2. Upload your audio file on the page.
3. (Optional) In the **hotwords** box, type names, jargon, or product terms that
   appear in the meeting, comma-separated — this improves recognition of those
   words.
4. Click **Transcribe** and wait. The first transcription after the app starts
   takes a bit longer (the AI model loads); after that it's fast.

**Step 3 — Save the transcript.**
Copy the result from the page into a text file called `transcript.txt`, in a
folder for this meeting (e.g. `meetings/26-03-08/transcript.txt`).

> **File too big?** The web page has an upload size limit. A long meeting can
> exceed it. If you hit "File too large", either trim the recording or use
> **Path B** below.

### Path B — The power way (full pipeline over SSH)

For long meetings, the project has a script that splits the audio on silence,
transcribes each chunk, cleans it up, and lays out the meeting folder for you —
all in one command. It runs on a GPU box.

This path is documented step-by-step in
[`runbook-process-recording.md`](runbook-process-recording.md). The short
version, once you're SSH'd into the GPU box and in the project folder:

```bash
./scripts/process_recording.sh "/path/to/2026-03-08 14-00-00.mkv"
```

That produces `meetings/<date>/transcript.txt` **and** a cleaned-up
`meetings/<date>/clean.txt`, ready to summarize.

> Don't have SSH access or aren't comfortable with the terminal? Use Path A, or
> ask whoever maintains the Transcriber to run Path B for you.

---

## Summarize a meeting

Once you have `transcript.txt` (from either path), summarizing is a single Claude
Code prompt. Open Claude Code **in the folder that contains your transcript** and
paste this (edit the participant names and the structure to taste):

> Read `transcript.txt`. It's a meeting transcript with speaker labels and may be
> in Russian. Write a summary to `summary.md` with these sections:
> - **Participants** — who was there and their roles
> - **Context** — what the meeting was about
> - **Topics discussed** — the main threads, each with a short explanation
> - **Decisions** — what was agreed
> - **Action items** — who needs to do what, by when
>
> Keep it concise and faithful to what was actually said. Write the summary in
> English.

Tips:
- If you already know who the speakers are, tell Claude in the same prompt
  (e.g. "Speaker 0 is Anna the PM, Speaker 1 is the client"). The summary comes
  out much better.
- For a cleaner input, use `clean.txt` instead of `transcript.txt` if you ran
  Path B — it has timestamps and noise stripped out.

---

## Name the speakers

The transcriber labels people as **Speaker 0**, **Speaker 1**, etc. — it can't
know their real names. There's one important quirk to understand first.

> **Important:** For long recordings the audio is split into chunks, and speaker
> numbering **restarts in every chunk**. So "Speaker 0" in the first part of the
> meeting may be a *different person* than "Speaker 0" later on. You can't just
> blind-replace "Speaker 0" → "Anna" across the whole file. Use the meaning of
> what's being said to figure out who's who in each part.

The reliable way is to let Claude Code read the transcript and reason about it.
Paste this into Claude Code in the folder with your transcript:

> Read `transcript.txt`. The speakers are labelled "Speaker 0", "Speaker 1", etc.,
> and the numbering may restart partway through (each chunk numbers speakers from
> 0 again), so the same number can mean different people in different parts.
>
> The real participants are: **[list the names and a clue for each — e.g. "Anna,
> the project manager, asks most of the questions" / "Dmitry, the developer,
> talks about the backend"]**.
>
> Using what each person says (topics, role references, speech style), work out
> who each "Speaker N" is in each part of the meeting, then write a new file
> `transcript_named.txt` with the real names in place of the speaker labels.
> Where you're genuinely unsure, keep the original label and add `(?)`.

Clues that help Claude (and you) tell speakers apart:
- **Role references** — "as the PM, I think…", "on the backend side…".
- **What only one person would say** — the client asks about price; the engineer
  explains the code.
- **Speech style** — some people are terse, some ramble.
- In Russian, **gendered verb/adjective forms** reveal whether a speaker is male
  or female (`сделал` vs `сделала`).

Once you have `transcript_named.txt`, you can summarize *that* file instead for a
summary that already uses real names.

---

## Putting it together (a typical run)

A complete flow for one meeting, the easy way:

1. Extract audio from the OBS file → `meeting.mp3` (Claude Code + ffmpeg).
2. Upload to `http://192.168.87.25:7860`, transcribe, save `transcript.txt`.
3. Name the speakers → `transcript_named.txt` (Claude Code prompt).
4. Summarize `transcript_named.txt` → `summary.md` (Claude Code prompt).

Keep all four files in one folder per meeting so they're easy to find later:

```
meetings/
└── 26-03-08/
    ├── meeting.mp3
    ├── transcript.txt
    ├── transcript_named.txt
    └── summary.md
```

## If something goes wrong

- **The transcriber page won't open** (`http://192.168.87.25:7860`): the app
  probably isn't running. See
  [`runbook-start-transcriber-windows-25.md`](runbook-start-transcriber-windows-25.md),
  or ask someone to start it at machine 25.
- **"File too large" on upload:** use Path B (the full pipeline), or trim/split
  the recording first.
- **`ffmpeg` not found:** ask Claude Code to install it for your operating system,
  or install it from <https://ffmpeg.org/download.html>.
- **The transcript looks garbled or has the wrong language:** add the meeting's
  key terms and names in the **hotwords** box and transcribe again.
- **Speakers are clearly mixed up:** that's expected on long recordings (see the
  quirk under *Name the speakers*). Give Claude more/better clues about who said
  what, and let it re-derive the names.
```
