# Chatterbox playback pipeline (streaming via ffplay)

Date: 2026-06-22
Status: approved, ready for implementation plan

## Problem

Chatterbox runs at RTF ~1 (~2-3s per fragment). Today the client buffers the
*entire* utterance, then plays it as one WAV through the Windows host
(`WinSoundPlayer`, used because WSLg's RDP audio bridge adds static to all
WSL-side playback). For multi-fragment Chatterbox text this means the user
waits for the whole utterance to synthesize before hearing anything.

Goal: start speaking after roughly one fragment instead of the whole
utterance, with no audible gaps between fragments ("seamless + sooner"), while
keeping playback on the clean Windows-native path.

## Constraints

- Single GPU. The server already serializes GPU work behind
  `asyncio.Semaphore(1)` and synthesizes fragments sequentially, streaming
  their bytes concatenated over one HTTP response. This does NOT change.
  "Parallelize" here means *pipeline* (overlap synth of fragment N+1 with
  playback of fragment N), not parallel GPU compute.
- WSLg's RDP audio bridge is unreliable (intermittent static on pacat AND
  paplay). Playback must stay on the Windows-native path.
- The built-in Windows `SoundPlayer` is file-based and cannot stream
  gaplessly. Seamless streaming needs a streaming player on the Windows side.
- CLAUDE.md anti-stutter rule: do not stream Chatterbox sub-fragment chunks.
  This design streams whole fragments (as the server already emits them) and
  adds a lead buffer; it does not sub-chunk a single generation.

## Approach (chosen: A)

Add a streaming player that pipes the server's already-sequential fragment
stream into a long-lived `ffplay.exe` on the Windows host. ffplay plays
natively on Windows (clean, bypasses the RDP path) and gaplessly. The pipeline
falls out for free: while ffplay plays fragment N's streamed bytes, the server
synthesizes fragment N+1. No server change, no fragment-boundary protocol, no
playback queue/threads.

Rejected alternatives:
- B (two-phase, no install): one ~0.35s seam after fragment 1 (powershell
  respawn). Not seamless.
- C (winmm waveOut P/Invoke, no install): seamless but fragile inline audio
  code, high maintenance risk.

ffmpeg is installed on the Windows host (Gyan.FFmpeg 8.1.1 via winget).

## Player interface (unifies the three players)

`_speak` stops buffering-then-playing and instead streams:

```
player.write(chunk)      # feed bytes as they arrive; gain applied here
player.end_utterance()   # mark "utterance complete"
```

- `FfplayPlayer` (new, primary on WSLg when ffplay present): `write` feeds
  ffplay stdin (through the lead buffer); `end_utterance` flushes any held
  lead bytes (continuous playback otherwise).
- `WinSoundPlayer` (existing, fallback when ffmpeg absent): `write`
  accumulates; `end_utterance` plays the whole WAV (today's behavior, incl.
  lead-in silence).
- `PacatPlayer` (native Linux): `write` streams to pacat; `end_utterance`
  no-op.

`_speak`: for each streamed chunk -> `player.write(chunk)`; at end ->
`player.end_utterance()`. One code path; each player decides stream vs buffer.

## FfplayPlayer

Launch (once per REPL session, persistent like pacat):

```
<ffplay> -hide_banner -loglevel error -nodisp -autoexit \
  -f f32le -ar 24000 -ch_layout mono -i pipe:0
```

(`-ch_layout mono`, not `-ac 1`: ffplay 8.1 rejects `-ac` for raw input.
Verified: f32le from stdin via WSL interop plays clean on Windows.)

- Float32 LE bytes written to stdin. Gain applied client-side first, with the
  same 4-byte realignment buffer as `PacatPlayer` (network chunks may split a
  float).
- `end_utterance` is a no-op for continuous playback except flushing held lead
  bytes.
- On REPL exit: close stdin, let `-autoexit` end ffplay, terminate if needed.

Lead buffer (cushion for RTF ~1):
- Time-based, `PIPELINE_LEAD_MS` (default ~1500ms) -> bytes =
  ms/1000 * 24000 * 4.
- `write` holds bytes until the lead threshold is reached, then flushes them to
  ffplay at once and streams continuously thereafter. ffplay then has a
  ~lead-sized cushion to ride over per-fragment synth jitter.
- The client sees a concatenated byte stream (no fragment markers), so the lead
  is purely time/byte based. Kokoro fills it near-instantly; Chatterbox uses it
  as the cushion.
- Reset per utterance (`end_utterance` flushes remainder; next utterance
  re-primes the lead).

Honest limit: at sustained RTF > 1 the cushion drains and a gap can appear.
Mitigation is raising `PIPELINE_LEAD_MS`. Not a hard guarantee -- "seamless in
practice when synth keeps pace, with a knob to bias toward safety."

## Discovery and selection (`make_player`, WSLg)

1. Probe `ffplay.exe` on PATH (present after a shell/WSL restart following the
   winget install).
2. Else glob the winget install path:
   `/mnt/c/Users/<user>/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg*/ffmpeg*/bin/ffplay.exe`
3. Found -> `FfplayPlayer` (primary on WSLg, both backends: Kokoro regains
   low-latency streaming, Chatterbox gets the pipeline).
4. Not found -> `WinSoundPlayer` (buffered fallback) + one-line hint:
   "install ffmpeg on Windows for seamless streaming."

Non-WSLg -> `PacatPlayer` as today.

## Error handling

- ffplay dies mid-stream (`BrokenPipeError` on write): print one line, keep the
  REPL alive, relaunch ffplay on the next utterance.
- ffplay not found at startup: fall back to `WinSoundPlayer` + hint.
- Server stream error mid-utterance: existing `httpx.HTTPError` handling in
  `_speak` stays; `end_utterance()` still called.
- Gain + 4-byte realignment identical to `PacatPlayer`.

## Config

- `PIPELINE_LEAD_MS = 1500` (lead-buffer cushion, ms).
- Reuses `OUTPUT_GAIN`, `SAMPLE_RATE`.

## Testing (all in `.venv-dev`, no GPU/torch)

- `FfplayPlayer` with an injected fake process/stdin:
  - `write` applies gain, 4-byte-aligns, and respects the lead buffer (withholds
    until threshold, then flushes).
  - `end_utterance` flushes the remainder.
  - `BrokenPipeError` on write is handled (no raise out).
- Discovery: PATH probe, else winget glob; found -> `FfplayPlayer`,
  missing -> `WinSoundPlayer`.
- `make_player` selection across WSLg + ffplay-present / ffplay-absent /
  non-WSLg.
- `_speak`: streams chunks incrementally and calls `end_utterance` exactly once
  (mock player records calls).

## Out of scope

- Server-side changes (fragment-boundary protocol, batching).
- True parallel GPU synthesis (single GPU; serialized by design).
- Replacing the Kokoro realtime benchmark gate.
