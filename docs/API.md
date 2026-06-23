# hosaka HTTP / WS API

Base URL: `http://127.0.0.1:8123` (loopback only). Also reachable remotely as the
`/hosaka` page inside exec-fn (`https://wai-lau.net/hosaka`), which reverse-proxies
the WebSocket over an SSH tunnel and gates it behind that app's auth. PCM is
float32 LE, 24 kHz, mono everywhere.

## GET /health
`{"status": "ok"}` once both models are loaded and warmed.

## GET /v1/voices
List of `{id, backend, source, description}`. `backend` is `kokoro` (presets) or
`chatterbox` (cloneable library clips). `source` is `preset|recording|bake|kokoro`.
`description` is a short blurb (preset character, or the bake prompt).

## POST /v1/audio/speech
Request body:
```json
{
  "input": "text to speak",
  "backend": "kokoro | chatterbox",
  "voice": "af_heart | <library voice id>",
  "params": {"exaggeration": 0.5, "cfg_weight": 0.4,
             "temperature": 0.8, "speed": 1.0}
}
```
Response: `200`, `Content-Type: application/octet-stream` — a chunked stream of
raw float32 LE PCM (24 kHz mono). Concatenate chunks and play, or pipe to a PCM
player. `400` for an unknown backend/voice. `503` only when the request queue is
full: requests normally wait in a bounded FIFO for the single GPU slot rather
than being rejected.

## WS /v1/audio/stream
Persistent-session variant for a web client. Send one JSON message per utterance
(same shape as `POST /v1/audio/speech`). The server replies with:
- a `{"type":"start"}` text frame,
- N binary frames of raw float32 LE PCM (24 kHz mono),
- a `{"type":"end"}` text frame.

A malformed / unknown-voice / over-cap request gets `{"type":"error","detail":…}`
and the socket stays open for the next utterance. Same `_GpuQueue` admission and
GPU serialization as the HTTP route.

## GET /app/
A bundled browser demo client — voice dropdown (with descriptions), param
sliders, and an AudioWorklet PCM player driving the WebSocket. Reference /
local-test UI; same-origin, so no CORS needed.

## POST /shutdown
Signals the server process to exit (used by the REPL `:quit --stop`).
