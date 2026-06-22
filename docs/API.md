# hosaka HTTP API

Base URL: `http://127.0.0.1:8123`. Local only. PCM is float32 LE, 24 kHz, mono.

## GET /health
Returns `{"status": "ok"}` once both models are loaded and warmed.

## GET /v1/voices
Returns a list of `{id, backend, source}`. `backend` is `kokoro` (presets) or
`chatterbox` (cloneable library clips). `source` is `preset|recording|bake|kokoro`.

## POST /v1/audio/speech
Request body:
```json
{
  "input": "text to speak",
  "backend": "kokoro | chatterbox",
  "voice": "af_heart | <library voice id>",
  "params": {"exaggeration": 0.5, "cfg_weight": 0.4,
             "temperature": 0.8, "speed": 1.0},
  "response_format": "pcm",
  "stream": true
}
```
Response: `200` with `Content-Type: application/octet-stream`, a chunked stream
of raw float32 LE PCM (24 kHz mono). Concatenate chunks and play, or feed a
PCM player directly. Returns `503` if the single GPU slot is busy, `400` for an
unknown backend.

Note for a future browser frontend: PCM + no CORS in v1. A browser client will
need a `wav`/`mp3` `response_format` and CORS headers added server-side.

## POST /shutdown
Signals the server process to exit (used by the REPL `:quit --stop`).
