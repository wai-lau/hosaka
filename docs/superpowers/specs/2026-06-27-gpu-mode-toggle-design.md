# GPU-mode toggle (homo/emo) button

**Date:** 2026-06-27
**Status:** Approved (design)
**Repos:** `hosaka` (home-box service) + `exec-fn` (droplet protected route/UI)

## Problem

The home-box RTX 5070 Ti (~16 GB VRAM) hosts two mutually exclusive GPU
workloads:

- **hosaka** TTS (`hosaka-server.service`, user unit, port 8123) -- Kokoro +
  Chatterbox resident in VRAM (~15 GB).
- **ollama** local LLM (`ollama.service`, system unit) for the emet dev box.

Only one fits at a time. Two CLI commands already arbitrate the GPU
(`~/bin/homo`, `~/bin/emo`):

- `homo` (hosaka mode): stop ollama, start hosaka-server.
- `emo` (emet mode): stop hosaka-server, start ollama.

Goal: flip the mode from a **button on a protected web route**, not just the
shell, so the GPU can be handed between TTS and the LLM remotely.

## Constraints discovered

1. **Self-kill.** `emo` stops `hosaka-server`. The toggle therefore CANNOT live
   inside hosaka-server: in `emo` mode that process is dead, so the button could
   never flip back. The switch must run in a separate, always-on host.
2. **No public ingress to the home box.** The home box (WSL2, behind NAT) is
   reachable only via the existing `hosaka-tunnel` reverse SSH tunnel that
   exposes home `:8123` on the droplet's docker-bridge gateway
   (`172.17.0.1:8123`). Any new control endpoint must be reverse-tunneled the
   same way.
3. **sudo password.** `homo`/`emo` run `sudo systemctl start/stop ollama`
   (ollama is a system unit). There is no NOPASSWD rule today (`sudo -n -l`
   empty), so a daemon cannot drive it. Must add a scoped NOPASSWD rule.
4. **Active remote users.** `emo` kills `hosaka-server`, cutting off in-flight
   and remote `wai-lau.net/hosaka` listeners mid-stream. exec-fn already tracks
   connected users as `_presence: set[WebSocket]` in `routes_tts.py`.

## Decisions

- **Control host:** a tiny always-on home-box service (`gpu-mode`) on its own
  port, reverse-tunneled to the droplet. Isolated -- it never touches the
  tuned hosaka GPU/`Semaphore(1)`/watchdog path. (Rejected: making
  hosaka-server stay up and unload/reload models -- too invasive to the
  carefully tuned resident-model path per hosaka CLAUDE.md.)
- **ollama permission:** scoped `/etc/sudoers.d` NOPASSWD rule for exactly
  `systemctl start ollama` and `systemctl stop ollama`, user `wai`. (Rejected:
  converting ollama to a user unit -- re-provisions the existing install.)
- **Active-user guard:** any action that STOPS hosaka-server (`emo` and `idle`)
  warns and requires confirmation when `_presence` is non-empty; `homo` (which
  starts hosaka) never needs it.
- **Auth:** the exec-fn route is owner-only (`require_auth` / `protected`
  router), NOT guest. Guests must never flip the GPU.

## Modes

Exactly four, mutually exclusive (XOR):

| Mode | Meaning | Derived from |
|------|---------|--------------|
| `homo` | hosaka TTS holds the GPU | hosaka-server up, ollama down |
| `emo`  | ollama LLM holds the GPU | ollama up, hosaka-server down |
| `idle` | GPU online but no payload running | both services down, home service reachable |
| `gone` | GPU unreachable or off | exec-fn cannot reach the home `gpu-mode` service |

`homo`/`emo`/`idle` are computed by the home `gpu-mode` service (it can see the
two units). `gone` is the exec-fn layer's label when the proxy call to the home
service fails -- the service itself never returns `gone`.

Both-services-up is NOT a mode: it is an invariant violation that cannot happen
because each toggle stops the other unit before starting the target. If ever
observed it is logged and re-read, never displayed.

## Architecture

### Component A -- `gpu-mode` service (hosaka repo, home box)

GPU-free, always-on. Runs under `.venv-dev` (fastapi + httpx, NO torch).

- **`scripts/gpu_mode.sh {homo|emo|idle|status}`** -- single source of truth for
  the systemctl logic.
  - `homo`: `sudo systemctl stop ollama` + `systemctl --user start hosaka-server`.
  - `emo`: `systemctl --user stop hosaka-server` + `sudo systemctl start ollama`.
  - `idle`: `systemctl --user stop hosaka-server` + `sudo systemctl stop ollama`
    (both down -> `idle` mode; frees the GPU entirely).
  - `status`: derive the mode from `systemctl is-active ollama` +
    `systemctl --user is-active hosaka-server` -> `homo` (hosaka up) | `emo`
    (ollama up) | `idle` (both down). Both up is an invariant violation (see
    Modes): logged + re-read, never returned. `gone` is not produced here.
  - Idempotent: already in target mode -> print + exit 0 (matches the current
    `~/bin/homo`/`emo` behaviour).
  - `~/bin/homo` and `~/bin/emo` are rewritten as one-line wrappers
    (`exec ~/src/hosaka/scripts/gpu_mode.sh homo|emo`) so CLI and service can
    never diverge.

- **`scripts/gpu_mode_server.py`** -- tiny FastAPI app, bound `127.0.0.1:8124`:
  - `GET  /mode` -> `{"mode": <status>}`
  - `POST /homo` -> run `gpu_mode.sh homo`, return `{"mode": <status>}`
  - `POST /emo`  -> run `gpu_mode.sh emo`,  return `{"mode": <status>}`
  - `POST /idle` -> run `gpu_mode.sh idle`, return `{"mode": <status>}`
  - Every route requires `Authorization: Bearer $GPU_MODE_TOKEN`
    (defense-in-depth on the tunnel hop; loopback + tunnel is the primary
    boundary). Missing/wrong token -> 401.
  - Subprocess failure -> 500 with a short stderr tail.

- **`scripts/start_gpu_mode.sh`** + **`gpu-mode.service`** (systemd user unit,
  linger-enabled like hosaka-server) -- execs uvicorn on the app.

- **Tunnel:** extend the existing `hosaka-tunnel` ssh invocation with a second
  forward `-R 127.0.0.1:8124:127.0.0.1:8124` (one tunnel, two forwards -- no new
  unit).

- **`deploy/gpu-mode.sudoers`** -- the scoped NOPASSWD rule:
  ```
  wai ALL=(root) NOPASSWD: /usr/bin/systemctl start ollama, /usr/bin/systemctl stop ollama
  ```
  Installed once, manually, as root (`visudo -c` validated, mode 0440 in
  `/etc/sudoers.d/`). Documented in README; not auto-applied.

### Component B -- exec-fn protected route + button (droplet)

Routes added in `routes_tts.py` (needs `_presence`), on the owner-only
`protected` router:

- **`GET /api/hosaka/mode`** -> proxy `http://$GPU_MODE_UPSTREAM/mode` with the
  Bearer token. Returns the home service's `homo`/`emo`/`idle`; on proxy failure
  (timeout / conn refused) returns `{"mode": "gone"}` (never 500; the UI greys
  the control).
- **`POST /api/hosaka/mode {action: "homo"|"emo"|"idle", force?: bool}`**:
  - `action in {"emo","idle"}` (both stop hosaka-server) and `len(_presence) > 0`
    and not `force` -> **409 `{"detail": "active_users", "count": n}`** (UI
    confirms, re-POSTs with `force: true`).
  - else proxy to `/homo` | `/emo` | `/idle` upstream, return `{"mode": <status>}`.

- **UI:** a 3-button segmented control in the `/hosaka` page header, rendered
  ONLY when `request.cookies["session"] == SESSION_TOKEN` (full-auth owner).
  Buttons left-to-right: **`emo`  `idle`  `homo`**.
  - **`gone`:** all three deactivated and grey (no current state to highlight).
  - **any other mode (`emo`/`idle`/`homo`):** the button matching the current
    mode = light background, dark text, unclickable (you are here). The other two
    = transparent background, light outline + light text, clickable (click to
    POST that `action` and switch).
  - A click whose `action` stops hosaka-server (`emo` or `idle`) while users are
    active -> confirm dialog ("N users streaming -- switch anyway?") using the
    409 count, then re-POST with `force: true`.
  - On success the returned `{mode}` re-renders which button is the filled/active
    one. The control polls `GET /api/hosaka/mode` (or refetches after each click)
    to stay current.

- **New env:** `GPU_MODE_UPSTREAM` (default `172.17.0.1:8124`), `GPU_MODE_TOKEN`
  (shared with the home service; provisioned in the droplet's env/secrets).

## Data flow

```
owner browser
  -> POST exec-fn /api/hosaka/mode {action:"emo"}        (owner cookie)
     -> exec-fn: len(_presence) > 0 ? -> 409 {count} -> browser confirm
        -> re-POST {action:"emo", force:true}
     -> exec-fn proxy POST 172.17.0.1:8124/emo            (Bearer token)
        -> tunnel -> home gpu-mode :8124
           -> gpu_mode.sh emo
              -> systemctl --user stop hosaka-server
              -> sudo systemctl start ollama   (NOPASSWD)
           -> status -> {"mode":"emo"}
     <- {"mode":"emo"} -> button updates
```
hosaka-server (8123) being down in `emo` mode is fine -- the toggle lives on
8124, which stays up across all modes.

## Error handling

| Condition | Behaviour |
|-----------|-----------|
| Home box / tunnel down | exec-fn `GET /api/hosaka/mode` -> `{"mode":"gone"}`; button disabled, "GPU unreachable or off" |
| `gpu_mode.sh` non-zero | 500 + short stderr tail; button shows error, mode re-fetched |
| `emo` with active users | 409 `{count}`; UI confirm -> re-POST `force:true` |
| Bad/absent Bearer token | 401 from gpu-mode service |
| Non-owner hits route | 401 from `require_auth` (route on `protected` router) |

## Security

- gpu-mode service: loopback-bound + reachable only via the reverse tunnel;
  Bearer shared-secret on every route.
- sudoers: NOPASSWD limited to two exact `systemctl` argvs for ollama only.
- exec-fn route: owner-only (`require_auth`); the button is not even rendered
  for guests.

## Testing

- **gpu-mode app** (`.venv-dev`, pytest): monkeypatch the subprocess runner ->
  assert `/mode` maps systemctl states to `homo|emo|idle` (and both-up ->
  re-read, never returned); `/homo`/`/emo`/`/idle` dispatch the right script arg;
  token gate returns 401 on bad/missing Bearer; subprocess failure surfaces 500.
  Matches hosaka's pure-logic `.venv-dev` pattern (no GPU import).
- **`gpu_mode.sh status`**: testable by stubbing `systemctl` on PATH; the
  homo/emo/idle branches are integration-only (documented, not unit-tested).
- **exec-fn route** (pytest): `action in {emo, idle}` with `_presence` non-empty
  -> 409 with count; `homo` never guards; `force` bypasses; proxy calls mocked
  via httpx transport; owner-only dependency rejects no-cookie requests.

## Out of scope

- No auto-switching / scheduling (a future cron could call `/homo` overnight).
- No multi-GPU or partial-VRAM coexistence.
- ollama stays a system unit (not converted to a user unit).
