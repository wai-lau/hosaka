import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from hosaka.chunking import pause_ms, split_fragments
from hosaka.config import (
    CHATTERBOX_MAX_CHARS,
    FIRST_FRAGMENT_MAX_CHARS,
    FRAGMENT_GROWTH,
    GEN_TIMEOUT_S,
    LEXICON_PATH,
    LLM_MODEL,
    MAX_QUEUE,
    PIPER_VOICES,
    RVC_VOICES,
    SAMPLE_RATE,
)
from hosaka.lexicon import Lexicon
from hosaka.library import VoiceLibrary
from hosaka.normalize import normalize_times
from hosaka.schemas import SpeechRequest, VoiceInfo, clamp_params
from hosaka.server.engines.base import EngineRegistry

WEB_DIR = Path(__file__).resolve().parent.parent / "web"  # bundled browser client

# Curated to a single preset by choice -- the other ~28 Kokoro voices are not
# offered. "nicole" is the display name; KOKORO_ALIASES maps it to the af_nicole
# embedding.
KOKORO_PRESETS = ["nicole"]

KOKORO_DESC = {
    "nicole": "American female, soft and intimate",
}


def stop_llm() -> None:
    try:
        subprocess.run(["ollama", "stop", LLM_MODEL], capture_output=True, timeout=30, check=False)
    except Exception:
        pass  # best-effort


def _do_shutdown():
    os.kill(os.getpid(), signal.SIGTERM)


def _is_fatal_cuda(exc: BaseException) -> bool:
    """A device-side assert / CUDA error corrupts the context process-wide.

    Once it fires, no further GPU work can succeed until the process restarts,
    so the only safe response is to exit and let the next launch respawn clean.
    """
    s = str(exc)
    return "CUDA error" in s or "device-side assert" in s


def _resolve(registry: EngineRegistry, library: VoiceLibrary, backend: str, voice: str):
    """Resolve the engine for (backend, voice), or return (None, error string).

    Validate the voice *before* any stream starts: a bad voice that only fails
    inside engine.stream() surfaces as a mid-stream close the client can't tell
    apart from a crash. Reject it cleanly up front instead. Shared by the HTTP
    and WebSocket routes.
    """
    try:
        engine = registry.get(backend)
    except KeyError:
        return None, f"unknown backend: {backend}"
    if backend == "kokoro":
        if voice not in KOKORO_PRESETS:
            return None, f"unknown kokoro voice: {voice}"
    elif backend == "piper":
        if voice not in engine.voice_ids:
            return None, f"unknown piper voice: {voice}"
    elif backend == "rvc":
        if voice not in engine.voice_ids:
            return None, f"unknown rvc voice: {voice}"
    elif voice and library.path_for(voice) is None:
        # chatterbox: "" is the model's own default voice; anything else must
        # resolve to a reference clip in the library.
        return None, f"unknown chatterbox voice: {voice}"
    return engine, None


def _fragments_for(backend: str, text: str) -> list[str]:
    """Split text into synth fragments. The Chatterbox quality path delivers
    each fragment whole at RTF ~0.8, so it uses the ramping cap -- a small
    first fragment for fast first-audio, growing to stay gapless (see
    chunking.split_fragments). The realtime Kokoro path streams sub-fragment
    audio itself, so it keeps the plain sentence-based split."""
    if backend == "chatterbox":
        return split_fragments(
            text,
            first_max_chars=FIRST_FRAGMENT_MAX_CHARS,
            max_chars=CHATTERBOX_MAX_CHARS,
            growth=FRAGMENT_GROWTH,
        )
    return split_fragments(text)


async def _pcm_frames(engine, voice, params, fragments, gpu_queue):
    """Stream PCM byte chunks for the fragments, holding the single GPU slot for
    the whole stream and releasing it (in finally) once the worker thread is
    done -- even on cancellation / client disconnect. The caller must have
    already admitted + acquired the slot. Exits the process on a fatal CUDA
    error. Shared by the HTTP and WebSocket routes.
    """
    loop = asyncio.get_running_loop()
    try:
        for frag in fragments:
            ms = pause_ms(frag)
            if ms is not None:
                # A dash pause: emit real silence (float32 zeros) instead of
                # synthesizing. The engine never sees the sentinel, so RVC can't
                # hallucinate phonemes into it.
                yield b"\x00\x00\x00\x00" * (SAMPLE_RATE * ms // 1000)
                continue
            queue: asyncio.Queue = asyncio.Queue()

            def produce(fragment=frag, queue=queue):
                try:
                    for chunk in engine.stream(fragment, voice, params):
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.tobytes())
                except BaseException as exc:  # surface engine failures
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            fut = loop.run_in_executor(None, produce)
            wedged = False
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), GEN_TIMEOUT_S)
                    except TimeoutError:
                        # No PCM for GEN_TIMEOUT_S: the worker thread is stuck on a
                        # GPU call that can't be cancelled, so it holds the single
                        # GPU slot forever and every later request would hang
                        # behind it. A Python thread can't be killed; the only
                        # clean recovery is to exit and let systemd respawn. Break
                        # out (don't await the hung fut below) and shut down.
                        wedged = True
                        break
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        if _is_fatal_cuda(item):
                            # GPU context is dead for the whole process; exit so
                            # the next launch starts clean.
                            _do_shutdown()
                        raise item
                    yield item
            finally:
                if wedged:
                    _do_shutdown()
                else:
                    # Hold the GPU slot until the worker thread has actually
                    # finished -- even on client disconnect -- so the single-GPU
                    # serialization invariant holds and no thread is orphaned.
                    await asyncio.shield(fut)
            if wedged:
                return
    finally:
        gpu_queue.release()


class _GpuQueue:
    """Bounded FIFO admission in front of the single GPU slot.

    Replaces the old "503 the moment the slot is busy" with a wait queue:
    concurrent callers line up on the semaphore (asyncio wakes waiters in FIFO
    order) instead of being rejected outright. The queue is *bounded* so a
    backlog cannot grow unbounded memory / unbounded wait -- past the cap we
    still return 503.

    ``try_admit`` is deliberately synchronous: the cap check and the counter
    bump are one atomic step (no ``await`` between them), so two concurrent
    requests cannot both slip past a nearly-full cap. Only *after* admission do
    we ``await`` the actual GPU slot. The slot itself is still a Semaphore(1),
    so the single-GPU serialization invariant is unchanged -- exactly one
    request touches the GPU at a time.
    """

    def __init__(self, max_depth: int):
        self._max = max_depth
        self._slot = asyncio.Semaphore(1)
        self._depth = 0  # admitted = waiting + the one currently running

    def try_admit(self) -> bool:
        if self._depth >= self._max:
            return False
        self._depth += 1
        return True

    def undo_admit(self) -> None:
        # Give back a reserved spot when we never reach release() (e.g. the
        # caller is cancelled while still waiting for the slot).
        self._depth -= 1

    async def acquire(self) -> None:
        await self._slot.acquire()

    def release(self) -> None:
        self._slot.release()
        self._depth -= 1


def create_app(
    registry: EngineRegistry,
    library: VoiceLibrary,
    do_warmup: bool = True,
    max_queue: int = MAX_QUEUE,
) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if do_warmup:
            stop_llm()
            registry.warmup_all()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.registry = registry
    app.state.library = library

    gpu_queue = _GpuQueue(max_queue)
    lexicon = Lexicon(LEXICON_PATH)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/voices")
    def voices():
        out = [
            VoiceInfo(
                id=p, backend="kokoro", source="preset", description=KOKORO_DESC.get(p, "")
            ).model_dump()
            for p in KOKORO_PRESETS
        ]
        # Library clips used only as an RVC source (e.g. Charlie's Chatterbox
        # clone) are not standalone voices -- hide them from the listing.
        rvc_sources = {
            s["source"] for s in RVC_VOICES.values() if s.get("source_backend") == "chatterbox"
        }
        out += [
            VoiceInfo(
                id=e.id,
                backend="chatterbox",
                source=e.source,
                description=e.params.get("description", ""),
                cb=True,
            ).model_dump()
            for e in library.list()
            if e.id not in rvc_sources
        ]
        if registry.piper is not None:
            out += [
                VoiceInfo(
                    id=vid,
                    backend="piper",
                    source="piper",
                    description=PIPER_VOICES.get(vid, {}).get("description", ""),
                ).model_dump()
                for vid in registry.piper.voice_ids
            ]
        if registry.rvc is not None:
            out += [
                VoiceInfo(
                    id=vid,
                    backend="rvc",
                    source="rvc",
                    description=RVC_VOICES.get(vid, {}).get("description", ""),
                    cb=RVC_VOICES.get(vid, {}).get("source_backend") == "chatterbox",
                    cb_params=(
                        RVC_VOICES.get(vid, {}).get("source_params")
                        if RVC_VOICES.get(vid, {}).get("source_backend") == "chatterbox"
                        else None
                    ),
                    speed=RVC_VOICES.get(vid, {}).get("speed", 1.0),
                ).model_dump()
                for vid in registry.rvc.voice_ids
            ]
        return out

    async def _admit_or(busy):
        # Reserve a queue spot (atomic cap-check), then wait our turn for the
        # single GPU slot. Returns True once the slot is held; on a full queue
        # calls busy() and returns False. If cancelled while still waiting,
        # undoes the admission. The caller owns releasing the slot via the
        # _pcm_frames finally once it starts streaming.
        if not gpu_queue.try_admit():
            await busy()
            return False
        try:
            await gpu_queue.acquire()
        except BaseException:
            gpu_queue.undo_admit()
            raise
        return True

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        engine, err = _resolve(registry, library, req.backend, req.voice)
        if err:
            raise HTTPException(status_code=400, detail=err)

        async def busy():
            raise HTTPException(status_code=503, detail="busy")

        await _admit_or(busy)
        params = clamp_params(req.params).model_dump()
        fragments = _fragments_for(req.backend, lexicon.apply(normalize_times(req.input)))
        return StreamingResponse(
            _pcm_frames(engine, req.voice, params, fragments, gpu_queue),
            media_type="application/octet-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    @app.websocket("/v1/audio/stream")
    async def stream(ws: WebSocket):
        # Persistent session: each JSON message is one utterance. Reply with a
        # {"type":"start"} marker, then raw PCM binary frames, then
        # {"type":"end"}. A malformed/unknown/over-cap request gets a
        # {"type":"error"} and leaves the socket open for the next utterance.
        # Admission, queueing and GPU serialization are shared with HTTP.
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                if not isinstance(msg, dict):
                    await ws.send_json({"type": "error", "detail": "expected a JSON object"})
                    continue
                try:
                    req = SpeechRequest(**msg)
                except ValidationError as exc:
                    await ws.send_json({"type": "error", "detail": str(exc)})
                    continue
                engine, err = _resolve(registry, library, req.backend, req.voice)
                if err:
                    await ws.send_json({"type": "error", "detail": err})
                    continue

                async def busy():
                    await ws.send_json({"type": "error", "detail": "busy"})

                if not await _admit_or(busy):
                    continue
                params = clamp_params(req.params).model_dump()
                fragments = _fragments_for(req.backend, lexicon.apply(normalize_times(req.input)))
                await ws.send_json({"type": "start"})
                async for chunk in _pcm_frames(engine, req.voice, params, fragments, gpu_queue):
                    await ws.send_bytes(chunk)
                await ws.send_json({"type": "end"})
        except WebSocketDisconnect:
            pass

    @app.post("/shutdown")
    def shutdown():
        _do_shutdown()
        return {"status": "stopping"}

    # Serve the bundled demo client at /app/ (html=True -> /app/ -> index.html).
    # Guarded so a stripped deploy without the web dir still boots.
    if WEB_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")

    return app
