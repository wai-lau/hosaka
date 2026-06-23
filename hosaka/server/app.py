import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from hosaka.chunking import split_fragments
from hosaka.config import LLM_MODEL, MAX_QUEUE
from hosaka.library import VoiceLibrary
from hosaka.schemas import SpeechRequest, VoiceInfo, clamp_params
from hosaka.server.engines.base import EngineRegistry

KOKORO_PRESETS = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bm_george",
]

# Short blurbs for the preset voices (prefix: a=American/b=British, f/m=gender).
KOKORO_DESC = {
    "af_heart": "American female, warm and friendly",
    "af_bella": "American female, bright and expressive",
    "af_nicole": "American female, soft and intimate",
    "af_sarah": "American female, clear and neutral",
    "am_adam": "American male, deep and steady",
    "am_michael": "American male, casual mid-range",
    "bf_emma": "British female, warm and refined",
    "bm_george": "British male, mature and measured",
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
    elif voice and library.path_for(voice) is None:
        # chatterbox: "" is the model's own default voice; anything else must
        # resolve to a reference clip in the library.
        return None, f"unknown chatterbox voice: {voice}"
    return engine, None


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
            try:
                while True:
                    item = await queue.get()
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
                # Hold the GPU slot until the worker thread has actually finished
                # -- even on client disconnect -- so the single-GPU serialization
                # invariant holds and no thread is orphaned.
                await asyncio.shield(fut)
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
        out += [
            VoiceInfo(
                id=e.id,
                backend="chatterbox",
                source=e.source,
                description=e.params.get("description", ""),
            ).model_dump()
            for e in library.list()
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
        fragments = split_fragments(req.input)
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
                fragments = split_fragments(req.input)
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

    return app
