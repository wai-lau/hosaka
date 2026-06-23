import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from hosaka.chunking import split_fragments
from hosaka.config import LLM_MODEL
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


def create_app(registry: EngineRegistry, library: VoiceLibrary, do_warmup: bool = True) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if do_warmup:
            stop_llm()
            registry.warmup_all()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.registry = registry
    app.state.library = library

    gpu_lock = asyncio.Semaphore(1)

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

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        try:
            engine = registry.get(req.backend)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"unknown backend: {req.backend}") from None
        # Validate the voice *before* the 200 stream starts. A bad voice that
        # only fails inside engine.stream() surfaces as a mid-body connection
        # close (the status line is already sent), which the client can't tell
        # apart from a crash. Reject it cleanly here instead.
        if req.backend == "kokoro":
            if req.voice not in KOKORO_PRESETS:
                raise HTTPException(status_code=400, detail=f"unknown kokoro voice: {req.voice}")
        elif req.voice and library.path_for(req.voice) is None:
            # chatterbox: "" is the model's own default voice; anything else
            # must resolve to a reference clip in the library.
            raise HTTPException(status_code=400, detail=f"unknown chatterbox voice: {req.voice}")
        # Acquire the single-GPU slot here, atomically with the busy check
        # (no await between them), so a second concurrent request reliably
        # gets 503 instead of racing. The lock is held until gen() finishes,
        # including the worker thread, then released in gen()'s finally.
        if gpu_lock.locked():
            raise HTTPException(status_code=503, detail="busy")
        await gpu_lock.acquire()

        params = clamp_params(req.params).model_dump()
        fragments = split_fragments(req.input)

        async def gen():
            loop = asyncio.get_running_loop()
            try:
                for frag in fragments:
                    queue: asyncio.Queue = asyncio.Queue()

                    def produce(fragment=frag, queue=queue):
                        try:
                            for chunk in engine.stream(fragment, req.voice, params):
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
                                    # GPU context is dead for the whole process;
                                    # exit so the next launch starts clean.
                                    _do_shutdown()
                                raise item
                            yield item
                    finally:
                        # Hold the GPU slot until the worker thread has actually
                        # finished — even on client disconnect — so the single-GPU
                        # serialization invariant holds and no thread is orphaned.
                        await asyncio.shield(fut)
            finally:
                gpu_lock.release()

        return StreamingResponse(
            gen(),
            media_type="application/octet-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    @app.post("/shutdown")
    def shutdown():
        _do_shutdown()
        return {"status": "stopping"}

    return app
