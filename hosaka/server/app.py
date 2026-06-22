import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from hosaka.config import LLM_MODEL
from hosaka.chunking import split_fragments
from hosaka.schemas import SpeechRequest, VoiceInfo, clamp_params
from hosaka.server.engines.base import EngineRegistry
from hosaka.library import VoiceLibrary

KOKORO_PRESETS = [
    "af_heart", "af_bella", "af_nicole", "af_sarah", "am_adam",
    "am_michael", "bf_emma", "bm_george",
]


def stop_llm() -> None:
    try:
        subprocess.run(["ollama", "stop", LLM_MODEL],
                       capture_output=True, timeout=30, check=False)
    except Exception:
        pass   # best-effort


def _do_shutdown():
    os.kill(os.getpid(), signal.SIGTERM)


def create_app(registry: EngineRegistry, library: VoiceLibrary,
               do_warmup: bool = True) -> FastAPI:

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
        out = [VoiceInfo(id=p, backend="kokoro", source="preset").model_dump()
               for p in KOKORO_PRESETS]
        out += [VoiceInfo(id=e.id, backend="chatterbox", source=e.source).model_dump()
                for e in library.list()]
        return out

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        try:
            engine = registry.get(req.backend)
        except KeyError:
            raise HTTPException(status_code=400,
                                detail=f"unknown backend: {req.backend}")
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

                    def produce(fragment=frag):
                        try:
                            for chunk in engine.stream(fragment, req.voice, params):
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, chunk.tobytes())
                        except BaseException as exc:   # surface engine failures
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
                                raise item
                            yield item
                    finally:
                        # Hold the GPU slot until the worker thread has actually
                        # finished — even on client disconnect — so the single-GPU
                        # serialization invariant holds and no thread is orphaned.
                        await asyncio.shield(fut)
            finally:
                gpu_lock.release()

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-cache"})

    @app.post("/shutdown")
    def shutdown():
        _do_shutdown()
        return {"status": "stopping"}

    return app
