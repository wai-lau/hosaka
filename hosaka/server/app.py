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
        if gpu_lock.locked():
            raise HTTPException(status_code=503, detail="busy")

        params = clamp_params(req.params).model_dump()
        fragments = split_fragments(req.input)

        async def gen():
            async with gpu_lock:
                loop = asyncio.get_running_loop()
                for frag in fragments:
                    queue: asyncio.Queue = asyncio.Queue()

                    def produce(fragment=frag):
                        try:
                            for chunk in engine.stream(fragment, req.voice, params):
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, chunk.tobytes())
                        finally:
                            loop.call_soon_threadsafe(queue.put_nowait, None)

                    loop.run_in_executor(None, produce)
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        yield item

        return StreamingResponse(gen(), media_type="application/octet-stream",
                                 headers={"X-Accel-Buffering": "no",
                                          "Cache-Control": "no-cache"})

    @app.post("/shutdown")
    def shutdown():
        _do_shutdown()
        return {"status": "stopping"}

    return app
