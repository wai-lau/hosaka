import asyncio
import os
import signal
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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
        # Admit to the bounded queue, then wait our turn for the single GPU
        # slot. try_admit is synchronous (atomic cap-check + reserve, no await
        # between them) so concurrent callers can't both slip past a full cap;
        # only past the cap do we 503. The slot is held until gen() finishes --
        # including its worker thread -- then released in gen()'s finally. If
        # we're cancelled while still waiting for the slot, undo the admission.
        if not gpu_queue.try_admit():
            raise HTTPException(status_code=503, detail="busy")
        try:
            await gpu_queue.acquire()
        except BaseException:
            gpu_queue.undo_admit()
            raise

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
                gpu_queue.release()

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
