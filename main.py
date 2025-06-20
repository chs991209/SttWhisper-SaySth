import base64
import io
import os
import logging
import asyncio

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)


# Input data model
class AudioPayload(BaseModel):
    data: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = WhisperModel("small", device="cuda", compute_type="float16")
    logging.info(f"Loading faster-whisper small model on laptop...")

    # except Exception as e:  # CUDA가 사용 불가할 때 CPU run으로 fallback합니다.
    #     print(f"CUDA not available, defaulting to CPU. Reason: {e}")
    #     model = WhisperModel("base", device="cpu")

    max_workers = min(4, (os.cpu_count() or 2))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    app.state.whisper_model = model
    app.state.executor = executor

    try:
        yield
    finally:
        executor.shutdown()
        logging.info("Executor shut down.")


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For public testing; restrict domains for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/stt")
async def transcribe_audio(request: Request):
    model: WhisperModel = request.app.state.whisper_model
    executor: ThreadPoolExecutor = request.app.state.executor

    try:
        audio_base64 = (await request.json())["data"]

        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")

        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    audio_file = io.BytesIO(audio_bytes)

    def _transcribe():
        segments, info = model.transcribe(audio_file)
        return " ".join([seg.text for seg in segments if seg.text])

    transcript = await asyncio.get_running_loop().run_in_executor(executor, _transcribe)

    return JSONResponse(content={"text": transcript})

