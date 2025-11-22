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
import httpx

from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

# Agentic AI 서버 설정 (환경 변수에서 읽어오기, 기본값: http://127.0.0.1:8002/execute-voice-command)
AGENTIC_AI_SERVER_URL = os.getenv(
    "AGENTIC_AI_SERVER_URL",
    "http://127.0.0.1:8002/execute-voice-command"
)
logging.info(f"Agentic AI Server URL: {AGENTIC_AI_SERVER_URL}")


# Input data model
class AudioPayload(BaseModel):
    data: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Jetson Orin 환경에 맞게 모델 로딩 (CUDA 시도 후 CPU fallback)
    model = None
    device = "cuda"
    compute_type = "int8"  # Jetson Orin에서 메모리 효율적
    
    try:
        # 먼저 CUDA로 시도 (Jetson Orin은 CUDA 지원)
        model = WhisperModel("base", device="cuda", compute_type=compute_type)
        logging.info(f"Loading faster-whisper base model on Jetson Orin with CUDA (compute_type={compute_type})...")
    except Exception as e:
        logging.warning(f"CUDA not available or failed, defaulting to CPU. Reason: {e}")
        try:
            # CUDA 실패 시 CPU로 fallback
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel("base", device="cpu", compute_type=compute_type)
            logging.info(f"Loading faster-whisper base model on CPU (compute_type={compute_type})...")
        except Exception as e2:
            logging.error(f"Failed to load model on CPU as well: {e2}")
            # 최후의 수단: float32로 시도
            try:
                model = WhisperModel("base", device="cpu", compute_type="float32")
                logging.info("Loading faster-whisper base model on CPU with float32...")
            except Exception as e3:
                logging.error(f"Failed to load model: {e3}")
                raise
    
    if model is None:
        raise RuntimeError("Failed to initialize Whisper model")

    max_workers = min(4, (os.cpu_count() or 2))
    executor = ThreadPoolExecutor(max_workers=max_workers)
    app.state.whisper_model = model
    app.state.executor = executor
    app.state.device = device

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


@app.post("/transcribe")
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

    # Agentic AI 서버로 transcription 결과 전송
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                AGENTIC_AI_SERVER_URL,
                json={"text": transcript},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logging.info(f"Successfully sent transcription to Agentic AI server: {transcript}")
    except httpx.HTTPError as e:
        logging.error(f"Failed to send transcription to Agentic AI server: {e}")
        # Agentic AI 서버로 전송 실패해도 transcription 결과는 반환
    except Exception as e:
        logging.error(f"Unexpected error while sending to Agentic AI server: {e}")

    return JSONResponse(content={"transcriptionResult": transcript})

