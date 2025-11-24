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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Agentic AI 서버 설정
AGENTIC_AI_SERVER_URL = os.getenv(
    "AGENTIC_AI_SERVER_URL",
    "http://127.0.0.1:8002/execute-voice-command"
)
logging.info(f"Target Agentic AI Server URL: {AGENTIC_AI_SERVER_URL}")


class AudioPayload(BaseModel):
    data: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Jetson Orin 최적화 모델 로딩 전략:
    1순위: CUDA + float16 (Orin Native 성능)
    2순위: CUDA + int8 (메모리 부족 시)
    3순위: CPU (최후의 수단)
    """
    model = None
    device = "cuda"
    compute_type = "float16"  # Jetson Orin은 float16이 가장 안정적이고 빠름
    model_size = "base"

    try:
        logging.info(f"Attempting to load '{model_size}' model on GPU (float16)...")
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        logging.info("✅ Success: Model loaded on Jetson Orin GPU (float16)")

    except Exception as e1:
        logging.warning(f"⚠️ float16 load failed: {e1}. Trying int8...")
        try:
            # 2순위: int8 시도
            compute_type = "int8"
            model = WhisperModel(model_size, device="cuda", compute_type="int8")
            logging.info("✅ Success: Model loaded on Jetson Orin GPU (int8)")

        except Exception as e2:
            logging.error(f"❌ GPU load failed: {e2}. Fallback to CPU...")
            try:
                # 3순위: CPU 시도
                device = "cpu"
                compute_type = "int8"  # CPU는 int8이 효율적
                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logging.warning("⚠️ Running on CPU. Performance will be limited.")

            except Exception as e3:
                logging.critical(f"🔥 Critical Error: Failed to load model on CPU: {e3}")
                raise RuntimeError("Could not initialize Whisper model.")

    # ThreadPool 설정 (Orin CPU 코어 수에 맞게 조절)
    max_workers = min(4, (os.cpu_count() or 2))
    executor = ThreadPoolExecutor(max_workers=max_workers)

    app.state.whisper_model = model
    app.state.executor = executor
    app.state.device = device

    yield  # 서버 실행 중

    # 종료 시 정리
    executor.shutdown()
    logging.info("Executor shut down.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/transcribe")
async def transcribe_audio(request: Request):
    model: WhisperModel = request.app.state.whisper_model
    executor: ThreadPoolExecutor = request.app.state.executor

    try:
        body = await request.json()
        audio_base64 = body.get("data")

        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")

        audio_bytes = base64.b64decode(audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # 메모리 내 파일 객체 생성
    audio_file = io.BytesIO(audio_bytes)

    def _transcribe():
        # beam_size=5는 정확도를 높이지만 속도를 약간 늦춤. 실시간성이 중요하면 1로 낮출 수 있음.
        segments, info = model.transcribe(audio_file, beam_size=5)
        return " ".join([seg.text for seg in segments])

    # 별도 스레드에서 추론 실행 (메인 루프 차단 방지)
    transcript = await asyncio.get_running_loop().run_in_executor(executor, _transcribe)

    logging.info(f"STT Result: {transcript}")

    # Agentic AI 서버로 결과 전송 (Fire-and-forget 방식이 아닌 비동기 대기)
    if transcript:
        asyncio.create_task(send_to_agent(transcript))

    return JSONResponse(content={"transcriptionResult": transcript})


async def send_to_agent(text: str):
    """결과를 에이전트 서버로 비동기 전송"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                AGENTIC_AI_SERVER_URL,
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            logging.info(f"Sent to Agent: {text}")
    except Exception as e:
        logging.error(f"Failed to send to Agent: {e}")