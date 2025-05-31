import os
from contextlib import asynccontextmanager

import executor
import tempfile
import whisper
import torch
import base64
import logging
import asyncio
from io import BytesIO # 'IO' 대신 'io'로 수정
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from tempfile import NamedTemporaryFile, TemporaryDirectory

forced_temp_root = "C:\\temp"
os.makedirs(forced_temp_root, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TemporaryDirectory를 사용하여 애플리케이션 종료 시 자동으로 임시 디렉토리가 삭제
temp_dir_manager = tempfile.TemporaryDirectory(dir=forced_temp_root)
temp_dir = temp_dir_manager.name
os.makedirs(temp_dir, exist_ok=True)
logger.info(f"Temporary directory: {temp_dir}")

if torch.cuda.is_available():
    device = "cuda"
    logger.info("Using GPU")
else:
    device = "cpu"
    logger.warning("Using CPU")

model = whisper.load_model("turbo").to(device)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 도메인만 허용할 것
    allow_credentials=True,
    allow_methods=["*"],  # 사용되는 HTTP 메소드만 허용
    allow_headers=["*"],  # 필요한 헤더만 허용
)
app.mount("/static", StaticFiles(directory="static"), name="static")

"""
@app.post('/stt')
async def transcript(file: UploadFile = File(...)):
    if file.content_type.startswith("audio") is False:
        return PlainTextResponse("not allowed file type.", status_code=400)

    file_name = None

    with NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp.seek(0)
        file_name = temp.name
    try:
        stt_result = model.transcribe(file_name)
        return JSONResponse(content={"text": stt_result["text"]}, status_code=200)
        #return JSONResponse(content={"text": "전북대 근처의 중국집 알려줘"}, status_code=200)
    except Exception as e:
        print(e)
    finally:
        os.remove(file_name)
"""

thread_pool_executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application started")
    yield # 이 시점에서 애플리케이션이 요청을 받기 시작
    logger.info("application ended")
    temp_dir_manager.cleanup()
    logger.info(f"Temporary directory '{temp_dir}'cleaned")
    thread_pool_executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor ended")

app = FastAPI(lifespan=lifespan)

@app.post("/stt_base64")
async def transcript_base64(request: Request):
    try:
        data = await request.json()
        audio_base64 = data.get("audio") or data.get("data")
        if audio_base64:
            logger.info(f"base64 audio data length: {len(audio_base64)} byte")
            # logger.debug(f"Base64 data sample (for debug): {audio_base64[:100]}")
        else :
            logger.warning("there is no audio data")
            raise HTTPException(status_code=400, detail="there is no audio data")

        # base64 decode
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Base64 decode failed: {e}")
            raise HTTPException(status_code=400, detail="invalid base64 audio data")



        with NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir) as temp_audio_file:
            temp_audio_file.write(audio_data)
            temp_audio_file_path = temp_audio_file.name
            logger.info(f"audio data saved temporary: {temp_audio_file_path}")

            # --- STT 모델 추론 비동기 처리: 변경된 스레드 풀 변수 이름 사용 ---
        try:
            stt_result = await asyncio.get_event_loop().run_in_executor(
                thread_pool_executor, model.transcribe, temp_audio_file_path
                )
        finally:
            # 임시 파일 삭제
            os.remove(temp_audio_file_path)

        logger.info(f"audio translate complete: '{stt_result['text'][:10]}' (10 characters)")

        return JSONResponse(content={"text": stt_result["text"]}, status_code=200) # 프론트엔드에게 다시 전달, 수정할 것

    except HTTPException as he:
        # FastAPI의 HTTPException은 직접 반환하여 클라이언트에게 적절한 상태 코드를 전달
        raise he
    except Exception as e:

        logger.error(f"서버 처리 중 예기치 않은 오류 발생: {e}", exc_info=True)  # exc_info=True로 스택 트레이스 포함
        raise HTTPException(status_code=500, detail="서버 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") # 클라이언트에게 오류 메시지 전달

"""
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
"""

if __name__ == "__main__":
    os.makedirs(temp_dir, exist_ok=True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7666)
