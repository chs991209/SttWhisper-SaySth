import os
import tempfile
import whisper
import torch
import base64
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

temp_dir = tempfile.mkdtemp()
os.makedirs(temp_dir, exist_ok=True)
print(f"Temp dir: {temp_dir}")
device = "cuda"
print(f"Using device: {device}")

model = whisper.load_model("turbo").to(device)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:5500"]처럼 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],  # "POST", "GET", "OPTIONS" 등 명시도 가능
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

"""
@app.post('/stt')
async def transcript(file: UploadFile = File(...)):
    if file.content_type.startswith("audio") is False:
        return PlainTextResponse("올바르지 않은 파일 형식입니다.", status_code=400)

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


@app.post("/stt_base64")
async def transcript_base64(request: Request):
    try:
        data = await request.json()
        audio_base64 = data.get("audio") or data.get("data")
        print("받은 base64 길이:", len(audio_base64))
        print("앞부분 샘플:", audio_base64[:100])

        if not audio_base64:
            return PlainTextResponse("오디오 데이터가 없습니다.", status_code=400)

        # base64 디코딩
        audio_data = base64.b64decode(audio_base64)

        with NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_data)
            temp.seek(0)
            file_name = temp.name

        stt_result = model.transcribe(file_name)
        logger.info(f"변환된 텍스트 (base64): {stt_result['text']}")
        os.remove(file_name)

        # 실제 결과로 교체하거나 디버깅용 샘플 텍스트 유지
        return JSONResponse(content={"text": stt_result["text"]}, status_code=200)

    except Exception as e:
        print(f"Error: {e}")
        return PlainTextResponse("서버 처리 중 오류 발생", status_code=500)

"""
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
"""

if __name__ == "__main__":
    os.makedirs(temp_dir, exist_ok=True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7666)
