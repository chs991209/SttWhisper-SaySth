import os
import tempfile
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

temp_dir = tempfile.mkdtemp()
os.makedirs(temp_dir, exist_ok=True)
print(f"Temp dir: {temp_dir}")
device = "cpu"
print(f"Using device: {device}")

model = whisper.load_model("tiny").to(device)  # "tiny"나 "small"도 가능
app = FastAPI()


@app.post("/stt")
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
        # return JSONResponse(content={"text": stt_result["text"]}, status_code=200)
        return JSONResponse(
            content={"text": "전북대 근처의 중국집 알려줘"}, status_code=200
        )  # 테스트용, 테스트 끝나면 위 코드로 변경할 것
    except Exception as e:
        print(e)
    finally:
        os.remove(file_name)


@app.get("/")
async def read_index():
    return FileResponse("index.html")


if __name__ == "__main__":
    os.makedirs(temp_dir, exist_ok=True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
