# SaySth-STT

FastAPI 기반의 음성 인식(STT) 서버로, faster-whisper을 사용하여 오디오를 텍스트로 변환하고 Agentic AI 서버로 결과를 전송합니다.

## 기능

- 오디오 파일을 base64로 받아서 텍스트로 변환
- faster-whisper 모델 사용 (CUDA/CPU 지원)
- Agentic AI 서버로 transcription 결과 자동 전송
- Jetson Orin 환경 지원

## Jetson Orin 설치 가이드

### 1. 사전 요구사항

- JetPack 5.x 이상
- Python 3.8 이상
- CUDA 12.6

### 2. PyTorch 설치 (필수)

Jetson Orin은 ARM64 아키텍처이므로 일반 PyPI의 PyTorch를 설치할 수 없습니다. NVIDIA의 Jetson 전용 빌드를 사용해야 합니다.

#### 방법 A: NVIDIA 포럼에서 wheel 파일 다운로드 (권장)

1. JetPack 버전 확인:
```bash
cat /etc/nv_tegra_release
```

2. NVIDIA 포럼에서 해당 JetPack 버전에 맞는 PyTorch wheel 파일 찾기:
   - https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

3. wheel 파일 다운로드 및 설치:
```bash
# 예시 (JetPack 5.x, Python 3.8 기준)
wget https://nvidia.box.com/shared/static/[파일명].whl -O torch-[version]-cp38-cp38-linux_aarch64.whl
pip3 install torch-[version]-cp38-cp38-linux_aarch64.whl
```

#### 방법 B: 자동 설치 스크립트 사용

```bash
wget https://raw.githubusercontent.com/dusty-nv/jetson-containers/master/install_pytorch.sh
bash install_pytorch.sh
```

#### PyTorch 설치 확인

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. 나머지 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택사항)

`.env` 파일 생성:
```bash
echo 'AGENTIC_AI_SERVER_URL="http://127.0.0.1:8002/execute-voice-command"' > .env
```

또는 환경 변수로 직접 설정:
```bash
export AGENTIC_AI_SERVER_URL="http://127.0.0.1:8002/execute-voice-command"
```

### 5. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API 사용법

### POST /transcribe

오디오 파일을 base64로 인코딩하여 전송:

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"data": "base64_encoded_audio_data"}'
```

응답:
```json
{
  "transcriptionResult": "변환된 텍스트"
}
```

## 환경 변수

- `AGENTIC_AI_SERVER_URL`: Agentic AI 서버 URL (기본값: `http://127.0.0.1:8002/execute-voice-command`)

## 문제 해결

### CUDA를 사용할 수 없는 경우

코드는 자동으로 CPU로 fallback합니다. 로그를 확인하여 어떤 디바이스를 사용하는지 확인하세요.

### PyTorch 설치 실패

- JetPack 버전과 Python 버전을 확인하세요
- NVIDIA 포럼에서 정확한 wheel 파일을 찾아 설치하세요
- `pip3 install --upgrade pip`로 pip를 최신 버전으로 업그레이드하세요

## 참고 자료

- [NVIDIA Jetson PyTorch 설치 가이드](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [faster-whisper 문서](https://github.com/guillaumekln/faster-whisper)
