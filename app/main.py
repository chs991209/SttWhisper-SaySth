from fastapi import FastAPI
from .config import lifespan
from .middleware import setup_middleware
from fastapi.staticfiles import StaticFiles
from .routes import stt_base64
#from .routes import index

app = FastAPI(lifespan=lifespan)

setup_middleware(app)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(stt_base64.router)
#app.include_router(index.router)