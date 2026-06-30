from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, health

settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Backend for the Hybrid Token Routing Agent",
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Hybrid Token Router Backend Running"}
