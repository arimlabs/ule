from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from app.db.init_db import init_db_on_fastapi_startup
from app.db.init_redis import init_redis, close_redis
from app.api.v1.routes import router
from app.pages.routes import router as pages_router

# Import models package to auto-register all models
import app.models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database and Redis
    await init_db_on_fastapi_startup()
    await init_redis()
    yield
    # Shutdown: Close Redis connection
    await close_redis()


app = FastAPI(lifespan=lifespan)

# Include routers
app.include_router(router=router)
app.include_router(router=pages_router)