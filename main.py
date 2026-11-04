from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)
