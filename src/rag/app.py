from contextlib import asynccontextmanager

from fastapi import FastAPI

# Routes
from rag.api.v1.textbook import router as textbook_router
from rag.db.session import engine, init_db


# Upon application startup, initialize the database
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield  # Yield control back to FastAPI to continue with the application lifecycle
    await engine.dispose()  # Dispose of the engine when the app shuts down


app = FastAPI(lifespan=lifespan)

app.include_router(textbook_router)
