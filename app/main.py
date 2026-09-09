from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import documents, facts, jobs, query, relationships
from app.db import init_db
from app.jobs import start_worker

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_worker()
    yield


app = FastAPI(title="Fact Knowledge Layer", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# API routes are registered before the static mount below, so they always take priority -
# the mount only serves what none of these routes matched.
app.include_router(documents.router)
app.include_router(jobs.router)
app.include_router(facts.router)
app.include_router(relationships.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
