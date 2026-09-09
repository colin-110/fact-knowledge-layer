from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, facts, jobs, query, relationships
from app.db import init_db
from app.jobs import start_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_worker()
    yield


app = FastAPI(title="Fact Knowledge Layer", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(jobs.router)
app.include_router(facts.router)
app.include_router(relationships.router)
app.include_router(query.router)


@app.get("/health")
def health():
    return {"status": "ok"}
