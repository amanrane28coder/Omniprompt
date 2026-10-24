from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .db import Base, engine

app = FastAPI(title="OmniPrompt")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
