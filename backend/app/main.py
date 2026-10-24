from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .db import Base, engine

app = FastAPI(title="OmniPrompt")
# Local dev only: allow any localhost/127.0.0.1 origin so both Vite dev
# (5173) and Vite preview (4173), on either hostname, can call the API.
# This is not for production deployment, where origins must be explicit.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
