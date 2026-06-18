"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.exceptions import register_exception_handlers
from api.routes import health, predict


def create_app():
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    register_exception_handlers(app)
    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["forecasting"])

    return app


app = create_app()
