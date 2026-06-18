"""API exception types and handlers."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


UNPROCESSABLE_STATUS_CODE = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


class ApiError(Exception):
    """Application-level API error with a stable response shape."""

    def __init__(self, status_code, code, message, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _error_response(status_code, code, message, details=None):
    content = {
        "error": {
            "code": code,
            "message": message,
        }
    }

    if details is not None:
        content["error"]["details"] = details

    return JSONResponse(status_code=status_code, content=content)


def register_exception_handlers(app: FastAPI):
    """Register API exception handlers."""

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        return _error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="ARTIFACT_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return _error_response(
            status_code=UNPROCESSABLE_STATUS_CODE,
            code="VALIDATION_FAILED",
            message=str(exc),
        )
