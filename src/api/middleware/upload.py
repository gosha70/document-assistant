import os
import logging
from fastapi import UploadFile, HTTPException

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


def validate_upload(file: UploadFile) -> None:
    """Validate an uploaded file for extension, size, and path safety."""
    settings = get_settings()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Path traversal protection
    basename = os.path.basename(file.filename)
    if basename != file.filename or ".." in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Extension check
    ext = basename.rsplit(".", 1)[-1].lower() if "." in basename else ""
    if ext not in settings.upload.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' is not allowed. Allowed: {settings.upload.allowed_extensions}",
        )

    # Size check (Content-Length header, if available)
    if file.size is not None:
        max_bytes = settings.upload.max_file_size_mb * 1024 * 1024
        if file.size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {settings.upload.max_file_size_mb} MB",
            )
