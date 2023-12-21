import shutil
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import CONFIGS, DOCS_DIR_PATH
from .logging import logger

app = FastAPI()


class PATHS:
    upload_file = "/upload/"


@app.middleware("http")
async def handling_exception(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(status_code=500, content={"error": e.__class__.__name__, "messages": e.args})


@app.post(path=PATHS.upload_file)
async def create_upload_file(file: UploadFile) -> Response:
    logger.info(f"uploading {file.filename}")
    with (DOCS_DIR_PATH / file.filename).open(mode="wb") as f:
        shutil.copyfileobj(file.file, f)
    return HTMLResponse(status_code=204)


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()
