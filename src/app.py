import uvicorn
from fastapi import FastAPI

from .config import get_config_model

app = FastAPI()


def main() -> None:
    cfg = get_config_model().uvicorn
    uvicorn.run(app, **cfg.model_dump())


if __name__ == "__main__":
    main()
