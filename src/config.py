from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel

SRC_PATH = Path(__file__).parent.resolve()
PROJECT_ROOT_PATH = SRC_PATH.parent
ASSETS_DIR_PATH = PROJECT_ROOT_PATH / "assets"
DOCS_DIR_PATH = ASSETS_DIR_PATH / "docs"


APPLICATION_NAME = "docu-master"

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml"],
    environments=True,
)


class UvicornConfig(BaseModel):
    host: str
    port: int


class RootConfig(BaseModel):
    uvicorn: UvicornConfig


CONFIGS = RootConfig.model_validate({k.lower(): v for k, v in settings.as_dict().items()})
