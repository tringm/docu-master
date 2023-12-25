from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel

SRC_PATH = Path(__file__).parent.resolve()
PROJECT_ROOT_PATH = SRC_PATH.parent
MODEL_DIR_PATH = PROJECT_ROOT_PATH / "models"

APPLICATION_NAME = "docu-master"

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml"],
    environments=True,
)


class UvicornConfig(BaseModel):
    host: str
    port: int


class ChromaDBConfig(BaseModel):
    database: str
    distance_score_threshold: float
    client_configs: dict


class LLMConfig(BaseModel):
    llm_name: str
    llm_configs: dict
    prompt_configs: dict


class DocsConfig(BaseModel):
    chunk_capacity: int | tuple[int, int]


class RootConfig(BaseModel):
    uvicorn: UvicornConfig
    chromadb: ChromaDBConfig
    llm: LLMConfig
    docs: DocsConfig
    log_level: str


CONFIGS = RootConfig.model_validate({k.lower(): v for k, v in settings.as_dict().items()})
