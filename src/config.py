from dynaconf import Dynaconf
from pydantic import BaseModel

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


def get_config_model() -> RootConfig:
    # dynaconf parser capitalizes root keys
    settings_dict = {k.lower(): v for k, v in settings.as_dict().items()}
    return RootConfig.model_validate(settings_dict)
