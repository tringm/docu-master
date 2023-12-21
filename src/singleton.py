from typing import Any, ClassVar


class ThreadUnsafeSingletonMeta(type):
    _instances: ClassVar[dict] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
