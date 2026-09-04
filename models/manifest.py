from dataclasses import dataclass
from typing import Any


@dataclass
class Manifest:
    status: str
    params: dict
    signature: str
    execution_time: float | None = None
    error: str | None = None