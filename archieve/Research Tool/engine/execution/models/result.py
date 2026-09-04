# engine/execution/models/result.py

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ExecutionResult:
    """Standardized result produced by executing one node."""

    node_id: str
    status: str
    output: Any = None
    error: Optional[str] = None