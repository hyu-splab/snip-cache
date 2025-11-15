import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMClientBase(ABC):
    """
    Abstract base class for LLM clients (OpenAI / Azure).
    Provides unified system function execution.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_response(
        self,
        input: str,
        format: str = "json_object",
        temperature: float = 0.1,
        tools: list[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
    ) -> Dict[str, Any]:
        """Send a prompt and return parsed LLM response (including function calls)."""
        pass

    @abstractmethod
    def reconnect(self) -> None:
        """Reconnect or reinitialize the client."""
        pass
