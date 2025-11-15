import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .base import LLMClientBase


class OpenAIClient(LLMClientBase):
    def __init__(self, api_key: str = "", model: str = "gpt-4o"):
        super().__init__()
        self.logger = logging.getLogger("OpenAIClient")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_response(
        self,
        input: str,
        format: str = "json_object",
        temperature: float = 0.1,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
    ) -> Dict[str, Any]:
        response = self.client.responses.create(
            input=input,
            model=self.model,
            temperature=temperature,
        )  # type: ignore

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        output = response.output[0]

        # Text Response
        if output.type == "message":
            text_output = output.content[0].text  # type: ignore
            return {"type": "text", "text": text_output, "usage": usage}

        else:
            self.logger.warning(f"Unknown response type: {output.type}")
            return {"type": "unknown", "raw": response.output, "usage": usage}

    def reconnect(self) -> None:
        self.logger.info("Reconnecting OpenAI client...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
