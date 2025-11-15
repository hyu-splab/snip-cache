import ast
import json
import logging
from typing import Any, Dict, List

from client.base import LLMClientBase
from prompt.prompt_builder import (
    build_AAD_expand_prompt,
    build_AAD_fix_prompt,
    build_AAD_init_prompt,
    build_ATD_fix_prompt,
    build_ATD_init_prompt,
    build_ATD_prompt_expend,
    build_extract_code_prompt,
    build_extract_fix_prompt,
    build_response_code_prompt,
    build_response_fix_prompt,
)


class SnippetGenerator:
    """
    Stateless LLM helper that generates code snippets using an external LLM client.
    """

    def __init__(self, client: LLMClientBase):
        """
        Args:
            client: An external LLM client object implementing `get_response(input, format, temperature, mode)`
                    - Example: OpenAI, Azure, Gemini, or any custom client.
        """
        self.client = client
        self.logger = logging.getLogger("SnipLLM")
        self.code_token_usage = {}
        self.json_token_usage = {}
        self.text_token_usage = {}

    #########################################################################
    # Common LLM Call Helper
    #########################################################################
    def _run_llm_request(
        self, prompt: str, format: str = "json_object", temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Executes a single LLM request and safely parses the result.
        """
        try:
            response = self.client.get_response(
                input=prompt,
                format=format,
                temperature=temperature,
            )
            text = response.get("text", "")
            usage = response.get("usage", {})
            self.logger.debug(f"LLM Usage: {usage}")

            if format == "code":
                parsed = self.safe_code_parse(text)
                for key in usage:
                    if key not in self.code_token_usage:
                        self.code_token_usage[key] = 0
                    self.code_token_usage[key] += usage[key]
                return {"result": parsed, "usage": usage}
            elif format == "json_object":
                parsed = self.safe_json_parse(text)
                for key in usage:
                    if key not in self.json_token_usage:
                        self.json_token_usage[key] = 0
                    self.json_token_usage[key] += usage[key]
                return {"result": parsed, "usage": usage}
            else:
                for key in usage:
                    if key not in self.text_token_usage:
                        self.text_token_usage[key] = 0
                    self.text_token_usage[key] += usage[key]
                return {"result": text, "usage": usage}
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            return {"error": str(e)}

    #########################################################################
    # Safe Parsing
    #########################################################################
    def safe_json_parse(self, text: str) -> Any:
        """
        Safely parse JSON-like text. Falls back to literal_eval if JSON fails.
        """
        text = text.replace("```json", "").replace("```", "")
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                self.logger.warning(f"safe_parse failed for text: {text[:100]}")
                return {}

    def safe_code_parse(self, text: str) -> Any:
        """
        Safely parse code text. Removes markdown formatting.
        """
        return text.replace("```python", "").replace("```", "")

    # Action Trigger
    def generate_action_trigger_dictionary(
        self, action: str, spec: str, samples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        prompt = build_ATD_init_prompt(action, spec, samples)
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    def expand_action_trigger_dictionary(
        self,
        action: str,
        spec: str,
        samples: List[Dict[str, str]],
        existing_atd: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = build_ATD_prompt_expend(action, spec, samples, existing_atd)
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    def fix_action_trigger(self, action: str, commands: str) -> Dict[str, Any]:
        prompt = build_ATD_fix_prompt(action, commands)
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    # Argument Dictionary
    def generate_action_argument_dictionary(
        self, action: str, spec: str, samples: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        prompt = build_AAD_init_prompt(action, spec, samples)
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    def expand_aad_value(
        self,
        action: str,
        argument_name: str,
        normalized_value: str,
        function_spec: str,
        samples: list,
    ) -> Dict[str, Any]:
        prompt = build_AAD_expand_prompt(
            action, argument_name, normalized_value, function_spec, samples
        )
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    def fix_action_argument(
        self, action: str, spec: str, command: str, missing_pairs: Dict[str, str]
    ) -> Dict[str, Any]:
        prompt = build_AAD_fix_prompt(action, spec, command, missing_pairs)
        result = self._run_llm_request(prompt, format="json_object")
        return result.get("result", {})

    def generate_extract_code(
        self,
        action: str,
        spec: str,
        samples: List[Dict[str, str]],
        aad: Dict[str, Any],
    ) -> str:
        prompt = build_extract_code_prompt(action, spec, samples, aad)
        result = self._run_llm_request(prompt, format="code")
        return result.get("result", "")

    def fix_extract_code(
        self,
        action: str,
        spec: str | None,
        samples: list[dict],
        failed_commands: list[str],
        aad: dict,
        failed_summary: str,
        previous_code: str,
    ) -> str:
        """Fix extract code using LLM, guided by failed cases."""
        prompt = build_extract_fix_prompt(
            action, spec, samples, failed_commands, aad, failed_summary, previous_code
        )
        result = self._run_llm_request(prompt, format="text")
        return result.get("result", "") if isinstance(result, dict) else result

    def generate_response_code(
        self,
        action: str,
        spec: str,
        samples: List[Dict[str, str]],
    ) -> str:
        prompt = build_response_code_prompt(action, spec, samples)
        result = self._run_llm_request(prompt, format="code")
        return result.get("result", "")

    def fix_response_code(
        self,
        action: str,
        spec: str,
        samples: list[dict],
        failed_commands: list[str],
        failed_summary: str,
        previous_code: str,
    ) -> str:
        """Fix response code using LLM, guided by failed cases."""
        prompt = build_response_fix_prompt(
            action, spec, samples, failed_commands, failed_summary, previous_code
        )
        result = self._run_llm_request(prompt, format="text")
        return result.get("result", "") if isinstance(result, dict) else result
