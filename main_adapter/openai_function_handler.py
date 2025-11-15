from .base_function_handler import BaseFunctionHandler
import logging


class OpenAIFunctionHandler(BaseFunctionHandler):
    """
    Base handler class that follows the OpenAI Function Calling specification.

    This class provides a standard interface for retrieving and validating
    function specifications compatible with OpenAI's tool-calling format.
    It does not define any specific function schema by default—subclasses
    must populate `self.specs` according to their own system's action set.

    Reference:
        https://platform.openai.com/docs/guides/function-calling
    """

    def __init__(self):
        super().__init__()
        self.specs = {}  # Subclasses will populate this
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_spec(self, action_name: str) -> dict:
        """Return the OpenAI-style function spec for the given action."""
        spec = self.specs.get(action_name)
        if not spec:
            raise ValueError(f"Unknown action: {action_name}")
        return spec

    def get_specs(self) -> dict:
        """Return the OpenAI-style function specs."""
        return self.specs

    def get_required_params(self, action_name: str) -> list[str]:
        """Extract required parameter names from the OpenAI function schema."""
        spec = self.get_spec(action_name)
        params = spec.get("parameters", {})
        required = params.get("required")
        if isinstance(required, list):
            return required
        else:
            self.logger.warning(
                f"No required parameters defined for action '{action_name}'."
            )
            return []
