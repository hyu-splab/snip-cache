from abc import ABC, abstractmethod


class BaseFunctionHandler(ABC):

    @abstractmethod
    def get_spec(self, action_name: str) -> dict:
        """Return specific function spec for the given action."""
        pass

    @abstractmethod
    def get_specs(self) -> dict:
        """Return all function specs."""
        pass

    @abstractmethod
    def get_required_params(self, action_name: str) -> list[str]:
        """Return a list of required parameters for the given action."""
        pass

    def validate_required_params(self, action_name: str, arguments: dict) -> bool:
        """Check if all required params are provided and not None."""
        required = self.get_required_params(action_name)
        return all(arguments.get(param) is not None for param in required)
