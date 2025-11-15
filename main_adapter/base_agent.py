from abc import ABC, abstractmethod


class BaseMainAgent(ABC):
    @abstractmethod
    def run(self, command: str) -> dict:
        """Process the input command and return the result as a dictionary."""
        pass
