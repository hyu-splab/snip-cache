from typing import Optional


class Policy:
    def __init__(
        self,
        min_samples: int = 3,
        max_retry: int = 1,
        auto_save_path: Optional[str] = None,
        reject_on_ambiguity: bool = True,
        generation_enable: bool = True,
        background_generation: bool = False,
        response_threshold: float = 0.8,
        validation_list: list = ["atd", "aad", "extract_code", "response_code"],
    ):
        self.min_samples = min_samples
        self.max_retry = max_retry
        self.auto_save_path = auto_save_path
        self.reject_on_ambiguity = reject_on_ambiguity
        self.generation_enable = generation_enable
        self.background_generation = background_generation
        self.response_threshold = response_threshold
        self.validation_list = validation_list

    def is_skip_validation(self, validation_type: str) -> bool:
        return validation_type not in self.validation_list
