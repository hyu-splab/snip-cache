import logging

from main_adapter.openai_function_handler import OpenAIFunctionHandler


class MockFunctionHandler(OpenAIFunctionHandler):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("FunctionHandler")
        self.specs = {
            "activate": {
                "type": "function",
                "name": "activate",
                "description": "Turn on a device (e.g., tv, lights) or play music. Optionally specify a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "The target name to activate or play. ('tv', 'lights', or 'music')"
                                "Any other value should be treated as invalid."
                            ),
                        },
                        "location_name": {
                            "type": "string",
                            "description": (
                                "The location of the target, 'bedroom', 'kitchen', or 'washroom'. "
                                "If no location is applicable, leave it empty or null."
                            ),
                        },
                    },
                    "required": ["object_name"],
                    "additionalProperties": False,
                },
            },
            "deactivate": {
                "type": "function",
                "name": "deactivate",
                "description": "Turn off a device (e.g., tv, lights) or stop/pause music. Optionally specify a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "The target name to deactivate or stop. ('tv', 'lights', or 'music')"
                                "Any other value should be treated as invalid."
                            ),
                        },
                        "location_name": {
                            "type": "string",
                            "description": (
                                "The location of the target, 'bedroom', 'kitchen', or 'washroom'. "
                                "If no location is applicable, leave it empty or null."
                                "Any other value should be treated as invalid."
                            ),
                        },
                    },
                    "required": ["object_name"],
                    "additionalProperties": False,
                },
            },
            "increase": {
                "type": "function",
                "name": "increase",
                "description": "Increase a controllable property such as heat or volume. Optionally specify a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "The target name to increase. 'heat' or 'volume'. "
                                "Any other value should be treated as invalid."
                            ),
                        },
                        "location_name": {
                            "type": "string",
                            "description": (
                                "The location of the target, 'bedroom', 'kitchen', or 'washroom'. "
                                "If no location is applicable, leave it empty or null."
                                "Any other value should be treated as invalid."
                            ),
                        },
                    },
                    "required": ["object_name"],
                    "additionalProperties": False,
                },
            },
            "decrease": {
                "type": "function",
                "name": "decrease",
                "description": "Decrease a controllable property such as heat or volume. Optionally specify a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "The target name to decrease. 'heat' or 'volume'. "
                                "Any other value should be treated as invalid."
                            ),
                        },
                        "location_name": {
                            "type": "string",
                            "description": (
                                "The location of the target, 'bedroom', 'kitchen', or 'washroom'. "
                                "If no location is applicable, leave it empty or null."
                                "Any other value should be treated as invalid."
                            ),
                        },
                    },
                    "required": ["object_name"],
                    "additionalProperties": False,
                },
            },
            "bring": {
                "type": "function",
                "name": "bring",
                "description": "Bring a specific object to the user, such as a newspaper, juice, socks, or shoes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": (
                                "The name of the object to bring. ('newspaper', 'juice', 'socks', or 'shoes')"
                                "Any other value should be treated as invalid."
                                "Any other value should be treated as invalid."
                            ),
                        }
                    },
                    "required": ["object_name"],
                    "additionalProperties": False,
                },
            },
            "change_language": {
                "type": "function",
                "name": "change_language",
                "description": (
                    "Change the system language to a specified one, such as Chinese, Korean, English, or German. "
                    "If no language is specified, it remains None."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "language_name": {
                            "description": (
                                "The target language to change to. ('Chinese', 'Korean', 'English', or 'German')"
                                "If no language is specified, use an empty or null value."
                                "Any other value should be treated as invalid."
                            ),
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        }
                    },
                    "additionalProperties": False,
                },
            },
        }

    ##############################################################################################
    # Functions (fluent-speech-corpus)
    ##############################################################################################

    def activate(self, object_name: str, location_name: str = None):  # type: ignore
        """
        This action allows you to activate either a 'tv' or 'lights', or to play 'music.'
        Because these objects may exist in various locations, a location_name is also required.
        However, if no location is applicable, location_name may be empty (None).

        Parameters:
        - object_name (str): The target name to activate or play. Must be one of 'tv', 'lights', or 'music.'
        - location_name (str): The location of the target, which should be one of 'bedroom', 'kitchen', 'washroom.' or None. In cases where no location is needed, this can be an empty value.

        Returns:
        - bool: True indicates that the operation was successful.
        """
        self.logger.info(f"function_call: activate({object_name})")
        if object_name and object_name.lower() in ["tv", "lights", "music"]:
            if location_name:
                if location_name.lower() in ["bedroom", "kitchen", "washroom"]:
                    return True
                else:
                    return False
            return True
        else:
            return False

    def deactivate(self, object_name: str, location_name: str = None):  # type: ignore
        """
        This action allows you to deactivate either a 'tv' or 'lights,' or to pause/stop 'music.'
        Because these objects may exist in various locations, a location_name is also required.
        However, if no location is applicable, location_name may be empty (None).

        Parameters:
        - object_name (str): The target name to deactivate or stop. Must be one of 'tv', 'lights', or 'music.' Any other value results in an error.
        - location_name (str): The location of the target, which should be one of 'bedroom', 'kitchen', 'washroom.' or None. In cases where no location is needed, this can be an empty value.

        Returns:
        - bool: True indicates that the operation was successful.

        """
        self.logger.debug(f"function_call: deactivate({object_name})")
        if object_name and object_name.lower() in ["tv", "lights", "music"]:
            if location_name:
                if location_name.lower() in ["bedroom", "kitchen", "washroom"]:
                    return True
                else:
                    return False
            return True
        else:
            return False

    def increase(self, object_name: str, location_name: str = None):  # type: ignore
        """
        This action allows you to increase either 'heat' or 'volume.'
        Because these objects may exist in various locations, a location_name is also required.
        However, if no location is applicable, location_name may be empty (None).

        Parameters:
        - object_name (str): The target name to increase. Must be either 'heat' or 'volume.' Any other value results in an error.
        - location_name (str): The location of the target, which should be one of 'bedroom', 'kitchen', 'washroom.' or None. In cases where no location is needed, this can be an empty value.

        Returns:
        - bool: True indicates that the operation was successful.
        """
        self.logger.debug(f"function_call: increase({object_name})")
        if object_name and object_name.lower() in ["heat", "volume"]:
            if location_name:
                if location_name.lower() in ["bedroom", "kitchen", "washroom"]:
                    return True
                else:
                    return False
            return True
        else:
            return False

    def decrease(self, object_name: str, location_name: str = None):  # type: ignore
        """
        This action allows you to decrease either 'heat' or 'volume.'
        Because these objects may exist in various locations, a location_name is also required.
        However, if no location is applicable, location_name may be empty (None).

        Parameters:
        - object_name (str): The target name to decrease. Must be either 'heat' or 'volume.' Any other value results in an error.
        - location_name (str): The location of the target, which should be one of 'bedroom', 'kitchen', 'washroom.' or None. In cases where no location is needed, this can be an empty value.

        Returns:
        - bool: True indicates that the operation was successful.
        """
        self.logger.debug(f"function_call: decrease({object_name})")
        if object_name and object_name.lower() in ["heat", "volume"]:
            if location_name:
                if location_name.lower() in ["bedroom", "kitchen", "washroom"]:
                    return True
                else:
                    return False
            return True
        else:
            return False

    def bring(self, object_name: str):
        """
        This action brings a specific object to the user. The object_name must be one of 'newspaper', 'juice', 'socks', or 'shoes'. Any other value should be treated as an error.

        Parameters:
        - object_name (str): The name of the object to bring. Must be 'newspaper', 'juice', 'socks', or 'shoes'

        Returns:
        - bool: True indicates that the operation was successful.
        """
        self.logger.debug(f"function_call: bring({object_name})")
        if object_name.lower() in ["newspaper", "juice", "socks", "shoes"]:
            return True
        else:
            return False

    def change_language(self, language_name: str = None):  # type: ignore
        """
        This action changes or set the language. The language_name can be one of 'Chinese', 'Korean', 'English', 'German', or an empty (None) value.
        If the user does not specify a language, it should remain empty (None).

        Parameters:
        - language_name (str): The target language to change to. If no language is specified, use an empty (None) value.

        Returns:
        - bool: True indicates that the operation was successful.
        """
        self.logger.debug(f"function_call: change_language({language_name})")

        if language_name == None or language_name.lower() in [
            "chinese",
            "korean",
            "english",
            "german",
        ]:
            return True
        else:
            return False
