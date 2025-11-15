import csv
import logging
import random
from typing import Any, Dict, List, Optional


class DemoCommandLoader:
    def __init__(self, command_csv=None, function_handler=None, file_name=""):
        self.logger = logging.getLogger("CommandLoader")
        self.records = []
        self.commands = []
        self.commands_info = []
        self.expected_results = {}
        self.function_handler = function_handler
        self.summary = {}
        self.file_name = file_name
        if command_csv:
            self.load_commands_from_csv(command_csv)

    def load_commands_from_csv(self, csv_file):
        """Load commands and their expected results from a CSV file."""
        # transcription,  action, object, location
        self.commands = []
        self.commands_info = []
        self.transcription = []
        self.action = []
        self.object = []
        self.location = []
        self.expected_results = {}
        # for mocking
        self.mocked_responses = {}
        with open(csv_file, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                command = row.get("transcription", None)
                action = row.get("action", None)
                if action:
                    action = action.replace(" ", "_")
                object_name = row.get("object", None)
                location_name = row.get("location", None)
                arguments = {}
                if object_name != "none" and object_name != None:
                    if "change_language" == action:
                        arguments["language_name"] = object_name
                    else:
                        arguments["object_name"] = object_name
                if location_name != "none" and location_name != None:
                    arguments["location_name"] = location_name
                weather_type = row.get("weather_type", None)
                if weather_type != "none" and weather_type != None:
                    arguments["weather_type"] = weather_type
                time = row.get("time", None)
                if time != "none" and time != None:
                    arguments["time"] = time
                reponse = row.get("llm_response", None)
                expected_result = {
                    "function_name": action,
                    "arguments": arguments,
                    "result": True,
                    "response": reponse,
                }
                self.commands.append(command)
                self.commands_info.append(
                    {
                        "id": idx,
                        "command": command,
                        "action": action,
                        "arguments": arguments,
                        "expected_result": expected_result,
                    }
                )
                self.expected_results[idx] = expected_result
                self.mocked_responses[command] = {
                    "action": action,
                    "arguments": arguments,
                    "response": expected_result.get("response", ""),
                }

    def get_commands(self):
        """Return the list of commands to execute."""
        return self.commands

    def get_random_ordered_dataset(self):
        dataset = self.commands
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)
        reordered = [dataset[i] for i in indices]
        return reordered

    def get_response(self, command):
        return self.mocked_responses.get(command, {})

    def get_summary(self):
        return self.summary

    def get_samples_by_action(
        self, action_name: str, n: int, random_state: Optional[int] = None
    ) -> List[Dict]:
        filtered_entries = [
            entry for entry in self.commands_info if entry.get("action") == action_name
        ]
        if not filtered_entries:
            return []

        if random_state is not None:
            random.seed(random_state)

        sampled_entries = random.sample(
            filtered_entries, k=min(n, len(filtered_entries))
        )
        results = []
        for entry in sampled_entries:
            results.append(
                {
                    "id": entry["id"],
                    "command": entry["command"],
                    "action": entry["action"],
                    "arguments": entry["arguments"],
                    "expected_result": entry["expected_result"],
                }
            )

        return results

    def reorder_dataset(self, group_size=200, sample_per_group=20):
        dataset = self.commands_info
        total_size = len(dataset)
        reordered = []
        used_indices = set()

        for start in range(0, total_size, group_size):
            end = min(start + group_size, total_size)
            group = list(range(start, end))
            sampled_indices = random.sample(group, min(sample_per_group, len(group)))
            used_indices.update(sampled_indices)
            reordered.extend(dataset[i] for i in sampled_indices)

        remaining = [dataset[i] for i in range(total_size) if i not in used_indices]
        reordered.extend(remaining)

        return reordered

    def is_correct(
        self,
        command: str,
        action: str,
        arguments: dict,
        exec_result: bool,
    ) -> bool:
        if exec_result != True:
            return False
        expected_response = self.get_response(command)
        self.logger.debug(f"expected: {expected_response}")
        self.logger.debug(f"actual  : 'action': '{action}', 'arguments': {arguments}")
        if expected_response.get("action", "") == action:
            expected_args = expected_response.get("arguments", {})
            for key, value in expected_args.items():
                actual_value = arguments.get(key, None)
                if str(actual_value).lower() != str(value).lower():
                    return False
        else:
            return False
        return True
