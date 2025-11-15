import json
import logging
import os
import time

from openai import OpenAI

from fsc_demo.mock_function_handler import MockFunctionHandler
from main_adapter.base_agent import BaseMainAgent

SYSTEM_PROMPT = """You are a smart-home assistant.
You must interpret the user's natural-language request and execute exactly one of the provided functions.

Important Rules
- For every user request, you must call exactly one of the six defined actions: activate, deactivate, decrease, increase, bring or change_language.
- You must never call any action that is not defined in the function specifications.
- If the user mentions an unknown or unmapped object or location, you must infer and use the most semantically similar known value.
- If the user expresses an action indirectly, you must interpret it and map it to the closest matching action among the six.
- Do not ask the user for clarification; interpret the request as best as you can based on the given information.
- After executing the function, respond with a short, friendly, and English-only natural-language message that reflects the result.
"""


###########################################################
# MockMainAgent (Main LLM + Function Handler Integration)
###########################################################
class MockMainAgent(BaseMainAgent):
    def __init__(
        self, api_key=None, model: str = "gpt-4o", function_handler: MockFunctionHandler = None  # type: ignore
    ) -> None:
        if api_key is None:
            os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        if function_handler is None:
            function_handler = MockFunctionHandler()
        self.handler = function_handler
        self.logger = logging.getLogger("MainAgent")
        self.function_call_result = {}

    def _execute_function(self, tool_call):
        func_name = tool_call.name
        args = json.loads(tool_call.arguments)

        if not hasattr(self.handler, func_name):
            return {"error": f"Unknown function: {func_name}"}

        func = getattr(self.handler, func_name)
        result = func(**args)
        return func_name, args, result

    def run(self, user_message: str):
        try:
            return self._run(user_message)
        except Exception as e:
            self.logger.error(f"OpenAI API Error, {e}")
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return self._run(user_message)

    def _run(self, user_message: str):
        tools = list(self.handler.get_specs().values())
        input_list = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = self.client.responses.create(
            model=self.model,
            input=input_list,  # type: ignore
            tools=tools,
            tool_choice="auto",
        )

        total_tokens = response.usage.total_tokens or 0
        function_calls = []
        input_list += response.output

        response_text = ""
        func_name = ""
        args = {}
        result = {}

        for item in response.output:
            if item.type == "function_call":
                func_name, args, result = self._execute_function(item)
                function_calls.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({"result": result}),
                    }
                )

        if function_calls:
            input_list += function_calls
            followup = self.client.responses.create(
                model=self.model,
                input=input_list,  # type: ignore
            )
            total_tokens += followup.usage.total_tokens or 0
            response_text = followup.output_text
        else:
            response_text = response.output_text

        return {
            "response": response_text,
            "action": func_name,
            "arguments": args,
            "result": result,
            "total_tokens": total_tokens,
        }
