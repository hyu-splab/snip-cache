import json
import re
from typing import Any, Dict, List


PROMPT_ATD = """Generate a list of natural language trigger expressions for the "{function_name}" action using the provided user command examples and function specification.  
Each trigger should clearly express the intent to perform the "{function_name}" action in natural language.

# Input
## User Command Samples
{samples}

## Function Specification
{function_spec}

# Task Steps
Before generating new triggers, first check whether the same or similar expressions already appear in the user command samples.
Always prioritize triggers that can be directly or semantically derived from those examples.

[Step 1]: Extract core action phrases  
- Review each user command and extract a concise phrase that represents the intent to perform the "{function_name}" action.
- Consider full-sentence meaning and context.  
- Single words can be triggers only if they clearly and unambiguously indicate the action.
- Focus mainly on the action intent, but short contextual words may be included when they are essential to convey a clear and natural expression of the action.
- Accept verb phrases or noun + verb combinations when they clarify meaning.
- Do not include determiners ("a", "the") or possessive pronouns ("my", "your").  
- Avoid short or vague triggers that could cause ambiguity (e.g., "prefer", "make it").


[Step 2]: Add supplementary expressions  
- Using the function specification, expand on Step 1 expressions with natural variations that convey the same action intent.  
- Each expression must correspond only to the "{function_name}" action and not overlap with other functions.  
- Do not add expressions that are identical or semantically redundant with Step 1.

[Step 3]: Expand to meet quantity requirements
- Extend the list to include {extend_number} distinct triggers.
- Ensure diverse phrasing while keeping meaning consistent and unambiguous.
- Avoid repetitive or semantically overlapping expressions.

[Step 4]: Filter out ambiguous or unsafe triggers
- Remove expressions that could imply unrelated or conflicting actions.
- Exclude overly generic or context-dependent terms when used alone.
- The final list must contain only clear and self-contained triggers.

[Step 5]: Output format
Return a single JSON object in the following structure:
{{
  "action_triggers": ["trigger_1", "trigger_2", ..., "trigger_n"]
}}
"""

PROMPT_ATD_INIT = """Extract short action-trigger phrases that express the "{function_name}" action.
Each trigger must be concise, stable, and appear as an exact substring of the user commands.

# User Command Samples
{samples}

# Core Rules
- Each trigger MUST be an exact contiguous substring from the user commands.
- Allowed trigger forms:
  * a single verb (only if unambiguous in the samples),  
  * a verb + particle/preposition (e.g., "turn on", "power off"), or  
  * an object + particle/preposition when this clearly expresses the action 
    (e.g., "lights on", "tv off").
- Use “verb + object” only when the object is essential to convey the action meaning
  (i.e., removing the object would change or obscure the intended action).
- Preferred trigger length is two words; if unavailable, use a clear one-word expression, and allow three-word triggers only when strictly necessary.

# Restrictions
- Do not extract generic or vague single-word verbs 
  (e.g., get, make, turn, set, do, have) unless the samples contain no more specific phrase.
- Do not extract long multi-word fragments that contain unnecessary objects or fillers.
- Exclude determiners (“the”, “a”, “an”), polite expressions (“please”, “can you”), and other fillers.
- Do not paraphrase, modify, shorten, or invent new expressions; only extract literal substrings.
- If multiple overlapping substrings could express the action, select the shortest, clearest action phrase.
- Exclude expressions that could reasonably represent a different action.
- Keep only well-formed, self-contained verb phrases that clearly and unambiguously express the "{function_name}" action. 
  Exclude any fragment that is grammatically incomplete or whose meaning depends on omitted surrounding words.
  
# Output Format
{{
  "action_triggers": ["trigger_1", "trigger_2", ...]
}}
"""


PROMPT_ATD_EXPAND = """Expand the given list of base action triggers for the "{function_name}" action.  
Your goal is to generate additional short, natural action-trigger phrases that express the same action intent.

# Input
## Base Action Triggers
{base_triggers}

## Reference User Command Samples
{samples}

## Function Specification
{function_spec}

# Task Steps
[Step 1]: Identify the action intent
- Review the base triggers and reference samples to understand the shared action meaning.
- Do not reuse, rephrase, or minimally alter any expression from the base list.

[Step 2]: Generate new triggers
- Produce new, concise action phrases suitable for token-based action matching.
- Prefer verb-centered expressions:
  * clear and unambiguous single verbs that directly express the "{function_name}" action, or
  * verb + particle/preposition forms.
- Be cautious with verbs whose meaning changes depending on on/off direction or contextual modifiers (e.g., switch, turn).  
  Such verbs should only be used when they clearly and uniquely express the "{function_name}" action in the samples.
- Avoid triggers that could reasonably be interpreted as expressing the opposite direction of the "{function_name}" action.
- Preferred trigger length is two words; if unavailable, use a clear one-word verb.  
  Allow three-word triggers only when strictly necessary.
- Include an object only when it is strictly required to express the "{function_name}" action and no shorter verb-centered expression exists in the samples. Avoid sample-specific or unnecessary objects.
- Do NOT include object-specific triggers; triggers must express only the action itself, independent of any device or item names.
- Do not output standalone nouns as triggers. Noun-based expressions are acceptable only when they appear within natural action-pattern forms that align with the "{function_name}" action.
- Exclude triggers that involve objects or elements unrelated to the "{function_name}" action. Related objects may appear only when they support a natural and commonly used way of expressing the intended action.
- Determiners ("the", "a", "an"), polite expressions ("please", "can you"), and filler text are prohibited.
- Do not output full sentences or long multi-word fragments.
- Avoid non-specific verbs that can refer to many different actions and therefore do not uniquely indicate the "{function_name}" action (e.g., get, make, set, do, have).
- Avoid invented, awkward, or unnatural expressions.
- Keep only well-formed, self-contained phrases that clearly and unambiguously express the "{function_name}" action.

[Step 3]: Expand to meet quantity requirements
- Only include triggers that do not appear in the base list.
- Generate new triggers until there are {extend_number} total distinct triggers.
- Maintain meaningful linguistic diversity without relying on trivial or repetitive variations.

# Output Format
Return one JSON object:
{{
  "action_triggers": ["trigger_1", "trigger_2", ..., "trigger_n"]
}}
"""


PROMPT_AAD = """Generate an action_argument_dictionary for the "{function_name}" function using the provided user command examples and function specification.
This dictionary defines, for each function argument, how its possible values (normalized values) can be expressed by users in natural language.

# Terminology
- argument: an input defined in the function specification.
- normalized value: a fixed canonical value used during function execution (must be explicitly defined in the function specification).
- user expression: a natural phrase that refers to a normalized value.
  → Each expression must correspond to one and only one normalized value.

# Input
## User Command Samples
{samples}

## Function Specification
{function_spec}

# Task Steps
Before generating new expressions, first check if equivalent or similar phrases already appear in the user command samples.  
Always prioritize expressions that can be directly or semantically derived from the provided examples.

[Step 1]: Identify arguments
List all arguments defined in the function specification.

[Step 2]: Extract normalized values
For each argument, identify valid normalized values explicitly defined in the function specification.
Include only categorical values; exclude open-ended types such as numbers or free text.

[Step 3]: Generate user expressions
For every normalized value:
- Create natural and diverse expressions that users might use to refer to it.
- Always include the normalized value itself.
- Avoid expressions that could also match other normalized values.
- All expressions from the provided examples must appear in the final output exactly as they are.

[Step 4]: Build the 'arguments_dictionary'
Structure the output as:
argument → normalized value → list of user expressions.

[Step 5]: Expand to meet quantity requirements
- Extend each normalized value list to include {sample_number}-{extend_number} distinct expressions.
- Ensure diverse phrasing while keeping meaning consistent and unambiguous.

[Step 6]: Filter out ambiguous or cross-value expressions
- Remove expressions that could imply or overlap with different normalized values.
- Exclude phrases that may cause confusion between arguments or lead to incorrect value assignments.
- Keep only expressions that clearly and uniquely represent the intended normalized value.

# Output Requirements
- Include all arguments from the function specification, even if not mentioned in user examples.
- For each normalized value, include a reasonable number of distinct expressions (at least as many as examples, up to {extend_number}).
- Remove ambiguous or overlapping expressions.
- Do not use determiners (e.g., "the", "a") or possessive pronouns (e.g., "my", "your").
- Prefer concise noun phrases; avoid unnecessary verbs unless needed for clarity.
- Use plural or synonymous variations when natural.

# Output Format
Return a single JSON object in the following structure:
{{
  "arguments_dictionary": {{
    "argument_name_1": {{
      "normalized_value_1": ["expression_1", ..., "expression_20"],
      "normalized_value_2": ["expression_20", ..., "expression_40"],
      ...
    }},
    "argument_name_2": {{
      ...
    }}
  }}
}}
"""

PROMPT_AAD_INIT = """Extract minimal normalized values for each argument in the "{function_name}" function using the provided function specification and user command examples.  
Your goal is to identify the smallest valid set of categorical normalized values that represent the possible options for each argument.

# Terminology
- argument: an input defined in the function specification.
- normalized value: a fixed canonical value used during function execution (must be explicitly defined in the function specification).
- user expression: a natural phrase that refers to a normalized value (not required in this step).

# Input
## User Command Samples
{samples}

## Function Specification
{function_spec}

# Task Steps
[Step 1]: Identify arguments  
List all arguments explicitly defined in the function specification.

[Step 2]: Extract candidate normalized values  
For each argument:
- Identify all valid categorical normalized values explicitly defined or implied in the function specification.  
- If some categorical values are not clearly listed but appear consistently in user commands, include them only if they align with the specification.  
- Do not infer new values beyond those directly supported by the function specification or examples.
- For non-categorical arguments (numeric, date/time, or free-text types), record their type label (e.g., "numeric type", "date/time type", "free-text type") instead of extracting normalized values.

[Step 3]: Verify minimality and distinctness  
- Remove duplicates or semantically overlapping values.  
- Keep only the minimal, distinct set of normalized values that clearly represent the argument's valid options.

# Output Requirements
- Include every argument listed in the function specification.  
- For each argument, provide only the distinct categorical normalized values.  
- Do not include user expressions or free-text phrases at this stage.  
- The extracted list must be minimal but sufficient to represent all valid discrete states.

# Output Format
Return a single JSON object in the following structure:
{{
  "normalized_values": {{
    "argument_name_1": ["normalized_value_1", "normalized_value_2", ...],
    "argument_name_2": ["normalized_value_1", "normalized_value_2", ...]
  }}
}}
"""


PROMPT_AAD_EXPAND_SINGLE = """Generate natural user expressions that refer to the normalized value "{normalized_value}" used in the "{function_name}" function.

# Input
Argument Name: {argument_name}
Normalized Value: {normalized_value}
{samples}

# Function Specification
{function_spec}

# Rules
- Create short and natural phrases that users might say to mean "{normalized_value}".
- Always include expressions that literally appear in the given samples.
- Add simple synonyms or common variations with the same meaning.
- Add semantic synonyms or alternative names that people naturally use to refer to this value (e.g., alternate labels or native-language terms).
- When both singular and plural noun phrases are natural for this value, include at least one singular-form expression and at least one plural-form expression in the output list.
- Keep only phrases that clearly and uniquely refer to "{normalized_value}".
- Exclude determiners ("a", "the") and possessive pronouns ("my", "your").
- Avoid verbs unless essential for natural phrasing.
- Return 5–10 distinct expressions.
- Exclude expressions that could belong to another argument category.  
  Use the 'function specification' to keep only phrases that uniquely refer to the normalized value for this argument.

# Output
{{
  "{normalized_value}": ["expression_1", "expression_2", ...]
}}
"""


PROMPT_ATD_FIX = """Extract all natural-language phrases from the given user commands that clearly and literally express the "{function_name}" action.  
Select only phrases that appear exactly in the text, not inferred or rephrased versions.  
Your goal is to identify every short expression in these commands that conveys the same action intent.
Excluding arguments, unless they are naturally required to express the action 

# Input
User Commands:
{samples}

# Instructions
- Identify every phrase that expresses the intent to perform the "{function_name}" action.  
- Focus only on verbs or short verb phrases that appear literally in the text.  
- Exclude nouns or argument-specific parts such as object names or locations.  
- Keep only concise and natural phrases that directly represent the action itself.  
- Do not infer, paraphrase, or generalize beyond what appears in the text.  
- Each extracted phrase must exist verbatim (substring-level match) in the commands.  
- Remove duplicates and meaningless fragments.  
- If no valid phrase is found, return an empty list.

# Output Format
Return the result in JSON:
{{
  "action_triggers": ["trigger_1", "trigger_2", ...]
}}
"""


PROMPT_AAD_FIX = """Your task is to extract user expressions in the command that correspond to specific function arguments and their normalized values for the "{function_name}" function.  
The command "{command}" implicitly contains the following missing argument–value pairs: {missing_pairs}.  
For each argument and normalized value, identify a verbatim expression in the text that clearly matches that normalized value.

## Function Specification
{function_spec}

# Instructions
- Identify which function arguments are mentioned in the command.  
- For each detected argument, determine the corresponding normalized value defined for that argument.  
- Extract exactly one user expression for each normalized value. The expression **must appear verbatim** in the command.  
- The expression should clearly and unambiguously represent that normalized value.   
- Normalized value: a fixed canonical value used during function execution (must be explicitly defined in the Function Specification).
- Use short phrases (no more than 4 words).  
- Do not include unrelated, generic, or ambiguous terms.  
- Do not infer or generate new expressions that are not present in the command.  
- Exclude determiners ("the", "a") and possessive pronouns ("my", "your").

# Output Format
Return the extracted result in JSON:
{{
    "argument_name": {{
        "normalized_value": "expression"
    }}
}}
"""


PROMPT_EXTRACT_CODE = """You are a code generator that writes robust and reliable Python functions to extract arguments required for actions based on speech recognition commands.

# Role and Objective
Generate a function named `extract_arguments` that takes a user command and a 'aad(action argument dictionary)' as input, and returns a dictionary of extracted argument values. Use the provided action specification and mapping rules to guide the implementation.

# Instructions
- Implement a function with the signature:
  def extract_arguments(command: str, aad: dict) -> dict:
- The function must return a dictionary of extracted arguments with correct types, based on user input and the action argument dictionary.
- Use only the allowed libraries for specific types (see below).
- Follow all matching rules to ensure extraction is reliable and consistent.
- Do not raise exceptions. Always return all required arguments, using None if extraction fails.
- Do not use substring or partial matching techniques.
- Do not include any comments in the code.

## Sub-categories for more detailed instructions

### Parameter Type Handling
- Enum types: use the provided `aad` to identify valid values.
- Numeric types: extract using `word2number` or regular expressions.
- Date/time types: use `dateparser.parse()`.
- Free-text arguments: omit if ambiguous or unreliable.

### Enum Matching Strategy
- Normalize the input using `.lower().strip()`.
- Tokenize using `.split()` for word-level comparison.
- Match single-word expressions only if they appear as standalone tokens.
- Match multi-word expressions using `'in'` on the normalized string.
- Never match substrings or partial overlaps (e.g., "room" should not match "bedroom").
- 'aad(action argument dictionary)' has a None type key.

### Output Format
The function must return a dictionary in this format:
{{
  "argument1": "value1",
  "argument2": 25,
  "argument3": None
}}

### Allowed Libraries
- `word2number` (only for "numeric type" parsing)
- `dateparser` (only for "date/time type" parsing)
- `re` (only for "numeric type" or "free-text type" types)
- Do not use `re` for enum-type matching.

### Error Handling
- Do not raise any exceptions.
- All expected arguments must appear in the output, even if some are None.

# Reasoning Steps
1. Read the action specification to understand argument names, types, and constraints.
2. Use the 'aad' to determine possible normalized values and associated expressions.
3. Apply matching logic for enum, number, and date types accordingly.
4. Fill in the output dictionary with extracted or None values.
5. Return the completed dictionary.

# Context

## Action Name
'{function_name}'

## Function Specification and Constraints
{function_spec}

## User Command Samples
{samples}

## Action Argument Dictionary (AAD)
{aad}
"""


PROMPT_RESPONSE_CODE = """Generate a complete Python function in the exact format below.

def generate_response(parameters: dict, action_result: bool) -> str:
    # Implement logic according to the following rules:
    # ...
    return response

# Function Specification
- The function must return a single English sentence that describes the result of an action.
- The response must reflect both the parameter values (when available) and the action result (True or False).
- Do not raise exceptions or print anything.
- Do not include comments, logs, or multilingual content.
- Always return a complete English sentence suitable for spoken dialogue.
- Use the Function Specification and Constraints as the authoritative source for understanding what the action does, what each parameter represents, and how the output should be conceptually expressed.

## Sub-categories for more detailed instructions

### Tone and Style
- Match the tone and phrasing of the provided user command samples.
- Maintain consistency in politeness, brevity, and formality.
- Avoid abrupt, robotic, or overly verbose sentences.

### Reasoning Steps
1. Read the Function Specification and Constraints to fully understand the intended behavior of this action and the meaning of each parameter.
2. Consider all possible combinations of parameter availability: full parameters, partial parameters, single parameters, inconsistent values, or no parameters at all.
3. Determine whether the action succeeded or failed based on the `action_result` flag.
4. If the action succeeded, generate a sentence that best reflects the action's meaning using whatever parameters are available.  
   - If parameters are sufficient, make the response more specific.  
   - If parameters are missing or minimal, produce a more general confirmation that still matches the action's semantics as defined by the specification.
5. If the action failed, return a helpful fallback message that acknowledges the failure without assuming parameter requirements.
6. Ensure that every possible combination of (action_result × parameter availability) produces a valid, natural-sounding sentence.
7. Do not treat any parameter as required; instead, adapt the sentence to whatever information is present, guided by the action's semantics defined in the Function Specification and Constraints.

### Scenario Handling
- Success with many parameters → more specific and contextual confirmation.
- Success with few or no parameters → more general but still semantically correct confirmation.
- Failure (logic or system) → brief and helpful fallback message.
- Logically inconsistent or unusable parameter values → acknowledge lightly and fallback gracefully, while still aligning with the action’s intended meaning.

### Response Generalization
- Do not hardcode specific user inputs.
- Ensure the message structure can generalize to future commands with similar parameter patterns.
- Make fallback expressions general (e.g., "Something went wrong while performing the action.").
- Do not impose artificial constraints on which parameters must appear; always adjust the sentence according to whatever combination of parameters is given.
- The Function Specification must guide the semantic structure of the response, even when parameters are missing.

# Output Format
Output only this function:
def generate_response(parameters: dict, action_result: bool) -> str:
    return response

# Context

## Action Name
{function_name}

## Function Specification and Constraints
{function_spec}

## User Command Samples
{samples}
"""


PROMPT_EXTRACT_FIX_CODE = """You are a Python code repair assistant for command-driven systems.

Your task is to FIX an existing function named `extract_parameters(command: str, aad: dict) -> dict`
that extracts structured arguments from natural language commands.

# Context
- Action: {action}
- Specification: {spec}

# Current Code
{previous_code}

# Samples (command, expected arguments)
{samples}

# Failed Commands
{failed_commands}

# Failed Summary
{failed_summary}

# AAD (Action Argument Dictionary)
{aad}

# Requirements
1. Keep the correct logic from the previous code.
2. Fix only what is necessary to handle failed commands.
3. Do not change the function name or signature.
4. Return a complete, executable Python function as plain text.
"""


PROMPT_RESPONSE_FIX_CODE = """You are a Python code repair assistant for natural language response generation.

Your task is to FIX an existing function named `generate_response(args: dict, result: Any) -> str`
that produces polite, natural, and accurate responses.

# Context
- Action: {action}
- Specification: {spec}

# Current Code
{previous_code}

# Samples (command, action, arguments)
{samples}

# Failed Commands
{failed_commands}

# Requirements
1. Keep the overall tone and structure of the existing code.
2. Adjust only what is necessary for the failed samples.
3. Ensure the function is valid Python code and returns a string.
4. Return ONLY the corrected full Python code (no explanations).
"""


###############################################################################################
# ATD: Action Trigger Dictionary
###############################################################################################
def build_ATD_prompt(action: str, spec: str, samples: List[Dict[str, Any]]) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    extend_number = min(len(samples) * 4, 60)
    return PROMPT_ATD.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
        extend_number=extend_number,
    )


def build_ATD_init_prompt(action: str, spec: str, samples: List[Dict[str, Any]]) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r',?\s*"action"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    return PROMPT_ATD_INIT.format(
        function_name=action,
        samples=sample_text,
    )


def build_ATD_prompt_expend(
    action: str, spec: str, samples: List[Dict[str, Any]], existing_atd: Dict[str, Any]
) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r',?\s*"action"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    extend_number = min(len(samples) + 10, 20)
    return PROMPT_ATD_EXPAND.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
        extend_number=extend_number,
        base_triggers=json.dumps(existing_atd, indent=2, ensure_ascii=False),
    )


def build_ATD_fix_prompt(action: str, commands: str) -> str:
    return PROMPT_ATD_FIX.format(function_name=action, samples=commands)


###############################################################################################
# AAD: Action Argument Dictionary
###############################################################################################
def build_AAD_prompt(action: str, spec: str, samples: List[Dict[str, Any]]) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    sample_number = max(len(samples) * 3, 5)
    extend_number = min(len(samples) * 3, 20)
    return PROMPT_AAD.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
        sample_number=sample_number,
        extend_number=extend_number,
    )


def build_AAD_init_prompt(action: str, spec: str, samples: List[Dict[str, Any]]) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    return PROMPT_AAD_INIT.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
    )


def build_AAD_expand_prompt(
    action: str,
    argument_name: str,
    normalized_value: str,
    function_spec: str,
    samples: list,
) -> str:

    sample_text = ""
    for sample in samples:
        text = "Reference User Command Samples: \n"
        text += json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r',?\s*"action"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"

    return PROMPT_AAD_EXPAND_SINGLE.format(
        function_name=action,
        argument_name=argument_name,
        samples=sample_text,
        function_spec=function_spec,
        normalized_value=normalized_value,
    )


def build_AAD_fix_prompt(
    action: str, spec: str, command: str, missing_pairs: Dict[str, str]
) -> str:
    return PROMPT_AAD_FIX.format(
        function_name=action,
        function_spec=spec,
        command=command,
        missing_pairs=json.dumps(missing_pairs),
    )


###############################################################################################
# Extract Code
###############################################################################################
def build_extract_code_prompt(
    action: str,
    spec: str,
    samples: List[Dict[str, Any]],
    aad: Dict[str, Any],
) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r',?\s*"action"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    return PROMPT_EXTRACT_CODE.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
        aad=json.dumps(aad, indent=2, ensure_ascii=False),
    )


def build_extract_fix_prompt(
    action: str,
    spec: str | None,
    samples: list[dict],
    failed_commands: list[str],
    aad: dict,
    failed_summary: str,
    previous_code: str,
) -> str:
    """Builds prompt for fixing extract code."""
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r',?\s*"response"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r',?\s*"action"\s*:\s*".*?"(?=,|\n|})', "", text)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"

    failed_commands_text = "\n".join(f"  - {cmd}" for cmd in failed_commands)

    return PROMPT_EXTRACT_FIX_CODE.format(
        action=action,
        spec=spec,
        previous_code=previous_code,
        samples=sample_text,
        failed_commands=failed_commands_text,
        failed_summary=failed_summary,
        aad=json.dumps(aad, indent=2, ensure_ascii=False),
    )


###############################################################################################
# Response Code
###############################################################################################
def build_response_code_prompt(
    action: str,
    spec: str,
    samples: List[Dict[str, Any]],
) -> str:
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"
    return PROMPT_RESPONSE_CODE.format(
        function_name=action,
        function_spec=spec,
        samples=sample_text,
    )


def build_response_fix_prompt(
    action: str,
    spec: str,
    samples: list[dict],
    failed_commands: list[str],
    failed_summary: str,
    previous_code: str,
) -> str:
    """Builds prompt for fixing response code."""
    sample_text = ""
    for sample in samples:
        text = json.dumps(sample, indent=2, ensure_ascii=False)
        text = re.sub(r"[{}\[\]\"]", "", text)
        text = re.sub(r"\s*,\s*", "\n", text)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        sample_text += "  - " + "\n    ".join(lines) + "\n"

    failed_commands_text = "\n".join(f"  - {cmd}" for cmd in failed_commands)

    return PROMPT_RESPONSE_FIX_CODE.format(
        action=action,
        spec=spec,
        previous_code=previous_code,
        samples=sample_text,
        failed_commands=failed_commands_text,
        failed_summary=failed_summary,
    )
