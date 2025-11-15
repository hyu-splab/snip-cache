import ast
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
from typing import Any, Callable, Dict, List, Optional

import dateparser
import text_to_num
import word2number

from main_adapter.base_agent import BaseMainAgent
from main_adapter.base_function_handler import BaseFunctionHandler
from utils.comparer import SentenceComparer
from utils.monitor import monitor as global_monitor

from .policy import Policy
from .snip_generator import SnippetGenerator

ATD = "action_triggers"
AAD = "action_argument_dictionary"
EXTRACT_CODE = "extract_code"
RESPONSE_CODE = "response_code"


@dataclass
class ExecResult:
    result: Any
    response: Any


class SnipCache:
    def __init__(
        self,
        functions: Dict[str, Callable[[Dict[str, Any]], Any]],
        spec: Dict[str, str | None],
        gen_llm: SnippetGenerator,
        policy: Optional[Policy] = None,
        function_handler: Optional[BaseFunctionHandler] = None,
        cache_name: str = "snip_cache",
    ):
        self.functions = functions
        self.spec = spec
        self.gen_llm = gen_llm
        self.cache_name = cache_name
        self.cache_path = f"./cache_store/{cache_name}_cache.json"
        self.comparer = SentenceComparer()
        self.policy = policy or Policy()
        self.data: Dict[str, Any] = {}
        self.samples: Dict[str, List[Dict[str, str]]] = {}
        self.logger = logging.getLogger("SnipCache")
        self.logger.setLevel(logging.DEBUG)
        self.fail_count = {}
        self.function_handler = function_handler

    def load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def save_cache(self) -> None:
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp = f"{self.cache_name}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    def lookup(self, command: str, main_agent: BaseMainAgent) -> Dict[str, Any]:
        cache_result = self.search(command)
        if cache_result.get("status") == "hit":
            return cache_result
        else:
            result = main_agent.run(command)
            if self.policy.background_generation:
                Thread(target=self.learn, args=(command, result), daemon=True).start()
            else:
                self.learn(command, result)
            return result

    def search(self, command: str):
        command = self.comparer.remove_custom_stopwords(command)
        self.logger.debug(f"[Search] start: '{command}'")

        matched_actions = []
        matched_info = {}

        for action, entry in self.data.items():
            atd = entry.get(ATD, [])
            aad = entry.get(AAD, {})
            extract_code = entry.get(EXTRACT_CODE, "")
            response_code = entry.get(RESPONSE_CODE, "")

            if not (atd and aad and extract_code.strip() and response_code.strip()):
                self.logger.warning(f"[Search] skip invalid cache: {action}")
                continue

            for trigger in atd:
                if self.comparer.contain_similarity(command, trigger) == 1.0:
                    self.logger.debug(
                        f"[Search] trigger match: action='{action}', trigger='{trigger}'"
                    )
                    matched_actions.append(action)
                    matched_info[action] = (aad, extract_code, response_code)
                    break

        if len(matched_actions) == 0:
            self.logger.debug("[Search] miss: no trigger matched")
            return {"status": "miss"}
        if len(matched_actions) > 1 and self.policy.reject_on_ambiguity:
            self.logger.debug(f"[Search] ambiguous: {matched_actions}")
            return {"status": "miss"}

        matched_action = matched_actions[0]
        matched_aad, matched_extract_code, matched_response_code = matched_info[
            matched_action
        ]

        args = self._run_extract_function(
            matched_extract_code, command, matched_aad, matched_action
        )
        if not args:
            self.logger.debug(f"[Search:{matched_action}] miss: extract_code failed")
            return {"status": "miss"}
        if not self._validate_function_argurements(matched_action, args):
            self.logger.debug(f"[Search:{matched_action}] miss: invalid args {args}")
            return {"status": "miss"}
        self.logger.debug(f"[Search:{matched_action}] extracted args={args}")

        try:
            exec_result = self._execute(matched_action, args, matched_response_code)
            self.logger.debug(f"[Search:{matched_action}] executed successfully")
            return {
                "status": "hit",
                "action": matched_action,
                "arguments": args,
                "result": exec_result.result,
                "response": exec_result.response,
            }
        except Exception as e:
            self.logger.debug(
                f"[Search:{matched_action}] response execution failed: {e}",
                exc_info=True,
            )
            return {
                "status": "miss",
                "action": matched_action,
                "arguments": args,
                "result": None,
                "response": None,
            }

    #####################################################################
    # Action triger dictionary
    #####################################################################
    def _generate_action_trigger_dictionary(
        self, action: str, spec: str, samples: List[Dict[str, str]]
    ):
        # 1.Initial generation
        global_monitor.mark_phase(f"[ATD:{action}] generation start")
        atd = self.gen_llm.generate_action_trigger_dictionary(action, spec, samples)
        self.logger.debug(f"ATD initial result: {atd}")

        # 2. Validation loop
        global_monitor.mark_phase(f"[ATD:{action}] validation start")
        fail_key = f"atd_{action}"
        self.fail_count[fail_key] = 0
        while self.fail_count[fail_key] < self.policy.max_retry:
            cycle = self.fail_count[fail_key] + 1
            global_monitor.mark_phase(f"[ATD:{action}] validation cycle {cycle}")

            validation, failed_command = self._validate_atd(action, atd, samples)
            if validation:
                self.logger.debug(f"ATD validation succeeded for action: {action}")
                break

            self.logger.debug(f"ATD validation failed for: {failed_command}")
            self.logger.debug(f"Retry {cycle}/{self.policy.max_retry}")

            commands = ", ".join(sample["command"] for sample in samples)
            atd_fix = self.gen_llm.fix_action_trigger(action, commands)
            self._merge_triggers(atd, atd_fix.get(ATD, []))
            self.fail_count[fail_key] += 1

        # 3. Expand
        expand_atd = self.gen_llm.expand_action_trigger_dictionary(
            action, spec, samples, atd
        )
        if action not in atd:
            atd[ATD].insert(0, action)
        self._merge_triggers(atd, expand_atd.get(ATD, []))
        global_monitor.mark_phase(
            f"[ATD:{action}] generation complete (tries={self.fail_count[fail_key]})"
        )
        return atd

    def _merge_triggers(self, atd: dict, new_triggers: List[str]) -> None:
        """Merge new triggers into ATD while preserving order and removing duplicates."""
        if not new_triggers:
            return
        atd.setdefault(ATD, [])
        seen = set()
        merged = [t for t in atd[ATD] + new_triggers if not (t in seen or seen.add(t))]
        atd[ATD] = merged

    def _validate_atd(self, action, atd, samples):
        if self.policy.is_skip_validation("atd"):
            return True, []
        failed_commands = []
        action_triggers = atd.get(ATD, [])
        for sample in samples:
            command = sample["command"].lower()
            is_valid = False
            for trigger in action_triggers:
                similarity = self.comparer.contain_similarity(command, trigger)
                if similarity == 1.0:
                    self.logger.debug(f"+ Trigger valid:'{trigger}' >> '{command}'")
                    is_valid = True
                    break
            if not is_valid:
                self.logger.debug(f"- Trigger invalid:'{command}'")
                failed_commands.append(command)
                self.fail_count.get(f"atd_{action}")
        if len(failed_commands) >= 1:
            self.logger.debug("Retrying with additional keywords.")
            return False, failed_commands
        return True, []

    #####################################################################
    # Action argument dictionary
    #####################################################################
    def _generate_action_argument_dictionary(
        self, action: str, spec: str, samples: List[Dict[str, str]]
    ):
        # 1.Initial generation
        global_monitor.mark_phase(f"[AAD:{action}] generation start")
        aad_outline = self.gen_llm.generate_action_argument_dictionary(
            action, spec, samples
        )
        outline = aad_outline["normalized_values"]
        self.logger.debug(f"AAD outline: {outline}")

        # 2. Expand expressions (aggressive parallel expansion)
        aad = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for argument_key, values in outline.items():
                if isinstance(values, str):
                    # "numeric type", "date/time type", "free-text type"
                    aad.setdefault(argument_key, {})[values] = [values]
                    continue
                for value in values:
                    filtered_samples = [
                        s
                        for s in samples
                        if value
                        and isinstance(s, dict)
                        and value.lower() in str(s.get("arguments", "")).lower()
                    ]
                    sample_input = filtered_samples or []
                    futures[
                        executor.submit(
                            self.gen_llm.expand_aad_value,
                            action,
                            argument_key,
                            value,
                            spec,
                            sample_input,
                        )
                    ] = (argument_key, value)

                for future in as_completed(futures):
                    argument_key, value = futures[future]
                    try:
                        result = future.result()
                        aad.setdefault(argument_key, {})[value] = result.get(
                            value, [value]
                        )
                    except Exception as e:
                        self.logger.error(
                            f"AAD expansion failed for {argument_key}:{value}: {e}",
                            exc_info=True,
                        )
                        aad.setdefault(argument_key, {})[value] = [value]
        self.logger.debug(f"AAD expanded: {aad}")

        # 3. validation loop
        global_monitor.mark_phase(f"[AAD:{action}] validation start")

        fail_key = f"aad_{action}"
        self.fail_count[fail_key] = 0
        while self.fail_count[fail_key] < self.policy.max_retry:
            global_monitor.mark_phase(
                f"[AAD:{action}] validation cycle {self.fail_count[fail_key] + 1}"
            )
            is_valid, details = self._validate_aad(action, aad, samples)
            if is_valid:
                self.logger.debug(f"AAD validation succeeded for action: {action}")
                break
            self.logger.debug(f"AAD validation failed for: {details}")
            self.logger.debug(
                f"Retry {self.fail_count[fail_key] + 1}/{self.policy.max_retry}"
            )

            failed_command = details.get("failed_commands", [])
            missing_pairs = details.get("missing_pairs", {})
            if not failed_command or not missing_pairs:
                self.logger.debug("No missing pairs found; cannot proceed with fix.")
                break
            partial_aads = []
            with ThreadPoolExecutor(
                max_workers=max(1, len(failed_command))
            ) as executor:
                futures = {
                    executor.submit(
                        self.gen_llm.fix_action_argument,  # PROMPT_AAD_SINGLE 기반
                        action,
                        spec,
                        cmd,
                        details.get("missing_pairs", {}).get(cmd, {}),
                    ): cmd
                    for cmd in failed_command
                }
                for future in as_completed(futures):
                    cmd = futures[future]
                    try:
                        aad_fix = future.result()
                        self.logger.debug(f"Partial AAD result for '{cmd}': {aad_fix}")
                        if isinstance(aad_fix, dict):
                            partial_aads.append(aad_fix)
                        else:
                            self.logger.warning(
                                f"AAD partial generation returned invalid format for '{cmd}': {aad_fix}"
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"AAD partial generation failed for '{cmd}': {e}",
                            exc_info=True,
                        )

            for partial in partial_aads:
                for arg, nv_dict in partial.items():
                    if arg not in aad:
                        aad[arg] = {}
                    for nv, expr in nv_dict.items():
                        if isinstance(expr, str):
                            expr = [expr]
                        if nv not in aad[arg]:
                            aad[arg][nv] = []
                        aad[arg][nv].extend(expr)

            for arg, nv_dict in aad.items():
                for nv, expr_list in nv_dict.items():
                    seen = set()
                    aad[arg][nv] = [
                        e for e in expr_list if not (e in seen or seen.add(e))
                    ]
            self.fail_count[fail_key] += 1

        # 4. Final normalization
        action_argument_dictionary = {}
        for argument_key, values in aad.items():
            safe_dict = {}
            for k, exprs in values.items():
                k_norm = k.lower() if isinstance(k, str) else k
                if exprs is None:
                    expr_list = []
                elif isinstance(exprs, str):
                    expr_list = [exprs]
                else:
                    expr_list = exprs
                expr_norm = [v.lower() if isinstance(v, str) else v for v in expr_list]
                if isinstance(k_norm, str) and k_norm not in expr_norm:
                    expr_norm.insert(0, k_norm)
                safe_dict[k_norm] = expr_norm
            action_argument_dictionary[argument_key] = safe_dict

        global_monitor.mark_phase(
            f"[AAD:{action}] generation complete (tries={self.fail_count[fail_key]})"
        )
        return action_argument_dictionary

    def _validate_aad(self, action, aad_dict, samples):
        if self.policy.is_skip_validation("aad"):
            return True, {}

        failed_commands = []
        all_missing_pairs = {}  # command → missing_pairs

        if not isinstance(aad_dict, dict):
            return False, {}

        for sample in samples:
            command_raw = sample.get("command", "")
            command = command_raw.lower()

            # Parse expected arguments
            expected_args = sample.get("arguments", {})
            if isinstance(expected_args, str):
                try:
                    expected_args = json.loads(expected_args)
                except Exception:
                    try:
                        expected_args = ast.literal_eval(expected_args)
                    except Exception:
                        expected_args = {}

            # Checklist of normalized values expected in this command
            checklist = [
                str(v).lower()
                for v in expected_args.values()
                if v is not None and str(v).strip()
            ]

            self.logger.debug(f"  AAD checklist: {checklist}")

            # Try to match provided user expressions from aad_dict
            for arg, nv_dict in aad_dict.items():
                if not checklist:
                    break
                if not isinstance(nv_dict, dict):
                    continue

                for nv, exprs in nv_dict.items():
                    if not checklist:
                        break
                    if nv is None:
                        continue

                    norm_value = str(nv).lower()

                    # Skip if this value is not expected
                    if norm_value not in checklist:
                        continue

                    expr_list = exprs if isinstance(exprs, list) else [exprs]
                    for expr in expr_list:
                        if not expr:
                            continue
                        expr_norm = str(expr).lower().strip()
                        if not expr_norm:
                            continue

                        # literal match
                        pattern = r"\b" + re.escape(expr_norm) + r"\b"
                        if re.search(pattern, command):
                            if norm_value in checklist:
                                checklist.remove(norm_value)
                            self.logger.debug(
                                f"  [{arg}] OK '{expr_norm}' → '{norm_value}'"
                            )
                            break

            # If checklist still has items → missing normalized values
            if checklist:
                missing_pairs = {}

                # Map missing normalized values back to argument_name via aad_dict
                for mv in checklist:
                    for arg, nv_dict in aad_dict.items():
                        # Check if argument contains this normalized value
                        if mv in [str(k).lower() for k in nv_dict.keys()]:
                            missing_pairs[arg] = mv

                self.logger.debug(
                    f"  AAD check failed >> Missing: {missing_pairs} in '{command_raw}'"
                )

                failed_commands.append(command_raw)
                all_missing_pairs[command_raw] = missing_pairs

            else:
                self.logger.debug(f"+ AAD check passed >> '{command_raw}'")

        # At least one failure → return all missing pairs
        if failed_commands:
            return False, {
                "failed_commands": failed_commands,
                "missing_pairs": all_missing_pairs,
            }

        return True, {}

    #####################################################################
    # Extract Code
    #####################################################################
    def _generate_extract_code(
        self, action: str, spec: str, aad: Dict[str, Any], samples: List[Dict[str, str]]
    ):
        aad = aad.get(AAD, aad)
        global_monitor.mark_phase(f"[ExtractCode:{action}] generation start")
        code = self.gen_llm.generate_extract_code(action, spec, samples, aad)
        is_valid = self._validate_extract_code(action, code, aad, samples)
        if not is_valid:
            self.logger.warning(
                f"[{action}] extract_code generation failed validation and will be ADD regenerated."
            )
            regenerate_aad = self._generate_action_argument_dictionary(
                action, spec, samples
            )
            is_valid = self._validate_extract_code(
                action, code, regenerate_aad, samples
            )
            if not is_valid:
                self.logger.error(
                    f"[{action}] extract_code regeneration also failed validation."
                )
                return None
        global_monitor.mark_phase(f"[ExtractCode:{action}] generation complete")
        return code

    def _validate_extract_code(
        self,
        action: str,
        extract_code: str,
        aad: Dict[str, Any],
        samples: List[Dict[str, str]],
    ) -> bool:
        """Validate extract_code using unified runner (same as search)."""
        if self.policy.is_skip_validation("extract_code"):
            return True
        fail_key = f"{action}_extract_code"
        self.fail_count[fail_key] = 0
        spec = self.spec.get(action, "")

        remaining_samples = samples[:]

        # validation loop
        global_monitor.mark_phase(f"[ExtractCode:{action}] validation start")
        while self.fail_count[fail_key] < self.policy.max_retry:
            global_monitor.mark_phase(
                f"[ExtractCode:{action}] validation cycle {self.fail_count[fail_key] + 1}"
            )
            failed_commands = []
            failed_details = []

            for sample in remaining_samples:
                cmd = sample["command"]
                result = self._run_extract_function(extract_code, cmd, aad, action)
                if not result:
                    failed_commands.append(cmd)
                    failed_details.append(
                        f"[Command] {cmd}\n[Error] extract_code failed\n"
                    )
                    continue

                expected = sample.get("arguments", {})
                if isinstance(expected, str):
                    try:
                        expected = json.loads(expected)
                    except Exception:
                        try:
                            expected = ast.literal_eval(expected)
                        except Exception:
                            expected = {}

                mismatched = []
                for k, v in expected.items():
                    rv = str(result.get(k, "")).strip().lower()
                    ev = str(v).strip().lower()
                    empty_expression = ["", "none", "null", None]
                    if rv in empty_expression and ev in empty_expression:
                        self.logger.debug(f"  [{k}] OK both empty")
                    elif rv != ev:
                        mismatched.append(f"{k}: expected='{ev}', got='{rv}'")

                if mismatched:
                    failed_commands.append(cmd)
                    failed_details.append(
                        f"[Command] {cmd}\n" + "\n".join(mismatched) + "\n"
                    )

            if not failed_commands:
                self.logger.debug(f"[{action}] extract_code validation succeeded.")
                global_monitor.mark_phase(
                    f"[ExtractCode:{action}] validation complete (tries={self.fail_count[fail_key]})"
                )
                return True

            failed_summary = "\n---\n".join(failed_details).replace("---\n", "")
            self.logger.debug(
                f"[{action}] extract_code validation failures:\n{failed_summary}"
            )

            self.fail_count[fail_key] += 1
            try:
                extract_code = self.gen_llm.fix_extract_code(
                    action,
                    spec,
                    samples,
                    failed_commands,
                    aad,
                    failed_summary,
                    previous_code=extract_code,
                )
                self.logger.debug(f"[{action}] fixed extract_code regenerated.")
            except Exception as e:
                self.logger.error(f"[{action}] fix_extract_code failed: {e}")
                return False

        self.logger.warning(
            f"[{action}] extract_code validation failed after max retries."
        )
        return False

    #####################################################################
    # Response Code
    #####################################################################
    def _generate_response_code(
        self, action: str, spec: str, samples: List[Dict[str, str]]
    ):
        global_monitor.mark_phase(f"[ResponseCode:{action}] generation start")
        code = self.gen_llm.generate_response_code(action, spec, samples)
        global_monitor.mark_phase(f"[ResponseCode:{action}] generation complete")
        return code

    def _validate_response_code(
        self,
        action: str,
        response_code: str,
        samples: List[Dict[str, str]],
    ) -> bool:
        """Validate response_code with compile caching and selective re-validation."""
        if self.policy.is_skip_validation("response_code"):
            return True
        fail_key = f"{action}_response_code"
        self.fail_count[fail_key] = 0
        spec = self.spec.get(action, "")

        remaining_samples = samples[:]
        passed_commands = set()

        # validation loop
        global_monitor.mark_phase(f"[ResponseCode:{action}] validation start")
        while self.fail_count[fail_key] < self.policy.max_retry:
            global_monitor.mark_phase(
                f"[ResponseCode:{action}] validation cycle {self.fail_count[fail_key] + 1}"
            )
            ok, fn = self._compile_response(response_code)
            if not ok:
                self.logger.warning(f"[{action}] response_code compilation failed.")
                return False

            failed_commands = []
            failed_details = []

            with ThreadPoolExecutor(
                max_workers=min(len(remaining_samples), os.cpu_count() or 4)
            ) as executor:
                futures = {}
                for s in remaining_samples:
                    cmd = s.get("command", "")
                    args = s.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    func = self.functions.get(action)
                    result = func(args) if func else None
                    futures[executor.submit(fn, args, result)] = (cmd, s)

                for future in as_completed(futures):
                    cmd, sample = futures[future]
                    try:
                        response = future.result()
                        expected = str(sample.get("response", "")).strip().lower()
                        actual = str(response).strip().lower()

                        if not actual:
                            failed_commands.append(cmd)
                            failed_details.append(
                                f"[Command] {cmd}\n[Error] empty response\n"
                            )
                            continue

                        if expected:
                            try:
                                similarity = self.comparer.semantic_similarity(
                                    expected, actual
                                )
                                if similarity < self.policy.response_threshold:
                                    failed_commands.append(cmd)
                                    failed_details.append(
                                        f"[Command] {cmd}\n[Expected] {expected}\n[Got] {actual}\n"
                                        f"[Similarity] {similarity:.2f} < {self.policy.response_threshold:.2f}\n"
                                    )
                                    self.logger.warning(
                                        f"[{action}] response_code validation failed for command: {cmd} (similarity={similarity:.2f})"
                                    )
                                else:
                                    passed_commands.add(cmd)

                            except Exception as e:
                                failed_commands.append(cmd)
                                failed_details.append(
                                    f"[Command] {cmd}\n[Error in similarity] {e}\n"
                                )

                    except Exception as e:
                        failed_commands.append(cmd)
                        failed_details.append(f"[Command] {cmd}\n[Error] {e}\n")

            if not failed_commands:
                self.logger.debug(f"[{action}] response_code validation succeeded.")
                global_monitor.mark_phase(
                    f"[ResponseCode:{action}] validation complete (tries={self.fail_count[fail_key]})"
                )
                return True

            failed_summary = "\n---\n".join(failed_details)
            self.logger.debug(
                f"[{action}] response_code validation failures:\n{failed_summary}"
            )

            self.fail_count[fail_key] += 1
            self.logger.debug(
                f"[{action}] response_code validation failed (attempt {self.fail_count[fail_key]}/{self.policy.max_retry})"
            )

            try:
                response_code = self.gen_llm.fix_response_code(
                    action,
                    spec,  # type: ignore
                    samples,
                    failed_commands,
                    failed_summary,
                    previous_code=response_code,
                )
                self.logger.debug(f"[{action}] fixed response_code regenerated.")
                remaining_samples = samples[:]
                passed_commands.clear()
            except Exception as e:
                self.logger.error(
                    f"[{action}] fix_response_code failed: {e}", exc_info=True
                )
                return False

        self.logger.warning(
            f"[{action}] response_code validation failed after max retries."
        )
        return False

    #####################################################################
    # Generation
    #####################################################################
    def learn(self, command: str, result: dict):
        command = self.comparer.remove_custom_stopwords(command)
        action = result.get("action", "unknown")
        arguments = result.get("arguments", {})
        response: str = result.get("response", "")

        if not self.policy.generation_enable:
            self.logger.debug("Auto generation disabled; skipping learning.")
            return False
        if self.data.get(action):
            self.logger.debug(
                f"Action '{action}' already has cache; skipping learning."
            )
            return False
        if len(command.split()) <= 1:
            self.logger.debug(f"Command too short; skipping learning: '{command}'")
            return False

        if action not in self.samples:
            self.samples[action] = []

        self.samples[action].append(
            {
                "command": command,
                "action": action,
                "arguments": str(arguments),
                "response": response,
            }
        )
        if len(self.samples[action]) < self.policy.min_samples:
            return False
        elif len(self.samples[action]) == self.policy.min_samples:
            global_monitor.mark_phase("[1] generate ATD AAD start!")
            self.logger.info(f"Generating cache for action: {action}")
            self.logger.debug(f"samples: {json.dumps(self.samples[action], indent=2)}")

            # Generate cache code
            action_type_spec = str(self.spec.get(action, ""))
            other_actions = str(self.spec.keys() - {action})
            samples = self.samples[action]
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=2) as executor:
                tasks = [
                    (
                        "ATD",
                        executor.submit(
                            self._generate_action_trigger_dictionary,
                            action,
                            action_type_spec,
                            samples,
                        ),
                    ),
                    (
                        "AAD",
                        executor.submit(
                            self._generate_action_argument_dictionary,
                            action,
                            action_type_spec,
                            samples,
                        ),
                    ),
                ]

                results = {}
                durations = {}

                for key, future in tasks:
                    durations[key] = {"start": time.time()}

                for future in as_completed([f for _, f in tasks]):
                    key = next(k for k, f in tasks if f == future)
                    try:
                        results[key] = future.result()
                        durations[key]["end"] = time.time()
                        elapsed = durations[key]["end"] - durations[key]["start"]
                        self.logger.debug(
                            f"{key} generation took {elapsed:.2f} seconds"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"{key} generation failed for {action}: {e}\n\n",
                            exc_info=True,
                        )
                        return False
            atd = results.get("ATD", {})
            aad = results.get("AAD", {})
            self.logger.info(f"ATD Final: {atd}")
            self.logger.info(f"AAD Final: {aad}")
            global_monitor.mark_phase("[2] generate ATD AAD done")
            global_monitor.mark_phase("[3] generate Code snippets start")
            with ThreadPoolExecutor(max_workers=2) as executor:
                tasks = [
                    (
                        "ExtractCode",
                        executor.submit(
                            self._generate_extract_code,
                            action,
                            action_type_spec,
                            aad,
                            samples,
                        ),
                    ),
                    (
                        "ResponseCode",
                        executor.submit(
                            self._generate_response_code,
                            action,
                            action_type_spec,
                            samples,
                        ),
                    ),
                ]
                results = {}
                durations = {}

                for key, future in tasks:
                    durations[key] = {"start": time.time()}

                for future in as_completed([f for _, f in tasks]):
                    key = next(k for k, f in tasks if f == future)
                    try:
                        results[key] = future.result()
                        durations[key]["end"] = time.time()
                        elapsed = durations[key]["end"] - durations[key]["start"]
                        self.logger.debug(f"Generated {key}: {results[key]}")
                        self.logger.info(f"{key} generation took {elapsed:.2f} seconds")
                    except Exception as e:
                        self.logger.error(
                            f"{key} generation failed for {action}: {e}\n\n",
                            exc_info=True,
                        )
                        return False
            extract_code = results.get("ExtractCode", "")
            response_code = results.get("ResponseCode", "")
            self.logger.debug(f"Final extract_code: {extract_code}")
            self.logger.debug(f"Final response_code: {response_code}")
            global_monitor.mark_phase("[4] generate Code done")
            self.logger.info(
                f"Final LLM code generation for {action} took {time.time() - start_time:.2f} seconds"
            )
            self._save_cache(action, atd, aad, extract_code, response_code)
            return True

    def _save_cache(self, action: str, atd, aad, extract_code: str, response_code: str):
        self.logger.debug(f"Save cache for action: {action}")
        self.logger.debug(f"- ATD: {atd}")
        self.logger.debug(f"- AAD: {aad}")
        self.logger.debug(f"- ExtractCode: {extract_code}")
        self.logger.debug(f"- ResponseCode: {response_code}")
        entry = self._ensure_entry(action)
        entry[ATD] = atd.get(ATD, [])
        entry[AAD] = aad
        entry[EXTRACT_CODE] = extract_code or ""
        entry[RESPONSE_CODE] = response_code or ""
        self.logger.info(f"{entry}")

    def _execute(
        self, action: str, function_args: dict, response_code: str
    ) -> ExecResult:
        args = {k: v for k, v in (function_args or {}).items() if v is not None}
        action_fn = self.functions.get(action)
        action_result = action_fn(**args)  # type: ignore
        ok, resp_fn = self._compile_response(response_code)
        response_obj = resp_fn(function_args, action_result) if ok else None
        return ExecResult(result=action_result, response=response_obj)

    def _build_template(self, command: str, parameters: Dict[str, Any]) -> str:
        text = command
        for k, v in (parameters or {}).items():
            if isinstance(v, str) and v.strip():
                pattern = re.compile(re.escape(v), flags=re.IGNORECASE)
                text = pattern.sub(f"{{{k}}}", text)
        return re.sub(r"\s+", " ", text).strip()

    def _validate_function_argurements(
        self, action_name: str, extracted: Dict[str, Any]
    ) -> bool:
        if self.function_handler is None:
            self.logger.warning(
                "Function handler is not set; skipping argument validation."
            )
            return True
        return self.function_handler.validate_required_params(
            action_name=action_name, arguments=extracted
        )

    def _run_extract_function(
        self,
        extract_code: str,
        command: str,
        aad: Dict[str, Any],
        action: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Unified runner for extract_code execution.
        Recompiles with safe_globals to ensure consistent behavior
        between validation and search.
        """
        safe_globals = {
            "re": re,
            "dateparser": dateparser,
            "text_to_num": text_to_num,
            "word2number": word2number,
        }
        local_env = {}
        try:
            exec(extract_code, safe_globals, local_env)
            fn = local_env.get("extract_arguments")
            if not callable(fn):
                if action:
                    self.logger.debug(
                        f"[{action}] extract_code missing callable 'extract_arguments'"
                    )
                return None

            result = fn(command, aad)
            if not isinstance(result, dict):
                if action:
                    self.logger.debug(
                        f"[{action}] extract_code invalid return type: {type(result)}"
                    )
                return None
            return result

        except Exception as e:
            if action:
                self.logger.debug(
                    f"[{action}] extract_code execution error: {e}", exc_info=True
                )
            return None

    def _compile_response(self, code: str):
        safe_globals = {}
        local_env = {}
        try:
            exec(code, safe_globals, local_env)
            fn = local_env.get("generate_response")
            return (True, fn) if callable(fn) else (False, lambda *_: None)
        except Exception:
            return False, lambda *_: None

    def _ensure_entry(self, action: str):
        entry = self.data.get(action)
        if entry is None:
            entry = {
                ATD: [],
                AAD: {},
                EXTRACT_CODE: "",
                RESPONSE_CODE: "",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self.data[action] = entry
        return entry
