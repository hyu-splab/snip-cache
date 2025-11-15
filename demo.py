import logging
import os
import time

from dotenv import load_dotenv

from client.openai_client import OpenAIClient
from core.policy import Policy
from core.snip_cache import SnipCache
from core.snip_generator import SnippetGenerator
from fsc_demo.command_loader import DemoCommandLoader
from fsc_demo.mock_agent import MockMainAgent
from fsc_demo.mock_function_handler import MockFunctionHandler
from utils.monitor import monitor

load_dotenv(override=True)


def set_log(file_name: str, logger_name: str = ""):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    log_dir = f"./logs/session/{file_name}"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)-7s][%(name)-10s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        f"{log_dir}/{file_name}.log", encoding="utf-8", mode="w"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    for noisy in ["httpx", "httpcore", "openai", "urllib3", "sentence_transformers"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def run_experiment(test_name: str, sampel_size: int):
    logger = set_log(file_name=test_name)

    total_count = 0
    hit_count = 0
    miss_count = 0
    false_hit_count = 0
    cache_gen_count = 0

    hit_time = 0.0
    miss_time = 0.0
    false_hit_time = 0.0
    total_time = 0.0
    cache_gen_time = 0.0
    cache_gen_times = []
    false_hit_details = []
    total_token_usage_main = 0

    # FSC Dataset
    demo_dataset_loader = DemoCommandLoader(
        command_csv="./fsc_demo/fluent_speech_commands_extend.csv"
    )
    reordered_dataset = demo_dataset_loader.get_random_ordered_dataset()

    # Demo System
    demo_function_handler = MockFunctionHandler()
    demo_functions = {
        "activate": demo_function_handler.activate,
        "deactivate": demo_function_handler.deactivate,
        "increase": demo_function_handler.increase,
        "decrease": demo_function_handler.decrease,
        "bring": demo_function_handler.bring,
        "change_language": demo_function_handler.change_language,
    }
    action_type_specification = demo_function_handler.get_specs()
    api_key = os.getenv("OPENAI_API_KEY", "unknown_key")
    main_agent = MockMainAgent(
        api_key=api_key,
        model="gpt-4o",
        function_handler=demo_function_handler,
    )

    # Snip Cache
    policy = Policy(min_samples=sampel_size)
    snip_llm = SnippetGenerator(OpenAIClient(api_key=api_key, model="gpt-4o"))
    cache = SnipCache(
        functions=demo_functions,
        spec=action_type_specification,
        gen_llm=snip_llm,
        policy=policy,
        function_handler=demo_function_handler,
        cache_name=test_name,
    )

    monitor.set_output_file(f"./logs/session/{test_name}/resource_usage.csv")
    monitor.start()

    start_time = time.time()

    for idx, command in enumerate(reordered_dataset):
        total_count += 1
        logger.info(f"USER >> ({idx}) '{command}'")

        loop_start = time.time()
        cache_result = cache.search(command)

        if cache_result.get("status") == "hit":
            # Hit Case
            hit_count += 1
            action = cache_result.get("action", "?")
            arguments = cache_result.get("arguments", {})
            exec_result = cache_result.get("result", False)
            response = cache_result.get("response", "")
            if demo_dataset_loader.is_correct(command, action, arguments, exec_result):  # type: ignore
                logger.info("Correct Hit")
                hit_time += time.time() - loop_start
            else:
                logger.info("False Hit")
                false_hit_count += 1
                false_hit_details.append(
                    {
                        "command": command,
                        "action": action,
                        "arguments": arguments,
                        "exec_result": exec_result,
                        "response": response,
                    }
                )
                false_hit_time += time.time() - loop_start
            logger.info(
                f"CACHE >> '{response}', {action}({arguments})={exec_result})\n"
            )
        else:
            # Miss Case
            miss_count += 1
            logger.info(f"Cache Miss (run MainAgent)")
            result = main_agent.run(command)
            reponse = result.get("response", "")
            action = result.get("action", "unknown")
            arguments = result.get("arguments", {})
            exec_result = result.get("result", False)
            token_usage = result.get("total_tokens", 0)
            total_token_usage_main += token_usage
            logger.info(
                f"MAIN AGENT >> '{reponse}', {action}({arguments})={exec_result})\n"
            )

            # Sometimes MainAgent fails to execute correctly; we skip those cases
            if demo_dataset_loader.is_correct(command, action, arguments, exec_result):
                gen_start = time.time()
                logger.debug("Learning snippet into Cache")
                is_gen = cache.learn(command, result)
                miss_time += time.time() - loop_start
                if is_gen:
                    gen_time = time.time() - gen_start
                    cache_gen_times.append(gen_time)
                    cache_gen_time += gen_time
                    cache_gen_count += 1

        loop_time = time.time() - loop_start
        total_time += loop_time

    end_time = time.time()
    monitor.stop()
    cache.save_cache()

    # Summary
    # Cache Accuracy
    correct_hit = hit_count - false_hit_count
    success_rate = (
        ((correct_hit + miss_count) / total_count) * 100 if total_count else 0
    )
    hit_rate = (hit_count / total_count) * 100 if total_count else 0
    false_hit_rate = (false_hit_count / hit_count) * 100 if hit_count else 0
    miss_rate = (miss_count / total_count) * 100 if total_count else 0

    # Average Times
    avg_time = total_time / total_count if total_count else 0
    avg_hit_time = hit_time / hit_count if hit_count else 0
    avg_miss_time = miss_time / miss_count if miss_count else 0
    avg_false_hit_time = false_hit_time / false_hit_count if false_hit_count else 0
    avg_cache_gen_time = cache_gen_time / cache_gen_count if cache_gen_count else 0
    total_elapsed = end_time - start_time

    # Token Usage
    total_token_usage_cache = (
        snip_llm.code_token_usage.get("total_tokens", 0)
        + snip_llm.json_token_usage.get("total_tokens", 0)
        + snip_llm.text_token_usage.get("total_tokens", 0)
    )
    total_token_usage = total_token_usage_cache + total_token_usage_main
    avg_total_tokens = total_token_usage / total_count if total_count else 0

    logger.info(" ")
    logger.info("========== TEST SUMMARY ==========")
    logger.info(f"Total Commands       : {total_count}")
    logger.info(f"Cache Hits           : {hit_count} ({hit_rate:.2f}%)")
    logger.info(f"Cache Misses         : {miss_count} ({miss_rate:.2f}%)")
    logger.info(f"False Hits           : {false_hit_count} ({false_hit_rate:.2f}%)")
    logger.info(f"Hit Ratio(w/fallback): {success_rate:.2f}%")
    logger.info(f"Average Time / Cmd      : {avg_time:.3f} sec")
    logger.info(f"Average Hit Time(Cache) : {avg_hit_time:.3f} sec")
    logger.info(f"Average Miss Time(Main) : {avg_miss_time:.3f} sec")
    logger.info(f"Average False Hit Time  : {avg_false_hit_time:.3f} sec")
    logger.info(f"Cache Generations       : {cache_gen_count}")
    logger.info(f"Cache Generation Times  : {cache_gen_times}")
    logger.info(f"Average Cache Gen Time  : {avg_cache_gen_time:.3f} sec")
    logger.info(f"Total Elapsed Time      : {total_elapsed:.3f} sec")
    logger.info(f"Tokens Usage(Cache)  : {total_token_usage_cache:,} tokens")
    logger.info(f"Tokens Usage(Main)   : {total_token_usage_main:,} tokens")
    logger.info(f"Total Tokens Usage   : {total_token_usage:,} tokens")
    logger.info(f"Average Tokens / Cmd : {avg_total_tokens:.2f} tokens")
    logger.info("========================================\n")
    for idx, detail in enumerate(false_hit_details):
        logger.info(f"--- False Hit Detail [{idx + 1}] ---")
        logger.info(detail)


if __name__ == "__main__":
    run_experiment("test", 3)
