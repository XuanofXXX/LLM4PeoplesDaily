import openai  # Needs version 1.x or higher for AsyncOpenAI
import asyncio
import aiofiles
import json  # For JSONL
from pathlib import Path

# --- Configuration ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"  # Or your actual API key if required by vLLM
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"  # Or the specific model ID you are using with vLLM

# Input JSONL file
INPUT_JSONL_FILE = Path("rmrb_data_deduplicate_new.jsonl")
# Key in the input JSONL that contains the article text
INPUT_JSONL_CONTENT_KEY = "content"
# Key in the input JSONL to use as an identifier (e.g., "id", "source_path"). Fallbacks to line number if not found.
INPUT_JSONL_ID_KEY = (
    "file_path"  # Or "source_path", or any other unique key in your input JSONL
)

# Output JSONL file for meaningful articles
OUTPUT_JSONL_FILE = Path("llm_processed_meaningful_articles.jsonl")

# Concurrency and Retry Settings
MAX_CONCURRENT_TASKS = (
    1000  # Adjust based on your vLLM server capacity and system resources
)
MAX_MODEL_RETRIES = 3  # Max retries for model if output is not "是" or "否"
RETRY_DELAY_SECONDS = 5  # Delay between retries

PROMPT_TEMPLATE = """
## Instruction
以下这段文字能否作为一篇有意义的文章或者报道？如果是，则只输出是，如果否，则只输出否。

## Article
{content}
""".strip()


async def get_model_decision(
    client: openai.AsyncOpenAI, article_content: str, item_id_for_log: str
) -> tuple[bool, str]:
    """
    Uses the Qwen 2.5 model via vLLM to determine if an article is meaningful.
    Retries if the output is not strictly "是" or "否".
    Applies fallback logic after max retries.
    Returns: (is_meaningful: bool, raw_model_output: str)
    """
    prompt = PROMPT_TEMPLATE.format(content=article_content)
    last_raw_answer = "ERROR_NO_RESPONSE_CAPTURED"

    for attempt in range(MAX_MODEL_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            current_raw_answer = response.choices[0].message.content.strip()
            last_raw_answer = current_raw_answer
            # print(
                # f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - Model raw response: '{current_raw_answer}'"
            # )

            if current_raw_answer == "是":
                return True, current_raw_answer
            if current_raw_answer == "否":
                return False, current_raw_answer

            if attempt < MAX_MODEL_RETRIES - 1:
                print(
                    f"Item ID: {item_id_for_log} - Unexpected response. Retrying in {RETRY_DELAY_SECONDS}s..."
                )
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                print(
                    f"Item ID: {item_id_for_log} - Max retries reached with non-standard final response: '{current_raw_answer}'. Applying fallback logic."
                )
                is_meaningful_fallback = not ("否" not in current_raw_answer)
                explanation = f"Fallback logic: '否' not in '{current_raw_answer}' -> Meaningful: {is_meaningful_fallback}"
                print(f"Item ID: {item_id_for_log} - {explanation}")
                return is_meaningful_fallback, current_raw_answer

        except openai.APIConnectionError as e:
            last_raw_answer = f"API_CONNECTION_ERROR: {e}"
            print(
                f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - Error connecting to API: {e}"
            )
        except openai.RateLimitError as e:
            last_raw_answer = f"RATE_LIMIT_ERROR: {e}"
            print(
                f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - Rate limit exceeded: {e}."
            )
        except openai.APIStatusError as e:
            last_raw_answer = f"API_STATUS_ERROR: {e.status_code} - {e.response}"
            print(
                f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - API status error: {e.status_code} - {e.response}"
            )
        except Exception as e:
            last_raw_answer = f"GENERAL_ERROR: {type(e).__name__} - {e}"
            print(
                f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - Error querying model: {type(e).__name__} - {e}"
            )

        if attempt < MAX_MODEL_RETRIES - 1:
            print(
                f"Item ID: {item_id_for_log} - Retrying after error/unexpected response in {RETRY_DELAY_SECONDS}s..."
            )
            await asyncio.sleep(RETRY_DELAY_SECONDS)
        elif attempt == MAX_MODEL_RETRIES - 1:
            print(
                f"Item ID: {item_id_for_log} - All retry attempts failed or ended with non-standard response. Final recorded raw answer: '{last_raw_answer}'. Defaulting to NOT meaningful."
            )
            return False, last_raw_answer

    print(
        f"Item ID: {item_id_for_log} - Unexpected exit from retry loop. Defaulting to NOT meaningful. Last raw answer: {last_raw_answer}"
    )
    return False, last_raw_answer


async def process_article_item(
    client: openai.AsyncOpenAI,
    article_json_obj: dict,  # Parsed JSON object from a line in the input file
    item_identifier: str,  # Identifier for logging and potentially for output
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """
    Extracts content, determines if it's meaningful using the model,
    and returns a dictionary for the output JSONL if meaningful, otherwise None.
    """
    async with semaphore:
        article_content = article_json_obj.get(INPUT_JSONL_CONTENT_KEY)

        if (
            not article_content
            or not isinstance(article_content, str)
            or not article_content.strip()
        ):
            print(
                f"Item ID: {item_identifier} - Missing, invalid, or empty content from key '{INPUT_JSONL_CONTENT_KEY}'. Skipping."
            )
            return None

        is_meaningful, raw_model_output = await get_model_decision(
            client, article_content, item_identifier
        )

        if is_meaningful:
            output_data = {
                "identifier": item_identifier,  # The ID from input JSONL or line number
                "content": article_content,
                "model_raw_output": raw_model_output,
            }
            # If you want to include all original fields from article_json_obj:
            # output_data = article_json_obj.copy()
            # output_data["model_raw_output"] = raw_model_output
            # output_data["llm_is_meaningful"] = True # Optional: add a specific flag
            return output_data
        else:
            # Item is not meaningful, or an error occurred in decision making
            return None


async def main():
    if not INPUT_JSONL_FILE.exists():
        print(f"Error: Input JSONL file '{INPUT_JSONL_FILE}' does not exist.")
        return

    print(f"Processing articles from: {INPUT_JSONL_FILE.resolve()}")
    print(f"Saving meaningful articles to: {OUTPUT_JSONL_FILE.resolve()}")
    print(
        f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}, Model retries: {MAX_MODEL_RETRIES}"
    )

    if OUTPUT_JSONL_FILE.exists():
        print(f"Output file {OUTPUT_JSONL_FILE} already exists. Appending new data.")
        # Consider OUTPUT_JSONL_FILE.unlink(missing_ok=True) if you want to overwrite

    async with openai.AsyncOpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    ) as client:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        jsonl_writer_lock = asyncio.Lock()
        tasks = []

        total_items_in_input = 0
        malformed_json_lines = 0

        # First pass to count items and prepare tasks (to get total count for progress)
        article_batch_data = []  # Store (json_obj, identifier)
        try:
            async with aiofiles.open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_in:
                line_number = 0
                async for line in f_in:
                    line_number += 1
                    total_items_in_input += 1
                    try:
                        json_obj = json.loads(line)
                        identifier = json_obj.get(INPUT_JSONL_ID_KEY)
                        if not identifier:  # Fallback identifier logic
                            identifier = json_obj.get(
                                "source_path", f"line_{line_number}"
                            )
                        article_batch_data.append((json_obj, identifier))
                    except json.JSONDecodeError:
                        print(
                            f"Warning: Malformed JSON on line {line_number} in '{INPUT_JSONL_FILE}'. Skipping this line."
                        )
                        malformed_json_lines += 1
                        continue
        except FileNotFoundError:
            print(
                f"Error: Input file {INPUT_JSONL_FILE} not found during second read phase (should not happen if first check passed)."
            )
            return

        if not article_batch_data:
            print("No valid articles found in the input JSONL file. Exiting.")
            return

        print(
            f"Found {len(article_batch_data)} valid articles (out of {total_items_in_input} lines, {malformed_json_lines} malformed) to process."
        )

        for json_obj, identifier in article_batch_data:
            tasks.append(process_article_item(client, json_obj, identifier, semaphore))

        meaningful_articles_count = 0
        processed_tasks_count = 0

        async with aiofiles.open(
            OUTPUT_JSONL_FILE, "a", encoding="utf-8"
        ) as jsonl_f_out:
            for future in asyncio.as_completed(tasks):
                processed_tasks_count += 1
                try:
                    result_dict = (
                        await future
                    )  # This is the dict from process_article_item or None
                    if result_dict:
                        async with jsonl_writer_lock:
                            await jsonl_f_out.write(
                                json.dumps(result_dict, ensure_ascii=False) + "\n"
                            )
                        meaningful_articles_count += 1

                    if processed_tasks_count % 100 == 0 or processed_tasks_count == len(
                        tasks
                    ):  # Log progress
                        print(
                            f"Progress: {processed_tasks_count}/{len(tasks)} tasks completed. Meaningful so far: {meaningful_articles_count}"
                        )

                except Exception as e:
                    # This exception would be from await future itself if the task raised an unhandled one
                    # (process_article_item should ideally handle its own errors and return None)
                    print(f"Error processing a task future: {e}")

    print(f"\n--- Processing Complete ---")
    print(f"Total lines in input file: {total_items_in_input}")
    if malformed_json_lines > 0:
        print(f"Malformed JSON lines skipped: {malformed_json_lines}")
    print(f"Valid articles processed: {len(article_batch_data)}")
    print(f"Meaningful articles saved to JSONL: {meaningful_articles_count}")
    print(f"Output JSONL file: {OUTPUT_JSONL_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
