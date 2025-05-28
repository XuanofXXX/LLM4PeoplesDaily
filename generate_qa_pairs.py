import openai
import asyncio
import aiofiles
import json
import csv
from pathlib import Path
from tqdm.asyncio import tqdm

# --- Configuration ---
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen3-32B/"  # 使用 Qwen3-32B 模型

# Input JSONL file
INPUT_JSONL_FILE = Path("data/llm_processed_meaningful_articles_v2_rag.jsonl")
# Key in the input JSONL that contains the article text
INPUT_JSONL_CONTENT_KEY = "contents"
# Key in the input JSONL to use as an identifier
INPUT_JSONL_ID_KEY = "identifier"

# Output TSV file for QA pairs (similar to example_ans.tsv format)
OUTPUT_TSV_FILE = Path("generated_qa_pairs.tsv")

# Concurrency and Retry Settings
MAX_CONCURRENT_TASKS = 200  # 降低并发数，避免对模型造成压力
MAX_MODEL_RETRIES = 5
RETRY_DELAY_SECONDS = 5

# 生成问答对的提示模板
PROMPT_TEMPLATE = """
## 指令
请根据以下文章内容，生成1-3个高质量的问答对。要求：

1. 问题要具体、明确，能够从文章中找到明确答案
2. 答案要简洁准确，不超过20个字
3. 问题应涵盖文章的关键信息，如时间、地点、人物、事件、数据等。问题提及的各类时间应该完整表述，如“2023年10月1日”而不是“10月1日”，“中国特别研讨会”而不是“研讨会”。
4. 避免生成过于简单或过于复杂的问题
5. 问题要有一定的知识价值，适合用作问答测试
6. 答案应直接回答问题，不需要额外解释

请严格按照以下格式输出，每行一个问答对，用": "分隔问题和答案。例如：
- 问题1: 答案1
- 问题2: 答案2
- 问题3: 答案3

## 文章内容
{content}
""".strip()


async def generate_qa_pairs(
    client: openai.AsyncOpenAI, article_content: str, item_id_for_log: str
) -> list[tuple[str, str]]:
    """
    使用模型生成问答对
    返回: [(question, answer), ...] 或空列表
    """
    # 限制文章长度，避免超过模型上下文限制
    max_content_length = 3000
    if len(article_content) > max_content_length:
        article_content = article_content[:max_content_length] + "..."
    
    date = item_id_for_log.split('/')[1]
    date = f"{date[:4]}年{date[4:6]}月{date[6:]}日" if len(date) == 8 else date
    article_content = f"{date}报告\n\n{article_content}"
    prompt = PROMPT_TEMPLATE.format(content=article_content)
    
    for attempt in range(MAX_MODEL_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "你是一个专业的问答对生成助手，能够根据文章内容生成高质量的问答对。"
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0,
            )

            raw_response = response.choices[0].message.content.strip()
            # print(f"Item ID: {item_id_for_log} - Model response: {raw_response}")
            # print(f"Item ID: {item_id_for_log}")
            
            # 解析模型输出的问答对
            raw_response = raw_response.split('</think>')[-1]
            qa_pairs = []
            lines = raw_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('- '):
                    # 处理以 "- " 开头的行
                    line = line[2:].strip()
                    parts = line.split(': ')
                    if len(parts) >= 2:
                        question = parts[-2].strip()
                        answer = parts[-1].strip()
                        # 过滤掉过短或不合理的问答对
                        if len(question) >= 10 and len(answer) >= 2 and len(answer) <= 50:
                            qa_pairs.append((question, answer))
            
            if qa_pairs:
                print(f"Item ID: {item_id_for_log} - Successfully generated {len(qa_pairs)} QA pairs")
                return qa_pairs
            else:
                print(f"Item ID: {item_id_for_log} - No valid QA pairs found in response")
                if attempt < MAX_MODEL_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                    
        except openai.APIConnectionError as e:
            print(f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - API connection error: {e}")
        except openai.RateLimitError as e:
            print(f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - Rate limit error: {e}")
        except openai.APIStatusError as e:
            print(f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - API status error: {e.status_code}")
        except Exception as e:
            print(f"Item ID: {item_id_for_log} - Attempt {attempt + 1}/{MAX_MODEL_RETRIES} - General error: {type(e).__name__} - {e}")
        
        if attempt < MAX_MODEL_RETRIES - 1:
            await asyncio.sleep(RETRY_DELAY_SECONDS)
    
    print(f"Item ID: {item_id_for_log} - Failed to generate QA pairs after all retries")
    return []


async def process_article_item(
    client: openai.AsyncOpenAI,
    article_json_obj: dict,
    item_identifier: str,
    semaphore: asyncio.Semaphore,
) -> list[tuple[str, str]]:
    """
    处理单篇文章，生成问答对
    返回问答对列表或空列表
    """
    async with semaphore:
        article_content = article_json_obj.get(INPUT_JSONL_CONTENT_KEY)

        if (
            not article_content
            or not isinstance(article_content, str)
            or not article_content.strip()
        ):
            print(f"Item ID: {item_identifier} - Missing or empty content")
            return []

        qa_pairs = await generate_qa_pairs(client, article_content, item_identifier)
        return qa_pairs


async def main():
    if not INPUT_JSONL_FILE.exists():
        print(f"Error: Input JSONL file '{INPUT_JSONL_FILE}' does not exist.")
        return

    print(f"Processing articles from: {INPUT_JSONL_FILE.resolve()}")
    print(f"Saving QA pairs to: {OUTPUT_TSV_FILE.resolve()}")
    print(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}, Model retries: {MAX_MODEL_RETRIES}")

    # 如果输出文件已存在，删除它
    if OUTPUT_TSV_FILE.exists():
        OUTPUT_TSV_FILE.unlink()

    async with openai.AsyncOpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
    ) as client:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        tasks = []

        # 读取并准备文章数据
        article_batch_data = []
        total_items = 0
        processed_items = 0
        
        try:
            async with aiofiles.open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_in:
                line_number = 0
                async for line in f_in:
                    line_number += 1
                    total_items += 1
                    try:
                        json_obj = json.loads(line)
                        identifier = json_obj.get(INPUT_JSONL_ID_KEY, f"line_{line_number}")
                        article_batch_data.append((json_obj, identifier))
                        
                        # 限制处理的文章数量以控制时间
                        # if len(article_batch_data) >= 50:  # 只处理前50篇文章作为示例
                        #     break
                            
                    except json.JSONDecodeError:
                        print(f"Warning: Malformed JSON on line {line_number}")
                        continue
        except FileNotFoundError:
            print(f"Error: Input file {INPUT_JSONL_FILE} not found")
            return

        if not article_batch_data:
            print("No valid articles found in the input file.")
            return

        print(f"Found {len(article_batch_data)} articles to process")

        # 创建处理任务
        for json_obj, identifier in article_batch_data:
            tasks.append(process_article_item(client, json_obj, identifier, semaphore))

        # 准备写入TSV文件
        all_qa_pairs = []
        
        # 处理任务并收集结果
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理文章"):
            processed_items += 1
            try:
                qa_pairs = await future
                all_qa_pairs.extend(qa_pairs)
                
                if processed_items % 10 == 0 or processed_items == len(tasks):
                    print(f"Progress: {processed_items}/{len(tasks)} articles processed. Total QA pairs: {len(all_qa_pairs)}")
                    
            except Exception as e:
                print(f"Error processing a task: {e}")

        # 写入TSV文件
        if all_qa_pairs:
            with open(OUTPUT_TSV_FILE, 'w', encoding='utf-8', newline='') as f_out:
                writer = csv.writer(f_out, delimiter='\t')
                for question, answer in all_qa_pairs:
                    writer.writerow([question, answer])
        
        print(f"\n--- Processing Complete ---")
        print(f"Total articles processed: {len(article_batch_data)}")
        print(f"Total QA pairs generated: {len(all_qa_pairs)}")
        print(f"Output TSV file: {OUTPUT_TSV_FILE.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())