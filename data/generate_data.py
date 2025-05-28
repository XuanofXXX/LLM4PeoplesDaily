# async_vllm_batch.py

import re
import sys
import asyncio
import torch    
from uuid import uuid4
import json
from tqdm.asyncio import tqdm_asyncio

from vllm import SamplingParams, AsyncLLMEngine, AsyncEngineArgs


TEMPLATE = """
## 角色 
你是一位出题专家，擅长从《人民日报》的新闻报道中制作填空题。

## 任务
请根据以下提供的《人民日报》文章片段（内容均选自2023年5月至2024年4月），生成5个高质量的填空题。

## 要求
1.  每个填空题都必须基于所提供的文章片段。
2.  将原文中的一个关键信息（如人名、地名、机构名、具体日期、数字、重要事件的核心词等）替换为 "_______"（三个下划线）作为题目。
3.  "_______" 所代表的内容即为该题的答案。
4.  答案必须是原文中精确的词语或短语。
5.  确保问题在缺少填空内容后依然通顺，并且指向性明确。
6.  生成的每个问题和答案请按照以下格式提供：
    Q: [填空后的句子]
    A: [被替换掉的原文内容]
    ---

## 文章片段
{content}

## 示例


请开始生成填空题。
""".strip()
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/fs/archive/share/yulan-team/YuLan-Mini"
# MODEL_PATH = "/fs/archive/share/yulan-team/YuLan-Mini"
INPUT_FILE = "/home/u20140041/xiachunxuan/eval_yulan/eval/instruction_following_eval/data/input_data.jsonl"
OUTPUT_FILE = f"answers_{MODEL_PATH}.jsonl"

async def batch_generate(model_path: str,
                         output_file: str,
                         max_tokens: int = 2000,
                         temperature: float = 0.7):
    # 1. 读 prompts
    with open(INPUT_FILE, 'r') as f:
        prompts = [json.loads(line)['prompt'] for line in f]

    def extract_size(path):
        m = re.search(r'(\d+(?:\.\d+)?B)', path)
        return m.group(1) if m else None

    # 2. 初始化异步引擎
    size = extract_size(model_path)
    engine_args = None
    
    if (size is not None and float(size[:-1]) < 20) or torch.cuda.device_count() == 1:
        engine_args =  AsyncEngineArgs(model=model_path, 
                                    disable_log_requests=True)
    else:
        engine_args = AsyncEngineArgs(model=model_path, 
                                    disable_log_requests=True, tensor_parallel_size=4,pipeline_parallel_size=1,
                                    distributed_executor_backend="mp")
    

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # 3. 并发提交
    async def run_query(prompt: str) -> str:
        req_id = uuid4().hex
        outputs = engine.generate(prompt,
                                  SamplingParams(max_tokens=max_tokens, temperature=temperature),
                                  req_id)
        last = None
        async for out in outputs:
            last = out
        return last.outputs[0].text

    tasks = [run_query(p) for p in prompts]
    answers = await tqdm_asyncio.gather(*tasks, desc=f"Gen {model_path}", total=len(prompts))

    # 4. 写结果
    with open(output_file, 'w') as fout:
        for prompt, ans in zip(prompts, answers):
            fout.write(json.dumps({"prompt": prompt, "response": ans}, ensure_ascii=False) + "\n")


async def run_query(engine: AsyncLLMEngine, prompt: str) -> str:
    request_id = uuid4().hex
    outputs = engine.generate(prompt, SamplingParams(max_tokens=2000, temperature=0.7), request_id)
    last_output = None
    async for out in outputs:
        last_output = out
    # 返回模型最终文字
    return last_output.outputs[0].text

async def main():
    with open(INPUT_FILE, 'r') as f:
        prompts = [json.loads(line)['prompt'] for line in f]

    # 2. 用异步引擎加载模型（内存里一次初始化，后续所有请求复用）
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=MODEL_PATH)
    )

    # 3. 并发提交所有请求并收集答案
    tasks = [run_query(engine, p) for p in prompts]
    answers = await tqdm_asyncio.gather(*tasks, desc="Generating", total=len(prompts))

    # 4. 写文件
    with open(OUTPUT_FILE, 'w') as fout:
        for prompt, ans in zip(prompts, answers):
            fout.write(json.dumps({"prompt": prompt, "response": ans}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    asyncio.run(main())
