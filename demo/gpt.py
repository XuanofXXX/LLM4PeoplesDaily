import json
import re
from rank_bm25 import BM25Okapi
import jieba
import os
from openai import AsyncOpenAI, OpenAI
import logging
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import functools

from utils import is_match, is_set_match


# --- Logger Configuration ---
def setup_logger():
    """设置同时输出到文件和终端的logger"""
    # 创建logger
    logger = logging.getLogger("rag_system")
    logger.setLevel(logging.INFO)

    # 如果logger已经有handler，则清除（避免重复添加）
    if logger.handlers:
        logger.handlers.clear()

    # 创建格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 创建文件handler
    log_filename = f"logs/rag_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)  # 确保日志目录存在
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 创建终端handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 初始化logger
logger = setup_logger()

# --- Configuration ---
JSONL_FILE_PATH = "data/llm_processed_meaningful_articles_v3_rag.jsonl"
EXAMPLE_ANS_PATH = "data/eval/generated_qa_pairs.jsonl"
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"  # 使用完整路径
MAX_NEW_TOKENS = 500
TOP_K_DOCUMENTS = 3

# Dense retrieval配置
DENSE_MODEL_NAME = (
    "/media/public/models/huggingface/BAAI/bge-large-zh-v1.5"  # 中文语义检索模型
)
RETRIEVAL_METHOD = "hybrid"  # 选项: "bm25", "dense", "hybrid"
BM25_WEIGHT = 0.3  # 混合检索中BM25的权重
DENSE_WEIGHT = 0.7  # 混合检索中dense retrieval的权重
CACHE_BASE_DIR = "cache"  # 基础缓存目录

# 迭代RAG配置
USE_ITERATIVE_RAG = True  # 是否使用迭代RAG (交替检索和生成)
MAX_ITERATIONS = 2  # 最大迭代次数
ITERATIVE_TOP_K = 5  # 迭代RAG中每次检索的文档数量

PROMPT_TEMPLATE = """
## 指令
你是一个基于检索的问答助手。请根据以下提供的上下文信息来回答问题。如果上下文信息不足以回答问题，请说明你无法从提供的信息中找到答案。请直接回答问题所提问的人名、地名、主题等等，不要任何多余的话。如果答案是多个人或多个单位，请用分号（；）分隔。

## 参考示例
问题：2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？
答案：湖南第一师范学院

问题：2024年是中国红十字会成立多少周年？
答案：120

问题：哪些单位在中国期刊高质量发展论坛的主论坛上做主题演讲？
答案：中国科协科技创新部；湖南省委宣传部；上海大学；《历史研究》；《读者》；《分子植物》；《问天少年》；南方杂志社；中华医学会杂志社大学。

## 上下文信息
{context}

## 问题
{query}
""".strip()

client = OpenAI(
    base_url=VLLM_API_BASE,
    model=MODEL_NAME,
    max_tokens=MAX_NEW_TOKENS,
    temperature=0.0,  # 设置为0.0以获得确定性输出
    request_timeout=60,  # 设置请求超时时间
)

def google_search(query):
    """
    模拟Google搜索，返回一个固定的结果。
    实际应用中可以替换为真实的搜索API调用。
    """
    return f"Google search results for: {query}"