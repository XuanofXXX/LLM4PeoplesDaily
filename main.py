import json
import re
import pandas as pd
from rank_bm25 import BM25Okapi
import jieba
import torch
import os
from openai import OpenAI
import logging
from datetime import datetime

from utils import cal_em, is_match, is_set_match


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
JSONL_FILE_PATH = "data/llm_processed_meaningful_articles_v2_rag.jsonl"
EXAMPLE_ANS_PATH = "data/test_ans.json"
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"  # 使用完整路径
MAX_NEW_TOKENS = 500
TOP_K_DOCUMENTS = 3

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


# --- 1. Data Loading and Preprocessing ---
def load_documents_from_jsonl(file_path):
    """从 JSONL 文件加载文档内容"""
    documents = []
    doc_ids = []
    logger.info(f"Loading documents from {file_path}...")

    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found!")
        return [], []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                content = (
                    data.get("contents")
                    or data.get("text")
                    or data.get("article")
                    or data.get("body")
                )
                if content:
                    documents.append(content)
                    doc_ids.append(
                        data.get("identifier") or data.get("id") or f"doc_{i}"
                    )
                else:
                    logger.warning(f"Missing content in line {i + 1}")
            except json.JSONDecodeError:
                logger.warning(f"Could not decode JSON in line {i + 1}")

    logger.info(f"Loaded {len(documents)} documents.")
    return documents, doc_ids


def load_test_queries(file_path):
    """从TSV文件加载测试问题"""
    if not os.path.exists(file_path):
        logger.warning(f"Test file {file_path} not found!")
        return []

    try:
        # 读取文件并手动解析，因为使用的是 ||| 分隔符而不是制表符
        if file_path.endswith(".tsv"):
            queries_and_answers = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and "|||" in line:
                        parts = line.split("|||")
                        if len(parts) >= 2:
                            query = parts[0].strip()
                            answer = parts[1].strip()
                            queries_and_answers.append((query, answer, ""))
                        else:
                            logger.warning(f"Invalid format in line {line_num}: {line}")
                    else:
                        logger.warning(f"No separator found in line {line_num}: {line}")
        elif file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                queries_and_answers = json.load(f)
                if not isinstance(queries_and_answers, list):
                    logger.warning(
                        f"Expected a list in {file_path}, got {type(queries_and_answers)}"
                    )
                    return []
                queries_and_answers = [
                    (q["question"], q["answer"], q.get("reference", ""))
                    for q in queries_and_answers
                ]

        logger.info(f"Loaded {len(queries_and_answers)} test queries from {file_path}")
        return queries_and_answers
    except Exception as e:
        logger.error(f"Error loading test queries: {e}")
        return []


def chinese_tokenizer(text):
    """使用 jieba 进行中文分词"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return list(jieba.cut(text))


# --- 2. BM25 Indexing ---
def build_bm25_index(documents):
    """构建 BM25 索引"""
    logger.info("Tokenizing documents for BM25...")
    tokenized_corpus = [chinese_tokenizer(doc) for doc in documents]
    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built.")
    return bm25


# --- 3. OpenAI Client Setup ---
def setup_openai_client():
    """设置 OpenAI 客户端连接到 VLLM 服务器"""
    logger.info(
        f"Setting up OpenAI client to connect to VLLM server at {VLLM_API_BASE}..."
    )
    client = OpenAI(
        api_key="EMPTY",  # VLLM 服务器不需要真实的 API key
        base_url=VLLM_API_BASE,
    )
    logger.info("OpenAI client setup complete.")
    return client


# --- 4. RAG Query Function ---
def rag_query(
    query, bm25_index, original_documents, doc_ids, client, top_k=TOP_K_DOCUMENTS
):
    """执行 RAG 查询"""
    logger.info(f"Performing RAG query: '{query}'")
    # 1. Retrieve relevant documents
    tokenized_query = chinese_tokenizer(query)
    doc_scores = bm25_index.get_scores(tokenized_query)

    top_n_indices = sorted(
        range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
    )[:top_k]
    retrieved_docs_content = [original_documents[i] for i in top_n_indices]
    retrieved_docs_ids = [doc_ids[i] for i in top_n_indices]
    logger.info(f"Retrieved {len(retrieved_docs_content)} documents.")
    for i, doc_id in enumerate(retrieved_docs_ids):
        logger.info(f"  Doc ID: {doc_id}, Score: {doc_scores[top_n_indices[i]]:.4f}")

    # 2. Truncate documents to fit within model limits
    max_doc_length = 2000  # 限制每个文档的最大字符数
    truncated_docs = []
    for doc in retrieved_docs_content:
        if len(doc) > max_doc_length:
            truncated_docs.append(doc[:max_doc_length] + "...")
        else:
            truncated_docs.append(doc)

    # 3. Construct context and messages

    context = "\n\n".join(
        [f"文档{i + 1}: {doc}" for i, doc in enumerate(truncated_docs)]
    )
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    # 使用标准的 OpenAI 消息格式
    messages = [
        {
            "role": "system",
            "content": "你是一个基于检索的问答助手。请根据以下提供的上下文信息来回答问题。如果上下文信息不足以回答问题，请说明你无法从提供的信息中找到答案。请用最简洁准确的语言回答问题。",
        },
        {"role": "user", "content": prompt},
    ]

    # logger.info(f"Sending request to VLLM server...")
    logger.info(f"Context length: {len(context)} characters")

    # 4. Generate answer using OpenAI API format
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0,
            top_p=0.8,
            stop=None,
        )

        answer = response.choices[0].message.content.strip()

        logger.info("LLM Answer:")
        logger.info(answer)
        return answer, retrieved_docs_ids, retrieved_docs_content

    except Exception as e:
        logger.error(f"Error during API call: {e}")
        return "抱歉，生成答案时遇到问题。", retrieved_docs_ids, retrieved_docs_content


# --- 5. Batch Testing Function ---
def run_batch_test(test_queries, bm25_index, documents, doc_ids, client):
    """批量运行测试查询"""
    logger.info(f"=== Running batch test on {len(test_queries)} queries ===")

    results = []
    for i, (query, expected_answer, reference) in enumerate(test_queries):
        logger.info(f"--- Test {i + 1}/{len(test_queries)} ---")
        logger.info(f"Query: {query}")
        logger.info(f"Expected: {expected_answer}")
        logger.info(f"Reference: {reference}")

        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            query, bm25_index, documents, doc_ids, client
        )

        results.append(
            {
                "is_correct": is_set_match(generated_answer, expected_answer)
                or (
                    is_match(pred=generated_answer, gold=expected_answer)
                    and "；" not in expected_answer
                ),
                "query": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "retrieved_docs": retrieved_ids,
                "retrieved_contents": retrieved_contents,
                "reference": reference,
            }
        )

        logger.info("-" * 50)

    return results


# --- Main Execution ---
if __name__ == "__main__":
    # 1. 加载数据
    documents, doc_ids = load_documents_from_jsonl(JSONL_FILE_PATH)

    if not documents:
        logger.error("No documents loaded. Exiting.")
        exit()

    # 2. 构建 BM25 索引
    bm25_index = build_bm25_index(documents)

    # 3. 设置 OpenAI 客户端
    try:
        client = setup_openai_client()
    except Exception as e:
        logger.error(f"Error setting up OpenAI client: {e}")
        logger.error("Please make sure VLLM server is running on http://localhost:8000")
        exit()

    # 4. 加载测试查询
    test_queries = load_test_queries(EXAMPLE_ANS_PATH)

    if test_queries:
        # 运行批量测试
        test_results = run_batch_test(
            test_queries, bm25_index, documents, doc_ids, client
        )

        # 保存测试结果
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        logger.info("Test results saved to test_results.json")
        from utils import cal_em

        pred = [result["generated_answer"] for result in test_results]
        ans = [result["expected_answer"] for result in test_results]

        logger.info(f"acc: {cal_em(pred, ans)}")
    else:
        logger.info("No test queries found, running example query...")
        test_query = "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"
        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            test_query, bm25_index, documents, doc_ids, client
        )
