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
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import hashlib

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
EXAMPLE_ANS_PATH = "data/eval/official_test_ans.json"
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"  # 使用完整路径
MAX_NEW_TOKENS = 500
TOP_K_DOCUMENTS = 3

# Dense retrieval配置
DENSE_MODEL_NAME = "/media/public/models/huggingface/BAAI/bge-large-zh-v1.5"  # 中文语义检索模型
RETRIEVAL_METHOD = "hybrid"  # 选项: "bm25", "dense", "hybrid"
BM25_WEIGHT = 0.3  # 混合检索中BM25的权重
DENSE_WEIGHT = 0.7  # 混合检索中dense retrieval的权重
CACHE_BASE_DIR = "cache"  # 基础缓存目录

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


# --- Cache Management ---
def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        return None
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_cache_dir(file_path, retrieval_method, dense_model_name):
    """根据文件哈希和配置生成缓存目录"""
    file_hash = calculate_file_hash(file_path)
    if not file_hash:
        return None
    
    # 使用文件哈希前8位作为目录名，避免过长
    cache_dir_name = f"{file_hash[:8]}_{retrieval_method}_{os.path.basename(dense_model_name)}"
    cache_dir = os.path.join(CACHE_BASE_DIR, cache_dir_name)
    return cache_dir, file_hash

def get_cache_paths(cache_dir):
    """获取缓存文件路径"""
    if not cache_dir:
        return None, None, None
    
    return {
        'embeddings': os.path.join(cache_dir, "document_embeddings.pkl"),
        'bm25': os.path.join(cache_dir, "bm25_index.pkl"),
        'info': os.path.join(cache_dir, "cache_info.json")
    }

def save_cache_info(cache_dir, file_hash, retrieval_method, dense_model_name, file_path):
    """保存缓存信息"""
    cache_info = {
        "file_hash": file_hash,
        "file_path": file_path,
        "retrieval_method": retrieval_method,
        "dense_model_name": dense_model_name,
        "timestamp": datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat()
    }
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_info_path = os.path.join(cache_dir, "cache_info.json")
    with open(cache_info_path, "w", encoding="utf-8") as f:
        json.dump(cache_info, f, ensure_ascii=False, indent=2)

def load_cache_info(cache_dir):
    """加载缓存信息"""
    if not cache_dir:
        return None
    
    cache_info_path = os.path.join(cache_dir, "cache_info.json")
    if not os.path.exists(cache_info_path):
        return None
    
    try:
        with open(cache_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load cache info from {cache_info_path}: {e}")
        return None

def update_cache_access_time(cache_dir):
    """更新缓存访问时间"""
    if not cache_dir:
        return
    
    cache_info = load_cache_info(cache_dir)
    if cache_info:
        cache_info["last_accessed"] = datetime.now().isoformat()
        cache_info_path = os.path.join(cache_dir, "cache_info.json")
        with open(cache_info_path, "w", encoding="utf-8") as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)

def is_cache_valid(cache_dir, file_path, retrieval_method, dense_model_name):
    """检查缓存是否有效"""
    cache_info = load_cache_info(cache_dir)
    if not cache_info:
        return False
    
    current_hash = calculate_file_hash(file_path)
    
    is_valid = (
        cache_info.get("file_hash") == current_hash and
        cache_info.get("retrieval_method") == retrieval_method and
        cache_info.get("dense_model_name") == dense_model_name
    )
    
    if is_valid:
        update_cache_access_time(cache_dir)
        logger.info(f"Using existing cache from {cache_dir}")
    
    return is_valid

def list_available_caches():
    """列出所有可用的缓存"""
    if not os.path.exists(CACHE_BASE_DIR):
        return []
    
    caches = []
    for cache_dir_name in os.listdir(CACHE_BASE_DIR):
        cache_dir = os.path.join(CACHE_BASE_DIR, cache_dir_name)
        if os.path.isdir(cache_dir):
            cache_info = load_cache_info(cache_dir)
            if cache_info:
                caches.append({
                    'dir': cache_dir,
                    'info': cache_info
                })
    
    return caches

def cleanup_old_caches(max_caches=10):
    """清理旧的缓存，保留最近使用的缓存"""
    caches = list_available_caches()
    if len(caches) <= max_caches:
        return
    
    # 按最后访问时间排序
    caches.sort(key=lambda x: x['info'].get('last_accessed', ''), reverse=True)
    
    # 删除最旧的缓存
    for cache in caches[max_caches:]:
        cache_dir = cache['dir']
        try:
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Cleaned up old cache: {cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup cache {cache_dir}: {e}")


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


# --- 2. Dense Retrieval Setup ---
def setup_dense_retrieval_model():
    """设置dense retrieval模型"""
    logger.info(f"Loading dense retrieval model: {DENSE_MODEL_NAME}")
    try:
        model = SentenceTransformer(DENSE_MODEL_NAME)
        logger.info("Dense retrieval model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load dense retrieval model: {e}")
        logger.info("Falling back to BM25 only.")
        return None


def build_document_embeddings(documents, dense_model, cache_path=None):
    """构建文档的dense embeddings"""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached document embeddings from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
            logger.info(f"Loaded {embeddings.shape[0]} cached embeddings.")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load embeddings cache: {e}")

    logger.info("Building document embeddings...")
    # 预处理文档：截断过长的文档
    processed_docs = []
    for doc in documents:
        if len(doc) > 512:  # 限制输入长度
            processed_docs.append(doc[:512])
        else:
            processed_docs.append(doc)

    embeddings = dense_model.encode(
        processed_docs, batch_size=32, show_progress_bar=True
    )

    # 缓存embeddings
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Document embeddings cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

    logger.info(f"Built embeddings for {len(embeddings)} documents.")
    return embeddings


# --- 3. BM25 Indexing ---
def build_bm25_index(documents, cache_path=None):
    """构建 BM25 索引，支持缓存"""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached BM25 index from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                bm25 = pickle.load(f)
            logger.info("BM25 index loaded from cache.")
            return bm25
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}")
    
    logger.info("Tokenizing documents for BM25...")
    tokenized_corpus = [chinese_tokenizer(doc) for doc in documents]
    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built.")
    
    # 缓存BM25索引
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(bm25, f)
            logger.info(f"BM25 index cached to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache BM25 index: {e}")
    
    return bm25


# --- 4. Retrieval Functions ---
def bm25_retrieve(query, bm25_index, top_k):
    """使用BM25检索"""
    tokenized_query = chinese_tokenizer(query)
    doc_scores = bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    return top_indices, doc_scores[top_indices]


def dense_retrieve(query, dense_model, document_embeddings, top_k):
    """使用dense retrieval检索"""
    query_embedding = dense_model.encode([query])
    similarities = cosine_similarity(query_embedding, document_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]


def hybrid_retrieve(
    query,
    bm25_index,
    dense_model,
    document_embeddings,
    top_k,
    bm25_weight=BM25_WEIGHT,
    dense_weight=DENSE_WEIGHT,
):
    """混合检索策略"""
    # 获取更多的候选文档用于重排序
    candidate_k = min(top_k * 3, len(document_embeddings))

    # BM25检索
    bm25_indices, bm25_scores = bm25_retrieve(query, bm25_index, candidate_k)

    # Dense检索
    dense_indices, dense_scores = dense_retrieve(
        query, dense_model, document_embeddings, candidate_k
    )

    # 归一化分数
    bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (
        bm25_scores.max() - bm25_scores.min() + 1e-8
    )
    dense_scores_norm = (dense_scores - dense_scores.min()) / (
        dense_scores.max() - dense_scores.min() + 1e-8
    )

    # 合并分数
    combined_scores = {}

    # 添加BM25分数
    for i, idx in enumerate(bm25_indices):
        combined_scores[idx] = bm25_weight * bm25_scores_norm[i]

    # 添加dense分数
    for i, idx in enumerate(dense_indices):
        if idx in combined_scores:
            combined_scores[idx] += dense_weight * dense_scores_norm[i]
        else:
            combined_scores[idx] = dense_weight * dense_scores_norm[i]

    # 排序并返回top_k
    sorted_indices = sorted(
        combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True
    )[:top_k]
    sorted_scores = [combined_scores[idx] for idx in sorted_indices]

    return np.array(sorted_indices), np.array(sorted_scores)


# --- 5. OpenAI Client Setup ---
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


# --- 6. RAG Query Function ---
def rag_query(
    query, retrieval_components, documents, doc_ids, client, top_k=TOP_K_DOCUMENTS
):
    """执行 RAG 查询"""
    logger.info(f"Performing RAG query: '{query}' using {RETRIEVAL_METHOD} retrieval")

    bm25_index, dense_model, document_embeddings = retrieval_components

    # 根据配置选择检索方法
    if RETRIEVAL_METHOD == "bm25":
        top_indices, scores = bm25_retrieve(query, bm25_index, top_k)
    elif RETRIEVAL_METHOD == "dense" and dense_model is not None:
        top_indices, scores = dense_retrieve(
            query, dense_model, document_embeddings, top_k
        )
    elif RETRIEVAL_METHOD == "hybrid" and dense_model is not None:
        top_indices, scores = hybrid_retrieve(
            query, bm25_index, dense_model, document_embeddings, top_k
        )
    else:
        logger.warning(
            "Dense model not available or invalid retrieval method, falling back to BM25"
        )
        top_indices, scores = bm25_retrieve(query, bm25_index, top_k)

    retrieved_docs_content = [documents[i] for i in top_indices]
    retrieved_docs_ids = [doc_ids[i] for i in top_indices]

    logger.info(
        f"Retrieved {len(retrieved_docs_content)} documents using {RETRIEVAL_METHOD}:"
    )
    for i, (doc_id, score) in enumerate(zip(retrieved_docs_ids, scores)):
        logger.info(f"  Doc ID: {doc_id}, Score: {score:.4f}")

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


# --- 7. Batch Testing Function ---
def run_batch_test(test_queries, retrieval_components, documents, doc_ids, client):
    """批量运行测试查询"""
    logger.info(f"=== Running batch test on {len(test_queries)} queries ===")

    results = []
    for i, (query, expected_answer, reference) in enumerate(test_queries):
        logger.info(f"--- Test {i + 1}/{len(test_queries)} ---")
        logger.info(f"Query: {query}")
        logger.info(f"Expected: {expected_answer}")
        logger.info(f"Reference: {reference}")

        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            query, retrieval_components, documents, doc_ids, client
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

    # 2. 获取缓存目录和路径
    cache_result = get_cache_dir(JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME)
    if cache_result:
        cache_dir, file_hash = cache_result
        cache_paths = get_cache_paths(cache_dir)
    else:
        cache_dir = None
        cache_paths = None
        file_hash = None

    # 3. 检查缓存有效性
    cache_valid = False
    if cache_dir:
        cache_valid = is_cache_valid(cache_dir, JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME)
    
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Cache valid: {cache_valid}")

    # 4. 列出可用缓存（用于调试）
    available_caches = list_available_caches()
    logger.info(f"Available caches: {len(available_caches)}")
    for cache in available_caches:
        cache_info = cache['info']
        logger.info(f"  - {os.path.basename(cache['dir'])}: {cache_info.get('file_path', 'Unknown')} "
                   f"(last accessed: {cache_info.get('last_accessed', 'Unknown')})")

    # 5. 构建检索索引
    logger.info(f"Building retrieval indices with method: {RETRIEVAL_METHOD}")

    # 构建BM25索引（使用缓存）
    bm25_cache_path = cache_paths['bm25'] if cache_valid and cache_paths else None
    bm25_index = build_bm25_index(documents, bm25_cache_path)

    # 设置dense retrieval（如果需要）
    dense_model = None
    document_embeddings = None

    if RETRIEVAL_METHOD in ["dense", "hybrid"]:
        dense_model = setup_dense_retrieval_model()
        if dense_model is not None:
            embeddings_cache_path = cache_paths['embeddings'] if cache_valid and cache_paths else None
            document_embeddings = build_document_embeddings(
                documents, dense_model, embeddings_cache_path
            )
        else:
            logger.warning("Dense model setup failed, using BM25 only")
            RETRIEVAL_METHOD = "bm25"

    # 6. 如果缓存无效或不存在，保存新的缓存
    if not cache_valid and cache_dir and file_hash:
        logger.info("Saving new cache...")
        save_cache_info(cache_dir, file_hash, RETRIEVAL_METHOD, DENSE_MODEL_NAME, JSONL_FILE_PATH)
        
        # 保存索引缓存
        if cache_paths:
            if not os.path.exists(cache_paths['bm25']):
                try:
                    with open(cache_paths['bm25'], "wb") as f:
                        pickle.dump(bm25_index, f)
                    logger.info(f"BM25 index cached to {cache_paths['bm25']}")
                except Exception as e:
                    logger.warning(f"Failed to cache BM25 index: {e}")
        
        # 清理旧缓存
        cleanup_old_caches()

    retrieval_components = (bm25_index, dense_model, document_embeddings)

    # 7. 设置 OpenAI 客户端
    try:
        client = setup_openai_client()
    except Exception as e:
        logger.error(f"Error setting up OpenAI client: {e}")
        logger.error("Please make sure VLLM server is running on http://localhost:8000")
        exit()

    # 8. 加载测试查询
    test_queries = load_test_queries(EXAMPLE_ANS_PATH)

    if test_queries:
        # 运行批量测试
        test_results = run_batch_test(
            test_queries, retrieval_components, documents, doc_ids, client
        )

        # 保存测试结果
        result_filename = f"test_results_{RETRIEVAL_METHOD}.json"
        with open(result_filename, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Test results saved to {result_filename}")

        from utils import cal_em

        pred = [result["generated_answer"] for result in test_results]
        ans = [result["expected_answer"] for result in test_results]

        logger.info(
            f"Accuracy with {RETRIEVAL_METHOD} retrieval: {cal_em(pred, ans):.4f}"
        )
    else:
        logger.info("No test queries found, running example query...")
        test_query = "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"
        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            test_query, retrieval_components, documents, doc_ids, client
        )
