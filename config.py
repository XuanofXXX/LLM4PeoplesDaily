import os

# 基础RAG配置
# JSONL_FILE_PATH = "data/llm_processed_meaningful_articles_v3_rag.jsonl"
JSONL_FILE_PATH = "data/corpus_v3.jsonl" # 重新爬取标签等等
EXAMPLE_ANS_PATH = "data/eval/official_test_ans.json"
# EXAMPLE_ANS_PATH = "data/eval/generated_qa_pairs.jsonl"
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 500

# 检索配置
TOP_K_DOCUMENTS = 5
DENSE_MODEL_NAME = "/media/public/models/huggingface/BAAI/bge-large-zh-v1.5"
RETRIEVAL_METHOD = "hybrid"  # 可选: "bm25", "dense", "hybrid"
BM25_WEIGHT = 0.3
DENSE_WEIGHT = 0.7

# 缓存配置
CACHE_BASE_DIR = "cache"

# 并发配置
MAX_CONCURRENT = 100

# Google搜索配置
ENABLE_GOOGLE_SEARCH = os.getenv("ENABLE_GOOGLE_SEARCH", "True").lower() == "true"
GOOGLE_SEARCH_TOPK = int(os.getenv("GOOGLE_SEARCH_TOPK", "3"))
USE_GOOGLE_FALLBACK = os.getenv("USE_GOOGLE_FALLBACK", "True").lower() == "true"

# 可以通过环境变量覆盖配置
# 例如：export ENABLE_GOOGLE_SEARCH=true
# 例如：export GOOGLE_SEARCH_TOPK=5
# 例如：export USE_GOOGLE_FALLBACK=true

# 交替检索和生成RAG配置文件
# 通过修改这些配置可以切换不同的RAG模式


# 检索配置

# # 交替检索和生成配置
# USE_ITERATIVE_RAG = False  # True: 使用迭代RAG, False: 使用传统RAG
# MAX_ITERATIONS = 2        # 最大迭代次数 (建议2-3次)
# ITERATIVE_TOP_K = 5       # 迭代RAG中每次检索的文档数量