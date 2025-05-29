import asyncio
from datetime import datetime

from logger_config import setup_logger
from cache_manager import CacheManager
from data_loader import load_documents_from_jsonl, load_test_queries
from retrieval import RetrievalSystem
from rag_query import RAGQuerySystem
from utils import cal_em

# 导入配置
from config import (
    JSONL_FILE_PATH,
    EXAMPLE_ANS_PATH,
    VLLM_API_BASE,
    MODEL_NAME,
    MAX_NEW_TOKENS,
    TOP_K_DOCUMENTS,
    DENSE_MODEL_NAME,
    RETRIEVAL_METHOD,
    BM25_WEIGHT,
    DENSE_WEIGHT,
    CACHE_BASE_DIR,
    MAX_CONCURRENT,
)

# 初始化时间戳和logger
CUR_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(CUR_TIME)

async def main():
    """主程序异步版本"""
    # 记录当前配置到日志
    logger.info("=== RAG System Configuration ===")
    logger.info(f"JSONL_FILE_PATH: {JSONL_FILE_PATH}")
    logger.info(f"EXAMPLE_ANS_PATH: {EXAMPLE_ANS_PATH}")
    logger.info(f"VLLM_API_BASE: {VLLM_API_BASE}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    logger.info(f"TOP_K_DOCUMENTS: {TOP_K_DOCUMENTS}")
    logger.info(f"DENSE_MODEL_NAME: {DENSE_MODEL_NAME}")
    logger.info(f"RETRIEVAL_METHOD: {RETRIEVAL_METHOD}")
    logger.info(f"BM25_WEIGHT: {BM25_WEIGHT}")
    logger.info(f"DENSE_WEIGHT: {DENSE_WEIGHT}")
    logger.info(f"CACHE_BASE_DIR: {CACHE_BASE_DIR}")
    logger.info(f"MAX_CONCURRENT: {MAX_CONCURRENT}")
    logger.info("=" * 40)

    # 1. 加载数据
    documents, doc_ids = load_documents_from_jsonl(JSONL_FILE_PATH)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # 2. 初始化缓存管理器
    cache_manager = CacheManager(CACHE_BASE_DIR)
    
    # 获取缓存目录和路径
    cache_result = cache_manager.get_cache_dir(JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME)
    if cache_result:
        cache_dir, file_hash = cache_result
        cache_paths = cache_manager.get_cache_paths(cache_dir)
    else:
        cache_dir = None
        cache_paths = None
        file_hash = None

    # 检查缓存有效性
    cache_valid = False
    if cache_dir:
        cache_valid = cache_manager.is_cache_valid(
            cache_dir, JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME
        )

    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Cache valid: {cache_valid}")

    # 3. 初始化检索系统
    retrieval_system = RetrievalSystem(DENSE_MODEL_NAME)
    
    # 构建检索索引
    logger.info(f"Building retrieval indices with method: {RETRIEVAL_METHOD}")

    # 构建BM25索引
    bm25_cache_path = cache_paths["bm25"] if cache_valid and cache_paths else None
    retrieval_system.build_bm25_index(documents, bm25_cache_path)

    # 构建Dense检索组件
    if RETRIEVAL_METHOD in ["dense", "hybrid"]:
        retrieval_system.setup_dense_model()
        if retrieval_system.dense_model is not None:
            embeddings_cache_path = (
                cache_paths["embeddings"] if cache_valid and cache_paths else None
            )
            retrieval_system.build_document_embeddings(documents, embeddings_cache_path)
        else:
            logger.warning("Dense model setup failed, using BM25 only")

    # 4. 保存缓存信息
    if not cache_valid and cache_dir and file_hash:
        logger.info("Saving new cache...")
        cache_manager.save_cache_info(
            cache_dir, file_hash, RETRIEVAL_METHOD, DENSE_MODEL_NAME, JSONL_FILE_PATH
        )
        cache_manager.cleanup_old_caches()

    # 5. 初始化RAG查询系统
    try:
        rag_system = RAGQuerySystem(VLLM_API_BASE, MODEL_NAME, MAX_NEW_TOKENS)
        rag_system.setup_client()
    except Exception as e:
        logger.error(f"Error setting up RAG system: {e}")
        logger.error("Please make sure VLLM server is running on http://localhost:8000")
        return

    # 6. 加载测试查询并运行测试
    test_queries = load_test_queries(EXAMPLE_ANS_PATH)

    if test_queries:
        # 运行批量测试
        test_results = await rag_system.run_batch_test(
            test_queries, retrieval_system, documents, doc_ids,
            MAX_CONCURRENT, TOP_K_DOCUMENTS, RETRIEVAL_METHOD, BM25_WEIGHT, DENSE_WEIGHT
        )

        # 保存测试结果
        result_filename = f"result/test_results_{RETRIEVAL_METHOD}_{CUR_TIME}.json"
        await rag_system.save_results_async(test_results, result_filename)
        logger.info(f"Test results saved to {result_filename}")

        # 计算准确率
        pred = [result["generated_answer"] for result in test_results]
        ans = [result["expected_answer"] for result in test_results]
        logger.info(f"Accuracy with {RETRIEVAL_METHOD} retrieval: {cal_em(pred, ans):.4f}")
    else:
        logger.info("No test queries found, running example query...")
        test_query = "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"
        generated_answer, retrieved_ids, retrieved_contents = await rag_system.rag_query(
            test_query, retrieval_system, documents, doc_ids, TOP_K_DOCUMENTS,
            RETRIEVAL_METHOD, BM25_WEIGHT, DENSE_WEIGHT
        )
        logger.info(f"Generated answer: {generated_answer}")

if __name__ == "__main__":
    # 运行异步主程序
    asyncio.run(main())
