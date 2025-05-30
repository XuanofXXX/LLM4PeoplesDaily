import json
import asyncio
from datetime import datetime

from logger_config import setup_logger
from cache_manager import CacheManager
from data_loader import load_documents_from_jsonl, load_test_queries
from retrieval import RetrievalSystem
from rag_query import RAGQuerySystem
from utils import cal_em
import pytz

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
    ENABLE_GOOGLE_SEARCH,
    GOOGLE_SEARCH_TOPK,
    USE_GOOGLE_FALLBACK,
    API_KEY,
)

# 初始化时间戳和logger
CUR_TIME = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
logger = setup_logger(CUR_TIME)


async def main(test_file_path=None):
    """主程序异步版本"""
    global EXAMPLE_ANS_PATH
    # 记录当前配置到日志
    EXAMPLE_ANS_PATH = test_file_path or EXAMPLE_ANS_PATH
    logger.info("=== RAG System Configuration ===")
    logger.info(f"JSONL_FILE_PATH: {JSONL_FILE_PATH}")
    logger.info(f"EXAMPLE_ANS_PATH: {EXAMPLE_ANS_PATH}")
    logger.info(f"VLLM_API_BASE: {VLLM_API_BASE}")
    logger.info(f"API_KEY: {API_KEY}")
    logger.info(f"MODEL_NAME: {MODEL_NAME}")
    logger.info(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    logger.info(f"TOP_K_DOCUMENTS: {TOP_K_DOCUMENTS}")
    logger.info(f"DENSE_MODEL_NAME: {DENSE_MODEL_NAME}")
    logger.info(f"RETRIEVAL_METHOD: {RETRIEVAL_METHOD}")
    logger.info(f"BM25_WEIGHT: {BM25_WEIGHT}")
    logger.info(f"DENSE_WEIGHT: {DENSE_WEIGHT}")
    logger.info(f"CACHE_BASE_DIR: {CACHE_BASE_DIR}")
    logger.info(f"MAX_CONCURRENT: {MAX_CONCURRENT}")
    logger.info(f"ENABLE_GOOGLE_SEARCH: {ENABLE_GOOGLE_SEARCH}")
    logger.info(f"GOOGLE_SEARCH_TOPK: {GOOGLE_SEARCH_TOPK}")
    logger.info(f"USE_GOOGLE_FALLBACK: {USE_GOOGLE_FALLBACK}")
    logger.info("=" * 40)

    # 1. 加载数据
    documents, doc_ids = load_documents_from_jsonl(JSONL_FILE_PATH)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return

    # 2. 初始化缓存管理器
    cache_manager = CacheManager(CACHE_BASE_DIR)

    # 获取缓存目录和路径
    cache_result = cache_manager.get_cache_dir(
        JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME
    )
    logger.info(f"Cache result: {cache_result}")
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
        # cache_manager.cleanup_old_caches()

    # 5. 初始化RAG查询系统（增加Google搜索支持）
    try:
        rag_system = RAGQuerySystem(
            VLLM_API_BASE,
            API_KEY,
            MODEL_NAME,
            MAX_NEW_TOKENS,
            enable_google_search=ENABLE_GOOGLE_SEARCH,
            google_search_topk=GOOGLE_SEARCH_TOPK,
        )
        rag_system.setup_client()
    except Exception as e:
        logger.error(f"Error setting up RAG system: {e}")
        logger.error("Please make sure VLLM server is running on http://localhost:8000")
        return

    # 6. 加载测试查询并运行测试
    test_queries = load_test_queries(EXAMPLE_ANS_PATH)

    if test_queries:
        # 运行批量测试（增加Google搜索参数）
        test_results = await rag_system.run_batch_test(
            test_queries,
            retrieval_system,
            documents,
            doc_ids,
            MAX_CONCURRENT,
            TOP_K_DOCUMENTS,
            RETRIEVAL_METHOD,
            BM25_WEIGHT,
            DENSE_WEIGHT,
            use_google_fallback=USE_GOOGLE_FALLBACK,
        )

        # 保存测试结果
        google_suffix = "_with_google" if ENABLE_GOOGLE_SEARCH else ""
        result_filename = (
            f"result/test_results_{RETRIEVAL_METHOD}{google_suffix}_{CUR_TIME}.json"
        )
        pred = [result["generated_answer"] for result in test_results]
        ans = [result["expected_answer"] for result in test_results]
        accuracy = cal_em(pred, ans)
        test_results.insert(
            0,
            {
                "retrieval_method": RETRIEVAL_METHOD,
                "DENSE_WEIGHT": DENSE_WEIGHT,
                "BM25_WEIGHT": BM25_WEIGHT,
                "top_k_documents": TOP_K_DOCUMENTS,
                "ENABLE_GOOGLE_SEARCH": ENABLE_GOOGLE_SEARCH,
                "GOOGLE_SEARCH_TOPK": GOOGLE_SEARCH_TOPK,
                "USE_GOOGLE_FALLBACK": USE_GOOGLE_FALLBACK,
                "accuracy": accuracy,
                "input_file": JSONL_FILE_PATH,
                "example_ans_path": EXAMPLE_ANS_PATH,
            },
        )
        await rag_system.save_results_async(test_results, result_filename)
        logger.info(f"Test results saved to {result_filename}")

        if ENABLE_GOOGLE_SEARCH:
            google_used_count = sum(
                1 for result in test_results if result.get("used_google", False)
            )
            logger.info(f"Google搜索使用次数: {google_used_count}/{len(test_results)}")

        logger.info(
            f"Accuracy with {RETRIEVAL_METHOD} retrieval{google_suffix}: {accuracy:.4f}"
        )
    else:
        logger.info("No test queries found, running example query...")
        test_query = "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"
        (
            generated_answer,
            retrieved_ids,
            retrieved_contents,
        ) = await rag_system.rag_query(
            test_query,
            retrieval_system,
            documents,
            doc_ids,
            TOP_K_DOCUMENTS,
            RETRIEVAL_METHOD,
            BM25_WEIGHT,
            DENSE_WEIGHT,
            USE_GOOGLE_FALLBACK,
        )
        logger.info(f"Generated answer: {generated_answer}")
    return test_results


def evaluate(queries):
    """
    评估函数，模拟处理查询并返回答案
    前80条为简单问答题，后20条为开放题
    """
    # 验证查询数量
    if len(queries) != 100:
        logger.warning(f"期望100条查询，实际收到{len(queries)}条")

    # 构建测试数据，前80条为简单问答，后20条为开放题
    qs = []
    for i, q in enumerate(queries):
        if i < 80:
            # 前80条：简单问答题，answer为空
            qs.append({"question": q, "answer": "", "reference": ""})
        else:
            # 后20条：开放题，answer为空，但会用不同的prompt处理
            qs.append({"question": q, "answer": "", "reference": ""})

    final_path = "/home/xiachunxuan/nlp-homework/data/eval/final_test.json"
    with open(final_path, "w") as f:
        json.dump(qs, f, ensure_ascii=False, indent=4)

    logger.info("评估开始：前80条为简单问答题，后20条为开放题")
    result = asyncio.run(main(final_path))

    qa_map = {}
    print(result[0])  # 打印配置信息

    for r in result:
        if "query" not in r:
            continue
        question = r["query"]
        answer = r["generated_answer"]
        retr_contents = r["retrieved_contents"]
        is_open = r.get("is_open_question", False)
        qa_map[question] = {"answer": answer, "retrieved_contents": retr_contents}

        # 记录问题类型到日志
        question_type = "开放题" if is_open else "问答题"
        logger.debug(f"{question_type}: {question[:50]}...")

    logger.info(f"评估完成，处理了{len(qa_map)}个问题")

    return_results = []
    for q in queries[:80]:
        if q in qa_map:
            answer = qa_map[q]["answer"]
            retrieved_contents = qa_map[q]["retrieved_contents"]
            return_results.append(answer)
        else:
            # return_results.append({
            #     "question": q,
            #     "answer": "未找到答案",
            #     "retrieved_contents": []
            # })
            raise ValueError(f"查询 '{q}' 未找到答案，请检查输入数据是否正确")
    for q in queries[-20:]:
        if q in qa_map:
            answer = qa_map[q]["answer"]
            retrieved_contents = qa_map[q]["retrieved_contents"]
            return_results.append(
                {
                    "ans": answer,
                    "reference": retrieved_contents,
                }
            )
        else:
            # return_results.append({
            #     "question": q,
            #     "answer": "未找到答案",
            #     "retrieved_contents": []
            # })
            raise ValueError(f"查询 '{q}' 未找到答案，请检查输入数据是否正确")

    # 按原始顺序返回答案
    return [qa_map.get(query, "未找到答案") for query in queries]


if __name__ == "__main__":
    # 运行异步主程序
    # result = asyncio.run(main())
    # print(result)
    # test_queries = [
    #     "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校?",
    #     "中国第一大贸易伙伴是？",
    #     "《共建“一带一路”：构建人类命运共同体的重大实践》白皮书什么时候发布？",
    # ]
    with open('data/eval/final_debug.json', 'r', encoding='utf-8') as f:
        test_queries = json.load(f)
    evaluate(test_queries)
