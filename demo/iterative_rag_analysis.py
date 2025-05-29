#!/usr/bin/env python3
"""
迭代式RAG对比分析系统
Iterative RAG Comparison and Analysis System

该模块提供了传统RAG与迭代式RAG方法的详细对比分析功能，
包括性能评估、效果分析和可视化结果展示。

主要功能：
1. 传统RAG vs 迭代RAG的并行对比测试
2. 详细的迭代过程分析和可视化
3. 性能指标统计和报告生成
4. 批量测试和结果导出

使用方法：
    python iterative_rag_analysis.py [options]

作者: GitHub Copilot
日期: 2024
"""

import asyncio
import os
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Any
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import functools

# 导入主要功能模块
from main import (
    load_documents_from_jsonl,
    build_bm25_index,
    setup_dense_retrieval_model,
    build_document_embeddings,
    setup_openai_client,
    rag_query,
    iterative_rag_query,
    load_test_queries,
    save_results_async,
    setup_logger,
    get_cache_dir,
    get_cache_paths,
    is_cache_valid,
)
from ..utils import is_match, is_set_match

# 配置参数
from config import (
    JSONL_FILE_PATH,
    EXAMPLE_ANS_PATH,
    DENSE_MODEL_NAME,
    RETRIEVAL_METHOD,
    MAX_CONCURRENT,
)

# 设置专用logger
logger = setup_logger()


@dataclass
class ComparisonResult:
    """对比分析结果数据类"""

    query: str
    expected_answer: str
    traditional_answer: str
    iterative_answer: str
    traditional_docs: List[str]
    iterative_docs: List[str]
    traditional_contents: List[str]
    iterative_contents: List[str]
    iterations: List[Dict[str, Any]]
    traditional_correct: bool
    iterative_correct: bool
    execution_time_traditional: float
    execution_time_iterative: float
    improvement_type: str  # "improved", "same", "degraded", "different"


@dataclass
class AnalysisConfig:
    """分析配置参数"""

    max_iterations: int = 3
    top_k: int = 3
    demo_queries_limit: int = 5
    detailed_analysis_limit: int = 3
    save_detailed_logs: bool = True
    generate_report: bool = True
    output_dir: str = "analysis_results"


class IterativeRAGAnalyzer:
    """迭代式RAG分析器主类"""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.documents = None
        self.doc_ids = None
        self.retrieval_components = None
        self.client = None
        self.comparison_results: List[ComparisonResult] = []

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 设置日志
        self.setup_analysis_logger()

    def setup_analysis_logger(self):
        """设置分析专用日志"""
        analysis_logger = logging.getLogger("iterative_rag_analysis")
        analysis_logger.setLevel(logging.INFO)

        if analysis_logger.handlers:
            analysis_logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 分析结果日志文件
        log_filename = os.path.join(
            self.config.output_dir,
            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        analysis_logger.addHandler(file_handler)
        analysis_logger.addHandler(console_handler)

        self.analysis_logger = analysis_logger

    async def initialize_system(self):
        """初始化RAG系统组件"""
        self.analysis_logger.info("=== 初始化迭代式RAG分析系统 ===")

        # 1. 加载文档
        self.analysis_logger.info("加载文档...")
        self.documents, self.doc_ids = load_documents_from_jsonl(JSONL_FILE_PATH)

        if not self.documents:
            raise RuntimeError("无法加载文档数据")

        self.analysis_logger.info(f"成功加载 {len(self.documents)} 个文档")

        # 2. 检查缓存
        cache_result = get_cache_dir(
            JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME
        )
        cache_valid = False
        cache_paths = None

        if cache_result:
            cache_dir, file_hash = cache_result
            cache_paths = get_cache_paths(cache_dir)
            cache_valid = is_cache_valid(
                cache_dir, JSONL_FILE_PATH, RETRIEVAL_METHOD, DENSE_MODEL_NAME
            )

        # 3. 构建检索组件
        self.analysis_logger.info("构建检索索引...")

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            # BM25索引
            bm25_cache_path = (
                cache_paths["bm25"] if cache_valid and cache_paths else None
            )
            bm25_task = loop.run_in_executor(
                executor,
                functools.partial(build_bm25_index, self.documents, bm25_cache_path),
            )

            # Dense组件
            dense_task = None
            if RETRIEVAL_METHOD in ["dense", "hybrid"]:

                def setup_dense_components():
                    dense_model = setup_dense_retrieval_model()
                    if dense_model is not None:
                        embeddings_cache_path = (
                            cache_paths["embeddings"]
                            if cache_valid and cache_paths
                            else None
                        )
                        document_embeddings = build_document_embeddings(
                            self.documents, dense_model, embeddings_cache_path
                        )
                        return dense_model, document_embeddings
                    return None, None

                dense_task = loop.run_in_executor(executor, setup_dense_components)

            # 等待任务完成
            bm25_index = await bm25_task

            if dense_task:
                dense_model, document_embeddings = await dense_task
            else:
                dense_model = None
                document_embeddings = None

        self.retrieval_components = (bm25_index, dense_model, document_embeddings)

        # 4. 设置客户端
        self.analysis_logger.info("设置LLM客户端...")
        self.client = setup_openai_client()

        self.analysis_logger.info("系统初始化完成")

    async def compare_single_query(
        self, query: str, expected_answer: str = "", reference: str = ""
    ) -> ComparisonResult:
        """对单个查询进行传统RAG vs 迭代RAG的对比分析"""
        import time

        self.analysis_logger.info(f"对比分析查询: {query}")

        # 执行传统RAG
        start_time = time.time()
        traditional_answer, traditional_docs, traditional_contents = await rag_query(
            query,
            self.retrieval_components,
            self.documents,
            self.doc_ids,
            self.client,
            top_k=self.config.top_k,
        )
        traditional_time = time.time() - start_time

        # 执行迭代RAG
        start_time = time.time()
        (
            iterative_answer,
            iterative_docs,
            iterative_contents,
            iterations,
        ) = await iterative_rag_query(
            query,
            self.retrieval_components,
            self.documents,
            self.doc_ids,
            self.client,
            top_k=self.config.top_k,
            max_iterations=self.config.max_iterations,
        )
        iterative_time = time.time() - start_time

        # 计算准确性（如果有预期答案）
        traditional_correct = False
        iterative_correct = False

        if expected_answer:
            traditional_correct = is_set_match(traditional_answer, expected_answer) or (
                is_match(traditional_answer, expected_answer)
                and "；" not in expected_answer
            )
            iterative_correct = is_set_match(iterative_answer, expected_answer) or (
                is_match(iterative_answer, expected_answer)
                and "；" not in expected_answer
            )

        # 分析改进类型
        improvement_type = self._analyze_improvement_type(
            traditional_answer, iterative_answer, traditional_correct, iterative_correct
        )

        result = ComparisonResult(
            query=query,
            expected_answer=expected_answer,
            traditional_answer=traditional_answer,
            iterative_answer=iterative_answer,
            traditional_docs=traditional_docs,
            iterative_docs=iterative_docs,
            traditional_contents=traditional_contents,
            iterative_contents=iterative_contents,
            iterations=iterations,
            traditional_correct=traditional_correct,
            iterative_correct=iterative_correct,
            execution_time_traditional=traditional_time,
            execution_time_iterative=iterative_time,
            improvement_type=improvement_type,
        )

        self._log_comparison_result(result)
        return result

    def _analyze_improvement_type(
        self,
        traditional_answer: str,
        iterative_answer: str,
        traditional_correct: bool,
        iterative_correct: bool,
    ) -> str:
        """分析改进类型"""
        if traditional_answer == iterative_answer:
            return "same"
        elif iterative_correct and not traditional_correct:
            return "improved"
        elif not iterative_correct and traditional_correct:
            return "degraded"
        else:
            return "different"

    def _log_comparison_result(self, result: ComparisonResult):
        """记录对比结果日志"""
        self.analysis_logger.info(f"查询: {result.query}")
        self.analysis_logger.info(f"传统RAG答案: {result.traditional_answer}")
        self.analysis_logger.info(f"迭代RAG答案: {result.iterative_answer}")
        self.analysis_logger.info(f"迭代次数: {len(result.iterations)}")
        self.analysis_logger.info(f"改进类型: {result.improvement_type}")
        self.analysis_logger.info(
            f"执行时间 - 传统: {result.execution_time_traditional:.2f}s, 迭代: {result.execution_time_iterative:.2f}s"
        )

        if result.expected_answer:
            self.analysis_logger.info(
                f"准确性 - 传统: {result.traditional_correct}, 迭代: {result.iterative_correct}"
            )

        self.analysis_logger.info("-" * 80)

    async def run_demo_comparison(self, demo_queries: List[str]):
        """运行演示查询的对比分析"""
        self.analysis_logger.info(
            f"=== 演示查询对比分析 ({len(demo_queries)} 个查询) ==="
        )

        demo_results = []
        for i, query in enumerate(demo_queries[: self.config.demo_queries_limit]):
            self.analysis_logger.info(f"\n--- 演示查询 {i + 1}/{len(demo_queries)} ---")
            result = await self.compare_single_query(query)
            demo_results.append(result)
            self.comparison_results.append(result)

        return demo_results

    async def run_batch_comparison(self, test_queries: List[Tuple[str, str, str]]):
        """运行批量测试查询的对比分析"""
        self.analysis_logger.info(
            f"=== 批量查询对比分析 ({len(test_queries)} 个查询) ==="
        )

        # 使用信号量控制并发
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def process_single_comparison(query_data):
            async with semaphore:
                query, expected_answer, reference = query_data
                return await self.compare_single_query(
                    query, expected_answer, reference
                )

        # 创建任务
        tasks = [process_single_comparison(query_data) for query_data in test_queries]

        # 执行批量对比
        from tqdm.asyncio import tqdm as async_tqdm

        batch_results = []

        for task in async_tqdm.as_completed(
            tasks, desc="批量对比分析", total=len(tasks)
        ):
            try:
                result = await task
                batch_results.append(result)
                self.comparison_results.append(result)
            except Exception as e:
                self.analysis_logger.error(f"处理查询时出错: {e}")

        return batch_results

    async def detailed_iterative_analysis(self, query: str):
        """详细分析单个查询的迭代过程"""
        self.analysis_logger.info("\n=== 详细迭代过程分析 ===")
        self.analysis_logger.info(f"分析查询: {query}")

        # 执行迭代RAG
        answer, docs, contents, iterations = await iterative_rag_query(
            query,
            self.retrieval_components,
            self.documents,
            self.doc_ids,
            self.client,
            top_k=self.config.top_k,
            max_iterations=self.config.max_iterations,
        )

        self.analysis_logger.info(f"最终答案: {answer}")
        self.analysis_logger.info(f"总迭代次数: {len(iterations)}")
        self.analysis_logger.info(f"累计检索文档: {len(docs)}")

        # 分析每次迭代
        for i, iteration in enumerate(iterations):
            self.analysis_logger.info(f"\n--- 迭代 {i + 1} 详细分析 ---")
            self.analysis_logger.info(f"查询: {iteration['query']}")
            self.analysis_logger.info(f"检索到的文档ID: {iteration['retrieved_ids']}")
            self.analysis_logger.info(f"本轮答案: {iteration['answer']}")

            # 显示检索到的文档片段
            for j, (doc_id, content) in enumerate(
                zip(iteration["retrieved_ids"], iteration["retrieved_contents"])
            ):
                self.analysis_logger.info(f"\n  文档 {j + 1} (ID: {doc_id}):")
                self.analysis_logger.info(f"  内容片段: {content[:200]}...")

        # 分析迭代效果
        if len(iterations) > 1:
            first_answer = iterations[0]["answer"]
            final_answer = answer

            self.analysis_logger.info("\n--- 迭代效果分析 ---")
            self.analysis_logger.info(f"第一轮答案: {first_answer}")
            self.analysis_logger.info(f"最终答案: {final_answer}")

            if first_answer != final_answer:
                self.analysis_logger.info("✓ 迭代过程改善了答案质量")
            else:
                self.analysis_logger.info("→ 第一轮答案已经较为完整")

        return {
            "query": query,
            "final_answer": answer,
            "iterations": iterations,
            "analysis": "completed",
        }

    def generate_statistics_report(self) -> Dict[str, Any]:
        """生成统计报告"""
        if not self.comparison_results:
            return {}

        total_queries = len(self.comparison_results)

        # 统计改进类型
        improvement_stats = {"improved": 0, "same": 0, "degraded": 0, "different": 0}

        # 统计准确性
        traditional_correct = 0
        iterative_correct = 0
        accuracy_comparisons = 0

        # 统计性能
        total_traditional_time = 0
        total_iterative_time = 0

        # 统计迭代次数
        iteration_counts = []

        for result in self.comparison_results:
            improvement_stats[result.improvement_type] += 1

            if result.expected_answer:  # 只有有预期答案的才统计准确性
                accuracy_comparisons += 1
                if result.traditional_correct:
                    traditional_correct += 1
                if result.iterative_correct:
                    iterative_correct += 1

            total_traditional_time += result.execution_time_traditional
            total_iterative_time += result.execution_time_iterative
            iteration_counts.append(len(result.iterations))

        # 计算统计指标
        stats = {
            "total_queries": total_queries,
            "improvement_statistics": improvement_stats,
            "improvement_percentages": {
                k: (v / total_queries) * 100 for k, v in improvement_stats.items()
            },
            "accuracy_statistics": {
                "queries_with_expected_answers": accuracy_comparisons,
                "traditional_accuracy": traditional_correct / accuracy_comparisons
                if accuracy_comparisons > 0
                else 0,
                "iterative_accuracy": iterative_correct / accuracy_comparisons
                if accuracy_comparisons > 0
                else 0,
                "accuracy_improvement": (iterative_correct - traditional_correct)
                / accuracy_comparisons
                if accuracy_comparisons > 0
                else 0,
            },
            "performance_statistics": {
                "average_traditional_time": total_traditional_time / total_queries,
                "average_iterative_time": total_iterative_time / total_queries,
                "time_overhead_ratio": total_iterative_time / total_traditional_time
                if total_traditional_time > 0
                else 0,
            },
            "iteration_statistics": {
                "average_iterations": sum(iteration_counts) / len(iteration_counts),
                "max_iterations": max(iteration_counts),
                "min_iterations": min(iteration_counts),
                "single_iteration_percentage": (
                    iteration_counts.count(1) / len(iteration_counts)
                )
                * 100,
            },
        }

        return stats

    async def save_results(self):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细对比结果
        detailed_results = []
        for result in self.comparison_results:
            detailed_results.append(
                {
                    "query": result.query,
                    "expected_answer": result.expected_answer,
                    "traditional_answer": result.traditional_answer,
                    "iterative_answer": result.iterative_answer,
                    "traditional_docs": result.traditional_docs,
                    "iterative_docs": result.iterative_docs,
                    "iterations": result.iterations,
                    "traditional_correct": result.traditional_correct,
                    "iterative_correct": result.iterative_correct,
                    "execution_time_traditional": result.execution_time_traditional,
                    "execution_time_iterative": result.execution_time_iterative,
                    "improvement_type": result.improvement_type,
                }
            )

        detailed_file = os.path.join(
            self.config.output_dir, f"detailed_comparison_{timestamp}.json"
        )
        await save_results_async(detailed_results, detailed_file)

        # 生成并保存统计报告
        stats = self.generate_statistics_report()
        stats_file = os.path.join(
            self.config.output_dir, f"statistics_report_{timestamp}.json"
        )
        await save_results_async(stats, stats_file)

        self.analysis_logger.info(f"详细结果已保存到: {detailed_file}")
        self.analysis_logger.info(f"统计报告已保存到: {stats_file}")

        return detailed_file, stats_file

    def print_summary_report(self):
        """打印摘要报告"""
        stats = self.generate_statistics_report()

        if not stats:
            self.analysis_logger.info("没有可用的分析结果")
            return

        self.analysis_logger.info("\n" + "=" * 80)
        self.analysis_logger.info("迭代式RAG对比分析摘要报告")
        self.analysis_logger.info("=" * 80)

        self.analysis_logger.info(f"总查询数量: {stats['total_queries']}")

        self.analysis_logger.info("\n--- 改进效果统计 ---")
        for improvement_type, count in stats["improvement_statistics"].items():
            percentage = stats["improvement_percentages"][improvement_type]
            self.analysis_logger.info(
                f"{improvement_type}: {count} ({percentage:.1f}%)"
            )

        if stats["accuracy_statistics"]["queries_with_expected_answers"] > 0:
            acc_stats = stats["accuracy_statistics"]
            self.analysis_logger.info("\n--- 准确性统计 ---")
            self.analysis_logger.info(
                f"有预期答案的查询: {acc_stats['queries_with_expected_answers']}"
            )
            self.analysis_logger.info(
                f"传统RAG准确率: {acc_stats['traditional_accuracy']:.3f}"
            )
            self.analysis_logger.info(
                f"迭代RAG准确率: {acc_stats['iterative_accuracy']:.3f}"
            )
            self.analysis_logger.info(
                f"准确率提升: {acc_stats['accuracy_improvement']:.3f}"
            )

        perf_stats = stats["performance_statistics"]
        self.analysis_logger.info("\n--- 性能统计 ---")
        self.analysis_logger.info(
            f"传统RAG平均耗时: {perf_stats['average_traditional_time']:.2f}s"
        )
        self.analysis_logger.info(
            f"迭代RAG平均耗时: {perf_stats['average_iterative_time']:.2f}s"
        )
        self.analysis_logger.info(
            f"时间开销比例: {perf_stats['time_overhead_ratio']:.2f}x"
        )

        iter_stats = stats["iteration_statistics"]
        self.analysis_logger.info("\n--- 迭代统计 ---")
        self.analysis_logger.info(
            f"平均迭代次数: {iter_stats['average_iterations']:.1f}"
        )
        self.analysis_logger.info(
            f"单次迭代比例: {iter_stats['single_iteration_percentage']:.1f}%"
        )

        self.analysis_logger.info("=" * 80)


async def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="迭代式RAG对比分析系统")
    parser.add_argument("--demo-only", action="store_true", help="仅运行演示查询")
    parser.add_argument("--batch-only", action="store_true", help="仅运行批量测试")
    parser.add_argument("--max-iterations", type=int, default=3, help="最大迭代次数")
    parser.add_argument("--top-k", type=int, default=3, help="检索文档数量")
    parser.add_argument("--output-dir", default="analysis_results", help="输出目录")

    args = parser.parse_args()

    # 创建配置
    config = AnalysisConfig(
        max_iterations=args.max_iterations, top_k=args.top_k, output_dir=args.output_dir
    )

    # 创建分析器
    analyzer = IterativeRAGAnalyzer(config)

    try:
        # 初始化系统
        await analyzer.initialize_system()

        # 定义演示查询
        demo_queries = [
            "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？",
            "哪些单位在中国期刊高质量发展论坛的主论坛上做主题演讲？",
            "2024年是中国红十字会成立多少周年？",
            "中国与东盟建立对话关系多少周年了？",
            "2023年诺贝尔文学奖获得者是谁？",
        ]

        # 运行分析
        if not args.batch_only:
            analyzer.analysis_logger.info("开始演示查询对比分析...")
            await analyzer.run_demo_comparison(demo_queries)

            # 详细分析第一个查询
            if demo_queries:
                await analyzer.detailed_iterative_analysis(demo_queries[0])

        if not args.demo_only:
            # 加载测试查询
            test_queries = load_test_queries(EXAMPLE_ANS_PATH)
            if test_queries:
                analyzer.analysis_logger.info("开始批量查询对比分析...")
                await analyzer.run_batch_comparison(test_queries)

        # 保存结果
        await analyzer.save_results()

        # 打印摘要报告
        analyzer.print_summary_report()

    except Exception as e:
        analyzer.analysis_logger.error(f"分析过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
