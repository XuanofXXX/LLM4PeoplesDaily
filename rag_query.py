import re
import asyncio
import json
import aiofiles
from openai import AsyncOpenAI
from utils import is_match, is_set_match
from tqdm.asyncio import tqdm as async_tqdm
import logging
from google import GoogleSearcher

logger = logging.getLogger("rag_system")

# 原始的简单问答题prompt模板
PROMPT_TEMPLATE = """
## 指令
你是一个基于检索的问答助手。请根据以下提供的上下文信息来回答问题。

请按照以下格式回答：
1. 首先在"## 思考"部分分析上下文信息，整理相关内容
2. 然后在"## 答案"部分给出最终答案

要求：
- 答案部分请直接回答问题所提问的人名、地名、主题等等。
- 如果答案是单个人或单位，请直接给出名称。
- 如果答案是多个人或多个单位，请用分号（；）分隔。
- 如果问题是开放题，则分点回答。

## 参考示例
```
问题：2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？

## 思考
从上下文信息中查找关于2024年3月18日习近平总书记湖南考察的相关内容。文档中提到习近平总书记在湖南考察期间首先来到了湖南第一师范学院，这是他此次考察的第一站。

## 答案
湖南第一师范学院
```

---
```
问题：哪些单位在中国期刊高质量发展论坛的主论坛上做主题演讲？

## 思考
需要从上下文中找到中国期刊高质量发展论坛主论坛的演讲单位信息。根据文档内容，参与主题演讲的单位包括多个机构和期刊社，包含中国科协科技创新部，湖南省委宣传部，上海大学，《历史研究》，《读者》，《分子植物》，《问天少年》，南方杂志社，中华医学会杂志社。

## 答案
中国科协科技创新部；湖南省委宣传部；上海大学；《历史研究》；《读者》；《分子植物》；《问天少年》；南方杂志社；中华医学会杂志社
```

## 上下文信息
{context}

## 问题
{query}
""".strip()

# 开放题的prompt模板
OPEN_QUESTION_PROMPT_TEMPLATE = """
## 指令
你是一个基于检索的问答助手。请根据以下提供的上下文信息来回答开放性问题。

请按照以下格式回答：
1. 首先在"## 思考"部分分析上下文信息，整理相关内容
2. 然后在"## 答案"部分给出详细的回答

要求：
- 这是一个开放性问题，需要综合分析和详细阐述
- 答案应该结构清晰，分点论述
- 每个要点都要有具体的事实支撑
- 可以适当展开分析，但要基于提供的上下文信息
- 语言表达要准确、流畅、有逻辑性

## 参考示例
```
问题：分析一下当前中国在国际合作方面的主要举措和成果？

## 思考
从上下文信息中整理中国在国际合作方面的相关内容，包括"一带一路"倡议、多边合作机制、对外开放政策等方面的举措和取得的成果。需要从政策制定、具体实施、合作成果等多个维度进行分析。

## 答案
当前中国在国际合作方面的主要举措和成果可以从以下几个方面来分析：

1. **"一带一路"倡议持续推进**
   - 与多个国家签署合作协议，推动基础设施互联互通
   - 在贸易投资、产能合作等领域取得显著成果

2. **多边合作机制不断完善**
   - 积极参与和推动区域全面经济伙伴关系协定（RCEP）等
   - 在联合国、G20等国际组织中发挥重要作用

3. **对外开放水平持续提升**
   - 设立更多自贸试验区，扩大对外开放领域
   - 举办进博会等大型国际展会，促进国际贸易合作

这些举措体现了中国构建人类命运共同体的理念，为全球发展作出了积极贡献。
```

## 上下文信息
{context}

## 问题
{query}
""".strip()


def extract_answer_from_response(response_text):
    """从LLM响应中提取最终答案，返回(答案, 是否成功提取)"""
    try:
        # 查找 ## Answer 部分
        answer_pattern = r"##\s*答案\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(answer_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if match:
            answer = match.group(1).strip()
            logger.debug(f"提取到的答案: {answer}")
            return answer, True
        else:
            # 如果没有找到 ## Answer 格式，尝试查找答案：格式
            answer_pattern2 = r"答案[：:]\s*(.*?)(?=\n|$)"
            match2 = re.search(answer_pattern2, response_text, re.IGNORECASE)
            if match2:
                answer = match2.group(1).strip()
                logger.debug(f"备用格式提取到的答案: {answer}")
                return answer, True
            else:
                # 如果都没找到，返回原始响应
                logger.warning("未能从响应中提取到标准格式的答案，将重新生成")
                return response_text.strip(), False
    except Exception as e:
        logger.error(f"答案提取出错: {e}")
        return response_text.strip(), False


class RAGQuerySystem:
    def __init__(
        self,
        api_base,
        api_key,
        model_name,
        max_new_tokens=512,
        enable_google_search=False,
        google_search_topk=3,
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.enable_google_search = enable_google_search
        self.google_search_topk = google_search_topk
        self.client = None

        # 初始化Google搜索器
        if self.enable_google_search:
            self.google_searcher = GoogleSearcher(
                topk=self.google_search_topk, sleep_interval=1.0
            )
            logger.info(f"Google搜索已启用，返回Top-{self.google_search_topk}结果")
        else:
            self.google_searcher = None

    def setup_client(self):
        """设置 OpenAI 客户端连接到 VLLM 服务器"""
        logger.info(
            f"Setting up AsyncOpenAI client to connect to server at {self.api_base}..."
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,  # VLLM 服务器不需要真实的 API key
            base_url=self.api_base,
        )
        logger.info("AsyncOpenAI client setup complete.")
        return self.client

    async def google_search_async(self, query):
        """异步执行Google搜索"""
        if not self.google_searcher:
            return []

        try:
            logger.info(f"执行Google搜索: {query}")
            # 使用异步搜索方法
            search_results = await self.google_searcher.asearch(
                query, self.google_search_topk
            )
            logger.info(f"Google搜索返回{len(search_results)}个结果")
            return search_results
        except Exception as e:
            logger.error(f"Google搜索出错: {e}")
            return []

    async def generate_answer_with_retry(self, messages, max_retries=10):
        """生成答案，如果格式不正确则重试"""
        # 第一次尝试，使用温度0
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0,
                top_p=0.8,
                stop=None,
            )

            raw_response = response.choices[0].message.content.strip()
            logger.debug("LLM Raw Response (第一次尝试):")
            logger.debug(raw_response)

            # 提取最终答案
            final_answer, extraction_success = extract_answer_from_response(raw_response)
            
            if extraction_success:
                logger.debug("首次生成成功提取答案")
                return final_answer
            
            # 如果提取失败，尝试重新生成
            logger.info("首次生成格式不正确，使用非零温度重新生成...")
            
            for retry_count in range(max_retries):
                logger.info(f"重试生成 {retry_count + 1}/{max_retries}")
                retry_response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=0.3,  # 使用非零温度
                    stop=None,
                )
                
                retry_raw_response = retry_response.choices[0].message.content.strip()
                logger.debug(f"LLM Raw Response (重试 {retry_count + 1}):")
                logger.debug(retry_raw_response)
                
                retry_answer, retry_extraction_success = extract_answer_from_response(retry_raw_response)
                
                if retry_extraction_success:
                    logger.info(f"重试 {retry_count + 1} 成功提取答案")
                    return retry_answer
            
            # 所有重试都失败，返回最后一次的结果
            logger.warning("所有重试都未能提取到标准格式答案，返回最后一次结果")
            return final_answer

        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return "抱歉，生成答案时遇到问题。"

    async def rag_query(
        self,
        query,
        retrieval_system,
        documents,
        doc_ids,
        top_k=5,
        retrieval_method="hybrid",
        bm25_weight=0.5,
        dense_weight=0.5,
        use_google_fallback=False,
        is_open_question=False,  # 新增参数，标识是否为开放题
    ):
        """执行 RAG 查询，支持Google搜索作为后备"""
        logger.debug(
            f"Performing RAG query: '{query}' using {retrieval_method} retrieval, open_question: {is_open_question}"
        )

        # 执行本地检索
        top_indices, scores = retrieval_system.retrieve(
            query, top_k, retrieval_method, bm25_weight, dense_weight
        )

        retrieved_docs_content = []
        retrieved_docs_ids = []
        context_sources = []  # 记录上下文来源

        if len(top_indices) > 0:
            retrieved_docs_content = [documents[i] for i in top_indices]
            retrieved_docs_ids = [doc_ids[i] for i in top_indices]
            context_sources.extend(
                [f"本地文档-{doc_id}" for doc_id in retrieved_docs_ids]
            )

            for i, (doc_id, score) in enumerate(zip(retrieved_docs_ids, scores)):
                logger.debug(f"  本地文档 ID: {doc_id}, Score: {score:.4f}")
        else:
            logger.warning("本地检索未找到相关文档")

        # 如果启用Google搜索且（本地没有结果或使用Google后备）
        if self.enable_google_search and (len(top_indices) == 0 or use_google_fallback):
            logger.info("执行Google搜索补充信息...")
            google_results = await self.google_search_async(query)

            for result in google_results:
                if result.get("contents") and len(result["contents"].strip()) > 50:
                    # 构建Google搜索结果的文档内容
                    google_doc = f"标题: {result.get('title', '无标题')}\n内容: {result['contents']}"
                    retrieved_docs_content.append(google_doc)
                    retrieved_docs_ids.append(f"google-{result.get('url', 'unknown')}")
                    context_sources.append(
                        f"Google搜索-{result.get('title', '未知标题')}"
                    )

        if len(retrieved_docs_content) == 0:
            logger.warning("本地检索和Google搜索都未找到相关信息")
            return "抱歉，未找到相关信息。", [], []

        # 处理文档截断
        max_doc_length = 3000  # 限制每个文档的最大字符数
        truncated_docs = []
        for doc in retrieved_docs_content:
            if len(doc) > max_doc_length:
                truncated_docs.append(doc[:max_doc_length//2] + "..." + doc[-max_doc_length//2:])
            else:
                truncated_docs.append(doc)

        # 构建上下文和消息
        context = "\n\n".join(
            [
                f"文档{i + 1} ({context_sources[i] if i < len(context_sources) else '未知来源'}): {doc}"
                for i, doc in enumerate(truncated_docs)
            ]
        )
        
        # 根据问题类型选择prompt模板
        if is_open_question:
            prompt = OPEN_QUESTION_PROMPT_TEMPLATE.format(context=context, query=query)
            logger.debug("使用开放题prompt模板")
        else:
            prompt = PROMPT_TEMPLATE.format(context=context, query=query)
            logger.debug("使用标准问答题prompt模板")

        # 使用标准的 OpenAI 消息格式
        messages = [
            {
                "role": "system",
                "content": "你是一个基于检索的问答助手。请严格按照用户要求的格式回答，先进行思考分析，然后给出最终答案。",
            },
            {"role": "user", "content": prompt},
        ]

        logger.debug(f"Context length: {len(context)} characters")
        logger.debug(
            f"使用了{len(retrieved_docs_content)}个文档，其中Google搜索结果: {len([s for s in context_sources if 'Google' in s])}个"
        )

        # 异步生成答案
        try:
            # 使用带重试的生成方法
            final_answer = await self.generate_answer_with_retry(messages)

            logger.debug("Final Answer:")
            logger.debug(final_answer)
            return final_answer, retrieved_docs_ids, retrieved_docs_content

        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return (
                "抱歉，生成答案时遇到问题。",
                retrieved_docs_ids,
                retrieved_docs_content,
            )

    async def save_results_async(self, results, filename):
        """异步保存测试结果到JSON文件"""
        async with aiofiles.open(filename, "w", encoding="utf-8") as f:
            await f.write(json.dumps(results, ensure_ascii=False, indent=2))

    async def run_batch_test(
        self,
        test_queries,
        retrieval_system,
        documents,
        doc_ids,
        max_concurrent=50,
        top_k=5,
        retrieval_method="hybrid",
        bm25_weight=0.5,
        dense_weight=0.5,
        use_google_fallback=False,
    ):
        """批量运行RAG测试查询，支持Google搜索"""
        logger.info(f"=== Running batch test on {len(test_queries)} queries ===")
        if self.enable_google_search:
            logger.info(f"Google搜索已启用，后备模式: {use_google_fallback}")

        async def process_single_query(i, query_data):
            query, expected_answer, reference = query_data
            # 判断是否为开放题（后20条，即索引80开始）
            is_open_question = i >= 80
            
            logger.debug(f"--- Test {i + 1}/{len(test_queries)} ---")
            logger.debug(f"Query: {query}")
            logger.debug(f"Expected: {expected_answer}")
            logger.debug(f"Open question: {is_open_question}")

            generated_answer, retrieved_ids, retrieved_contents = await self.rag_query(
                query,
                retrieval_system,
                documents,
                doc_ids,
                top_k,
                retrieval_method,
                bm25_weight,
                dense_weight,
                use_google_fallback,
                is_open_question,  # 传递开放题标识
            )

            result = {
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
                "used_google": any("google-" in doc_id for doc_id in retrieved_ids),
                "is_open_question": is_open_question,  # 记录是否为开放题
            }

            logger.debug("-" * 50)
            return result

        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(i, query_data):
            async with semaphore:
                return await process_single_query(i, query_data)

        # 创建所有任务
        tasks = [
            process_with_semaphore(i, query_data)
            for i, query_data in enumerate(test_queries)
        ]

        results = []
        for task in async_tqdm.as_completed(tasks, desc="处理查询", total=len(tasks)):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                # 创建错误结果
                i = len(results)
                if i < len(test_queries):
                    query, expected_answer, reference = test_queries[i]
                    results.append(
                        {
                            "is_correct": False,
                            "query": query,
                            "expected_answer": expected_answer,
                            "generated_answer": f"处理出错: {str(e)}",
                            "retrieved_docs": [],
                            "retrieved_contents": [],
                            "reference": reference,
                        }
                    )

        return results
