import json
import os
import re
import jieba
import logging

logger = logging.getLogger("rag_system")

# 设置环境变量以避免tokenizers警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化jieba以避免多进程警告
jieba.initialize()

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
                    or data.get("content")
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
        queries_and_answers = []
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
        elif file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    queries_and_answers.append(
                        (
                            item.get("question", ""),
                            item.get("answer", ""),
                            item.get("reference", ""),
                        )
                    )

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
