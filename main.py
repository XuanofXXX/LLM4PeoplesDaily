import json
import re
import pandas as pd
from rank_bm25 import BM25Okapi
import jieba
import torch
import os
from openai import OpenAI

# --- Configuration ---
JSONL_FILE_PATH = 'data/llm_processed_meaningful_articles_v2_rag.jsonl'
EXAMPLE_ANS_PATH = 'data/test_ans.json'
VLLM_API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "/media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct"  # 使用完整路径
MAX_NEW_TOKENS = 500
TOP_K_DOCUMENTS = 3

# --- 1. Data Loading and Preprocessing ---
def load_documents_from_jsonl(file_path):
    """从 JSONL 文件加载文档内容"""
    documents = []
    doc_ids = []
    print(f"Loading documents from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found!")
        return [], []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                content = data.get('contents') or data.get('text') or data.get('article') or data.get('body')
                if content:
                    documents.append(content)
                    doc_ids.append(data.get('identifier') or data.get('id') or f'doc_{i}')
                else:
                    print(f"Warning: Missing content in line {i+1}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON in line {i+1}")
    
    print(f"Loaded {len(documents)} documents.")
    return documents, doc_ids

def load_test_queries(file_path):
    """从TSV文件加载测试问题"""
    if not os.path.exists(file_path):
        print(f"Warning: Test file {file_path} not found!")
        return []
    
    try:
        # 读取文件并手动解析，因为使用的是 ||| 分隔符而不是制表符
        if file_path.endswith('.tsv'):
            queries_and_answers = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and '|||' in line:
                        parts = line.split('|||')
                        if len(parts) >= 2:
                            query = parts[0].strip()
                            answer = parts[1].strip()
                            queries_and_answers.append((query, answer, ""))
                        else:
                            print(f"Warning: Invalid format in line {line_num}: {line}")
                    else:
                        print(f"Warning: No separator found in line {line_num}: {line}")
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                queries_and_answers = json.load(f)
                if not isinstance(queries_and_answers, list):
                    print(f"Warning: Expected a list in {file_path}, got {type(queries_and_answers)}")
                    return []
                queries_and_answers = [(q['query'], q['answer'], q.get('reference', '')) for q in queries_and_answers]

        print(f"Loaded {len(queries_and_answers)} test queries from {file_path}")
        return queries_and_answers
    except Exception as e:
        print(f"Error loading test queries: {e}")
        return []

def chinese_tokenizer(text):
    """使用 jieba 进行中文分词"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    return list(jieba.cut(text))

# --- 2. BM25 Indexing ---
def build_bm25_index(documents):
    """构建 BM25 索引"""
    print("Tokenizing documents for BM25...")
    tokenized_corpus = [chinese_tokenizer(doc) for doc in documents]
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    print("BM25 index built.")
    return bm25

# --- 3. OpenAI Client Setup ---
def setup_openai_client():
    """设置 OpenAI 客户端连接到 VLLM 服务器"""
    print(f"Setting up OpenAI client to connect to VLLM server at {VLLM_API_BASE}...")
    client = OpenAI(
        api_key="EMPTY",  # VLLM 服务器不需要真实的 API key
        base_url=VLLM_API_BASE,
    )
    print("OpenAI client setup complete.")
    return client

# --- 4. RAG Query Function ---
def rag_query(query, bm25_index, original_documents, doc_ids, client, top_k=TOP_K_DOCUMENTS):
    """执行 RAG 查询"""
    print(f"\nPerforming RAG query: '{query}'")
    # 1. Retrieve relevant documents
    tokenized_query = chinese_tokenizer(query)
    doc_scores = bm25_index.get_scores(tokenized_query)
    
    top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    retrieved_docs_content = [original_documents[i] for i in top_n_indices]
    retrieved_docs_ids = [doc_ids[i] for i in top_n_indices]
    print(f"Retrieved {len(retrieved_docs_content)} documents.")
    for i, doc_id in enumerate(retrieved_docs_ids):
        print(f"  Doc ID: {doc_id}, Score: {doc_scores[top_n_indices[i]]:.4f}")
    
    # 2. Truncate documents to fit within model limits
    max_doc_length = 2000  # 限制每个文档的最大字符数
    truncated_docs = []
    for doc in retrieved_docs_content:
        if len(doc) > max_doc_length:
            truncated_docs.append(doc[:max_doc_length] + "...")
        else:
            truncated_docs.append(doc)
    
    # 3. Construct context and messages
    context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(truncated_docs)])
    
    # 使用标准的 OpenAI 消息格式
    messages = [
        {
            "role": "system",
            "content": "你是一个基于检索的问答助手。请根据以下提供的上下文信息来回答问题。如果上下文信息不足以回答问题，请说明你无法从提供的信息中找到答案。请用最简洁准确的语言回答问题。"
        },
        {
            "role": "user", 
            "content": f"上下文信息：\n{context}\n\n问题：{query}"
        }
    ]
    
    print(f"\nSending request to VLLM server...")
    print(f"Context length: {len(context)} characters")
    
    # 4. Generate answer using OpenAI API format
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            top_p=0.8,
            stop=None
        )
        
        answer = response.choices[0].message.content.strip()
        
        print("\nLLM Answer:")
        print(answer)
        return answer, retrieved_docs_ids, retrieved_docs_content
        
    except Exception as e:
        print(f"Error during API call: {e}")
        return "抱歉，生成答案时遇到问题。", retrieved_docs_ids, retrieved_docs_content

# --- 5. Batch Testing Function ---
def run_batch_test(test_queries, bm25_index, documents, doc_ids, client):
    """批量运行测试查询"""
    print(f"\n=== Running batch test on {len(test_queries)} queries ===")
    
    results = []
    for i, (query, expected_answer, reference) in enumerate(test_queries):
        print(f"\n--- Test {i+1}/{len(test_queries)} ---")
        print(f"Query: {query}")
        print(f"Expected: {expected_answer}")
        print(f"Reference: {reference}")
        
        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            query, bm25_index, documents, doc_ids, client
        )
        
        results.append({
            'query': query,
            'expected_answer': expected_answer,
            'generated_answer': generated_answer,
            'retrieved_docs': retrieved_ids,
            'retrieved_contents': retrieved_contents,
            'reference': reference
        })
        
        print("-" * 50)
    
    return results

# --- Main Execution ---
if __name__ == '__main__':
    # 1. 加载数据
    documents, doc_ids = load_documents_from_jsonl(JSONL_FILE_PATH)
    
    if not documents:
        print("No documents loaded. Exiting.")
        exit()

    # 2. 构建 BM25 索引
    bm25_index = build_bm25_index(documents)

    # 3. 设置 OpenAI 客户端
    try:
        client = setup_openai_client()
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        print("Please make sure VLLM server is running on http://localhost:8000")
        exit()

    # 4. 加载测试查询
    test_queries = load_test_queries(EXAMPLE_ANS_PATH)
    
    if test_queries:
        # 运行批量测试
        test_results = run_batch_test(test_queries, bm25_index, documents, doc_ids, client)
        
        # 保存测试结果
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print(f"\nTest results saved to test_results.json")
    else:
        print("\nNo test queries found, running example query...")
        test_query = "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"
        generated_answer, retrieved_ids, retrieved_contents = rag_query(
            test_query, bm25_index, documents, doc_ids, client
        )
