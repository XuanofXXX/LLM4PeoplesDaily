# LLM4PeoplesDaily (人民日报大模型问答系统)
本repo是一个基于大语言模型 (LLM) 的问答系统，旨在利用《人民日报》的文章语料库来回答问题。它集成了多种检索策略和缓存机制，并使用 VLLM 提供高效的 LLM 服务。使得这是一种high effeciency, low cost的RAG系统。


## 实验方法
使用清晰的代码管理和模块化设计，便于扩展和维护。接下来是我所用的主要方法

### 数据采集与处理

用于获取《人民日报》文章的脚本 (scripts/fetch_data.py, scripts/fetch_data_v2.py)。![爬虫脚本来源于](https://github.com/caspiankexin/people-daily-crawler-date)，在此基础上，发现爬取内容只有正文，而缺失作者信息，故在此基础上完成了对作者信息的补充。数据选取上，我们选取了22年到至今的文章，因为发现有些例子里面的答案并非在23/5~24/3之间，所以我这里全都爬下来了。

用于数据去重和初步处理的 Jupyter Notebook (data/data_process.ipynb)。对原始的数据进行去重，清理，并转换为 JSONL 格式。处理后的数据格式为 {"date": "YYYYMMDD", "content": "...", "file_path": "..."}。

基于 LLM 的文章筛选，以保留有意义的内容 (data/filter_by_llm.py)。发现有许多版面是只有`责任编辑`和`广告`的内容文章，因此在筛选时会将这些内容过滤掉。

使用 LLM 从文章生成问答对的脚本 (scripts/generate_qa_pairs.py)。本来是想自己造一些数据作为训练集和验证集进行微调的，后面发现`QwQ`造数据的能力不太行，而且微调有耗时耗力，故放弃，直接incontext learning。

根据日期范围切分语料库数据的脚本 (scripts/split_corpus.py)。考虑到所有文章都存起来的话，有点不太好，于是我们提供了一个脚本，可以根据日期范围来切分语料库数据。

### 灵活的检索策略:

我这里使用了多种检索策略。
- BM25: 经典的稀疏检索算法。

- 稠密检索 (Dense Retrieval): 使用句子转换器 (例如 BAAI/bge-large-zh-v1.5)。

- 混合检索 (Hybrid Retrieval): 结合 BM25 和稠密检索的评分，权重可配置。

### Google 搜索集成 

可以使用 Google 搜索作为备选方案或增强检索结果。但是这种不太稳定，有时候镜像站会挂掉，对网络的要求比较高。包含搜索结果的缓存功能 (google.py)。

### 评估与分析:

用于答案评估的精确匹配 (EM) 和基于集合匹配的工具函数 (utils.py)。

主脚本 (main.py) 支持批量测试和结果保存。

配置管理:

通过 config.py 和环境变量进行集中配置。

全面的日志记录功能 (logger_config.py)。

项目结构
```sh
.
├── cache/                    # 存储缓存数据的目录 (嵌入向量, 索引)
├── data/
│   ├── corpus_v4.jsonl       # 主要语料库文件示例 (可配置)
│   ├── eval/                 # 评估数据 (例如 official_test_ans.json)
│   ├── data_process.ipynb    # 数据处理的 Jupyter Notebook
│   └── filter_by_llm.py      # 基于 LLM 的文章筛选脚本
├── logs/                     # 日志文件目录
├── result/                   # 测试结果存储目录
├── scripts/
│   ├── fetch_data.py         # 获取人民日报文章的脚本
│   ├── fetch_data_v2.py      # 获取数据的备用脚本
│   ├── generate_qa_pairs.py  # 生成问答对的脚本
│   ├── split_corpus.py       # 切分语料库的脚本
│   └── vllm/                 # 启动不同模型的 VLLM 服务器脚本
│       ├── qwen2.5_0.5B.sh
│       ├── qwen2.5_7B.sh
│       └── qwen3_32B.sh
├── cache_manager.py          # 缓存管理机制
├── config.py                 # 配置文件 (路径, 模型, 参数)
├── data_loader.py            # 加载文档和测试问题
├── google.py                 # Google 搜索集成
├── logger_config.py          # 日志设置
├── main.py                   # 运行 RAG 系统和评估的主脚本
├── rag_query.py              # 处理 RAG 查询和 LLM 交互
├── retrieval.py              # 实现检索策略 (BM25, Dense, Hybrid)
├── utils.py                  # 工具函数 (例如答案匹配)
└── test.sh                   # 使用不同配置运行测试的示例 Shell 脚本
```
根据我的不同实验结果来看，在不开启google Search的情况下，这种方法的效果最好：


```
=== RAG System Configuration ===
JSONL_FILE_PATH: data/corpus_v3.jsonl # 主要语料库，包含22年到至今的人民日报文章
EXAMPLE_ANS_PATH: data/eval/official_test_ans.json
VLLM_API_BASE: http://localhost:8000/v1
MODEL_NAME: /media/public/models/huggingface/Qwen/Qwen2.5-7B-Instruct
MAX_NEW_TOKENS: 500
TOP_K_DOCUMENTS: 5
DENSE_MODEL_NAME: /media/public/models/huggingface/BAAI/bge-large-zh-v1.5
RETRIEVAL_METHOD: hybrid
BM25_WEIGHT: 0.7
DENSE_WEIGHT: 0.3
CACHE_BASE_DIR: cache
MAX_CONCURRENT: 100
ENABLE_GOOGLE_SEARCH: False
GOOGLE_SEARCH_TOPK: 3
USE_GOOGLE_FALLBACK: False
========================================
Loading documents from data/corpus_v3.jsonl...
Loaded 84388 documents.
Using existing cache from cache/ee05a3d7_hybrid_bge-large-zh-v1.5
Cache directory: cache/ee05a3d7_hybrid_bge-large-zh-v1.5
Cache valid: True
Building retrieval indices with method: hybrid
Loading cached BM25 index from cache/ee05a3d7_hybrid_bge-large-zh-v1.5/bm25_index.pkl
BM25 index loaded from cache.
Loading dense retrieval model: /media/public/models/huggingface/BAAI/bge-large-zh-v1.5
Dense retrieval model loaded successfully.
Loading cached document embeddings from cache/ee05a3d7_hybrid_bge-large-zh-v1.5/document_embeddings.pkl
Loaded 84388 cached embeddings.
Setting up AsyncOpenAI client to connect to VLLM server at http://localhost:8000/v1...
AsyncOpenAI client setup complete.
Loaded 25 test queries from data/eval/official_test_ans.json
```

但是如果开启 Google Search 的话，效果就会更好了，但我认为这个就设计到tool use的领域了，感觉不太适合本次实验🤔。所以最后也就没用了。

## 优化方法
### 缓存系统:

缓存文档嵌入和 BM25 索引，以加快初始化速度 (cache_manager.py)。

基于文件哈希和配置管理缓存的有效性。

包含缓存清理机制。

### VLLM 集成:
使用 VLLM 提供高效的 LLM 服务，支持多种模型 (config.py 中配置)。

### 使用异步操作
在检索和 LLM 交互中使用异步操作 (rag_query.py)，提高响应速度。

## 实验流程

准备数据:

使用 scripts/ 目录下的脚本获取和处理《人民日报》文章。

主要语料库应为 JSONL 文件，路径由 config.py 中的 JSONL_FILE_PATH 指定。每行应为一个 JSON 对象，包含文章内容 (键可以是 contents, content, text, article, 或 body) 和一个标识符 (键 identifier 或 id)。

评估问题可以是 JSON, JSONL, 或 TSV 格式，由 EXAMPLE_ANS_PATH 指定。

使用 scripts/vllm/ 目录下的脚本启动模型的 VLLM 服务器

VLLM API 端点在 config.py (VLLM_API_BASE) 中配置。

所有的配置都可以通过config.py 或环境变量进行修改。