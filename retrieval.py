import os
import torch
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import chinese_tokenizer
import logging
import concurrent.futures
import threading

logger = logging.getLogger("rag_system")


class RetrievalSystem:
    def __init__(self, dense_model_name=None):
        self.dense_model_name = dense_model_name
        self.dense_model = None
        self.bm25_index = None
        self.document_embeddings = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 添加线程锁以确保线程安全
        self._bm25_lock = threading.Lock()
        self._dense_lock = threading.Lock()

    def setup_dense_model(self):
        """设置dense retrieval模型"""
        if not self.dense_model_name:
            return None

        logger.info(f"Loading dense retrieval model: {self.dense_model_name}")
        try:
            self.dense_model = SentenceTransformer(
                self.dense_model_name, device=self.device
            )
            logger.info("Dense retrieval model loaded successfully.")
            return self.dense_model
        except Exception as e:
            logger.error(f"Failed to load dense retrieval model: {e}")
            logger.info("Falling back to BM25 only.")
            return None

    def build_document_embeddings(self, documents, cache_path=None):
        """构建文档的dense embeddings"""
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached document embeddings from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    self.document_embeddings = pickle.load(f)
                logger.info(
                    f"Loaded {self.document_embeddings.shape[0]} cached embeddings."
                )
                return self.document_embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings cache: {e}")

        if not self.dense_model:
            logger.warning("Dense model not available")
            return None

        logger.info("Building document embeddings...")
        # 预处理文档：截断过长的文档
        processed_docs = []
        for doc in documents:
            if len(doc) > 512:  # 限制输入长度
                processed_docs.append(doc[:512])
            else:
                processed_docs.append(doc)

        self.document_embeddings = self.dense_model.encode(
            processed_docs, batch_size=32, show_progress_bar=True
        )

        # 缓存embeddings
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.document_embeddings, f)
                logger.info(f"Document embeddings cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache embeddings: {e}")

        logger.info(f"Built embeddings for {len(self.document_embeddings)} documents.")
        return self.document_embeddings

    def build_bm25_index(self, documents, cache_path=None):
        """构建 BM25 索引，支持缓存"""
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached BM25 index from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    self.bm25_index = pickle.load(f)
                logger.info("BM25 index loaded from cache.")
                return self.bm25_index
            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}")

        logger.info("Tokenizing documents for BM25...")
        tokenized_corpus = [chinese_tokenizer(doc) for doc in documents]
        logger.info("Building BM25 index...")
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built.")

        # 缓存BM25索引
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.bm25_index, f)
                logger.info(f"BM25 index cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache BM25 index: {e}")

        return self.bm25_index

    def bm25_retrieve(self, query, top_k):
        """使用BM25检索（线程安全版本）"""
        if not self.bm25_index:
            logger.error("BM25 index not built")
            return np.array([]), np.array([])

        with self._bm25_lock:
            tokenized_query = chinese_tokenizer(query)
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            return top_indices, doc_scores[top_indices]

    def dense_retrieve(self, query, top_k):
        """使用dense retrieval检索（线程安全版本）"""
        if not self.dense_model or self.document_embeddings is None:
            logger.error("Dense model or embeddings not available")
            return np.array([]), np.array([])

        with self._dense_lock:
            query_embedding = self.dense_model.encode([query])
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return top_indices, similarities[top_indices]

    def hybrid_retrieve(self, query, top_k, bm25_weight=0.5, dense_weight=0.5):
        """混合检索策略（并行版本）"""
        if (
            not self.bm25_index
            or not self.dense_model
            or self.document_embeddings is None
        ):
            logger.warning(
                "Missing components for hybrid retrieval, falling back to available method"
            )
            if self.bm25_index:
                return self.bm25_retrieve(query, top_k)
            elif self.dense_model and self.document_embeddings is not None:
                return self.dense_retrieve(query, top_k)
            else:
                return np.array([]), np.array([])

        # 获取更多的候选文档用于重排序
        candidate_k = min(top_k * 3, len(self.document_embeddings))

        # 并行执行BM25和Dense检索
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 提交BM25检索任务
            bm25_future = executor.submit(self.bm25_retrieve, query, candidate_k)
            # 提交Dense检索任务
            dense_future = executor.submit(self.dense_retrieve, query, candidate_k)

            # 等待两个任务完成
            bm25_indices, bm25_scores = bm25_future.result()
            dense_indices, dense_scores = dense_future.result()

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

    def retrieve(
        self, query, top_k, method="hybrid", bm25_weight=0.5, dense_weight=0.5
    ):
        """统一的检索接口"""
        if method == "bm25" or (method == "hybrid" and dense_weight == 0):
            return self.bm25_retrieve(query, top_k)
        elif method == "dense" or (method == "hybrid" and bm25_weight == 0):
            return self.dense_retrieve(query, top_k)
        elif method == "hybrid":
            return self.hybrid_retrieve(query, top_k, bm25_weight, dense_weight)
        else:
            logger.warning(f"Unknown retrieval method: {method}, falling back to BM25")
            return self.bm25_retrieve(query, top_k)
