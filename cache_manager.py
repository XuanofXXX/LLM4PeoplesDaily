import os
import json
import hashlib
import shutil
from datetime import datetime
import logging

logger = logging.getLogger("rag_system")

class CacheManager:
    def __init__(self, cache_base_dir):
        self.cache_base_dir = cache_base_dir
    
    def calculate_file_hash(self, file_path):
        """计算文件的MD5哈希值"""
        if not os.path.exists(file_path):
            return None

        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def get_cache_dir(self, file_path, retrieval_method, dense_model_name):
        """根据文件哈希和配置生成缓存目录"""
        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return None

        # 使用文件哈希前8位作为目录名，避免过长
        cache_dir_name = (
            f"{file_hash[:8]}_{retrieval_method}_{os.path.basename(dense_model_name)}"
        )
        cache_dir = os.path.join(self.cache_base_dir, cache_dir_name)
        return cache_dir, file_hash

    def get_cache_paths(self, cache_dir):
        """获取缓存文件路径"""
        if not cache_dir:
            return None

        return {
            "embeddings": os.path.join(cache_dir, "document_embeddings.pkl"),
            "bm25": os.path.join(cache_dir, "bm25_index.pkl"),
            "info": os.path.join(cache_dir, "cache_info.json"),
        }

    def save_cache_info(self, cache_dir, file_hash, retrieval_method, dense_model_name, file_path):
        """保存缓存信息"""
        cache_info = {
            "file_hash": file_hash,
            "file_path": file_path,
            "retrieval_method": retrieval_method,
            "dense_model_name": dense_model_name,
            "timestamp": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }

        os.makedirs(cache_dir, exist_ok=True)
        cache_info_path = os.path.join(cache_dir, "cache_info.json")
        with open(cache_info_path, "w", encoding="utf-8") as f:
            json.dump(cache_info, f, ensure_ascii=False, indent=2)

    def load_cache_info(self, cache_dir):
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

    def update_cache_access_time(self, cache_dir):
        """更新缓存访问时间"""
        if not cache_dir:
            return

        cache_info = self.load_cache_info(cache_dir)
        if cache_info:
            cache_info["last_accessed"] = datetime.now().isoformat()
            cache_info_path = os.path.join(cache_dir, "cache_info.json")
            with open(cache_info_path, "w", encoding="utf-8") as f:
                json.dump(cache_info, f, ensure_ascii=False, indent=2)

    def is_cache_valid(self, cache_dir, file_path, retrieval_method, dense_model_name):
        """检查缓存是否有效"""
        cache_info = self.load_cache_info(cache_dir)
        if not cache_info:
            return False

        current_hash = self.calculate_file_hash(file_path)

        is_valid = (
            cache_info.get("file_hash") == current_hash
            and cache_info.get("retrieval_method") == retrieval_method
            and cache_info.get("dense_model_name") == dense_model_name
        )

        if is_valid:
            self.update_cache_access_time(cache_dir)
            logger.info(f"Using existing cache from {cache_dir}")

        return is_valid

    def list_available_caches(self):
        """列出所有可用的缓存"""
        if not os.path.exists(self.cache_base_dir):
            return []

        caches = []
        for cache_dir_name in os.listdir(self.cache_base_dir):
            cache_dir = os.path.join(self.cache_base_dir, cache_dir_name)
            if os.path.isdir(cache_dir):
                cache_info = self.load_cache_info(cache_dir)
                if cache_info:
                    caches.append({"dir": cache_dir, "info": cache_info})

        return caches

    def cleanup_old_caches(self, max_caches=10):
        """清理旧的缓存，保留最近使用的缓存"""
        caches = self.list_available_caches()
        if len(caches) <= max_caches:
            return

        # 按最后访问时间排序
        caches.sort(key=lambda x: x["info"].get("last_accessed", ""), reverse=True)

        # 删除最旧的缓存
        for cache in caches[max_caches:]:
            cache_dir = cache["dir"]
            try:
                shutil.rmtree(cache_dir)
                logger.info(f"Cleaned up old cache: {cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup cache {cache_dir}: {e}")
