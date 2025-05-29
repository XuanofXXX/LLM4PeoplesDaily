#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from googlesearch import search, asearch
from typing import List, Union, Dict, Optional
import time
import requests
from bs4 import BeautifulSoup
import re
import asyncio
import json
import os
import hashlib
from datetime import datetime, timedelta

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


class GoogleSearcher:
    def __init__(
        self, 
        topk: int = 10, 
        batch_size: int = 1, 
        sleep_interval: float = 1.0,
        cache_dir: Optional[str] = None,
        cache_expire_hours: int = 24,
        use_memory_cache: bool = True
    ):
        """
        Google搜索器

        参数:
        topk: 默认返回结果数量
        batch_size: 批处理大小（为了避免被限制，建议保持为1）
        sleep_interval: 搜索间隔时间（秒），避免请求过于频繁
        cache_dir: 缓存文件存储目录，如果为None则只使用内存缓存
        cache_expire_hours: 缓存过期时间（小时）
        use_memory_cache: 是否使用内存缓存
        """
        self.topk = topk
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 缓存设置
        self.cache_dir = cache_dir
        self.cache_expire_hours = cache_expire_hours
        self.use_memory_cache = use_memory_cache
        self.memory_cache = {} if use_memory_cache else None
        
        # 创建缓存目录
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, query: str, num: int) -> str:
        """生成缓存键"""
        cache_str = f"{query}_{num}"
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        print(f"  - 默认返回结果数: {self.topk}")
        print(f"  - 搜索间隔: {self.sleep_interval}秒")
        print(f"  - 缓存目录: {self.cache_dir or '未设置'}")
        print(f"  - 缓存过期时间: {self.cache_expire_hours}小时")
        print(f"  - 内存缓存: {'启用' if self.use_memory_cache else '禁用'}")

    def _get_cache_key(self, query: str, num: int) -> str:
        """生成缓存键"""
        cache_str = f"{query}_{num}"
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _is_cache_valid(self, cache_data: Dict) -> bool:
        """检查缓存是否有效"""
        if 'timestamp' not in cache_data:
            return False
        
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        expire_time = cache_time + timedelta(hours=self.cache_expire_hours)
        return datetime.now() < expire_time

    def _get_from_cache(self, query: str, num: int) -> Optional[List[Dict[str, str]]]:
        """从缓存中获取搜索结果"""
        cache_key = self._get_cache_key(query, num)
        
        # 首先检查内存缓存
        if self.use_memory_cache and cache_key in self.memory_cache:
            cache_data = self.memory_cache[cache_key]
            if self._is_cache_valid(cache_data):
                print(f"从内存缓存获取结果: {query}")
                return cache_data['results']
            else:
                # 移除过期的内存缓存
                del self.memory_cache[cache_key]
        
        # 检查文件缓存
        if self.cache_dir:
            cache_file = self._get_cache_file_path(cache_key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if self._is_cache_valid(cache_data):
                        print(f"从文件缓存获取结果: {query}")
                        # 同时更新内存缓存
                        if self.use_memory_cache:
                            self.memory_cache[cache_key] = cache_data
                        return cache_data['results']
                    else:
                        # 删除过期的文件缓存
                        os.remove(cache_file)
                except (json.JSONDecodeError, KeyError, IOError):
                    # 缓存文件损坏，删除它
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
        
        return None

    def _save_to_cache(self, query: str, num: int, results: List[Dict[str, str]]):
        """保存搜索结果到缓存"""
        cache_key = self._get_cache_key(query, num)
        cache_data = {
            'query': query,
            'num': num,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到内存缓存
        if self.use_memory_cache:
            self.memory_cache[cache_key] = cache_data
        
        # 保存到文件缓存
        if self.cache_dir:
            cache_file = self._get_cache_file_path(cache_key)
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except IOError as e:
                print(f"保存缓存文件失败: {e}")

    def clear_cache(self, query: Optional[str] = None):
        """清除缓存"""
        if query is None:
            # 清除所有缓存
            if self.use_memory_cache:
                self.memory_cache.clear()
            if self.cache_dir and os.path.exists(self.cache_dir):
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, file))
            print("已清除所有缓存")
        else:
            # 清除特定查询的缓存
            for num in [1, 5, 10, 20, 50]:  # 常见的num值
                cache_key = self._get_cache_key(query, num)
                if self.use_memory_cache and cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                if self.cache_dir:
                    cache_file = self._get_cache_file_path(cache_key)
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
            print(f"已清除查询 '{query}' 的缓存")

    async def acrawl_webpage(url: str) -> Dict[str, str]:
        """
        爬取网页内容并提取信息
        参数:
        url: 网页URL
        返回:
        Dict: 包含url, title, contents, author, date的字典
        """
        result = {
            "url": url,
            "title": "",
            "contents": "",
            "author": "google",
            "date": "无",
        }

        try:
            # 发起请求
            response = await asyncio.to_thread(
                requests.get, url, headers=HEADERS, timeout=10
            )
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 提取标题
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text().strip()

            # 移除脚本和样式标签
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # 提取正文内容
            # 优先从常见的内容容器中提取
            content_selectors = [
                "article",
                "main",
                ".content",
                "#content",
                ".post-content",
                ".entry-content",
                ".article-content",
                ".news-content",
                "p",
            ]

            text_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text_content += element.get_text() + " "
                    if len(text_content.strip()) > 100:  # 如果找到足够内容就停止
                        break

            # 如果上述方法没找到足够内容，直接从body提取
            if len(text_content.strip()) < 100:
                body = soup.find("body")
                if body:
                    text_content = body.get_text()

            # 清理文本
            text_content = re.sub(r"\s+", " ", text_content).strip()

            # 截取前500个字符
            if len(text_content) > 500:
                result["contents"] = text_content[:500]
            else:
                result["contents"] = text_content

        except requests.RequestException as e:
            print(f"请求错误 {url}: {e}")
        except Exception as e:
            print(f"解析错误 {url}: {e}")

        return result

    async def asingle_search(self, query: str, num: int = 10) -> List[Dict[str, str]]:
        """
        单个查询搜索并爬取内容（异步版本）
        """
        # 首先尝试从缓存获取
        cached_results = self._get_from_cache(query, num)
        if cached_results is not None:
            return cached_results
        
        results = []
        try:
            # 使用异步搜索
            search_results = []
            async for url in asearch(query, num_results=num, region="CN", lang="zh-CN"):
                search_results.append(url)
                if len(search_results) >= num:
                    break

            # 异步并发爬取所有URL
            if search_results:
                tasks = [self.acrawl_webpage(url) for url in search_results]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                # 过滤掉异常结果
                results = [r for r in results if isinstance(r, dict)]
                
                # 保存到缓存
                self._save_to_cache(query, num, results)
                print(f"已缓存搜索结果: {query}")
                
        except Exception as e:
            print(f"搜索出错 '{query}': {e}")
            print("提示: 可能是网络问题或搜索频率过高，请稍后再试")

        return results

    async def abatch_search(
        self, query: Union[str, List[str]], num: int = 10
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        """
        批量搜索并爬取内容（异步版本）

        参数:
        query: 搜索关键词，可以是字符串或字符串列表
        num: 返回结果数量

        返回:
        如果query是字符串，返回List[Dict]
        如果query是列表，返回List[List[Dict]]
        """
        if isinstance(query, str):
            query = [query]

        # 使用异步方式并发处理所有查询
        tasks = [self.asingle_search(q, num) for q in query]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"查询 '{query[i]}' 出错: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)

        # 如果原始输入是单个字符串，返回单个结果列表
        if len(query) == 1:
            return valid_results[0]
        return valid_results

    async def asearch(self, query: Union[str, List[str]], num: int = None):
        """
        异步搜索接口
        """
        if num is None:
            num = self.topk
        return await self.abatch_search(query, num)

    def _crawl_webpage(self, url: str) -> Dict[str, str]:
        """
        爬取网页内容并提取信息
        参数:
        url: 网页URL
        返回:
        Dict: 包含url, title, contents, author, date的字典
        """
        result = {
            "url": url,
            "title": "",
            "contents": "",
            "author": "google",
            "date": "无",
        }

        try:
            # 发起请求
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 解析HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # 提取标题
            title_tag = soup.find("title")
            if title_tag:
                result["title"] = title_tag.get_text().strip()

            # 移除脚本和样式标签
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # 提取正文内容
            # 优先从常见的内容容器中提取
            content_selectors = [
                "article",
                "main",
                ".content",
                "#content",
                ".post-content",
                ".entry-content",
                ".article-content",
                ".news-content",
                "p",
            ]

            text_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text_content += element.get_text() + " "
                    if len(text_content.strip()) > 100:  # 如果找到足够内容就停止
                        break

            # 如果上述方法没找到足够内容，直接从body提取
            if len(text_content.strip()) < 100:
                body = soup.find("body")
                if body:
                    text_content = body.get_text()

            # 清理文本
            text_content = re.sub(r"\s+", " ", text_content).strip()

            # 截取前500个字符
            if len(text_content) > 500:
                result["contents"] = text_content[:500]
            else:
                result["contents"] = text_content

        except requests.RequestException as e:
            print(f"请求错误 {url}: {e}")
        except Exception as e:
            print(f"解析错误 {url}: {e}")

        return result

    def _single_search(self, query: str, num: int = None) -> List[Dict[str, str]]:
        """
        单个查询搜索并爬取内容
        参数:
        query: 搜索关键词
        num: 返回结果数量，如果为None则使用self.topk

        返回:
        List[Dict]: 包含网页信息的字典列表
        """
        if num is None:
            num = self.topk

        # 首先尝试从缓存获取
        cached_results = self._get_from_cache(query, num)
        if cached_results is not None:
            return cached_results

        results = []
        try:
            print(f"正在搜索: {query}")
            # 执行搜索
            search_results = search(query, num_results=num, region="CN", lang="zh-CN")
            # 收集并爬取结果
            count = 0
            for url in search_results:
                print(f"正在爬取第{count + 1}个网页: {url}")
                webpage_data = self._crawl_webpage(url)
                results.append(webpage_data)
                count += 1
                if count >= num:
                    break
                # 爬取间隔，避免过于频繁
                time.sleep(0.5)
            
            # 保存到缓存
            if results:
                self._save_to_cache(query, num, results)
                print(f"已缓存搜索结果: {query}")

        except Exception as e:
            print(f"搜索出错 '{query}': {e}")
            print("提示: 可能是网络问题或搜索频率过高，请稍后再试")

        return results

    def _batch_search(
        self, query: Union[str, List[str]], num: int = None
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        """
        批量搜索并爬取内容

        参数:
        query: 搜索关键词，可以是字符串或字符串列表
        num: 返回结果数量，如果为None则使用self.topk

        返回:
        如果query是字符串，返回List[Dict]
        如果query是列表，返回List[List[Dict]]
        """
        if isinstance(query, str):
            query = [query]

        if num is None:
            num = self.topk

        results = []

        for i, q in enumerate(query):
            # 搜索并爬取单个查询
            single_results = self._single_search(q, num)
            results.append(single_results)

            # 添加延迟避免被限制
            if i < len(query) - 1:  # 不在最后一次搜索后等待
                time.sleep(self.sleep_interval)

        # 如果原始输入是单个字符串，返回单个结果列表
        if len(query) == 1:
            return results[0]

        # 如果是多个查询，返回结果列表的列表
        return results

    def search(self, query: Union[str, List[str]], num: int = None):
        """
        同步搜索接口
        """
        return self._batch_search(query, num)


def main():
    print("=== Google 搜索和爬取工具 ===")

    # 创建搜索器，启用缓存
    searcher = GoogleSearcher(
        topk=3, 
        sleep_interval=1.0,
        cache_dir="cache/search_cache",  # 设置缓存目录
        cache_expire_hours=24,       # 缓存24小时
        use_memory_cache=True        # 启用内存缓存
)

    # 单个查询测试
    print("\n--- 单个查询测试 ---")
    single_query = "2023年是中国和所罗门群岛建交多少年？"
    single_results = searcher.search(single_query)
    print(f"搜索结果数量: {len(single_results)}")

    for i, result in enumerate(single_results, 1):
        print(f"\n结果 {i}:")
        print(f"URL: {result['url']}")
        print(f"标题: {result['title']}")
        print(f"作者: {result['author']}")
        print(f"日期: {result['date']}")
        print(f"内容: {result['contents'][:200]}...")
    
    # 测试缓存功能
    print("\n--- 缓存测试 ---")
    print("再次执行相同搜索（应该从缓存获取）:")
    single_results_cached = searcher.search(single_query)
    print(f"缓存搜索结果数量: {len(single_results_cached)}")


if __name__ == "__main__":
    main()
