#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from googlesearch import search
from typing import List, Union, Dict
import time
import requests
from bs4 import BeautifulSoup
import re


class GoogleSearcher:
    def __init__(
        self, topk: int = 10, batch_size: int = 1, sleep_interval: float = 1.0
    ):
        """
        Google搜索器

        参数:
        topk: 默认返回结果数量
        batch_size: 批处理大小（为了避免被限制，建议保持为1）
        sleep_interval: 搜索间隔时间（秒），避免请求过于频繁
        """
        self.topk = topk
        self.batch_size = batch_size
        self.sleep_interval = sleep_interval
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

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

        results = []
        try:
            print(f"正在搜索: {query}")
            # 执行搜索
            search_results = search(query, num_results=num)
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
        搜索接口
        """
        return self._batch_search(query, num)


def main():
    print("=== Google 搜索和爬取工具 ===")

    # 创建搜索器
    searcher = GoogleSearcher(topk=3, sleep_interval=1.0)

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
        print(f"内容: {result['contents'][:500]}...")


if __name__ == "__main__":
    main()
