'''
代码名称：异步爬取人民日报数据为txt文件
编写日期：2025年1月1日
作者：github（caspiankexin）
版本：第4版（异步版本）
可爬取的时间范围：2024年12月起
注意：此代码仅供交流学习，不得作为其他用途。
'''

import aiohttp
import asyncio
import bs4
import os
import datetime
import time
from pathlib import Path
import logging
from typing import List, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RMRBSpider:
    def __init__(self, max_concurrent=5, delay=1):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.session = None
        self.headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=self.max_concurrent)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_url(self, url: str) -> str:
        """异步获取网页内容"""
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                content = await response.text(encoding='utf-8')
                return content
        except Exception as e:
            logger.error(f"获取URL失败 {url}: {e}")
            raise

    async def get_page_list(self, year: str, month: str, day: str) -> List[str]:
        """获取当天报纸的各版面链接列表"""
        url = f'http://paper.people.com.cn/rmrb/pc/layout/{year}{month}/{day}/node_01.html'
        try:
            html = await self.fetch_url(url)
            bsobj = bs4.BeautifulSoup(html, 'html.parser')
            
            temp = bsobj.find('div', attrs={'id': 'pageList'})
            if temp:
                pageList = temp.ul.find_all('div', attrs={'class': 'right_title-name'})
            else:
                pageList = bsobj.find('div', attrs={'class': 'swiper-container'}).find_all('div', attrs={'class': 'swiper-slide'})
            
            linkList = []
            for page in pageList:
                link = page.a["href"]
                page_url = f'http://paper.people.com.cn/rmrb/pc/layout/{year}{month}/{day}/{link}'
                linkList.append(page_url)
            
            return linkList
        except Exception as e:
            logger.error(f"获取版面列表失败 {year}-{month}-{day}: {e}")
            return []

    async def get_title_list(self, year: str, month: str, day: str, page_url: str) -> List[str]:
        """获取报纸某一版面的文章链接列表"""
        try:
            html = await self.fetch_url(page_url)
            bsobj = bs4.BeautifulSoup(html, 'html.parser')
            
            temp = bsobj.find('div', attrs={'id': 'titleList'})
            if temp:
                titleList = temp.ul.find_all('li')
            else:
                titleList = bsobj.find('ul', attrs={'class': 'news-list'}).find_all('li')
            
            linkList = []
            for title in titleList:
                tempList = title.find_all('a')
                for temp in tempList:
                    link = temp["href"]
                    if 'content' in link:
                        article_url = f'http://paper.people.com.cn/rmrb/pc/content/{year}{month}/{day}/{link}'
                        linkList.append(article_url)
            
            return linkList
        except Exception as e:
            logger.error(f"获取文章列表失败 {page_url}: {e}")
            return []

    # def get_content(self, html: str) -> str:
    #     """解析HTML网页，获取新闻的文章内容"""
    #     try:
    #         bsobj = bs4.BeautifulSoup(html, 'html.parser')
            
    #         # 获取文章标题
    #         title = bsobj.h3.text + '\n' + bsobj.h1.text + '\n' + bsobj.h2.text + '\n'
            
    #         # 获取文章内容
    #         pList = bsobj.find('div', attrs={'id': 'ozoom'}).find_all('p')
    #         content = ''
    #         for p in pList:
    #             content += p.text + '\n'
            
    #         return title + content
    #     except Exception as e:
    #         logger.error(f"解析文章内容失败: {e}")
    #         return ""

    def get_content(self, html: str) -> str:
        """
        功能：解析 HTML 网页，获取新闻的文章内容
        参数：html 网页内容
        """
        try:
            bsobj = bs4.BeautifulSoup(html, "html.parser")
            with open('tmp_html.html', 'w', encoding='utf-8') as f:
                f.write(html)

            # 首先尝试查找 div.article 元素
            article_div = bsobj.find("div", attrs={"class": "article"})

            if article_div:
                # 提取标题信息
                h3 = article_div.find("h3")
                h1 = article_div.find("h1")
                h2 = article_div.find("h2")

                title_parts = []
                if h3 and h3.text.strip():
                    title_parts.append(h3.text.strip())
                if h1 and h1.text.strip():
                    title_parts.append(h1.text.strip())
                if h2 and h2.text.strip():
                    title_parts.append(h2.text.strip())

                title = "\n".join(title_parts) + "\n" if title_parts else ""

                # 提取日期信息
                date_span = article_div.find("span", attrs={"class": "date"})
                date_info = (
                    date_span.text.strip() + "\n"
                    if date_span and date_span.text.strip()
                    else ""
                )

                # 提取作者
                writer_span = article_div.find("p", attrs={"class": "sec"})
                writer_info = (
                    writer_span.text.strip() + "\n"
                    if writer_span and writer_span.text.strip()
                    else ""
                )
                # 提取图片说明文字（在table中的p标签）
                image_captions = []
                tables = article_div.find_all("table")
                for table in tables:
                    caption_p = table.find("p")
                    if caption_p and caption_p.text.strip():
                        image_captions.append(caption_p.text.strip())

                image_caption_text = (
                    "\n".join(image_captions) + "\n" if image_captions else ""
                )

                # 提取正文内容（在ozoom div中的p标签）
                content_parts = []
                ozoom = article_div.find("div", attrs={"id": "ozoom"})
                if ozoom:
                    # 查找所有p标签，排除在table内的p标签（那些是图片说明）
                    p_tags = ozoom.find_all("p")
                    for p in p_tags:
                        # 检查p标签是否在table内
                        if not p.find_parent("table"):
                            text = p.text.strip()
                            if text:
                                content_parts.append(text)

                main_content = "\n".join(content_parts) if content_parts else ""

                # 组合所有内容
                full_content = (
                    title + writer_info + date_info + image_caption_text + main_content
                )
                return full_content.strip()

            else:
                # 如果没有找到div.article，回退到原来的逻辑
                # 获取文章 标题
                h3 = bsobj.h3.text if bsobj.h3 else ""
                h1 = bsobj.h1.text if bsobj.h1 else ""
                h2 = bsobj.h2.text if bsobj.h2 else ""
                title = f"{h3}\n{h1}\n{h2}\n"

                # 获取文章 内容
                ozoom = bsobj.find("div", attrs={"id": "ozoom"})
                if ozoom:
                    pList = ozoom.find_all("p")
                    content = ""
                    for p in pList:
                        content += p.text + "\n"
                else:
                    content = ""

                # 返回结果 标题+内容
                return title + content

        except Exception as e:
            print(f"解析内容失败: {e}")
            return ""


    async def save_file(self, content: str, path: str, filename: str):
        """异步保存文件"""
        try:
            # 确保目录存在
            Path(path).mkdir(parents=True, exist_ok=True)
            
            # 异步写入文件
            file_path = Path(path) / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"保存文件失败 {path}/{filename}: {e}")

    async def download_article(self, url: str, year: str, month: str, day: str, 
                             page_no: int, title_no: int, destdir: str):
        """下载单篇文章"""
        try:
            # 添加延迟避免被封IP
            await asyncio.sleep(self.delay)
            
            html = await self.fetch_url(url)
            content = self.get_content(html)
            
            if content:
                path = f"{destdir}/{year}{month}{day}/"
                filename = f"{year}{month}{day}-{page_no:02d}-{title_no:02d}.txt"
                await self.save_file(content, path, filename)
                logger.info(f"保存文章: {filename}")
            
        except Exception as e:
            logger.error(f"下载文章失败 {url}: {e}")

    async def download_rmrb(self, year: str, month: str, day: str, destdir: str):
        """爬取指定日期的人民日报内容"""
        try:
            page_list = await self.get_page_list(year, month, day)
            
            tasks = []
            for page_no, page_url in enumerate(page_list, 1):
                title_list = await self.get_title_list(year, month, day, page_url)
                
                for title_no, article_url in enumerate(title_list, 1):
                    task = self.download_article(
                        article_url, year, month, day, page_no, title_no, destdir
                    )
                    tasks.append(task)
            
            # 使用信号量控制并发数
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def bounded_download(task):
                async with semaphore:
                    await task
            
            # 执行所有下载任务
            bounded_tasks = [bounded_download(task) for task in tasks]
            await asyncio.gather(*bounded_tasks, return_exceptions=True)
            
            logger.info(f"完成日期 {year}-{month}-{day} 的爬取")
            
        except Exception as e:
            logger.error(f"爬取日期 {year}-{month}-{day} 失败: {e}")

def gen_dates(b_date: datetime.datetime, days: int):
    """生成日期序列"""
    day = datetime.timedelta(days=1)
    for i in range(days):
        yield b_date + day * i

def get_date_list(begin_date: str, end_date: str) -> List[datetime.datetime]:
    """获取日期列表"""
    start = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")
    
    data = []
    for d in gen_dates(start, (end - start).days + 1):
        data.append(d)
    
    return data

async def main():
    """主函数"""
    print("欢迎使用人民日报异步爬虫，请输入以下信息：")
    # begin_date = input('请输入开始日期(YYYYMMDD):') or '20241201'
    # end_date = input('请输入结束日期(YYYYMMDD):') or time.strftime('%Y%m%d', time.localtime())
    # destdir = input("请输入数据保存的地址：") or '/home/xiachunxuan/nlp-homework/cache/d/'
    begin_date = '20241201'
    end_date = time.strftime('%Y%m%d', time.localtime())
    destdir = '/home/xiachunxuan/nlp-homework/data/rmrb_data/'
    
    
    # 配置参数
    max_concurrent = int(input("请输入最大并发数(默认100):") or "100")
    delay = float(input("请输入请求间隔秒数(默认1):") or "1")
    
    date_list = get_date_list(begin_date, end_date)
    
    async with RMRBSpider(max_concurrent=max_concurrent, delay=delay) as spider:
        for d in date_list:
            year = str(d.year)
            month = f"{d.month:02d}"
            day = f"{d.day:02d}"
            
            await spider.download_rmrb(year, month, day, destdir)
            print(f"爬取完成：{year}{month}{day}")
    
    print("数据爬取完成！")

if __name__ == '__main__':
    asyncio.run(main())

