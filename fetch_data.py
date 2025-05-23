"""
功能：爬取人民日报的文章
爬取范围：2022年至2024年11月
作者GitHub：@caspiankexin
软件编写时间：2021年
注意：此代码仅供交流学习，不得作为其他用途。
"""

import aiohttp
import asyncio
import bs4
import os
import datetime
import aiofiles
from typing import List


class RMRBSpider:
    def __init__(self, max_concurrent_requests: int = 10):
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        }
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """
        功能：异步访问 url 的网页，获取网页内容并返回
        参数：aiohttp会话，目标网页的 url
        返回：目标网页的 html 内容
        """
        async with self.semaphore:
            try:
                async with session.get(url, headers=self.headers) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                print(f"获取URL失败 {url}: {e}")
                return ""

    async def get_page_list(
        self, session: aiohttp.ClientSession, year: str, month: str, day: str
    ) -> List[str]:
        """
        功能：异步获取当天报纸的各版面的链接列表
        参数：aiohttp会话，年，月，日
        """
        url = f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/nbs.D110000renmrb_01.htm"
        html = await self.fetch_url(session, url)

        if not html:
            return []

        bsobj = bs4.BeautifulSoup(html, "html.parser")
        temp = bsobj.find("div", attrs={"id": "pageList"})

        if temp:
            pageList = temp.ul.find_all("div", attrs={"class": "right_title-name"})
        else:
            container = bsobj.find("div", attrs={"class": "swiper-container"})
            if container:
                pageList = container.find_all("div", attrs={"class": "swiper-slide"})
            else:
                return []

        linkList = []
        for page in pageList:
            if page.a and page.a.get("href"):
                link = page.a["href"]
                url = (
                    f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/{link}"
                )
                linkList.append(url)

        return linkList

    async def get_title_list(
        self,
        session: aiohttp.ClientSession,
        year: str,
        month: str,
        day: str,
        page_url: str,
    ) -> List[str]:
        """
        功能：异步获取报纸某一版面的文章链接列表
        参数：aiohttp会话，年，月，日，该版面的链接
        """
        html = await self.fetch_url(session, page_url)

        if not html:
            return []

        bsobj = bs4.BeautifulSoup(html, "html.parser")
        temp = bsobj.find("div", attrs={"id": "titleList"})

        if temp:
            titleList = temp.ul.find_all("li")
        else:
            news_list = bsobj.find("ul", attrs={"class": "news-list"})
            if news_list:
                titleList = news_list.find_all("li")
            else:
                return []

        linkList = []
        for title in titleList:
            tempList = title.find_all("a")
            for temp in tempList:
                if temp.get("href"):
                    link = temp["href"]
                    if "nw.D110000renmrb" in link:
                        url = f"http://paper.people.com.cn/rmrb/html/{year}-{month}/{day}/{link}"
                        linkList.append(url)

        return linkList

    def get_content(self, html: str) -> str:
        """
        功能：解析 HTML 网页，获取新闻的文章内容
        参数：html 网页内容
        """
        try:
            bsobj = bs4.BeautifulSoup(html, "html.parser")

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
        """
        功能：异步将文章内容 content 保存到本地文件中
        参数：要保存的内容，路径，文件名
        """
        # 如果没有该文件夹，则自动生成
        if not os.path.exists(path):
            os.makedirs(path)

        # 异步保存文件
        async with aiofiles.open(path + filename, "w", encoding="utf-8") as f:
            await f.write(content)

    async def process_article(
        self,
        session: aiohttp.ClientSession,
        url: str,
        year: str,
        month: str,
        day: str,
        destdir: str,
    ):
        """
        功能：处理单篇文章
        """
        try:
            # 获取新闻文章内容
            html = await self.fetch_url(session, url)
            if not html:
                return

            content = self.get_content(html)
            if not content.strip():
                return

            # 生成保存的文件路径及文件名
            temp = url.split("_")[2].split(".")[0].split("-")
            pageNo = temp[1]
            titleNo = temp[0] if int(temp[0]) >= 10 else "0" + temp[0]
            path = f"{destdir}/{year}{month}{day}/"
            fileName = f"{year}{month}{day}-{pageNo}-{titleNo}.txt"

            # 保存文件
            await self.save_file(content, path, fileName)
            print(f"保存文章: {fileName}")

        except Exception as e:
            print(f"处理文章失败 {url}: {e}")

    async def download_rmrb_day(
        self,
        session: aiohttp.ClientSession,
        year: str,
        month: str,
        day: str,
        destdir: str,
    ):
        """
        功能：异步爬取《人民日报》网站 某年 某月 某日 的新闻内容，并保存在 指定目录下
        参数：aiohttp会话，年，月，日，文件保存的根目录
        """
        try:
            print(f"开始爬取 {year}-{month}-{day}")

            # 获取版面列表
            page_list = await self.get_page_list(session, year, month, day)

            if not page_list:
                print(f"未找到 {year}-{month}-{day} 的版面")
                return

            # 并发获取所有版面的文章链接
            title_tasks = []
            for page in page_list:
                task = self.get_title_list(session, year, month, day, page)
                title_tasks.append(task)

            title_lists = await asyncio.gather(*title_tasks, return_exceptions=True)

            # 收集所有文章链接
            all_urls = []
            for title_list in title_lists:
                if isinstance(title_list, list):
                    all_urls.extend(title_list)

            if not all_urls:
                print(f"未找到 {year}-{month}-{day} 的文章")
                return

            # 并发处理所有文章
            article_tasks = []
            for url in all_urls:
                task = self.process_article(session, url, year, month, day, destdir)
                article_tasks.append(task)

            await asyncio.gather(*article_tasks, return_exceptions=True)
            print(f"完成爬取 {year}-{month}-{day}")

        except Exception as e:
            print(f"爬取日期 {year}-{month}-{day} 出现错误：{e}")

    async def download_rmrb_range(self, begin_date: str, end_date: str, destdir: str):
        """
        功能：异步爬取日期范围内的所有新闻
        """
        date_list = get_date_list(begin_date, end_date)

        # 创建aiohttp会话
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            # 为避免对服务器造成过大压力，我们按天串行处理，但每天内部并发处理
            for d in date_list:
                year = str(d.year)
                month = str(d.month) if d.month >= 10 else "0" + str(d.month)
                day = str(d.day) if d.day >= 10 else "0" + str(d.day)

                await self.download_rmrb_day(session, year, month, day, destdir)

                # 适当延迟，避免被封IP
                await asyncio.sleep(1)


def gen_dates(b_date, days):
    day = datetime.timedelta(days=1)
    for i in range(days):
        yield b_date + day * i


def get_date_list(begin_date: str, end_date: str):
    """
    获取日期列表
    :param begin_date: 开始日期
    :param end_date: 结束日期
    :return: 开始日期和结束日期之间的日期列表
    """
    start = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end = datetime.datetime.strptime(end_date, "%Y%m%d")

    data = []
    for d in gen_dates(start, (end - start).days):
        data.append(d)

    return data


async def main():
    # 输入起止日期，爬取之间的新闻
    print("欢迎使用异步人民日报爬虫，请输入以下信息：")
    begin_date = input("请输入开始日期(YYYYMMDD):") or '20220101'
    end_date = input("请输入结束日期(YYYYMMDD):") or '20241231'
    destdir = input("请输入数据保存的地址：") or "rmrb_data/"

    spider = RMRBSpider(max_concurrent_requests=10)
    await spider.download_rmrb_range(begin_date, end_date, destdir)

    print("数据爬取完成!")


if __name__ == "__main__":
    # 安装依赖提示
    try:
        import aiohttp
        import aiofiles
    except ImportError:
        print("请先安装依赖包:")
        print("pip install aiohttp aiofiles")
        exit(1)

    asyncio.run(main())
