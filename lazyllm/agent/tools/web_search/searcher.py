from bs4 import BeautifulSoup
import requests
from typing import List,  Literal
from datetime import datetime
from time import sleep
import traceback
import json
import random

from lazyllm import LOG as logger

from .configs import SEARCX_HOST, SEARCX_PORT, SEARCX_PREFERENCES, GOOGLE_KEY, GOOGLE_CX
from .utils import format_date, HEADERS_LIST, SearchResult

class SearchKw:
    def __init__(self,
                 file_type:Literal["pdf", "doc", "xls", "ppt", "rtf"] = None,
                 time_period:Literal["day", "week", "month", "year"] = None,
                 time_range:str = None, # 类似于1970-01-01的两个时间点，以","分隔
                 site:str = None) -> None:
        self.file_type = file_type
        self.time_period = time_period
        self.time_range = self._process_time_range(time_range)
        self.site = site
    
    def _process_time_range(self, time_range:str) -> List[datetime]:
        
        time_range = self._get_time_range(time_range)
        start = datetime.strptime('1970-01-01', "%Y-%m-%d")
        end = datetime.now()
        if not time_range or not isinstance(time_range, list):
            return None
        elif len(time_range) == 1:
            start = self._get_datetime(time_range[0]) or start
        else:
            start = self._get_datetime(time_range[0]) or start
            end = self._get_datetime(time_range[-1]) or end
        
        return [start, end]

    @staticmethod
    def _get_time_range(time_range:str) -> List[str]:
        if not time_range:
            return None
        try:
            time_range = [x.strip() for x in time_range.split(',')]
        except:
            logger.warning(f"====== 日期范围格式错误 ======\ntime_range:{time_range}")
        return time_range

    @staticmethod
    def _get_datetime(date_str:str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            logger.warning(f"====== 日期格式错误 ======\ndate_str:{date_str}")
            return None
        

class BaseSearcher:
    skip_keywords = ['视频','百度贴吧','翻译']

    max_web_pages = 2
    timeout = 10

    def search(self, 
               query:str, 
               file_type:Literal["pdf", "doc", "xls", "ppt", "rtf"] = None,
               time_period:Literal["day", "week", "month", "year"] = None,
               time_range:List[str] = None, 
               site:str = None,
               max_res_num:int=5,
               **kwargs) -> List[SearchResult]:
        search_kw = SearchKw(file_type=file_type, time_period=time_period, time_range=time_range, site=site)
        results = self._search(query, search_kw, max_res_num)
        return results[:max_res_num]

    # 随机获取一个请求头
    @property
    def headers(self):
        choiced = random.choice(HEADERS_LIST)
        return {'User-Agent': choiced}

    def _http_get(self, url: str) -> requests.Response:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            logger.debug(f'========= http request =========\nurl:{url} \nresponse status code: {response.status_code}')
            return response
        except Exception as e:
            logger.error(f'========= http request error =========\nurl:{url} \nerror:{e}')
            return None

    def get_soup(self, url) -> BeautifulSoup:
        response = self._http_get(url)
        if not response:
            return None
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup
    
    def _is_skip(self, query, title):
        for skip_keyword in self.skip_keywords:
            if skip_keyword not in query and skip_keyword in title:
                return True
        return False

    def _search_one_page(self, 
               query:str, 
               search_kw:SearchKw,
               pn:int=0) -> List[SearchResult]:
        raise NotImplementedError
    
    def _search(self, 
                query:str, 
                search_kw:SearchKw,
                max_res_num:int=3) -> List[SearchResult]:
        search_results = self._search_one_page(query, search_kw, 0)
        pn = 1
        while len(search_results) < max_res_num and pn < self.max_web_pages:
            sleep(2)
            new_search_results = self._search_one_page(query, search_kw, pn)
            search_results.extend(new_search_results)
            pn += 1
        return search_results
    
    def _get_search_kw_str(self, search_kw:SearchKw) -> str:
        kw_str = ''
        if search_kw.file_type:
            kw_str += f' filetype:{search_kw.file_type}'
        if search_kw.site:
            kw_str += f' site:{search_kw.site}'
        return kw_str


class GoogleSearch(BaseSearcher):
    timeout = 5
    # google_key 和 google_cx 的获取方式参考：https://zhuanlan.zhihu.com/p/174666017
    def __init__(self, each_page_num=10) -> None:
        super().__init__()
        self.google_key = GOOGLE_KEY
        self.google_cx = GOOGLE_CX
        self.each_page_num = each_page_num
        self.base_url = 'https://www.googleapis.com/customsearch/v1?key={google_key}&q={query}&cx={google_cx}&start={pn}&num={each_page_num}'

    def _search_one_page(self, query:str, search_kw:SearchKw, pn:int=0) -> List[SearchResult]:
        if not (self.google_key and self.google_cx):
            raise ValueError("If you want to use GoogleSearch, you need to set environment variables: LAZYLLM_GOOGLE_KEY and LAZYLLM_GOOGLE_CX")
        url = self.base_url.format(google_key=self.google_key, 
                                   query=query, 
                                   google_cx=self.google_cx, 
                                   pn=pn*self.each_page_num, 
                                   each_page_num=self.each_page_num
                                ) + self._get_search_kw_str(search_kw)
        
        response = self._http_get(url)
        if not response: return [], []

        results = json.loads(response.text).get('items', [])

        search_results = []
        for result in results:
            try:
                title,href = result['title'],result['link']
                if self._is_skip(query, title): continue
                introduction:str = result['snippet']
                date = format_date(introduction[:20].split(' ... ', maxsplit=1)[0])
            except:
                logger.warning(f"====== search_result 解析错误 ======\nresult:{result}\n{traceback.format_exc()}")
            else:
                search_results.append(SearchResult(**{
                    'title': title.replace('\n','').strip(),
                    'href': href,
                    'date': date,
                    'introduction': introduction.replace('\n','').strip(),
                    'citation': pn*len(results) + len(search_results) + 1
                }))
        return search_results

    def _get_search_kw_str(self, search_kw:SearchKw) -> str:
        kw_str = ''
        if search_kw.file_type:
            kw_str += f'&fileType={search_kw.file_type}'
        if search_kw.site:
            kw_str += f'&siteSearch={search_kw.site}&siteSearchFilter=i'
        if search_kw.time_range:
            start_date = search_kw.time_range[0].strftime("%Y%m%d")
            end_date = search_kw.time_range[1].strftime("%Y%m%d")
            kw_str += f'&sort=date:r:{start_date}:{end_date}'
        return kw_str

class SearXNGSearch(BaseSearcher):
    timeout = 20
    def __init__(self) -> None:
        super().__init__()
        self.base_url = "http://"+SEARCX_HOST+":"+str(SEARCX_PORT)+"/?preferences="+SEARCX_PREFERENCES+"&category_general=1&language=auto&safesearch=2&format=json&pageno={pn}&q={query}"
    
    def _search_one_page(self, query:str, search_kw:SearchKw, pn:int=0) -> List[SearchResult]:
        """result样例：
        {
            "query": "商汤科技",
            "results": [
                {
                    "url": "https://www.sensetime.com/cn",
                    "title": "SenseTime | 商汤科技-坚持原创，让AI引领人类进步",
                    "content": "商汤科技宣布将与《三体》全球永久唯一的版权方三体宇宙携手合作，共同探索“AI+科幻”新范式，打造线下沉浸式娱乐新业态。 了解更多. 《“平衡发展”的人工智能治理白皮书》发布. 商汤此次进一步提出了发展“负责任且可评估”的人工智能，并将其作为商汤开展人工智能治理的愿景目标，打造伦理治理闭环。 了解更多. 客户案例. 查看所有案例. 智 …",
                    "engine": "bing",
                    "parsed_url": [
                        "https",
                        "www.sensetime.com",
                        "/cn",
                        "",
                        "",
                        ""
                    ],
                    "template": "default.html",
                    "engines": [
                        "bing",
                        "presearch"
                    ],
                    "positions": [
                        1,
                        1
                    ],
                    "score": 4.0,
                    "category": "general"
                }
                ...
            ]
        }
        """
        url = self.base_url.format(query=query, pn=pn+1) + self._get_search_kw_str(search_kw)
        response = self._http_get(url)
        if not response: return [], []

        results = response.json().get('results', [])
        search_results = []
        for result in results:
            try:
                title,href = result['title'],result['url']
                if self._is_skip(query, title): continue
                introduction = result['content']
            except:
                logger.warning(f"====== search_result 解析错误 ======\nresult:{result}\n{traceback.format_exc()}")
            else:
                search_results.append(SearchResult(**{
                    'title': title.replace('\n','').strip(),
                    'href': href,
                    'date': None,
                    'introduction': introduction,
                    'citation': pn*len(results) + len(search_results) + 1
                }))
        return search_results
    
    def _get_search_kw_str(self, search_kw:SearchKw) -> str:
        kw_str = super()._get_search_kw_str(search_kw)
        if search_kw.time_period and search_kw.time_period in ["day", "week", "month", "year"]:
            kw_str += f'&time_range={search_kw.time_period}'
        return kw_str