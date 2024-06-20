from typing import Literal, Dict, Union

from ..base import BaseTool
from .searcher import GoogleSearch, SearXNGSearch


class WebSearch(BaseTool):

    name = "web_search"

    searcher_map:Dict[str, Union[GoogleSearch, SearXNGSearch]] = {
        'google' : GoogleSearch(),
        'searx': SearXNGSearch(),
    }

    default_searcher = "searx"

    def __init__(self):
        pass

    def call(self, 
             keywords:str,
             engine_type:Literal['google','searx'] = default_searcher,
             sort_type:Literal["relevance", "time"] = "relevance",
             **kwargs):
        """
        使用搜索引擎联网搜索，当大模型不知道如何回答问题时，会尝试使用搜索引擎联网搜索

        Args:
            keywords (str): 关键词，作为搜索引擎的输入
            engine_type (Literal['google','searx']): 使用哪个搜索引擎，google指谷歌，searx指SearXNG搜索引擎，默认使用"searx"搜索
            sort_type (Literal["relevance", "time"]): 搜索结果按照什么顺序排序，"relevance"指按照相关度排序，"time"指按照时间排序，默认是"relevance"
        """
        searcher = self.searcher_map.get(engine_type, self.searcher_map[self.default_searcher])

        # 1. 搜索
        search_results = searcher.search(keywords, 
                                        file_type=kwargs.get('file_type'), 
                                        time_period=kwargs.get('time_period'), 
                                        time_range=kwargs.get('time_range'), 
                                        site=kwargs.get('site'), 
                                        max_res_num=kwargs.get('top_k', 8))
        if not search_results: return f"没有搜索到与{keywords}相关的结果，请换一个关键词再试"
        
        # 2. 排序
        search_results = sorted(search_results, key=lambda x: x.date, reverse=True) if sort_type == "time" else search_results

        return [item.model_dump() for item in search_results if item.introduction]