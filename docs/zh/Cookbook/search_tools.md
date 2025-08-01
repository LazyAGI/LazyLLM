# 本项目展示了如何使用Google Custom Search JSON API实现一个智能搜索代理
# 可通过API查询获取最新的网络搜索结果并格式化输出

## !!! abstract "通过本节您将学习到以下内容"
## - 如何使用Google Custom Search JSON API进行网络搜索
## - 如何封装搜索功能为可复用的Agent类
## - 如何格式化输出搜索结果

## 项目依赖

### 确保安装以下依赖：
```bash
pip install lazyllm
```
## 代码实现
``` python
from lazyllm.tools.tools import *
from typing import Dict, Optional, List
import json
## Step 1: 初始化搜索代理

**功能说明：**
- 创建GoogleSearchAgent实例，配置API访问参数
- 初始化底层GoogleSearch工具

**参数说明**
- api_key (str): Google Custom Search JSON API密钥
- search_engine_id (str): 搜索引擎ID
- timeout (int): 请求超时时间，默认为10秒
- proxies (Optional[Dict]): 代理设置，默认为None
class GoogleSearchAgent:
def __init__(self, api_key: str, search_engine_id: str,
             timeout: int = 10, proxies: Optional[Dict] = None):
    self.search_tool = GoogleSearch(
        custom_search_api_key=api_key,
        search_engine_id=search_engine_id,
        timeout=timeout,
        proxies=proxies
    )
## Step 2: 执行搜索查询
**功能说明：**
- 发送搜索请求到Google API, 处理返回结果并格式化
def search(self, query: str,
           date_restrict: str = 'm1',
           max_results: int = 5) -> List[Dict]:
    try:
        results = self.search_tool.forward(
            query=query,
            date_restrict=date_restrict
        )

        if results and 'items' in results:
            return [
                {
                    'title': item.get('title', 'No title'),
                    'link': item.get('link', '#'),
                    'snippet': item.get('snippet', 'No description')
                }
                for item in results['items'][:max_results]
            ]
        return []
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []
## Step 3: 格式化打印搜索结果
def print_results(self, results: List[Dict]):
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} results:")
    for i, item in enumerate(results, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   {item['link']}")
        print(f"   {item['snippet']}")
```

## 示例运行结果

#### 示例场景：
```python
if __name__ == "__main__":
    API_KEY = "YOUR_API_KEY"
    SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"
    search_agent = GoogleSearchAgent(API_KEY, SEARCH_ENGINE_ID)
    query = "latest AI advancements 2023"
    results = search_agent.search(query, max_results=3)
    print(f"Search results for: '{query}'")
    search_agent.print_results(results)
```
**问题**
"Search results for: 'latest AI advancements 2023'"

**程序控制台输出：**

```
Found 3 results:

1. McKinsey technology trends outlook 2025 | McKinsey
   https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-top-trends-in-tech
   Jul 22, 2025 ... The surging demand for compute-intensive workloads, especially from gen AI, robotics, and immersive environments, is creating new demands on global ...

2. MIT Technology Review
   https://www.technologyreview.com/
   2 days ago ... Fast-learning robots: 10 Breakthrough Technologies 2025. AI advances are rapidly speeding up the process of training robots, and helping them do new tasks ...

3. OpenAI News | OpenAI
   https://openai.com/news/
   Jul 21, 2025 ... Stay up to speed on the rapid advancement of AI technology and the benefits it offers to humanity ... Latest Advancements. OpenAI o3 · OpenAI o4-mini · GPT-4o ...

```