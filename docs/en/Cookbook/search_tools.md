# This project demonstrates how to build an intelligent search agent using the Google Custom Search JSON API
# It allows querying the latest web search results via the API and formatting the output

## !!! abstract "In this section, you will learn the following:"
## - How to perform web searches using the Google Custom Search JSON API
## - How to encapsulate the search functionality into a reusable Agent class
## - How to format and display search results

## Project Dependencies

### Make sure the following dependencies are installed:
```bash
pip install lazyllm
```
## Code Implementation
``` python
from lazyllm.tools.tools import *
from typing import Dict, Optional, List
import json
## Step 1: Initialize the Search Agent

**Function Description：**
- Create an instance of GoogleSearchAgent and configure the API access parameters
- Initialize the underlying GoogleSearch tool

**Parameter Description**
- api_key (str): Google Custom Search JSON API key
- search_engine_id (str): Search engine ID
- timeout (int): Request timeout in seconds, default is 10
- proxies (Optional[Dict]): Proxy settings, default is None
class GoogleSearchAgent:
def __init__(self, api_key: str, search_engine_id: str,
             timeout: int = 10, proxies: Optional[Dict] = None):
    self.search_tool = GoogleSearch(
        custom_search_api_key=api_key,
        search_engine_id=search_engine_id,
        timeout=timeout,
        proxies=proxies
    )
## Step 2: Perform a Search Query
**Function Description：**
- Send a search request to the Google API, process the returned results, and format them
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
## Step 3: Format and Print Search Results
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

## Example Output

#### Example Scenario：
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
**Question**
"Search results for: 'latest AI advancements 2023'"

**Console Output：**

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