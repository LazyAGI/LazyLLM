# flake8: noqa E501
import importlib
import functools
from .. import utils

_tools_module = importlib.import_module('lazyllm.tools.tools')
add_chinese_doc = functools.partial(utils.add_chinese_doc, module=_tools_module)
add_english_doc = functools.partial(utils.add_english_doc, module=_tools_module)
add_example = functools.partial(utils.add_example, module=_tools_module)

add_chinese_doc('SearchBase', '''
所有搜索工具的基类，定义统一接口与返回格式。

子类需实现 search，将各后端 API 的原始响应规范为包含 title、url、snippet、source（及可选 extra）的字典列表。
''')

add_english_doc('SearchBase', '''
Base class for all search tools with unified interface and result format.

Subclasses must implement search and normalize raw API responses to a list of dicts with keys: title, url, snippet, source, and optionally extra.
''')

add_example('SearchBase', '''
from lazyllm.tools.tools import SearchBase, ArxivSearch
engine = ArxivSearch()
results = engine.forward('transformer')
''')

add_chinese_doc('SearchBase.search', '''
执行搜索并返回统一格式结果。子类必须重写此方法，参数由各子类自行定义。

Returns:
    List[Dict[str, Any]]: 每条结果至少包含 title (str), url (str), snippet (str), source (str)；
    可选 extra (dict) 存放各引擎扩展字段（如 authors、pageid、score）。snippet 仅为摘要预览，
    不包含完整正文；需正文可调用 get_content(item) 或 get_contents(items)。
''')

add_english_doc('SearchBase.search', '''
Run search and return unified results. Subclasses must override this method; parameters are defined by each subclass.

Returns:
    List[Dict[str, Any]]: Each item has at least title (str), url (str), snippet (str), source (str);
    optional extra (dict) for engine-specific fields (e.g. authors, pageid, score). snippet is only a
    preview; for full body use get_content(item) or get_contents(items).
''')

add_chinese_doc('SearchBase.forward', '''
调用 search(query, **kwargs) 并返回结果。参数会透传给当前子类的 search 方法。
''')

add_english_doc('SearchBase.forward', '''
Calls search(query, **kwargs) and returns the result. Arguments are passed through to the subclass search method.
''')

add_chinese_doc('SearchBase.get_content', '''
根据单条搜索结果（search/forward 返回的 item）获取正文文本。

默认行为：请求 item 的 url，将响应 HTML 转为纯文本返回。子类可重写以使用 API 获取正文（如 Wikipedia 词条全文、arXiv 摘要、Stack Overflow 问答正文等）。

Args:
    item (Dict[str, Any]): 至少包含 url 的搜索结果项（_make_result 格式）。

Returns:
    str: 正文文本；失败或无 url 时返回空字符串。
''')

add_english_doc('SearchBase.get_content', '''
Fetch full body text for a single search result item (as returned by search/forward).

Default: GET the item url and convert response HTML to plain text. Subclasses may override to use APIs (e.g. Wikipedia full page, arXiv abstract, Stack Overflow Q&A body).

Args:
    item (Dict[str, Any]): Search result item with at least url (_make_result format).

Returns:
    str: Body text; empty string on failure or when url is missing.
''')

add_chinese_doc('SearchBase.get_contents', '''
根据多条搜索结果批量获取正文文本。

Args:
    items (List[Dict[str, Any]]): 搜索结果列表（_make_result 格式）。

Returns:
    List[str]: 与 items 一一对应的正文文本列表。
''')

add_english_doc('SearchBase.get_contents', '''
Fetch full body text for multiple search result items.

Args:
    items (List[Dict[str, Any]]): List of search result items (_make_result format).

Returns:
    List[str]: List of body texts, one per item.
''')

add_example('SearchBase.get_content', '''
from lazyllm.tools.tools import ArxivSearch
engine = ArxivSearch()
results = engine.forward('transformer')
if results:
    text = engine.get_content(results[0])
''')

add_example('SearchBase.get_contents', '''
from lazyllm.tools.tools import WikipediaSearch
wiki = WikipediaSearch()
items = wiki.search('machine learning')
texts = wiki.get_contents(items[:3])
''')

add_chinese_doc('GoogleSearch', '''
通过 Google Custom Search 搜索关键词。需配置 API key 与搜索引擎 ID。

如何申请 API Key 与搜索引擎 ID：
1. 打开 Google Cloud 控制台 https://console.cloud.google.com/ ，创建或选择项目。
2. 启用 "Custom Search API"，在 "凭据" 中创建 API 密钥，即得到 custom_search_api_key。
3. 打开 Programmable Search Engine https://programmablesearchengine.google.com/ 创建搜索引擎，在 "控制面板" 中可查看 "搜索引擎 ID"（cx），即 search_engine_id。

Args:
    custom_search_api_key (str): 用户申请的 Google API key。
    search_engine_id (str): 用户创建的检索用搜索引擎 id。
    timeout (int): 请求超时秒数，默认 10。
    proxies (Dict[str, str], optional): 代理配置，格式见 https://www.python-httpx.org/advanced/proxies。
    source_name (str): 结果中的来源标识，默认 "google"。
''')

add_english_doc('GoogleSearch', '''
Search via Google Custom Search. Requires API key and search engine ID.

How to get API key and search engine ID:
1. Go to Google Cloud Console https://console.cloud.google.com/ , create or select a project.
2. Enable "Custom Search API", create an API key under "Credentials" to get custom_search_api_key.
3. Go to Programmable Search Engine https://programmablesearchengine.google.com/ to create a search engine; the "Search engine ID" (cx) is in the control panel.

Args:
    custom_search_api_key (str): Google API key.
    search_engine_id (str): Search engine ID for retrieval.
    timeout (int): Request timeout in seconds, default 10.
    proxies (Dict[str, str], optional): Proxy config, see https://www.python-httpx.org/advanced/proxies.
    source_name (str): Source identifier in results, default "google".
''')

add_example('GoogleSearch', '''
from lazyllm.tools.tools import GoogleSearch
key = '<your_google_search_api_key>'
cx = '<your_search_engine_id>'
google = GoogleSearch(custom_search_api_key=key, search_engine_id=cx)
''')

add_chinese_doc('GoogleSearch.search', '''
执行 Google 搜索。

Args:
    query (str): 检索关键词。
    date_restrict (str): 内容时效，默认 "m1"（一个月内）。格式见 Google Custom Search API 文档。
    search_engine_id (str, optional): 检索用搜索引擎 id；为空则使用构造时传入的值。

Returns:
    List[Dict[str, Any]]: 统一格式的搜索结果列表。
''')

add_english_doc('GoogleSearch.search', '''
Execute Google search.

Args:
    query (str): Search keywords.
    date_restrict (str): Content freshness, default "m1" (past month). See Google Custom Search API docs.
    search_engine_id (str, optional): Search engine ID; if empty, uses constructor value.

Returns:
    List[Dict[str, Any]]: List of results in unified format.
''')

add_example('GoogleSearch.search', '''
from lazyllm.tools.tools import GoogleSearch
google = GoogleSearch('<api_key>', '<search_engine_id>')
res = google.search('machine learning', date_restrict='m1')
''')

add_chinese_doc('GoogleSearch.get_content', '''
使用 item 的 url 请求页面并将 HTML 转为纯文本返回（基类默认行为）。
''')

add_english_doc('GoogleSearch.get_content', '''
Fetches the item url and returns HTML as plain text (base default).
''')

add_chinese_doc('TencentSearch', '''
腾讯搜索接口封装，调用腾讯云内容搜索服务（SearchPro）。

如何申请 SecretId 与 SecretKey：
1. 登录腾讯云控制台 https://console.cloud.tencent.com/ 。
2. 进入 "访问管理" -> "API 密钥管理" https://console.cloud.tencent.com/cam/capi ，创建或查看密钥，得到 SecretId 与 SecretKey。
3. 需开通腾讯云 "文本内容安全" 或对应搜索类产品并确保账号有 SearchPro 调用权限。

Args:
    secret_id (str): 腾讯云 API 密钥 ID。
    secret_key (str): 腾讯云 API 密钥。
    source_name (str): 结果中的来源标识，默认 "tencent"。
''')

add_english_doc('TencentSearch', '''
Tencent search wrapper for Tencent Cloud content search (SearchPro).

How to get SecretId and SecretKey:
1. Log in to Tencent Cloud Console https://console.cloud.tencent.com/ .
2. Go to "Access Management" -> "API Key Management" https://console.cloud.tencent.com/cam/capi to create or view keys (SecretId, SecretKey).
3. Enable the relevant search/product (e.g. text content safety) and ensure the account has SearchPro API permission.
''')

add_example('TencentSearch', '''
from lazyllm.tools.tools import TencentSearch
searcher = TencentSearch(secret_id='<your_secret_id>', secret_key='<your_secret_key>')
''')

add_chinese_doc('TencentSearch.search', '''
执行腾讯云搜索。

Args:
    query (str): 用户查询内容。

Returns:
    List[Dict[str, Any]]: 统一格式的搜索结果列表；出错时返回空列表。
''')

add_english_doc('TencentSearch.search', '''
Execute Tencent Cloud search.

Args:
    query (str): User query string.

Returns:
    List[Dict[str, Any]]: List of results in unified format; empty list on error.
''')

add_example('TencentSearch.search', '''
from lazyllm.tools.tools import TencentSearch
searcher = TencentSearch(secret_id='<id>', secret_key='<key>')
res = searcher.search('calculus')
''')

add_chinese_doc('TencentSearch.get_content', '''
使用 item 的 url 请求页面并将 HTML 转为纯文本返回（基类默认行为）。
''')

add_english_doc('TencentSearch.get_content', '''
Fetches the item url and returns HTML as plain text (base default).
''')

add_chinese_doc('BingSearch', '''
Azure Bing Web Search API v7 封装。需要订阅密钥。

如何申请订阅密钥（Subscription Key）：
1. 登录 Azure 门户 https://portal.azure.com/ 。
2. 创建 "Bing Search v7" 资源（或 "Cognitive Services" 多服务资源），在 "密钥和终结点" 中可查看 key1/key2，即 subscription_key。
3. 官方文档与定价：https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview 。

Args:
    subscription_key (str): Azure 订阅密钥。
    endpoint (str, optional): API 端点，默认 https://api.bing.microsoft.com/v7.0/search。
    timeout (int): 请求超时秒数，默认 10。
    source_name (str): 结果来源标识，默认 "bing"。
''')

add_english_doc('BingSearch', '''
Azure Bing Web Search API v7. Requires subscription key.

How to get subscription key:
1. Log in to Azure Portal https://portal.azure.com/ .
2. Create a "Bing Search v7" resource (or "Cognitive Services" multi-service resource); under "Keys and endpoint" you will see key1/key2, use either as subscription_key.
3. Docs and pricing: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview .
''')

add_example('BingSearch', '''
from lazyllm.tools.tools import BingSearch
bing = BingSearch(subscription_key='<your_key>')
res = bing('python tutorial', count=5)
''')

add_chinese_doc('BingSearch.get_content', '''
使用 item 的 url 请求页面并将 HTML 转为纯文本返回（基类默认行为）。
''')

add_english_doc('BingSearch.get_content', '''
Fetches the item url and returns HTML as plain text (base default).
''')

add_chinese_doc('BingSearch.search', '''
执行 Bing 网页搜索。

Args:
    query (str): 搜索关键词。
    count (int): 返回条数，默认 10，最大 50。

Returns:
    List[Dict[str, Any]]: 统一格式的搜索结果列表。
''')

add_english_doc('BingSearch.search', '''
Execute Bing web search.

Args:
    query (str): Search query.
    count (int): Number of results, default 10, max 50.

Returns:
    List[Dict[str, Any]]: List of results in unified format.
''')

add_chinese_doc('BochaSearch', '''
博查 AI 网页搜索 API 封装。

如何申请 API Key：
1. 打开博查 AI 开放平台 https://open.bochaai.com/ 。
2. 使用微信扫码或账号注册登录，进入 "API 密钥管理" / "API Keys" 页面。
3. 创建密钥即可得到 api_key，用于请求头 Authorization: Bearer <api_key>。

Args:
    api_key (str): 博查 API key。
    base_url (str): API 根地址，默认 https://api.bochaai.com。
    timeout (int): 请求超时秒数，默认 15。
    source_name (str): 结果来源标识，默认 "bocha"。
''')

add_english_doc('BochaSearch', '''
Bocha AI Web Search API wrapper.

How to get API key:
1. Go to Bocha AI Open Platform https://open.bochaai.com/ .
2. Sign in (e.g. via WeChat or account), go to "API Keys" / "API 密钥管理".
3. Create a key to get api_key; use it as Authorization: Bearer <api_key> in requests.
''')

add_example('BochaSearch', '''
from lazyllm.tools.tools import BochaSearch
bocha = BochaSearch(api_key='<your_key>')
res = bocha('machine learning', count=5)
''')

add_chinese_doc('BochaSearch.get_content', '''
使用 item 的 url 请求页面并将 HTML 转为纯文本返回（基类默认行为）。
''')

add_english_doc('BochaSearch.get_content', '''
Fetches the item url and returns HTML as plain text (base default).
''')

add_chinese_doc('BochaSearch.search', '''
执行博查网页搜索。

Args:
    query (str): 搜索内容。
    count (int): 结果数量，默认 10。
    freshness (str, optional): 时间范围，如 oneDay、oneWeek、oneMonth。
    summary (bool): 是否返回摘要，默认 False。

Returns:
    List[Dict[str, Any]]: 统一格式的搜索结果列表。
''')

add_english_doc('BochaSearch.search', '''
Execute Bocha web search.

Args:
    query (str): Search query.
    count (int): Number of results, default 10.
    freshness (str, optional): Time range, e.g. oneDay, oneWeek, oneMonth.
    summary (bool): Whether to request summary, default False.

Returns:
    List[Dict[str, Any]]: List of results in unified format.
''')

add_chinese_doc('StackOverflowSearch', '''
通过 Stack Exchange API 2.3 搜索 Stack Overflow 等问题。可不带 key 使用，带 key 配额更高。

如何申请 API Key（可选）：
1. 打开 Stack Exchange API 应用注册页 https://stackexchange.com/oauth/dialog?client_id=YOUR_APP_ID&scope=no_expiry&redirect_uri=https://stackexchange.com/oauth/login_success （需先在 https://stackexchange.com/sites 创建应用获取 client_id）。
2. 或直接使用无 key 调用，每日配额较低；在 https://api.stackexchange.com/docs 查看 "Request a Key" 说明申请 key 以提高配额。

Args:
    site (str): 站点，默认 "stackoverflow"。
    key (str, optional): Stack Exchange API key。
    timeout (int): 请求超时秒数，默认 10。
    source_name (str): 结果来源标识，默认 "stackoverflow"。
''')

add_english_doc('StackOverflowSearch', '''
Search Stack Overflow (or other Stack Exchange sites) via API 2.3. Key optional for higher quota.

How to get API key (optional):
1. Use the site without a key for lower quota, or request a key at https://api.stackexchange.com/docs (see "Request a Key") for higher quota.
2. To register an app and get client_id: https://stackexchange.com/sites (create application); then use OAuth flow to get key if needed.
''')

add_example('StackOverflowSearch', '''
from lazyllm.tools.tools import StackOverflowSearch
so = StackOverflowSearch()
res = so('python asyncio', count=5)
''')

add_chinese_doc('StackOverflowSearch.search', '''
搜索 Stack Exchange 问题。

Args:
    query (str): 搜索关键词。
    count (int): 返回条数，默认 10。
    sort (str): 排序方式，默认 "relevance"。

Returns:
    List[Dict[str, Any]]: 统一格式结果，extra 中可含 score、answer_count、is_answered 等。
''')

add_chinese_doc('StackOverflowSearch.get_content', '''
通过 Stack Exchange API 获取问题正文及采纳答案正文（需 item 的 url 含 question id）；返回纯文本，失败时回退为请求 url 页面。
''')

add_english_doc('StackOverflowSearch.get_content', '''
Fetches question body and accepted answer body via Stack Exchange API (requires question id in item url); returns plain text, falls back to fetching url on failure.
''')

add_english_doc('StackOverflowSearch.search', '''
Search Stack Exchange questions.

Args:
    query (str): Search query.
    count (int): Number of results, default 10.
    sort (str): Sort order, default "relevance".

Returns:
    List[Dict[str, Any]]: Results in unified format; extra may include score, answer_count, is_answered.
''')

add_chinese_doc('SemanticScholarSearch', '''
Semantic Scholar 学术论文搜索。API key 可选，用于提高限流。

如何申请 API Key（可选）：
1. 打开 Semantic Scholar 官网 https://www.semanticscholar.org/ ，点击 "For Developers" 或直接访问 https://api.semanticscholar.org/ 。
2. 在 "Get an API Key" 页面 https://www.semanticscholar.org/product/api 填写申请，获取 API key 后可在请求头中设置 x-api-key 以提高限流额度。不申请也可使用，仅限流更严格。

Args:
    api_key (str, optional): Semantic Scholar API key。
    timeout (int): 请求超时秒数，默认 15。
    source_name (str): 结果来源标识，默认 "semantic_scholar"。
''')

add_english_doc('SemanticScholarSearch', '''
Search academic papers via Semantic Scholar API. API key optional for higher rate limit.

How to get API key (optional):
1. Go to Semantic Scholar https://www.semanticscholar.org/ -> "For Developers" or https://api.semanticscholar.org/ .
2. Apply for an API key at https://www.semanticscholar.org/product/api ("Get an API Key"); use it as x-api-key header for higher rate limits. You can also use the API without a key with stricter limits.
''')

add_example('SemanticScholarSearch', '''
from lazyllm.tools.tools import SemanticScholarSearch
sch = SemanticScholarSearch()
res = sch('transformer attention', limit=5)
''')

add_chinese_doc('SemanticScholarSearch.search', '''
搜索学术论文。

Args:
    query (str): 搜索关键词。
    limit (int): 返回条数，默认 10。
    fields (str, optional): API 返回字段，默认包含 title, url, abstract, authors, year, citationCount。

Returns:
    List[Dict[str, Any]]: 统一格式结果，extra 可含 authors、year、citationCount。
''')

add_english_doc('SemanticScholarSearch.search', '''
Search academic papers.

Args:
    query (str): Search query.
    limit (int): Number of results, default 10.
    fields (str, optional): API response fields.

Returns:
    List[Dict[str, Any]]: Results in unified format; extra may include authors, year, citationCount.
''')

add_chinese_doc('SemanticScholarSearch.get_content', '''
优先用 extra.paperId 调 API 获取论文摘要；无 paperId 时返回 item.snippet，再失败则请求 url 页面。
''')

add_english_doc('SemanticScholarSearch.get_content', '''
Uses extra.paperId to fetch abstract via API when available; otherwise returns item.snippet, then falls back to fetching url.
''')

add_chinese_doc('GoogleBooksSearch', '''
Google Books API 书籍检索。API key 可选，带 key 配额更高。

如何申请 API Key（可选）：
1. 打开 Google Cloud 控制台 https://console.cloud.google.com/ ，创建或选择项目。
2. 启用 "Books API"（Books API），在 "凭据" 中创建 "API 密钥"，即得到 api_key。不申请也可调用，但配额较低。

Args:
    api_key (str, optional): Google Books API key。
    timeout (int): 请求超时秒数，默认 10。
    source_name (str): 结果来源标识，默认 "google_books"。
''')

add_english_doc('GoogleBooksSearch', '''
Search books via Google Books API. API key optional for higher quota.

How to get API key (optional):
1. Go to Google Cloud Console https://console.cloud.google.com/ , create or select a project.
2. Enable "Books API", create an "API key" under "Credentials" to get api_key. You can also call without a key with lower quota.
''')

add_example('GoogleBooksSearch', '''
from lazyllm.tools.tools import GoogleBooksSearch
books = GoogleBooksSearch()
res = books('deep learning', max_results=5)
''')

add_chinese_doc('GoogleBooksSearch.get_content', '''
使用 item 的 url 请求页面并将 HTML 转为纯文本返回（基类默认行为）。
''')

add_english_doc('GoogleBooksSearch.get_content', '''
Fetches the item url and returns HTML as plain text (base default).
''')

add_chinese_doc('GoogleBooksSearch.search', '''
检索书籍。

Args:
    query (str): 搜索关键词。
    max_results (int): 返回条数，默认 10。

Returns:
    List[Dict[str, Any]]: 统一格式结果，extra 可含 authors、publishedDate、pageCount。
''')

add_english_doc('GoogleBooksSearch.search', '''
Search books.

Args:
    query (str): Search query.
    max_results (int): Number of results, default 10.

Returns:
    List[Dict[str, Any]]: Results in unified format; extra may include authors, publishedDate, pageCount.
''')

add_chinese_doc('ArxivSearch', '''
arXiv 预印本搜索。无需 API key，直接使用。

无需在官网申请 token；直接实例化即可调用。官网与 API 说明：https://info.arxiv.org/help/api/index.html 。

Args:
    timeout (int): 请求超时秒数，默认 15。
    source_name (str): 结果来源标识，默认 "arxiv"。
''')

add_english_doc('ArxivSearch', '''
Search arXiv preprints. No API key or token required.

No sign-up or token needed; instantiate and call. API docs: https://info.arxiv.org/help/api/index.html .
''')

add_example('ArxivSearch', '''
from lazyllm.tools.tools import ArxivSearch
arxiv = ArxivSearch()
res = arxiv('attention is all you need', max_results=5)
''')

add_chinese_doc('ArxivSearch.search', '''
搜索 arXiv 预印本。

Args:
    query (str): 搜索关键词。
    max_results (int): 返回条数，默认 10。
    sort_by (str): 排序字段，默认 "relevance"。

Returns:
    List[Dict[str, Any]]: 统一格式结果，extra 可含 authors、published。
''')

add_english_doc('ArxivSearch.search', '''
Search arXiv preprints.

Args:
    query (str): Search query.
    max_results (int): Number of results, default 10.
    sort_by (str): Sort field, default "relevance".

Returns:
    List[Dict[str, Any]]: Results in unified format; extra may include authors, published.
''')

add_chinese_doc('ArxivSearch.get_content', '''
从 item.url 解析 arXiv id，调用 export API 获取完整摘要文本；失败时回退为请求 url 页面。
''')

add_english_doc('ArxivSearch.get_content', '''
Parses arXiv id from item.url, fetches full abstract via export API; falls back to fetching url on failure.
''')

add_chinese_doc('WikipediaSearch', '''
Wikipedia 全文搜索，基于 MediaWiki API。无需 API key，直接使用。

无需在官网申请 token；直接实例化即可。使用各语言站点即可搜索对应语言，如英文 https://en.wikipedia.org 、中文 https://zh.wikipedia.org 。API 说明：https://www.mediawiki.org/wiki/API:Search 。

Args:
    base_url (str): 站点根地址，默认 "https://en.wikipedia.org"，可改为 zh.wikipedia.org 等。
    timeout (int): 请求超时秒数，默认 10。
    source_name (str): 结果来源标识，默认 "wikipedia"。
''')

add_english_doc('WikipediaSearch', '''
Wikipedia full-text search via MediaWiki API. No API key or token required.

No sign-up or token needed; instantiate and call. Use different base_url for language variants (e.g. https://en.wikipedia.org , https://zh.wikipedia.org ). API docs: https://www.mediawiki.org/wiki/API:Search .

Args:
    base_url (str): Site base URL, default "https://en.wikipedia.org"; use e.g. https://zh.wikipedia.org for other languages.
    timeout (int): Request timeout in seconds, default 10.
    source_name (str): Source identifier, default "wikipedia".
''')

add_example('WikipediaSearch', '''
from lazyllm.tools.tools import WikipediaSearch
wiki = WikipediaSearch()
res = wiki('transformer machine learning', limit=5)
''')

add_chinese_doc('WikipediaSearch.search', '''
执行 Wikipedia 搜索。

Args:
    query (str): 搜索关键词。
    limit (int): 返回条数，默认 10，最大 500。

Returns:
    List[Dict[str, Any]]: 统一格式结果，extra 可含 pageid。
''')

add_english_doc('WikipediaSearch.search', '''
Execute Wikipedia search.

Args:
    query (str): Search query.
    limit (int): Number of results, default 10, max 500.

Returns:
    List[Dict[str, Any]]: Results in unified format; extra may include pageid.
''')

add_example('WikipediaSearch.search', '''
from lazyllm.tools.tools import WikipediaSearch
wiki = WikipediaSearch(base_url='https://en.wikipedia.org')
res = wiki.search('machine learning', limit=5)
''')

add_chinese_doc('WikipediaSearch.get_content', '''
当 item.extra 含 pageid 时，使用 MediaWiki API 获取词条全文（纯文本）；否则回退为请求 url 页面。
''')

add_english_doc('WikipediaSearch.get_content', '''
When item.extra has pageid, fetches full page text via MediaWiki API; otherwise falls back to fetching url.
''')
