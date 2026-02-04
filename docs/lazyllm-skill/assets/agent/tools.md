# 工具

提供一些内置的工具用于Agent直接调用

##GoogleSearch

通过 Google 搜索指定的关键词。

参数:

- custom_search_api_key (str) – 用户申请的 Google API key。
- search_engine_id (str) – 用户创建的用于检索的搜索引擎 id。
- timeout (int, default: 10 ) – 搜索请求的超时时间，单位是秒，默认是 10。
- proxies (Dict[str, str], default: None ) – 请求时所用的代理服务。格式参考 https://www.python-httpx.org/advanced/proxies。

```python
from lazyllm.tools.tools import GoogleSearch

key = '<your_google_search_api_key>'
cx = '<your_search_engine_id>'

google = GoogleSearch(custom_search_api_key=key, search_engine_id=cx)
res = google(query='商汤科技', date_restrict='m1')
```

## TencentSearch

腾讯搜索接口封装类，用于调用腾讯云的内容搜索服务。
提供对腾讯云搜索API的封装，支持关键词搜索和结果处理。

参数:

- secret_id (str) – 腾讯云API密钥ID，用于身份认证
- secret_key (str) – 腾讯云API密钥，用于身份认证

```python
from lazyllm.tools.tools import TencentSearch
secret_id = '<your_secret_id>'
secret_key = '<your_secret_key>'
searcher = TencentSearch(secret_id, secret_key)
res = searcher('calculus')
```
