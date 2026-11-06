import json
from typing import Any, Dict, List, Optional

import lazyllm
from lazyllm.tools.agent.toolsManager import fc_register
from lazyllm.tools.mcp import MCPClient
from lazyllm.tools.mcp.utils import patch_sync

from .base import SearchBase, _make_result


class ParallelSearch(SearchBase):

    _UA = f'LazyLLM/{lazyllm.__version__} (https://github.com/LazyAGI/LazyLLM; parallel-search)'

    def __init__(self, timeout: int = 30, source_name: str = 'parallel', enabled: bool = True):
        super().__init__(source_name=source_name, skip_auth=True)
        self._enabled = enabled
        self._timeout = timeout
        self._content_headers = {'User-Agent': self._UA}
        self._client = MCPClient('https://mcp.parallel.ai/mcp', headers=self._content_headers,
                                 timeout=timeout, transport='streamable-http')

    def __key_source__(self) -> bool:
        return self._enabled

    def _fetch_content_text(self, item: Dict[str, Any]) -> str:
        return super()._fetch_content_text(item) if self._enabled else ''

    @fc_register(host_file='NONE')
    def search(self, query: str, max_results: int = 10,
               search_queries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        '''Search the web anonymously using Parallel's default Fast mode.

        Args:
            query (str): One natural-language search objective.
            max_results (int): Maximum returned results, applied locally.
            search_queries (Optional[List[str]]): Related keyword queries; defaults to query.

        Returns:
            List[Dict[str, Any]]: Search results with titles, URLs, and excerpt previews.
        '''
        if not self._enabled:
            return []
        if not query.strip():
            raise ValueError('query must not be empty')
        if type(max_results) is not int or max_results <= 0:
            raise ValueError('max_results must be a positive integer')
        queries = search_queries if search_queries is not None else [query]
        if not queries or any(not isinstance(q, str) or not q.strip() for q in queries):
            raise ValueError('search_queries must contain non-empty strings')
        result = patch_sync(self._client.call_tool)(
            'web_search', {'objective': query, 'search_queries': queries})
        if getattr(result, 'isError', getattr(result, 'is_error', False)):
            raise RuntimeError('Parallel search returned an MCP tool error')
        data = getattr(result, 'structuredContent', None)
        if data is None:
            data = getattr(result, 'structured_content', None)
        if data is None:
            data = json.loads('\n'.join(item.text for item in result.content if item.type == 'text'))
        return [_make_result(title=item.get('title') or '', url=item['url'],
                             snippet='\n'.join(item.get('excerpts') or []), source=self.source_name)
                for item in data['results'][:max_results]]
