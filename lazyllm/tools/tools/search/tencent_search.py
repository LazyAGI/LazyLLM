from typing import Dict, Any, List

import lazyllm

from .base import SearchBase, _make_result


class TencentSearch(SearchBase):

    def __init__(self, secret_id: str, secret_key: str, source_name: str = 'tencent'):
        super().__init__(source_name=source_name)
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        self._cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = 'tms.tencentcloudapi.com'
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        self._headers = {'X-TC-Action': 'SearchPro'}
        self._client = CommonClient(
            'tms', '2020-12-29', self._cred, '', profile=client_profile)

    def search(self, query: str) -> List[Dict[str, Any]]:
        try:
            res_dict = self._client.call_json(
                'SearchPro', {'Query': query, 'Mode': 2}, headers=self._headers)
            pages = res_dict.get('Response', {}).get('Pages') or []
        except Exception as err:
            lazyllm.LOG.error('Request Tencent Search meets error: %s', err)
            return []
        out: List[Dict[str, Any]] = []
        for p in pages:
            title = p.get('Title') or p.get('title') or ''
            url = p.get('Url') or p.get('url') or p.get('Link') or p.get('link') or ''
            snippet = p.get('Snippet') or p.get('snippet') or p.get('Description') or p.get('description') or ''
            out.append(_make_result(title=title, url=url, snippet=snippet, source=self.source_name))
        return out
