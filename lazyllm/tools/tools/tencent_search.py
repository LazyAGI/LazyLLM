import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.common import package


class TencentSearch(ModuleBase):
    """
Tencent search interface wrapper class for calling Tencent Cloud content search services.

Provides encapsulation of Tencent Cloud search API, supporting keyword search and result processing.

Args:
    secret_id (str): Tencent Cloud API key ID for authentication
    secret_key (str): Tencent Cloud API key for authentication



Examples:
    
    from lazyllm.tools.tools import TencentSearch
    secret_id = '<your_secret_id>'
    secret_key = '<your_secret_key>'
    searcher = TencentSearch(secret_id, secret_key)
    """
    def __init__(self, secret_id, secret_key):
        super().__init__()
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        self.cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = 'tms.tencentcloudapi.com'
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.headers = {'X-TC-Action': 'SearchPro'}
        self.common_client = CommonClient(
            'tms', '2020-12-29', self.cred, '', profile=clientProfile)

    def forward(self, query: str):
        """
Searches for the query entered by the user.

Args:
    query (str): The content that the user wants to query.

**Returns:**

- package: Object containing search results, returns empty package if error occurs


Examples:
    
    from lazyllm.tools.tools import TencentSearch
    secret_id = '<your_secret_id>'
    secret_key = '<your_secret_key>'
    searcher = TencentSearch(secret_id, secret_key)
    res = searcher('calculus')
    """
        try:
            res_dict = self.common_client.call_json('SearchPro', {'Query': query, 'Mode': 2}, headers=self.headers)
            res = package(res_dict['Response']['Pages'])
        except Exception as err:
            lazyllm.LOG.error('Request Tencent Search meets error: ', err)
            res = package()
        return res
