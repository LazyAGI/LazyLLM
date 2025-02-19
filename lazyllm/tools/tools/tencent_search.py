import lazyllm
from lazyllm.module import ModuleBase
from lazyllm.common import package


class TencentSearch(ModuleBase):
    def __init__(self, secret_id, secret_key):
        super().__init__()
        from tencentcloud.common.common_client import CommonClient
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        self.cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tms.tencentcloudapi.com"
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.headers = {"X-TC-Action": "SearchPro"}
        self.common_client = CommonClient(
            "tms", '2020-12-29', self.cred, "", profile=clientProfile)

    def forward(self, query: str):
        try:
            res_dict = self.common_client.call_json("SearchPro", {'Query': query, 'Mode': 2}, headers=self.headers)
            res = package(res_dict["Response"]["Pages"])
        except Exception as err:
            lazyllm.LOG.error("Request Tencent Search meets error: ", err)
            res = package()
        return res
