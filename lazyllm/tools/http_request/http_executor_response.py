import httpx

class HttpExecutorResponse:
    headers: dict[str, str]
    response: httpx.Response

    def __init__(self, response: httpx.Response = None):
        self.response = response
        self.headers = dict(response.headers) if isinstance(self.response, httpx.Response) else {}

    @property
    def is_file(self) -> bool:
        """
        check if response is file
        """
        content_type = self.get_content_type()
        file_content_types = ['image', 'audio', 'video']

        return any(v in content_type for v in file_content_types)

    def get_content_type(self) -> str:
        return self.headers.get('content-type', '')

    def extract_file(self) -> tuple[str, bytes]:
        """
        extract file from response if content type is file related
        """
        if self.is_file:
            return self.get_content_type(), self.body

        return '', b''

    @property
    def content(self) -> str:
        if isinstance(self.response, httpx.Response):
            return self.response.text
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')

    @property
    def body(self) -> bytes:
        if isinstance(self.response, httpx.Response):
            return self.response.content
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')

    @property
    def status_code(self) -> int:
        if isinstance(self.response, httpx.Response):
            return self.response.status_code
        else:
            raise ValueError(f'Invalid response type {type(self.response)}')
