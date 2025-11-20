from pathlib import Path
import requests
import shutil
import uuid
from lazyllm import LOG
from lazyllm.module import ServerModule
from .graphrag_service_impl import GraphRAGServiceImpl
from urllib.parse import urlparse
from typing import List

class GraphRagServerModule(ServerModule):
    def __init__(self, kg_dir: str, *args, **kwargs):
        Path(kg_dir).mkdir(parents=True, exist_ok=True)
        self._kg_dir = kg_dir
        self._graphrag_service_impl = GraphRAGServiceImpl(kg_dir=str(self._kg_dir))
        # this url can only be set by graphrag server module
        self._service_url = self._get_shared_url_from_file()
        super().__init__(self._graphrag_service_impl, *args, **kwargs)

    @property
    def service_url(self) -> str:
        return self._service_url

    def _get_shared_url_from_file(self):
        '''Get the GraphRAGServiceImpl URL from the file'''
        url = None
        url_file = Path(self._kg_dir) / 'url.txt'
        if not url_file.exists():
            return None

        with open(url_file, 'r') as f:
            url = f.read().strip()
            if not url.startswith('http'):
                LOG.warning(f'URL from url.txt is not a valid URL: {url}')
                return None

        # Check if the URL is accessible
        try:
            response = requests.get(f'{url}/docs', timeout=10)
            if response.status_code == 200:
                return url
            else:
                LOG.warning(f'URL from url.txt returned status code {response.status_code}: {url}')
        except requests.exceptions.RequestException as e:
            LOG.warning(f'URL from url.txt is not accessible: {url}, error: {str(e)}')
        url_file.unlink(missing_ok=True)
        return None

    def start(self):
        super().start()
        parsed = urlparse(self._url)
        root_url = ''
        if parsed.port:
            root_url = f'http://{parsed.hostname}:{parsed.port}'
        else:
            root_url = f'http://{parsed.hostname}'

        # refresh the shared url
        shared_url = self._get_shared_url_from_file()
        if shared_url:
            self._service_url = shared_url
        else:
            with open(Path(self._kg_dir) / 'url.txt', 'w') as f:
                f.write(root_url)
            self._service_url = root_url

    def stop(self, clean=False):
        super().stop()
        is_root_server = self._url and self._url == self._service_url
        if is_root_server and clean:
            shutil.rmtree(self._kg_dir)

    def prepare_files(self, files: List[str], regenerate_config: bool = True):
        '''Copy files to self._kg_dir/input/ with renamed format: filename_{uuid}.ext'''
        input_dir = Path(self._kg_dir) / 'input'
        shutil.rmtree(input_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            source_file = Path(file_path)
            if not source_file.exists():
                continue

            file_name = source_file.stem
            ext = source_file.suffix
            # Generate new filename: filename_{uuid}.ext
            new_filename = f'{file_name}_{uuid.uuid4().hex}{ext}'
            dest_file = input_dir / new_filename

            # Copy file to destination
            shutil.copy2(source_file, dest_file)
        try:
            GraphRAGServiceImpl.init_root_dir(self._kg_dir, force=regenerate_config)
        except Exception as e:
            LOG.error(f'Error initializing root directory: {str(e)}')
            raise e

    def create_index(self, override: bool = True) -> dict:
        api_url = f'{self.service_url}/graphrag/create_index'

        # Send POST request to create_index endpoint
        response = requests.post(
            api_url,
            params={'override': override},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        LOG.info(f'create_index response: {response.json()}')
        response.raise_for_status()
        return response.json()

    def index_status(self, task_id: str) -> dict:
        api_url = f'{self.service_url}/graphrag/index_status'
        response = requests.post(
            api_url,
            params={'task_id': task_id},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        LOG.info(f'index status response: {response.json()}')
        response.raise_for_status()
        return response.json()

    @staticmethod
    def query_by_url(
        graphrag_server_url: str,
        query: str,
        search_method: str = 'local',
        community_level: int = 2,
        response_type: str = 'Multiple Paragraphs',
    ) -> dict:
        api_url = f'{graphrag_server_url}/graphrag/query'
        payload = {
            'query': query,
            'search_method': search_method,
            'community_level': community_level,
            'response_type': response_type,
        }
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
        response.raise_for_status()
        return response.json()

    def query(
        self,
        query: str,
        search_method: str = 'local',
        community_level: int = 2,
        response_type: str = 'Multiple Paragraphs',
    ) -> dict:
        return self.query_by_url(self.service_url, query, search_method, community_level, response_type)

    def forward(self, query: str) -> str:
        ans = self.query(query)
        return ans['answer']
