from pathlib import Path
import requests
import shutil
import uuid
from lazyllm.module import ServerModule
from lazyllm.components.deploy.graphrag.graphrag_wrapper import GraphRagWrapper
from urllib.parse import urlparse
from typing import List

class GraphRagServerModule(ServerModule):
    def __init__(self, kg_dir, *args, **kwargs):
        kg_dir.mkdir(parents=True, exist_ok=True)
        self._kg_dir = kg_dir
        graphrag_wrapper = GraphRagWrapper(kg_dir=str(self._kg_dir))
        super().__init__(m=graphrag_wrapper, *args, **kwargs)

    def get_graphrag_url(self):
        return self._graphrag_server._url

    def start(self):
        super().start()
        parsed = urlparse(self._url)
        root_url = ''
        if parsed.port:
            root_url = f'http://{parsed.hostname}:{parsed.port}'
        else:
            root_url = f'http://{parsed.hostname}'
        with open(Path(self._kg_dir) / 'url.txt', 'w') as f:
            f.write(root_url)

    def stop(self, clean=False):
        is_root_server = self._url == self._graphrag_server._url
        if is_root_server and clean:
            shutil.rmtree(self._kg_dir)
        else:
            with open(Path(self._kg_dir) / 'url.txt', 'w') as _: pass
        super().stop()

    def prepare_files(self, files: List[str]):
        """Copy files to self._kg_dir/input/ with renamed format: filename_{uuid}.ext"""
        input_dir = Path(self._kg_dir) / 'input'
        input_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            source_file = Path(file_path)
            if not source_file.exists():
                continue

            file_name = source_file.stem
            ext = source_file.suffix
            # Generate new filename: filename_{uuid}.ext
            new_filename = f"{file_name}_{uuid.uuid4().hex}{ext}"
            dest_file = input_dir / new_filename

            # Copy file to destination
            shutil.copy2(source_file, dest_file)

    def create_index(self, override: bool = True) -> dict:
        graphrag_url = self.get_graphrag_url()
        api_url = f'{graphrag_url}/graphrag/create_index'

        # Send POST request to create_index endpoint
        response = requests.post(
            api_url,
            params={'override': override},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def index_status(self, task_id: str) -> dict:
        graphrag_url = self.get_graphrag_url()
        api_url = f'{graphrag_url}/graphrag/index_status'
        response = requests.post(
            api_url,
            params={'task_id': task_id},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def query(
        self,
        query: str,
        search_method: str = 'local',
        community_level: int = 2,
        response_type: str = 'Multiple Paragraphs',
    ) -> dict:
        graphrag_url = self.get_graphrag_url()
        api_url = f'{graphrag_url}/graphrag/query'
        payload = {
            'query': query,
            'search_method': search_method,
            'community_level': community_level,
            'response_type': response_type,
        }
        response = requests.post(api_url, json=payload, headers={'Content-Type': 'application/json'}, timeout=300)
        response.raise_for_status()
        return response.json()
