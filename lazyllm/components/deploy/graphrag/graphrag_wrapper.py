# coding: utf-8
from pathlib import Path
import logging
from enum import Enum
import pandas as pd  # noqa: NID001, NID002
from typing import Optional, Dict, Any
from dataclasses import dataclass

from lazyllm import FastapiApp as app
import lazyllm
from fastapi import HTTPException  # noqa: NID001, NID002
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import shutil
import asyncio
import requests

# GraphRAG imports
# thirdparty import not working
# from thirdparty import graphrag
from graphrag.api import global_search, local_search, drift_search
from graphrag.config.load_config import load_config
from graphrag.config.enums import IndexingMethod
from graphrag.cli.index import index_cli

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexStatus(str, Enum):
    PROCESSING = 'processing'
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'

class IndexStatusResponse(BaseModel):
    task_id: str
    root_dir: str
    status: IndexStatus
    created_at: datetime
    updated_at: datetime
    error_message: str = Field(default='', description='Error message if the task failed')

class CreateIndexResponse(BaseModel):
    task_id: str
    message: str = Field(default='', description='Message to the user')


@dataclass
class IndexState:
    """Index state containing all GraphRAG data"""
    config: Any
    communities: pd.DataFrame
    community_reports: pd.DataFrame
    entities: pd.DataFrame
    relationships: pd.DataFrame
    text_units: pd.DataFrame
    covariates: Optional[pd.DataFrame] = None

class QueryRequest(BaseModel):
    query: str = Field(..., description='Search query string')
    search_method: str = Field(
        default='local', description='Search method to use(global, local, drift)', pattern='^(global|local|drift)$'
    )
    community_level: int = Field(default=2, description='Community level to use(0, 1, 2)', ge=0, le=2)
    response_type: str = Field(
        default='Multiple Paragraphs',
        description='Free-form description of the response format (eg: "Single Sentence", "List of 3-7 Points", etc.)',
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description='Answer to the query')

class GraphRagWrapper:
    def __init__(self, kg_dir: str):
        self._kg_dir = kg_dir
        self._url = self._get_shared_url()
        self._tasks: Dict[str, Any] = {}
        self._index_state: Optional[IndexState] = None

    def _clean_index_state(self):
        with open(Path(self._kg_dir) / 'url.txt', 'w') as _: pass
        self._url = None
        self._index_state = None

    def index_ready(self) -> bool:
        return self._index_state is not None

    def _get_shared_url(self):
        """Check if the URL is accessible"""
        url = None
        url_file = Path(self._kg_dir) / 'url.txt'
        with open(url_file, 'r') as f:
            url = f.read().strip()
            if not url.startswith('http'):
                lazyllm.LOG.warning(f'URL from url.txt is not a valid URL: {url}')
                return None
        if not url:
            return None
        # Check if the URL is accessible
        try:
            response = requests.get(f'{url}/docs', timeout=5)
            if response.status_code == 200:
                return url
            else:
                lazyllm.LOG.warning(f'URL from url.txt returned status code {response.status_code}: {url}')
        except requests.exceptions.RequestException as e:
            lazyllm.LOG.warning(f'URL from url.txt is not accessible: {url}, error: {str(e)}')
        return None

    @app.post('/graphrag/create_index', response_model=CreateIndexResponse)
    async def create_index(self, override: bool = True):
        """Index a new document into the knowledge graph"""
        self._url = self._get_shared_url()
        if self._url:
            raise HTTPException(
                status_code=400, detail=f'Server already exists. No need to create index again. URL: {self._url}')
        task_id = str(uuid.uuid4())
        self._clean_index_state()
        # Initialize task status
        self._tasks[task_id] = IndexStatusResponse(
            task_id=task_id,
            root_dir=self._kg_dir,
            status=IndexStatus.PENDING,
            error_message='',
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        # Delete old fiels
        output_dir = Path(self._kg_dir) / 'output'
        cache_dir = Path(self._kg_dir) / 'cache'
        logs_dir = Path(self._kg_dir) / 'logs'
        folders_to_delete = [output_dir, cache_dir, logs_dir]
        for folder in folders_to_delete:
            if folder.exists():
                if override:
                    shutil.rmtree(folder)
                else:
                    self._tasks[task_id].status = IndexStatus.FAILED
                    self._tasks[task_id].error_message = f'Directory exists: {folder}'
                    return CreateIndexResponse(
                        task_id=task_id, message=f'Task {task_id} failed: Directory exists: {folder}')

        asyncio.create_task(self._run_index_cli(task_id,))
        return CreateIndexResponse(task_id=task_id, message=f'Task {task_id} created.')

    async def _run_index_cli(self, task_id: str):
        """run graphrag index task"""
        task_info = self._tasks[task_id]
        task_info['status'] = IndexStatus.PROCESSING
        task_info['updated_at'] = datetime.now()
        try:
            index_cli(
                root_dir=self._kg_dir,
                verbose=False,
                memprofile=False,
                cache=True,
                config_filepath=None,
                dry_run=False,
                skip_validation=False,
                output_dir=None,
                method=IndexingMethod.Standard.value,
            )
            index_log_file = Path(self._kg_dir) / 'logs' / 'indexing-engine.log'

            # Read the last two lines of the log file and check for success message
            if index_log_file.exists():
                with open(index_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get the last two lines
                    last_two_lines = ''.join(lines[-2:]) if len(lines) >= 2 else ''.join(lines)
                    if 'All workflows completed successfully' in last_two_lines:
                        task_info['status'] = IndexStatus.COMPLETED
                        lazyllm.LOG.info(f'Index task {task_id} completed successfully')
                        # Load index state
                        self._index_state = self._load_index_state()
                    else:
                        task_info['status'] = IndexStatus.FAILED
                        task_info['error_message'] = f'Indexing Failed. Please check logs {index_log_file}'
            else:
                lazyllm.LOG.warning(f'Log file not found: {index_log_file}')
                task_info['status'] = IndexStatus.FAILED
                task_info['error_message'] = f'Log file not found: {index_log_file}'

            task_info['updated_at'] = datetime.now()
        except Exception as e:
            lazyllm.LOG.error(f'Error creating index task: {str(e)}')
            task_info['status'] = IndexStatus.FAILED
            task_info['error_message'] = str(e)
            task_info['updated_at'] = datetime.now()

    def _load_index_state(self) -> IndexState:
        """Load index state from the knowledge graph directory"""
        try:
            kg_dir = Path(self._kg_dir)
            config = load_config(root_dir=kg_dir)
            graph_store_dir = kg_dir / 'output'

            communities = pd.read_parquet(graph_store_dir / 'communities.parquet')
            community_reports = pd.read_parquet(graph_store_dir / 'community_reports.parquet')
            entities = pd.read_parquet(graph_store_dir / 'entities.parquet')
            relationships = pd.read_parquet(graph_store_dir / 'relationships.parquet')
            text_units = pd.read_parquet(graph_store_dir / 'text_units.parquet')

            use_covariates = (graph_store_dir / 'covariates.parquet').exists()
            covariates = pd.read_parquet(graph_store_dir / 'covariates.parquet') if use_covariates else None
        except Exception as e:
            lazyllm.LOG.error(f'Error loading index state: {str(e)}')
            return None

        return IndexState(
            config=config,
            communities=communities,
            community_reports=community_reports,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            covariates=covariates
        )

    @app.post('/graphrag/index_status', response_model=IndexStatusResponse)
    async def index_status(self, task_id: str):
        """Get the status of an index task"""
        task_info = self._tasks.get(task_id, None)
        if not task_info:
            raise HTTPException(status_code=404, detail=f'Task not found: {task_id}')
        return task_info

    @app.post('/graphrag/query', response_model=QueryResponse)
    async def query(self, request: QueryRequest):
        """Process a GraphRAG query using the specified search method"""
        if not self.index_ready():
            raise HTTPException(status_code=400, detail='Index not created yet. Run index first.')
        try:
            search_method = request.search_method.lower()

            if search_method == 'global':
                answer, _ = await global_search(
                    config=self._index_state.config,
                    entities=self._index_state.entities,
                    communities=self._index_state.communities,
                    community_reports=self._index_state.community_reports,
                    community_level=request.community_level,
                    dynamic_community_selection=False,
                    response_type=request.response_type,
                    query=request.query,
                )
            elif search_method == 'local':
                answer, _ = await local_search(
                    config=self._index_state.config,
                    entities=self._index_state.entities,
                    communities=self._index_state.communities,
                    community_reports=self._index_state.community_reports,
                    text_units=self._index_state.text_units,
                    relationships=self._index_state.relationships,
                    covariates=self._index_state.covariates,
                    community_level=request.community_level,
                    response_type=request.response_type,
                    query=request.query,
                )
            elif search_method == 'drift':
                answer, _ = await drift_search(
                    config=self._index_state.config,
                    entities=self._index_state.entities,
                    communities=self._index_state.communities,
                    community_reports=self._index_state.community_reports,
                    text_units=self._index_state.text_units,
                    relationships=self._index_state.relationships,
                    community_level=request.community_level,
                    response_type=request.response_type,
                    query=request.query,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f'Invalid search method: {search_method}. Must be one of: global, local, drift',
                )
            return QueryResponse(answer=answer)

        except Exception as e:
            lazyllm.LOG.error(f'Error processing query: {str(e)}')
            raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
