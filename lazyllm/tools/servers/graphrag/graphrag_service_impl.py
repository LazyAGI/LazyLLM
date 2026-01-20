# coding: utf-8
from pathlib import Path
import logging
from enum import Enum
from lazyllm.thirdparty import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass
from lazyllm.thirdparty import fastapi
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import shutil
import asyncio
from concurrent.futures import ProcessPoolExecutor
from lazyllm.thirdparty import graphrag

from lazyllm import FastapiApp as app
from lazyllm import LOG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _IndexStatus(str, Enum):
    PROCESSING = 'processing'
    PENDING = 'pending'
    COMPLETED = 'completed'
    FAILED = 'failed'

class _IndexStatusResponse(BaseModel):
    task_id: str
    root_dir: str
    status: _IndexStatus
    created_at: datetime
    updated_at: datetime
    error_message: str = Field(default='', description='Error message if the task failed')

class _CreateIndexResponse(BaseModel):
    task_id: str
    message: str = Field(default='', description='Message to the user')


@dataclass
class _IndexState:
    '''Index state containing all GraphRAG data'''
    config: Any
    communities: 'pd.DataFrame'
    community_reports: 'pd.DataFrame'
    entities: 'pd.DataFrame'
    relationships: 'pd.DataFrame'
    text_units: 'pd.DataFrame'
    covariates: 'Optional[pd.DataFrame]' = None

class _QueryRequest(BaseModel):
    query: str = Field(..., description='Search query string')
    search_method: str = Field(
        default='local', description='Search method to use(global, local, drift)', pattern='^(global|local|drift)$'
    )
    community_level: int = Field(default=2, description='Community level to use(0, 1, 2)', ge=0, le=2)
    response_type: str = Field(
        default='Multiple Paragraphs',
        description='Free-form description of the response format (eg: "Single Sentence", "List of 3-7 Points", etc.)',
    )


class _QueryResponse(BaseModel):
    answer: str = Field(..., description='Answer to the query')

class GraphRAGServiceImpl:
    def __init__(self, kg_dir: str):
        self._kg_dir = kg_dir
        self._tasks: Dict[str, Any] = {}
        self._index_state: Optional[_IndexState] = None
        self._process_executor: Optional[ProcessPoolExecutor] = None
        self._background_tasks: Dict[str, asyncio.Task] = {}  # Store background tasks to prevent garbage collection
        self._running_tasks: set[asyncio.Task] = set()

    def _clean_index_state(self):
        self._index_state = None

    def index_ready(self) -> bool:
        return self._index_state is not None

    def _get_process_executor(self) -> ProcessPoolExecutor:
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(max_workers=1)
            LOG.info('ProcessPoolExecutor created')
        return self._process_executor

    def cleanup(self):
        for task in self._running_tasks:
            task.cancel()
        self._running_tasks.clear()
        if hasattr(self, '_process_executor') and self._process_executor is not None:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None
            LOG.info('ProcessPoolExecutor has been shut down')

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    @staticmethod
    def init_root_dir(kg_dir: str, force: bool = True):
        '''Initialize the root directory for a knowledge graph'''
        if not Path(kg_dir).exists():
            raise Exception(f'Root directory {kg_dir} does not exist. Please prepare it first.')
        graphrag.cli.initialize.initialize_project_at(path=kg_dir, force=True)

    @app.post('/graphrag/create_index', response_model=_CreateIndexResponse)
    async def create_index(self, override: bool = True):
        '''Index a new document into the knowledge graph'''
        if self._index_state and not override:
            raise fastapi.HTTPException(status_code=400, detail='Index already exists. No need to create index again.')
        task_id = str(uuid.uuid4())
        self._clean_index_state()
        # Initialize task status
        self._tasks[task_id] = _IndexStatusResponse(
            task_id=task_id,
            root_dir=self._kg_dir,
            status=_IndexStatus.PENDING,
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
            if folder.exists() and override:
                shutil.rmtree(folder)

        try:
            background_task = asyncio.create_task(self._run_index_cli(task_id))
            self._running_tasks.add(background_task)
            # Store the task to prevent garbage collection
            background_task.add_done_callback(
                lambda t: (
                    self._running_tasks.remove(background_task) if background_task in self._running_tasks else None
                )
            )
        except Exception as e:
            LOG.error(f'Error creating index task: {str(e)}')
            task_info = self._tasks[task_id]
            self._tasks[task_id] = task_info.model_copy(update={
                'status': _IndexStatus.FAILED,
                'error_message': f'Failed to create task: {str(e)}',
                'updated_at': datetime.now()
            })
            return _CreateIndexResponse(task_id=task_id, message=f'Task {task_id} failed: {str(e)}')

        return _CreateIndexResponse(task_id=task_id, message=f'Task {task_id} created.')

    async def _run_index_cli(self, task_id: str):
        '''run graphrag index task'''
        task_info = self._tasks[task_id]
        self._tasks[task_id] = task_info.model_copy(update={
            'status': _IndexStatus.PROCESSING,
            'updated_at': datetime.now()
        })
        index_log_file = Path(self._kg_dir) / 'logs' / 'indexing-engine.log'
        if not index_log_file.exists():
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    self._get_process_executor(),
                    self._run_index_cli_sync,
                    self._kg_dir
                )
            except SystemExit as e:
                # SystemExit: 0 graphrag index cli returns 0 for successful completion
                if e.code == 0:
                    LOG.info('Index CLI completed with SystemExit: 0 (normal exit)')
                else:
                    # Non-zero exit code indicates failure
                    LOG.error(f'Index CLI exited with code {e.code}')
                    self._tasks[task_id] = task_info.model_copy(update={
                        'status': _IndexStatus.FAILED,
                        'error_message': f'Index CLI exited with code {e.code}',
                        'updated_at': datetime.now()
                    })
                    return
        try:
            # Check log file for success message.
            # Read the last two lines of the log file and check for success message
            if index_log_file.exists():
                with open(index_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_two_lines = ''.join(lines[-2:]) if len(lines) >= 2 else ''.join(lines)
                    if 'All workflows completed successfully' in last_two_lines:
                        self._tasks[task_id] = task_info.model_copy(update={
                            'status': _IndexStatus.COMPLETED,
                            'updated_at': datetime.now()
                        })
                        LOG.info(f'Index task {task_id} completed successfully')
                        self._index_state = self._load_index_state()
                    else:
                        self._tasks[task_id] = task_info.model_copy(update={
                            'status': _IndexStatus.FAILED,
                            'error_message': f'Indexing Failed. Please check logs {index_log_file}',
                            'updated_at': datetime.now()
                        })
            else:
                LOG.warning(f'Log file not found: {index_log_file}')
                self._tasks[task_id] = task_info.model_copy(update={
                    'status': _IndexStatus.FAILED,
                    'error_message': f'Log file not found: {index_log_file}',
                    'updated_at': datetime.now()
                })
        except Exception as e:
            LOG.error(f'Error creating index task: {str(e)}')
            self._tasks[task_id] = task_info.model_copy(update={
                'status': _IndexStatus.FAILED,
                'error_message': str(e),
                'updated_at': datetime.now()
            })

    @staticmethod
    def _run_index_cli_sync(kg_dir: str):
        graphrag.cli.index.index_cli(
            root_dir=Path(kg_dir),
            verbose=False,
            memprofile=False,
            cache=True,
            config_filepath=None,
            dry_run=False,
            skip_validation=False,
            output_dir=None,
            method=graphrag.config.enums.IndexingMethod.Standard.value,
        )

    def _load_index_state(self) -> _IndexState:
        '''Load index state from the knowledge graph directory'''
        try:
            kg_dir = Path(self._kg_dir)
            config = graphrag.config.load_config.load_config(root_dir=kg_dir)
            graph_store_dir = kg_dir / 'output'

            communities = pd.read_parquet(graph_store_dir / 'communities.parquet')
            community_reports = pd.read_parquet(graph_store_dir / 'community_reports.parquet')
            entities = pd.read_parquet(graph_store_dir / 'entities.parquet')
            relationships = pd.read_parquet(graph_store_dir / 'relationships.parquet')
            text_units = pd.read_parquet(graph_store_dir / 'text_units.parquet')

            use_covariates = (graph_store_dir / 'covariates.parquet').exists()
            covariates = pd.read_parquet(graph_store_dir / 'covariates.parquet') if use_covariates else None
        except Exception as e:
            LOG.error(f'Error loading index state: {str(e)}')
            return None

        return _IndexState(
            config=config,
            communities=communities,
            community_reports=community_reports,
            entities=entities,
            relationships=relationships,
            text_units=text_units,
            covariates=covariates
        )

    @app.post('/graphrag/index_status', response_model=_IndexStatusResponse)
    async def index_status(self, task_id: str):
        '''Get the status of an index task'''
        task_info = self._tasks.get(task_id, None)
        if not task_info:
            raise fastapi.HTTPException(status_code=404, detail=f'Task not found: {task_id}')
        return task_info

    @app.post('/graphrag/query', response_model=_QueryResponse)
    async def query(self, request: _QueryRequest):
        '''Process a GraphRAG query using the specified search method'''
        if not self.index_ready():
            raise fastapi.HTTPException(status_code=400, detail='Index not created yet. Run index first.')
        try:
            search_method = request.search_method.lower()

            if search_method == 'global':
                answer, _ = await graphrag.api.query.graphrag.api.query.global_search(
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
                answer, _ = await graphrag.api.query.local_search(
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
                answer, _ = await graphrag.api.query.drift_search(
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
                raise fastapi.HTTPException(
                    status_code=400,
                    detail=f'Invalid search method: {search_method}. Must be one of: global, local, drift',
                )
            return _QueryResponse(answer=answer)

        except Exception as e:
            LOG.error(f'Error processing query: {str(e)}')
            raise fastapi.HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
