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

global_search = graphrag.api.query.drift_search
local_search = graphrag.api.query.local_search
drift_search = graphrag.api.query.drift_search
load_config = graphrag.config.load_config.load_config
IndexingMethod = graphrag.config.enums.IndexingMethod
index_cli = graphrag.cli.index.index_cli
initialize_project_at = graphrag.cli.initialize.initialize_project_at
HTTPException = fastapi.HTTPException

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
    '''Index state containing all GraphRAG data'''
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

class GraphRAGServiceImpl:
    def __init__(self, kg_dir: str):
        self._kg_dir = kg_dir
        self._tasks: Dict[str, Any] = {}
        self._index_state: Optional[IndexState] = None
        # 不在初始化时创建进程池，延迟到实际需要时创建
        self._process_executor: Optional[ProcessPoolExecutor] = None

    def _clean_index_state(self):
        self._index_state = None

    def index_ready(self) -> bool:
        return self._index_state is not None

    def _get_process_executor(self) -> ProcessPoolExecutor:
        '''懒加载进程池执行器，在需要时才创建'''
        if self._process_executor is None:
            self._process_executor = ProcessPoolExecutor(max_workers=1)
            LOG.info('ProcessPoolExecutor created')
        return self._process_executor

    def cleanup(self):
        '''清理资源，关闭进程池执行器'''
        if hasattr(self, '_process_executor') and self._process_executor is not None:
            self._process_executor.shutdown(wait=True)
            self._process_executor = None
            LOG.info('ProcessPoolExecutor has been shut down')

    def __del__(self):
        '''析构函数，确保资源被正确清理'''
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构时的异常

    @staticmethod
    def init_root_dir(kg_dir: str, force: bool = True):
        '''Initialize the root directory for a knowledge graph'''
        if not Path(kg_dir).exists():
            raise Exception(f'Root directory {kg_dir} does not exist. Please prepare it first.')
        initialize_project_at(path=kg_dir, force=True)

    @app.post('/graphrag/create_index', response_model=CreateIndexResponse)
    async def create_index(self, override: bool = True):
        '''Index a new document into the knowledge graph'''
        if self._index_state and not override:
            raise HTTPException(
                status_code=400, detail='Index already exists. No need to create index again.')
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
            if folder.exists() and override:
                shutil.rmtree(folder)

        def handle_task_exception(task: asyncio.Task):
            '''Handle exceptions from the background task'''
            try:
                task.result()
            except Exception as e:
                if task_id in self._tasks:
                    LOG.error(f'Error in index task {task_id}: {str(e)}')
                    task_info = self._tasks[task_id]
                    self._tasks[task_id] = task_info.model_copy(update={
                        'status': IndexStatus.FAILED,
                        'error_message': f'Task execution failed: {str(e)}',
                        'updated_at': datetime.now()
                    })

        try:
            background_task = asyncio.create_task(self._run_index_cli(task_id))
            background_task.add_done_callback(handle_task_exception)
        except Exception as e:
            LOG.error(f'Error creating index task: {str(e)}')
            task_info = self._tasks[task_id]
            self._tasks[task_id] = task_info.model_copy(update={
                'status': IndexStatus.FAILED,
                'error_message': f'Failed to create task: {str(e)}',
                'updated_at': datetime.now()
            })
            return CreateIndexResponse(task_id=task_id, message=f'Task {task_id} failed: {str(e)}')

        return CreateIndexResponse(task_id=task_id, message=f'Task {task_id} created.')

    async def _run_index_cli(self, task_id: str):
        '''run graphrag index task'''
        task_info = self._tasks[task_id]
        self._tasks[task_id] = task_info.model_copy(update={
            'status': IndexStatus.PROCESSING,
            'updated_at': datetime.now()
        })
        try:
            index_log_file = Path(self._kg_dir) / 'logs' / 'indexing-engine.log'
            if not index_log_file.exists():
                # 使用进程池执行器在独立进程中执行 index_cli，避免信号处理问题
                # 在需要时才获取进程池执行器
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._get_process_executor(),
                    self._run_index_cli_sync,
                    self._kg_dir
                )

            # Read the last two lines of the log file and check for success message
            if index_log_file.exists():
                with open(index_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_two_lines = ''.join(lines[-2:]) if len(lines) >= 2 else ''.join(lines)
                    if 'All workflows completed successfully' in last_two_lines:
                        self._tasks[task_id] = task_info.model_copy(update={
                            'status': IndexStatus.COMPLETED,
                            'updated_at': datetime.now()
                        })
                        LOG.info(f'Index task {task_id} completed successfully')
                        self._index_state = self._load_index_state()
                    else:
                        self._tasks[task_id] = task_info.model_copy(update={
                            'status': IndexStatus.FAILED,
                            'error_message': f'Indexing Failed. Please check logs {index_log_file}',
                            'updated_at': datetime.now()
                        })
            else:
                LOG.warning(f'Log file not found: {index_log_file}')
                self._tasks[task_id] = task_info.model_copy(update={
                    'status': IndexStatus.FAILED,
                    'error_message': f'Log file not found: {index_log_file}',
                    'updated_at': datetime.now()
                })
        except Exception as e:
            LOG.error(f'Error creating index task: {str(e)}')
            self._tasks[task_id] = task_info.model_copy(update={
                'status': IndexStatus.FAILED,
                'error_message': str(e),
                'updated_at': datetime.now()
            })
        finally:
            self.cleanup()

    @staticmethod
    def _run_index_cli_sync(kg_dir: str):
        '''同步执行 index_cli 的辅助函数，在独立进程中运行'''
        from pathlib import Path
        from lazyllm.thirdparty import graphrag
        IndexingMethod = graphrag.config.enums.IndexingMethod
        index_cli = graphrag.cli.index.index_cli

        index_cli(
            root_dir=Path(kg_dir),
            verbose=False,
            memprofile=False,
            cache=True,
            config_filepath=None,
            dry_run=False,
            skip_validation=False,
            output_dir=None,
            method=IndexingMethod.Standard.value,
        )

    def _load_index_state(self) -> IndexState:
        '''Load index state from the knowledge graph directory'''
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
            LOG.error(f'Error loading index state: {str(e)}')
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
        '''Get the status of an index task'''
        task_info = self._tasks.get(task_id, None)
        if not task_info:
            raise HTTPException(status_code=404, detail=f'Task not found: {task_id}')
        return task_info

    @app.post('/graphrag/query', response_model=QueryResponse)
    async def query(self, request: QueryRequest):
        '''Process a GraphRAG query using the specified search method'''
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
            LOG.error(f'Error processing query: {str(e)}')
            raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
