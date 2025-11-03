# coding: utf-8
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import argparse
import pandas as pd  # noqa: NID001, NID002

from fastapi import FastAPI, HTTPException  # noqa: NID001, NID002
from pydantic import BaseModel, Field
import uvicorn  # noqa: NID001, NID002

# GraphRAG imports
from graphrag.api import global_search, local_search, drift_search
from graphrag.config.load_config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRagRequest(BaseModel):
    query: str = Field(..., description='Search query string')
    search_method: str = Field(
        default='local', description='Search method to use(global, local, drift)', pattern='^(global|local|drift)$'
    )
    community_level: int = Field(default=2, description='Community level to use(0, 1, 2)', ge=0, le=2)
    response_type: str = Field(
        default='Multiple Paragraphs',
        description='Free-form description of the response format (eg: "Single Sentence", "List of 3-7 Points", etc.)',
    )


class GraphRagResponse(BaseModel):
    answer: str = Field(..., description='Answer to the query')


def validate_kg_dir(kg_dir: str) -> bool:
    """Validate that the knowledge graph directory exists and contains required files"""
    graph_dir = Path(kg_dir) / 'output'

    if not graph_dir.exists():
        return False

    # Check for common GraphRAG output files
    required_files = [
        'communities.parquet',
        'community_reports.parquet',
        # 'documents.parquet',
        'entities.parquet',
        'relationships.parquet',
        'text_units.parquet',
    ]

    # optional_files = ['covariates.parquet',]

    for file_name in required_files:
        if not (graph_dir / file_name).exists():
            logger.error(f'Warning: {file_name} not found in {kg_dir}')
            return False

    return True


# GraphRAG data will be stored in app.state
def create_app(args):

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.args = args

        kg_dir = Path(args.kg_dir)
        logger.info(f'Loading GraphRAG data from: {args.kg_dir}')
        app.state.config = load_config(root_dir=kg_dir)

        graph_store_dir = kg_dir / 'output'
        app.state.communities = pd.read_parquet(graph_store_dir / 'communities.parquet')
        app.state.community_reports = pd.read_parquet(graph_store_dir / 'community_reports.parquet')
        app.state.entities = pd.read_parquet(graph_store_dir / 'entities.parquet')
        app.state.relationships = pd.read_parquet(graph_store_dir / 'relationships.parquet')
        app.state.text_units = pd.read_parquet(graph_store_dir / 'text_units.parquet')

        use_covariates = (graph_store_dir / 'covariates.parquet').exists()
        app.state.covariates = pd.read_parquet(graph_store_dir / 'covariates.parquet') if use_covariates else None
        yield

    app = FastAPI(lifespan=lifespan, title='GraphRAG Service')

    @app.post('/query', response_model=GraphRagResponse)
    async def query(request: GraphRagRequest):
        """
        Process a GraphRAG query using the specified search method
        """
        try:
            search_method = request.search_method.lower()

            if search_method == 'global':
                answer, _ = await global_search(
                    config=app.state.config,
                    entities=app.state.entities,
                    communities=app.state.communities,
                    community_reports=app.state.community_reports,
                    community_level=request.community_level,
                    dynamic_community_selection=False,
                    response_type=request.response_type,
                    query=request.query,
                )
            elif search_method == 'local':
                answer, _ = await local_search(
                    config=app.state.config,
                    entities=app.state.entities,
                    communities=app.state.communities,
                    community_reports=app.state.community_reports,
                    text_units=app.state.text_units,
                    relationships=app.state.relationships,
                    covariates=app.state.covariates,
                    community_level=request.community_level,
                    response_type=request.response_type,
                    query=request.query,
                )
            elif search_method == 'drift':
                answer, _ = await drift_search(
                    config=app.state.config,
                    entities=app.state.entities,
                    communities=app.state.communities,
                    community_reports=app.state.community_reports,
                    text_units=app.state.text_units,
                    relationships=app.state.relationships,
                    community_level=request.community_level,
                    response_type=request.response_type,
                    query=request.query,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f'Invalid search method: {search_method}. Must be one of: global, local, drift',
                )

            return GraphRagResponse(answer=answer)

        except Exception as e:
            logger.error(f'Error processing query: {str(e)}')
            raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')

    return app


def parse_args():
    parser = argparse.ArgumentParser(description='GraphRAG Service')
    parser.add_argument('--kg_dir', type=str, required=True, help='Path to knowledge graph directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=9011, help='Port to run the service on')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not validate_kg_dir(args.kg_dir):
        logger.error(f'Invalid knowledge graph directory: {args.kg_dir}')
        exit(1)

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
