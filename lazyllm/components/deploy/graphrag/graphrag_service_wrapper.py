import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRagServiceWrapper:
    """GraphRAG Service Wrapper class that manages the GraphRAG service as a subprocess"""

    def __init__(self, graphrag_executable: str, kg_dir: str,
                 host: str = '0.0.0.0', port: int = 9011, start_timeout: int = 60):
        self._kg_dir = Path(kg_dir).resolve()
        self._host = host
        self._port = port
        self._python_executable = (Path(graphrag_executable) / '../python').resolve()
        if not self._python_executable.exists():
            raise FileNotFoundError(f'python nott found for {graphrag_executable}')
        self._start_timeout = start_timeout
        self._process: Optional[subprocess.Popen] = None
        self._base_url = f'http://{host}:{port}'

        if not self._kg_dir.exists():
            raise ValueError(f'Knowledge graph directory does not exist: {kg_dir}')

        # Validate python_executable can import required dependencies
        self._validate_dependencies()

        self._start_service()

    def _validate_dependencies(self):
        """Validate that the python_executable can import required dependencies"""
        required_modules = ['fastapi', 'uvicorn']

        for module in required_modules:
            try:
                result = subprocess.run(
                    [self._python_executable, '-c', f'import {module}'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    raise ImportError(f'Failed to import {module} for {self._python_executable}')

            except subprocess.TimeoutExpired:
                raise ImportError(f'Timeout while trying to import {module} for {self._python_executable}')
            except Exception as e:
                raise ImportError(f'Error while trying to import {module} for {self._python_executable}: {str(e)}')

    def _start_service(self):
        """Start the GraphRAG service as a subprocess"""
        try:
            current_dir = Path(__file__).parent
            service_script = current_dir / 'graphrag_service.py'

            if not service_script.exists():
                raise FileNotFoundError(f'GraphRAG service script not found: {service_script}')

            cmd = [
                str(self._python_executable),
                str(service_script),
                '--kg_dir', str(self._kg_dir),
                '--host', self._host,
                '--port', str(self._port)
            ]

            logger.info(f'Starting GraphRAG service with command: {" ".join(cmd)}')

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=current_dir
            )

            self._wait_for_service()
            logger.info(f'GraphRAG service started successfully on {self._base_url}')

        except Exception as e:
            logger.error(f'Failed to start GraphRAG service: {str(e)}')
            self._cleanup()
            raise

    def _wait_for_service(self):
        """Wait for the service to be ready"""
        start_time = time.time()

        while time.time() - start_time < self._start_timeout:
            try:
                # Try to connect to the service
                response = requests.get(f'{self._base_url}/docs', timeout=5)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException:
                pass

            # Check if process is still running
            if self._process and self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise RuntimeError(f'Service process exited unexpectedly. stdout: {stdout}, stderr: {stderr}')

            time.sleep(3)

        raise TimeoutError(f'Service failed to start within {self._start_timeout} seconds')

    def is_running(self) -> bool:
        """Check if the service is running"""
        if not self._process:
            return False

        # Check if process is still running
        if self._process.poll() is not None:
            return False

        try:
            response = requests.get(f'{self._base_url}/docs', timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def query(self, query: str, search_method: str = 'local',
              community_level: int = 2, response_type: str = 'Multiple Paragraphs') -> Dict[str, Any]:
        if not self.is_running():
            raise RuntimeError('Service is not running')

        try:
            payload = {
                'query': query,
                'search_method': search_method,
                'community_level': community_level,
                'response_type': response_type
            }

            response = requests.post(f'{self._base_url}/query', json=payload, timeout=300)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f'Failed to query service: {str(e)}')

    def _cleanup(self):
        """Clean up the subprocess"""
        if self._process:
            try:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

                logger.info('GraphRAG service stopped')

            except Exception as e:
                logger.error(f'Error stopping service: {str(e)}')
            finally:
                self._process = None

    def stop(self):
        """Stop the GraphRAG service"""
        self._cleanup()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._cleanup()

    def __repr__(self):
        return f'GraphRagServer(kg_dir="{self._kg_dir}", host="{self._host}", port={self._port})'


# Example usage and testing
if __name__ == '__main__':
    # python lazyllm/components/deploy/graphrag/graphrag_service_wrapper.py \
    # --kg_dir=$KG_DIR \
    # --graphrag_executable=$GRAPHRAG_EXECUTABLE_BIN --query='xxxx'
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description='GraphRAG Service Wrapper')
        parser.add_argument('--graphrag_executable', type=str, required=True, help='graphrag executable to use')
        parser.add_argument('--kg_dir', type=str, required=True, help='Path to knowledge graph directory')
        parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
        parser.add_argument('--port', type=int, default=9011, help='Port to run the service on')
        parser.add_argument('--start_timeout', type=int, default=60, help='Timeout for service startup')
        parser.add_argument('--query', type=str, help='Test query to run')
        return parser.parse_args()

    args = parse_args()

    try:
        with GraphRagServiceWrapper(
            graphrag_executable=args.graphrag_executable,
            kg_dir=args.kg_dir,
            host=args.host,
            port=args.port,
            start_timeout=args.start_timeout
        ) as server:
            logger.info(f'Server created: {server}')

            if args.query:
                logger.info(f'Running test query: {args.query}')
                result = server.query(args.query)
                logger.info(f'Query result: {result}')

            logger.info('Server is running. Press Ctrl+C to stop...')
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info('Shutting down...')

    except Exception as e:
        raise RuntimeError(f'Error: {str(e)}')
