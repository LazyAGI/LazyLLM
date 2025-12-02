import os
import json
import time
import random
import mimetypes
import tempfile
import importlib.util

from lazyllm import LOG
from lazyllm.thirdparty import boto3

from typing import Optional, Union
from io import BytesIO

INSERT_MAX_RETRIES = 10

def upload_data_to_s3(  # noqa:C901
    data: Union[bytes, str],
    bucket_name: str,
    object_key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    use_minio: bool = False,
    endpoint_url: Optional[str] = None,
    signature_version: str = 's3v4',
    max_memory_size: int = 100 * 1024 * 1024,  # 100MB threshold
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    client_kwargs = {}
    spec = importlib.util.find_spec('botocore.client')
    if spec is None:
        raise ImportError(
            'Please install boto3 to use botocore module. '
            'You can install it with `pip install boto3`'
        )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    Config = m.Config

    spec_exceptions = importlib.util.find_spec('botocore.exceptions')
    if spec_exceptions is None:
        raise ImportError(
            'Please install boto3 to use botocore module. '
            'You can install it with `pip install boto3`'
        )
    m_exceptions = importlib.util.module_from_spec(spec_exceptions)
    spec_exceptions.loader.exec_module(m_exceptions)
    ClientError = m_exceptions.ClientError

    client_kwargs['config'] = Config(connect_timeout=10, read_timeout=300,
                                     retries={'max_attempts': 10, 'mode': 'adaptive'}, max_pool_connections=10,)
    if use_minio:
        client_kwargs['endpoint_url'] = endpoint_url

    s3 = session.client('s3', **client_kwargs)

    transfer_config = boto3.s3.transfer.TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=3,
        use_threads=True
    )

    content_type, _ = mimetypes.guess_type(object_key)
    extra_args = {'ContentType': content_type or 'application/octet-stream'}

    if isinstance(data, (bytes, bytearray)):
        raw_bytes = data
    elif isinstance(data, str):
        raw_bytes = data.encode('utf-8')
    elif isinstance(data, list):
        jsonl_str = '\n'.join(json.dumps(d, ensure_ascii=False) for d in data)
        raw_bytes = jsonl_str.encode('utf-8')
    else:
        raise TypeError(f'Unsupported data type: {type(data)}')

    data_size = len(raw_bytes)

    # Decision: If the value is less than the threshold, upload to memory; otherwise, upload a temporary file.
    if data_size <= max_memory_size:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                fileobj = BytesIO(raw_bytes)
                fileobj.seek(0)
                s3.upload_fileobj(fileobj, bucket_name, object_key, ExtraArgs=extra_args, Config=transfer_config)
                break
            except ClientError as e:
                error_code = e.response['Error'].get('Code')
                if attempt == max_retries - 1:
                    raise
                if error_code in ['SlowDown', 'SlowDownWrite', 'RequestLimitExceeded', 'IncompleteBody']:
                    wait_time = (2 ** attempt) + (random.randint(0, 1000) / 1000)
                    LOG.warning(f'Upload encountered {error_code}, retrying in {wait_time:.2f}s... '
                                f'(Attempt {attempt + 1}/{max_retries})')
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt)
                LOG.warning(f'Upload failed with {e!r}, retrying in {wait_time:.2f}s... '
                            f'(Attempt {attempt + 1}/{max_retries})')
                time.sleep(wait_time)

    else:
        suffix = os.path.splitext(object_key)[1] or ''
        mode = 'wb' if isinstance(data, (bytes, bytearray)) else 'w'
        tmp_file = tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False)
        try:
            if suffix == '.jsonl' and isinstance(data, list):
                for d in data:
                    tmp_file.write(json.dumps(d, ensure_ascii=False) + '\n')
            else:
                tmp_file.write(data)
            tmp_file.flush()
            tmp_file.close()
            s3.upload_file(
                Filename=tmp_file.name,
                Bucket=bucket_name,
                Key=object_key,
            )
        except Exception as e:
            LOG.error(f'Upload Failed: {e}')
            raise
        finally:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

def download_data_from_s3(
    bucket_name: str,
    object_key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    use_minio: bool = False,
    endpoint_url: Optional[str] = None,
    signature_version: str = 's3v4',
    mode: str = 'rb',
    encoding: str = 'utf-8'
) -> bytes:

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    client_kwargs = {}
    if use_minio:
        spec = importlib.util.find_spec('botocore.client')
        if spec is None:
            raise ImportError(
                'Please install boto3 to use botocore module. '
                'You can install it with `pip install boto3`'
            )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        Config = m.Config
        client_kwargs['endpoint_url'] = endpoint_url
        client_kwargs['config'] = Config(signature_version=signature_version)
    s3 = session.client('s3', **client_kwargs)

    tmp = tempfile.NamedTemporaryFile(mode='wb', delete=False)
    try:
        s3.download_file(Bucket=bucket_name, Key=object_key, Filename=tmp.name)
        tmp.close()

        open_mode = mode
        with open(tmp.name, open_mode, encoding=None if 'b' in mode else encoding) as f:
            content = f.read()
        return content

    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass

def fibonacci_backoff(max_retries: int = INSERT_MAX_RETRIES):
    a, b = 1, 1
    for _ in range(max_retries):
        yield a
        a, b = b, a + b

def create_file_path(path: str, prefix: str = '') -> str:
    if prefix and not os.path.isabs(path):
        return os.path.join(prefix, path)
    return path
