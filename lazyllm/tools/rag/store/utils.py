import os
import json
import tempfile
import importlib.util

from lazyllm import LOG
from lazyllm.thirdparty import boto3

from typing import Optional, Union

INSERT_MAX_RETRIES = 10

def upload_data_to_s3(
    data: Union[bytes, str],
    bucket_name: str,
    object_key: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    use_minio: bool = False,
    endpoint_url: Optional[str] = None,
    signature_version: str = 's3v4'
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )
    client_kwargs = {}
    if use_minio:
        spec = importlib.util.find_spec("botocore.client")
        if spec is None:
            raise ImportError(
                "Please install boto3 to use botocore module. "
                "You can install it with `pip install boto3`"
            )
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        Config = m.Config
        client_kwargs['endpoint_url'] = endpoint_url
        client_kwargs['config'] = Config(signature_version=signature_version)
    s3 = session.client('s3', **client_kwargs)
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
        LOG.error(f"Upload Failed: {e}")
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
        spec = importlib.util.find_spec("botocore.client")
        if spec is None:
            raise ImportError(
                "Please install boto3 to use botocore module. "
                "You can install it with `pip install boto3`"
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

def create_file_path(path: str, prefix: str = "") -> str:
    if prefix and not os.path.isabs(path):
        return os.path.join(prefix, path)
    return path
