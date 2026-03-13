# Copyright (c) 2026 LazyAGI. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple

from lazyllm import thirdparty

from ..base import LazyLLMFSBase, CloudFSBufferedFile

try:
    from botocore.exceptions import ClientError
except ImportError:
    ClientError = Exception  # type: ignore[misc, assignment]


class S3FS(LazyLLMFSBase):

    protocol = 's3'

    def __init__(self, token: str = '', base_url: Optional[str] = None,
                 access_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None,
                 region_name: Optional[str] = None,
                 **storage_options):
        self._access_key = access_key or token
        self._secret_key = secret_key or ''
        self._endpoint_url = endpoint_url
        self._region_name = region_name
        self._s3_client = None
        super().__init__(token=token or access_key or '', base_url=base_url, **storage_options)

    def _setup_auth(self) -> None:
        boto3 = thirdparty.boto3
        session = boto3.session.Session()
        kwargs: Dict[str, Any] = {}
        if self._access_key:
            kwargs['aws_access_key_id'] = self._access_key
        if self._secret_key:
            kwargs['aws_secret_access_key'] = self._secret_key
        if self._endpoint_url:
            kwargs['endpoint_url'] = self._endpoint_url
        if self._region_name:
            kwargs['region_name'] = self._region_name
        self._s3_client = session.client('s3', **kwargs)

    def _parse_s3_path(self, path: str) -> Tuple[str, str]:
        parts = self._parse_path(path)
        if not parts:
            return '', ''
        bucket = parts[0]
        key = '/'.join(parts[1:]) if len(parts) > 1 else ''
        return bucket, key

    def ls(self, path: str, detail: bool = True, **kwargs) -> List:
        bucket, prefix = self._parse_s3_path(path)
        if not bucket:
            return self._list_buckets(detail)
        return self._list_objects(bucket, prefix, detail)

    def info(self, path: str, **kwargs) -> Dict[str, Any]:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            return self._entry('/', ftype='directory')
        if not key:
            try:
                self._s3_client.head_bucket(Bucket=bucket)
                return self._entry(f'/{bucket}', ftype='directory')
            except ClientError as e:
                if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchBucket'):
                    raise FileNotFoundError(path) from e
                raise
        try:
            resp = self._s3_client.head_object(Bucket=bucket, Key=key)
            mtime = resp['LastModified'].timestamp() if resp.get('LastModified') else None
            return self._entry(
                name=path, size=resp.get('ContentLength', 0),
                ftype='file', mtime=mtime,
                etag=resp.get('ETag', '').strip('"'),
                content_type=resp.get('ContentType', ''),
            )
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return self._entry(path, ftype='directory')
            raise

    def _open(self, path: str, mode: str = 'rb',
              block_size: Optional[int] = None,
              autocommit: bool = True,
              cache_options: Optional[Dict] = None,
              **kwargs) -> CloudFSBufferedFile:
        return CloudFSBufferedFile(
            self, path, mode=mode,
            block_size=block_size or self.blocksize,
            autocommit=autocommit, cache_options=cache_options,
        )

    def mkdir(self, path: str, create_parents: bool = True, **kwargs) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            raise ValueError('path must include at least a bucket name')
        if not key:
            self._s3_client.create_bucket(Bucket=bucket)
            return
        prefix = key.rstrip('/') + '/'
        self._s3_client.put_object(Bucket=bucket, Key=prefix, Body=b'')

    def rmdir(self, path: str) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket:
            return
        if not key:
            self._s3_client.delete_bucket(Bucket=bucket)
            return
        prefix = key.rstrip('/') + '/'
        self._s3_client.delete_object(Bucket=bucket, Key=prefix)

    def rm_file(self, path: str) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise ValueError(f'invalid S3 path: {path!r}')
        self._s3_client.delete_object(Bucket=bucket, Key=key)

    def _download_range(self, path: str, start: int, end: int) -> bytes:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise FileNotFoundError(path)
        range_header = f'bytes={start}-{end - 1}'
        resp = self._s3_client.get_object(Bucket=bucket, Key=key, Range=range_header)
        return resp['Body'].read()

    def _upload_data(self, path: str, data: bytes) -> None:
        bucket, key = self._parse_s3_path(path)
        if not bucket or not key:
            raise ValueError(f'invalid S3 path: {path!r}')
        self._s3_client.put_object(Bucket=bucket, Key=key, Body=data)

    def _platform_supports_webhook(self) -> bool:
        return True

    def _register_webhook(self, webhook_url: str, events: List[str], path: str) -> Dict[str, Any]:
        bucket, _ = self._parse_s3_path(path)
        if not bucket:
            raise ValueError('path must include a bucket name')
        s3_events = [
            's3:ObjectCreated:*',
            's3:ObjectRemoved:*',
        ]
        config: Dict[str, Any] = {
            'TopicConfigurations': [{
                'TopicArn': webhook_url,
                'Events': s3_events,
            }],
        }
        self._s3_client.put_bucket_notification_configuration(
            Bucket=bucket,
            NotificationConfiguration=config,
        )
        return {'bucket': bucket, 'events': s3_events, 'webhook_url': webhook_url}

    def _list_buckets(self, detail: bool) -> List:
        resp = self._s3_client.list_buckets()
        buckets = resp.get('Buckets', [])
        if detail:
            return [
                self._entry(
                    name=f'/{b["Name"]}', ftype='directory',
                    mtime=b['CreationDate'].timestamp() if b.get('CreationDate') else None,
                )
                for b in buckets
            ]
        return [f'/{b["Name"]}' for b in buckets]

    def _list_objects(self, bucket: str, prefix: str, detail: bool) -> List:
        kwargs: Dict[str, Any] = {'Bucket': bucket, 'Delimiter': '/'}
        if prefix:
            kwargs['Prefix'] = prefix.rstrip('/') + '/'
        results = []
        while True:
            resp = self._s3_client.list_objects_v2(**kwargs)
            for cp in resp.get('CommonPrefixes', []):
                raw = cp['Prefix'].rstrip('/')
                name = f'/{bucket}/{raw}'
                if detail:
                    results.append(self._entry(name=name, ftype='directory'))
                else:
                    results.append(name)
            for obj in resp.get('Contents', []):
                key = obj['Key']
                name = f'/{bucket}/{key}'
                mtime = obj['LastModified'].timestamp() if obj.get('LastModified') else None
                if detail:
                    results.append(self._entry(
                        name=name, size=obj.get('Size', 0),
                        ftype='file', mtime=mtime,
                        etag=obj.get('ETag', '').strip('"'),
                    ))
                else:
                    results.append(name)
            if resp.get('IsTruncated'):
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            else:
                break
        return results
