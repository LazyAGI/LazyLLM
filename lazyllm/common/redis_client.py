import lazyllm
from lazyllm.thirdparty import redis

lazyllm.config.add('redis_url', str, '', 'REDIS_URL', description='The URL of the Redis server.')
lazyllm.config.add('redis_recheck_delay', int, 5, 'REDIS_RECHECK_DELAY',
                   description='The delay of the Redis server check.')

_redis_url = lazyllm.config['redis_url']
_redis_client = None

if _redis_url:
    _redis_client = redis.Redis.from_url(_redis_url)
    assert _redis_client.ping(), (
        'Found reids config but can not connect, please check your config `LAZYLLM_REDIS_URL`.')


class RedisClient(object):
    def __init__(self, prefix: str = ''):
        self._prefix = prefix

    @staticmethod
    def get_instance(prefix: str = ''):
        return RedisClient(prefix)

    def __getitem__(self, prefix: str) -> 'RedisClient':
        assert prefix and isinstance(prefix, str), 'prefix shoule be a non-empty string'
        return RedisClient.get_instance(self._get_key(prefix))

    def _get_key(self, key):
        assert key and isinstance(key, str), 'key shoule be a non-empty string'
        return f'{self._prefix}@{key}' if self._prefix else key

    def get(self, key):
        return _redis_client.get(self._get_key(key))

    def set(self, key, value):
        return _redis_client.set(self._get_key(key), value)

    def delete(self, key):
        redis_client.delete(self._get_key(key) if isinstance(key, str) else [self._get_key(k) for k in key])

    def __getattr__(self, key):
        return getattr(redis_client, key)

    def __bool__(self):
        return bool(_redis_client)


redis_client = RedisClient()
