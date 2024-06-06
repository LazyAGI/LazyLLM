import lazyllm
from lazyllm.thirdparty import redis

lazyllm.config.add("redis_url", str, "", "REDIS_URL")
lazyllm.config.add("redis_recheck_delay", int, 5, "REDIS_RECHECK_DELAY")

redis_url = lazyllm.config["redis_url"]

redis_client = None

if redis_url:
    redis_client = redis.Redis.from_url(redis_url)
    assert (
        redis_client.ping()
    ), "Found reids config but can not connect, please check your config `LAZYLLM_REDIS_URL`."


def get_redis(key):
    url = redis_client.get(key)
    return url.decode("utf-8") if url else None
