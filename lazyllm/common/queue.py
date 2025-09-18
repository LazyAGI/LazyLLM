import sqlite3
import threading
from abc import ABC, abstractmethod
from .globals import globals
from ..configs import config
import os
from typing import Type
from lazyllm.thirdparty import redis
from filelock import FileLock

config.add('default_fsqueue', str, 'sqlite', 'DEFAULT_FSQUEUE')
config.add('fsqredis_url', str, '', 'FSQREDIS_URL')

class FileSystemQueue(ABC):
    """Abstract base class for file system-based queues.

FileSystemQueue is an abstract base class that provides a file system-based queue operation interface. It supports multiple backend implementations (such as SQLite, Redis) for message passing and data flow control in distributed environments.

This class implements the singleton pattern, ensuring only one queue instance per class name, and provides thread-safe queue operations.

Args:
    klass (str, optional): Class name identifier for the queue. Defaults to ``'__default__'``.

**Returns:**

- FileSystemQueue: Queue instance (singleton pattern)
"""

    __queue_pool__ = dict()

    def __init__(self, *, klass='__default__'):
        super().__init__()
        self._class = klass

    def __new__(cls, *args, **kw):
        klass = kw.get('klass', '__default__')
        if klass not in __class__.__queue_pool__:
            if cls is __class__:
                __class__.__queue_pool__[klass] = cls.__default_queue__(*args, **kw)
            else:
                __class__.__queue_pool__[klass] = super().__new__(cls)
        return __class__.__queue_pool__[klass]

    @classmethod
    def get_instance(cls, klass):
        """Get the queue instance for the specified class name.

This method returns the queue object bound to the given class name. If the class name has not been registered, it will be initialized automatically.

Args:
    klass (str): Queue class name identifier, must not be ``'__default__'``.

**Returns:**

- FileSystemQueue: Queue instance bound to the specified class name.
"""
        assert isinstance(klass, str) and klass != '__default__'
        return cls(klass=klass)

    @classmethod
    def set_default(cls, queue: Type):
        """Set the default queue implementation.

This method specifies the default queue class, used as the backend implementation when `klass` is not provided.

Args:
    queue (Type): Default queue class.
"""
        cls.__default_queue__ = queue

    @property
    def sid(self):
        return f'{globals._sid}-{self._class}'

    def enqueue(self, message):
        """Add a message to the queue.

This method adds the specified message to the tail of the queue, following the First-In-First-Out (FIFO) principle.

Args:
    message: The message content to be added to the queue.


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='enqueue_test')
    >>> queue.enqueue(123)
    >>> queue.peek()
    '123'
    """
        return self._enqueue(self.sid, message)
    def dequeue(self, limit=None):
        """Retrieve messages from the queue.

This method retrieves messages from the head of the queue and removes them, with the option to specify the number of messages to retrieve at once.

Args:
    limit (int, optional): Maximum number of messages to retrieve at once. If None, retrieves all messages. Defaults to None.

**Returns:**

- list: List of retrieved messages.


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='dequeue_test')
    >>> for i in range(5):
    ...     queue.enqueue(f"Message{i}")
    >>> all_messages = queue.dequeue()
    >>> all_messages
    ['Message0', 'Message1', 'Message2', 'Message3', 'Message4']
    """
        return self._dequeue(self.sid, limit=limit)
    def peek(self):
        """Retrieve the next message in the queue without removing it.

**Returns:**

- Any: The next available message in the queue, or ``None`` if the queue is empty.


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='peek_test')
    >>> queue.enqueue("First message")
    >>> queue.enqueue("Second message")
    >>> first_message = queue.peek()
    >>> first_message
    'First message'
    >>> queue.peek()
    'First message'
    """
        return self._peek(self.sid)
    def size(self):
        """Get the number of messages in the queue.

**Returns:**

- int: The current number of messages in the queue.


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='size_test')
    >>> queue.size()
    0
    >>> queue.enqueue("Message1")
    >>> queue.size()
    1
    >>> queue.enqueue("Message2")
    >>> queue.size()
    2
    >>> queue.dequeue()
    ['Message1', 'Message2']
    >>> queue.size()
    0
    """
        return self._size(self.sid)
    def init(self):
        """Initialize the queue.

This method clears all messages in the current queue, equivalent to calling ``clear()``.
"""
        self.clear()

    def clear(self):
        """Clear the queue.

Removes all messages from the queue, resetting it to an empty state.


Examples:
    >>> import lazyllm
    >>> queue = lazyllm.FileSystemQueue(klass='clear_test')
    >>> for i in range(10):
    ...     queue.enqueue(f"Message{i}")
    >>> queue.size()
    10
    >>> queue.clear()
    >>> queue.size()
    0
    >>> queue.peek() is None
    True
    """
        self._clear(self.sid)

    @abstractmethod
    def _enqueue(self, id, message): pass

    @abstractmethod
    def _dequeue(self, id, limit=None): pass

    @abstractmethod
    def _peek(self, id): pass

    @abstractmethod
    def _size(self, id): pass

    @abstractmethod
    def _clear(self, id): pass

# true means one connection can be used in multiple thread
# refer to: https://sqlite.org/compile.html#threadsafe
def sqlite3_check_threadsafety() -> bool:
    conn = sqlite3.connect(':memory:')
    res = conn.execute('''
        select * from pragma_compile_options
        where compile_options like 'THREADSAFE=%'
    ''').fetchall()
    conn.close()
    return True if res[0][0] == 'THREADSAFE=1' else False

class SQLiteQueue(FileSystemQueue):
    """Persistent file system queue backed by SQLite.
This class extends FileSystemQueue and stores queue data in an SQLite database. Messages are ordered by a position field to preserve FIFO behavior. The class supports concurrent-safe operations including enqueue, dequeue, peek, size checking, and clearing the queue.
The queue database is saved at ~/.lazyllm_filesystem_queue.db, with a file lock mechanism ensuring safe access in multi-process environments.

Args:
    klass (str): Name of the queue category used to logically separate queues. Default is '__default__'.
"""
    def __init__(self, klass='__default__'):
        super(__class__, self).__init__(klass=klass)
        self.db_path = os.path.expanduser(os.path.join(config['home'], '.lazyllm_filesystem_queue.db'))
        self._lock = FileLock(self.db_path + '.lock')
        self._check_same_thread = not sqlite3_check_threadsafety()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock, sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS queue (
                id TEXT NOT NULL,
                position INTEGER NOT NULL,
                message TEXT NOT NULL,
                PRIMARY KEY (id, position)
            )
            ''')
            conn.commit()

    def _enqueue(self, id, message):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT MAX(position) FROM queue WHERE id = ?
                ''', (id,))
                max_pos = cursor.fetchone()[0]
                next_pos = 0 if max_pos is None else max_pos + 1
                cursor.execute('''
                INSERT INTO queue (id, position, message)
                VALUES (?, ?, ?)
                ''', (id, next_pos, message))
                conn.commit()

    def _dequeue(self, id, limit=None):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                if limit:
                    cursor.execute('SELECT message, position FROM queue WHERE id = ? '
                                   'ORDER BY position ASC LIMIT ?', (id, limit))
                else:
                    cursor.execute('SELECT message, position FROM queue WHERE id = ? '
                                   'ORDER BY position ASC', (id,))

                rows = cursor.fetchall()
                if not rows:
                    return []
                messages = [row[0] for row in rows]
                cursor.execute('DELETE FROM queue WHERE id = ? AND position IN '
                               f'({",".join([str(row[1]) for row in rows])})', (id, ))
                conn.commit()
                return messages

    def _peek(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT message FROM queue WHERE id = ? ORDER BY position ASC LIMIT 1
                ''', (id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return row[0]

    def _size(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT COUNT(*) FROM queue WHERE id = ?
                ''', (id,))
                return cursor.fetchone()[0]

    def _clear(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path, check_same_thread=self._check_same_thread) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM queue WHERE id = ?
                ''', (id,))
                conn.commit()


class RedisQueue(FileSystemQueue):
    """
Redis-backed file system queue (inherits from FileSystemQueue) for cross-process/node message passing and queue management. It initializes its underlying storage using a configured Redis URL and employs thread-safe setup logic.

Args:
    klass (str): Classification name for the queue instance to distinguish different queues. Defaults to '__default__'.
"""
    def __init__(self, klass='__default__'):
        super(__class__, self).__init__(klass=klass)
        self.redis_url = config['fsqredis_url']
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            assert (
                conn.ping()
            ), 'Found fsque reids config but can not connect, please check your config `LAZYLLM_FSQREDIS_URL`.'
            if not conn.exists(self.sid):
                conn.rpush(self.sid, '<start>')

    def _enqueue(self, id, message):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            conn.rpush(id, message)

    def _dequeue(self, id, limit=None):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            if limit:
                limit = limit + 1
                vals = conn.lrange(id, 1, limit)
                conn.ltrim(id, limit, -1)
            else:
                vals = conn.lrange(id, 1, -1)
                conn.ltrim(id, 0, 0)
            if not vals:
                return []
            return [val.decode('utf-8') for val in vals]

    def _peek(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            val = conn.lindex(id, 1)
            if val is None:
                return None
            return val.decode('utf-8')

    def _size(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            rsize = conn.llen(id)
            return rsize - 1  # empty : [ <start> ]

    def _clear(self, id):
        with self._lock:
            conn = redis.Redis.from_url(self.redis_url)
            conn.delete(id)

fsquemap = {
    'sqlite': SQLiteQueue,
    'redis': RedisQueue
}

FileSystemQueue.set_default(fsquemap.get(config['default_fsqueue'].lower()))
