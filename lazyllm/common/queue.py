import sqlite3
import threading
from abc import ABC, abstractmethod
from .globals import globals
from ..configs import config
import os

class FileSystemQueue(ABC):

    __queue_pool__ = dict()

    def __init__(self, *, klass='__default__'):
        super().__init__()
        self._class = klass

    def __new__(cls, *args, **kw):
        klass = kw.get('klass', '__default__')
        if klass not in __class__.__queue_pool__:
            if cls is __class__:
                __class__.__queue_pool__[klass] = SQLiteQueue(*args, **kw)
            else:
                __class__.__queue_pool__[klass] = super().__new__(cls)
        return __class__.__queue_pool__[klass]

    @classmethod
    def get_instance(cls, klass):
        assert isinstance(klass, str) and klass != '__default__'
        return cls(klass=klass)

    @property
    def sid(self):
        return f'{globals._sid}-{self._class}'

    def enqueue(self, message): return self._enqueue(self.sid, message)
    def dequeue(self, limit=None): return self._dequeue(self.sid, limit=limit)
    def peek(self): return self._peek(self.sid)
    def size(self): return self._size(self.sid)
    def init(self): self.clear()

    def clear(self):
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


class SQLiteQueue(FileSystemQueue):
    def __init__(self, klass='__default__'):
        super(__class__, self).__init__(klass=klass)
        self.db_path = os.path.expanduser(os.path.join(config['home'], '.lazyllm_filesystem_queue.db'))
        self._lock = threading.Lock()
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
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
        """Retrieve and remove all messages from the queue."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT COUNT(*) FROM queue WHERE id = ?
                ''', (id,))
                return cursor.fetchone()[0]

    def _clear(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM queue WHERE id = ?
                ''', (id,))
                conn.commit()
