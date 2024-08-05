import sqlite3
import threading
from abc import ABC, abstractmethod

class FileSystemQueue(ABC):

    def __new__(cls, *args, **kw):
        if cls is __class__:
            return SQLiteQueue()
        else:
            return super().__new__(cls, *args, **kw)

    @abstractmethod
    def enqueue(self, id, message): pass

    @abstractmethod
    def dequeue(self, id): pass

    @abstractmethod
    def peek(self, id): pass

    @abstractmethod
    def size(self, id): pass

    @abstractmethod
    def clear(self, id): pass


class SQLiteQueue(FileSystemQueue):
    def __init__(self, db_path='queue.db'):
        self.db_path = db_path
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

    def enqueue(self, id, message):
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

    def dequeue(self, id, limit=None):
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

    def peek(self, id):
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

    def size(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                SELECT COUNT(*) FROM queue WHERE id = ?
                ''', (id,))
                return cursor.fetchone()[0]

    def clear(self, id):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                DELETE FROM queue WHERE id = ?
                ''', (id,))
                conn.commit()
