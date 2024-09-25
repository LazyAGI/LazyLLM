import logging
from typing import Dict
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base

logger = logging.getLogger(__name__)

class DBManager:
    """
    A class to manage database connections, sessions, and table operations.

    Attributes:
        engine (Engine): SQLAlchemy engine instance.
        SessionLocal (scoped_session): Scoped session factory.
        Base (declarative_base): Base class for declarative model definitions.
    """
    _instances: Dict[str, 'DBManager'] = {}

    def __init__(self, database_url: str):
        """
        Initialize the DBManager with the given database URL.

        Args:
            database_url (str): The database URL to connect to.
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=self.engine))
        self.Base = declarative_base()

    def create_tables(self):
        """
        Create all tables in the database.
        """
        self.Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """
        Drop all tables in the database.
        """
        self.Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def session(self):
        """
        Provide a transactional scope around a series of operations.

        Yields:
            Session: A SQLAlchemy session object.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Session rollback because of exception: %s", e)
            raise
        finally:
            session.close()

    @staticmethod
    def create_db(db_name: str = None) -> 'DBManager':
        """
        Create or get a cached DBManager instance for the given database URL.

        Args:
            database_url (str): The database URL to connect to.

        Returns:
            DBManager: The DBManager instance.
        """
        import os
        db_name = db_name or "default.db"
        database_url = os.path.join("/yhm/jisiyuan/LazyLLM/dataset", db_name)
        database_url = f"sqlite:///{database_url}"

        if database_url not in DBManager._instances:
            DBManager._instances[database_url] = DBManager(database_url)
        return DBManager._instances[database_url]

    @lru_cache
    @staticmethod
    def create_db_tables():
        """
        Create tables for all cached DBManager instances.
        """
        for _, db in DBManager._instances.items():
            db.create_tables()
