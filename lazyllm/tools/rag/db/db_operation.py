import logging
from typing import Type, TypeVar, Optional, List, Callable

from sqlalchemy.orm import scoped_session, DeclarativeMeta, Session
from sqlalchemy.sql import and_

logger = logging.getLogger(__name__)


T = TypeVar('T', bound='DBOperations')


class CustomMeta(type):
    """
    Custom metaclass to add additional keyword arguments as class attributes.
    """
    def __new__(cls, name, bases, dct, **kwargs):
        new_cls = super().__new__(cls, name, bases, dct)
        for key, value in kwargs.items():
            setattr(new_cls, key, value)
        return new_cls


class DBMergeClass(DeclarativeMeta, CustomMeta):
    """
    A merging class that combines DeclarativeMeta and CustomMeta.
    """
    pass


class DBOperations(metaclass=CustomMeta):
    """
    A class that provides basic database operations such as create, get, and filter.

    Attributes:
        session (scoped_session): SQLAlchemy session factory.
    """
    session: Optional[scoped_session[Session]] = None

    @classmethod
    def add_node(cls, node: T):
        """
        Add a new node to the table and commit it to the database.

        Args:
            **kwargs: Arbitrary keyword arguments for the node instance.
        Returns:
            Optional[T]: The created node or None if the operation fails.
        """
        with cls.session() as db_session:
            db_session.add(node)

    @classmethod
    def add_or_replace_node(cls, node: T, **kwargs):
        """
        Add a new node to the table and commit it to the database.
        If a node matching the filter conditions exists, it will be deleted first.

        Args:
            node (T): The node instance to be added.
            filter_conditions (Dict[str, Any]): A dictionary of conditions to filter existing nodes.

        Returns:
            Optional[T]: The created node or None if the operation fails.
        """
        with cls.session() as db_session:
            # Check if a node matching the filter conditions exists
            filter_conditions = cls.get_filter_conditions(**kwargs)
            existing_node = db_session.query(cls).filter(and_(*filter_conditions)).first()
            if existing_node:
                # Delete the existing node
                db_session.delete(existing_node)
                db_session.commit()

            # Add the new node
            db_session.add(node)
            db_session.commit()
            db_session.refresh(node)
            return node

    @classmethod
    def del_node(cls, **kwargs) -> bool:
        """
        Delete an instance of the model based on the provided filters.
        Args:
            **kwargs: Arbitrary keyword arguments for filtering the query.
            Returns:
            bool: True if the instance is deleted successfully, False otherwise.
        """
        with cls.session() as db_session:
            # Create filter conditions based on kwargs
            filter_conditions = cls.get_filter_conditions(**kwargs)
            # Combine all filter conditions using AND
            query = db_session.query(cls).filter(and_(*filter_conditions))
            items = query.all()
            for item in items:
                db_session.delete(item)
            db_session.commit()
            return True

    @classmethod
    def create(cls: Type[T], **kwargs) -> Optional[T]:
        """
        Create a new instance of the model and commit it to the database.

        Args:
            **kwargs: Arbitrary keyword arguments for the model instance.

        Returns:
            Optional[T]: The created instance or None if the operation fails.
        """
        with cls.session() as db_session:
            instance = cls(**kwargs)
            db_session.add(instance)
            db_session.commit()
            db_session.refresh(instance)
            return instance

    @classmethod
    def first(cls: Type[T], **kwargs) -> Optional[T]:
        """
        Retrieve a single instance of the model based on the provided filters.

        Args:
            **kwargs: Arbitrary keyword arguments for filtering the query.

        Returns:
            Optional[T]: The retrieved instance or None if not found.
        """
        with cls.session() as db_session:
            filter_conditions = cls.get_filter_conditions(**kwargs)
            item = db_session.query(cls).filter(and_(*filter_conditions)).first()
            if item:
                db_session.expunge(item)
            return item

    @classmethod
    def all(cls: Type[T], **kwargs) -> List[T]:
        """
        Retrieve multiple instances of the model based on the provided filters.

        Args:
            **kwargs: Arbitrary keyword arguments for filtering the query.

        Returns:
            List[T]: A list of retrieved instances.
        """
        with cls.session() as db_session:
            filter_conditions = cls.get_filter_conditions(**kwargs)
            items = db_session.query(cls).filter(and_(*filter_conditions)).all()
            for item in items:
                db_session.expunge(item)
            return items

    @classmethod
    def filter_by(cls: Type[T], skip: int = 0, limit: int = 10, order_by = None, **kwargs) -> List[T]:
        """
        Retrieve multiple instances of the model based on the provided filters.

        Args:
            **kwargs: Arbitrary keyword arguments for filtering the query.
            order_by: Order by clause. User.name.asc()

        Returns:
            List[T]: A list of retrieved instances.
        """
        with cls.session() as db_session:
            filter_conditions = cls.get_filter_conditions(**kwargs)
            query = db_session.query(cls).filter(and_(*filter_conditions))
            if order_by:
                query = query.order_by(order_by)
            items = query.offset(skip).limit(limit).all()
            for item in items:
                db_session.expunge(item)
            return items

    @classmethod
    def update(cls: Type[T], fun: Callable[[T], None], **kwargs) -> None:
        """
        Applies the update function to all instance matching the filter criteria.

        Args:
            update_fun (Callable[[T], None]): Function to update each matching instance.
            **kwargs: Filter criteria to query the instance (e.g., column=value pairs).
        """
        with cls.session() as db_session:
            filter_conditions = cls.get_filter_conditions(**kwargs)
            items = db_session.query(cls).filter(and_(*filter_conditions))
            for item in items:
                fun(item)
            db_session.commit()

    def set(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def get_filter_conditions(cls, **kwargs) -> list:
        filter_conditions = []
        for key, value in kwargs.items():
            if isinstance(value, list):
                filter_conditions.append(getattr(cls, key).in_(value))
            else:
                filter_conditions.append(getattr(cls, key) == value)
        return filter_conditions