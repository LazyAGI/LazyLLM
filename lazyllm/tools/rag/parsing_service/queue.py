from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from lazyllm import LOG
from ...sql import SqlManager
from ..utils import _orm_to_dict


class _SQLBasedQueue:
    '''a generic queue implementation based on SQL, use table to store messages, support FIFO and priority'''

    def __init__(self, table_name: str, columns: List[Dict[str, Any]], db_config: Dict[str, Any],
                 order_by: str = None, order_desc: bool = False):
        self._table_name = table_name
        self._columns = columns
        self._db_config = db_config
        self._order_by = order_by
        self._order_desc = order_desc

        try:
            self._sql_manager = SqlManager(
                **self._db_config,
                tables_info_dict={
                    'tables': [
                        {
                            'name': self._table_name,
                            'comment': f'Queue table: {self._table_name}',
                            'columns': self._columns
                        }
                    ]
                }
            )
            LOG.info(f'[SQLBasedQueue] Queue {self._table_name} initialized successfully')
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to initialize queue {self._table_name}: {e}')
            raise

    def _build_query(self, session, filter_by: Dict[str, Any] = None):
        TableCls = self._sql_manager.get_table_orm_class(self._table_name)
        query = session.query(TableCls)

        if filter_by:
            for key, value in filter_by.items():
                query = query.filter(getattr(TableCls, key) == value)

        order_field = self._order_by
        if not order_field:
            primary_keys = [key.name for key in TableCls.__table__.primary_key]
            order_field = primary_keys[0] if primary_keys else None

        if order_field:
            order_column = getattr(TableCls, order_field)
            if self._order_desc:
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column.asc())

        return query

    def enqueue(self, **kwargs) -> Dict[str, Any]:
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                new_record = TableCls(**kwargs)
                session.add(new_record)
                session.flush()
                result = _orm_to_dict(new_record)
                LOG.info(f'[SQLBasedQueue] Enqueued to {self._table_name}')
                return result
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to enqueue to {self._table_name}: {e}')
            raise

    def dequeue(self, filter_by: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        '''dequeue a message from the queue'''
        try:
            with self._sql_manager.get_session() as session:
                query = self._build_query(session, filter_by)
                record = query.with_for_update().first()

                if not record:
                    return None

                result = _orm_to_dict(record)
                session.delete(record)

                LOG.info(f'[SQLBasedQueue] Dequeued from {self._table_name}')
                return result
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to dequeue from {self._table_name}: {e}')
            raise

    def claim(self, worker_id: str, lease_duration: float,
              status_waiting: str = 'WAITING', status_working: str = 'WORKING',
              filter_by: Dict[str, Any] = None, include_task_types: List[str] = None,
              exclude_task_types: List[str] = None) -> Optional[Dict[str, Any]]:
        '''claim a message from the queue without removing it'''
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                query = self._build_query(session, filter_by)
                now = datetime.now()
                if include_task_types:
                    query = query.filter(TableCls.task_type.in_(include_task_types))
                if exclude_task_types:
                    query = query.filter(~TableCls.task_type.in_(exclude_task_types))
                query = query.filter(
                    (TableCls.status == status_waiting)
                    | ((TableCls.status == status_working)
                       & ((TableCls.lease_expires_at < now)
                          | (TableCls.lease_expires_at.is_(None))))
                )
                record = query.with_for_update().first()
                if not record:
                    return None

                record.status = status_working
                record.worker_id = worker_id
                record.lease_expires_at = now + timedelta(seconds=lease_duration)
                record.updated_at = now
                session.flush()
                result = _orm_to_dict(record)
                LOG.info(f'[SQLBasedQueue] Claimed from {self._table_name}')
                return result
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to claim from {self._table_name}: {e}')
            raise

    def extend_lease(self, task_id: str, worker_id: str, lease_duration: float) -> bool:
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                now = datetime.now()
                record = session.query(TableCls).filter(
                    TableCls.task_id == task_id,
                    TableCls.worker_id == worker_id
                ).first()
                if not record:
                    return False
                record.lease_expires_at = now + timedelta(seconds=lease_duration)
                record.updated_at = now
                session.flush()
                LOG.debug(f'[SQLBasedQueue] Extended lease for {self._table_name}')
                return True
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to extend lease for {self._table_name}: {e}')
            raise

    def delete(self, filter_by: Dict[str, Any]) -> int:
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                query = session.query(TableCls)
                if filter_by:
                    for key, value in filter_by.items():
                        query = query.filter(getattr(TableCls, key) == value)
                count = query.delete(synchronize_session=False)
                LOG.info(f'[SQLBasedQueue] Deleted {count} records from {self._table_name}')
                return count
        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to delete from {self._table_name}: {e}')
            raise

    def peek(self, filter_by: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        try:
            with self._sql_manager.get_session() as session:
                query = self._build_query(session, filter_by)
                record = query.first()

                if not record:
                    return None

                result = _orm_to_dict(record)
                LOG.debug(f'[SQLBasedQueue] Peeked from {self._table_name}')
                return result

        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to peek from {self._table_name}: {e}')
            raise

    def size(self, filter_by: Dict[str, Any] = None) -> int:
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                query = session.query(TableCls)

                if filter_by:
                    for key, value in filter_by.items():
                        query = query.filter(getattr(TableCls, key) == value)

                count = query.count()
                LOG.debug(f'[SQLBasedQueue] Size of {self._table_name}: {count}')
                return count

        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to get size of {self._table_name}: {e}')
            raise

    def clear(self, filter_by: Dict[str, Any] = None) -> int:
        try:
            with self._sql_manager.get_session() as session:
                TableCls = self._sql_manager.get_table_orm_class(self._table_name)
                query = session.query(TableCls)

                if filter_by:
                    for key, value in filter_by.items():
                        query = query.filter(getattr(TableCls, key) == value)

                count = query.delete(synchronize_session=False)

                LOG.info(f'[SQLBasedQueue] Cleared {count} records from {self._table_name}')
                return count

        except Exception as e:
            LOG.error(f'[SQLBasedQueue] Failed to clear {self._table_name}: {e}')
            raise
