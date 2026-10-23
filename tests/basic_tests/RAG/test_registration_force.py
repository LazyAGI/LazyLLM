from datetime import datetime
import sqlite3

import pytest

from lazyllm import dump_obj, load_obj
from lazyllm.tools.rag.data_loaders import DirectoryReader
from lazyllm.tools.rag.parsing_service.base import ALGORITHM_TABLE_INFO, NODE_GROUP_TABLE_INFO
from lazyllm.tools.rag.parsing_service.server import DocumentProcessor
from lazyllm.tools.sql import SqlManager


def test_force_preserves_node_group_identity_and_persisted_data(tmp_path):
    path = str(tmp_path / 'existing.db')
    db = SqlManager(db_type='sqlite', user=None, password=None, host=None, port=None,
                    db_name=path, tables_info_dict={'tables': [ALGORITHM_TABLE_INFO, NODE_GROUP_TABLE_INFO]})
    group = db.get_table_orm_class('lazyllm_node_group')
    with db.get_session() as session:
        session.add(group(id='existing-block-id', name='block', signature='628ceaa9889d22d9',
                          info_pickle=dump_obj({'lazy_mode': True}),
                          created_at=datetime.now(), updated_at=datetime.now()))
    # A persisted payload sentinel detects unintended clearing of the shared DB.
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE retained_chunks (id TEXT, content TEXT, vector BLOB)')
        conn.execute('INSERT INTO retained_chunks VALUES (?, ?, ?)', ('doc-1', 'old text', b'old-vector'))

    impl = DocumentProcessor._Impl.__new__(DocumentProcessor._Impl)
    impl._db_manager = db
    with pytest.raises(ValueError, match='different signature'):
        with db.get_session() as session:
            impl._upsert_single_node_group(session, 'block', '54a052f055f479c6', {}, 'update')
    with db.get_session() as session:
        identity = impl._upsert_single_node_group(session, 'block', '54a052f055f479c6', {'version': 1}, 'force')
        assert identity == 'existing-block-id'
    with db.get_session() as session:
        row = session.query(group).one()
        assert row.signature == '54a052f055f479c6'
        assert load_obj(row.info_pickle) == {'version': 1, 'lazy_mode': True}
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM retained_chunks').fetchall() == [('doc-1', 'old text', b'old-vector')]


def test_force_replaces_reader_but_update_rejects_changed_fingerprint():
    impl = DocumentProcessor._Impl.__new__(DocumentProcessor._Impl)
    old = DirectoryReader(None, local_readers={'*.txt': str})
    new = DirectoryReader(None, local_readers={'*.txt': bytes})
    impl._reader = old
    with pytest.raises(ValueError, match='reader signature mismatch'):
        impl._validate_reader(new, 'update')
    assert impl._reader is old
    impl._validate_reader(new, 'force')
    assert impl._reader is new
