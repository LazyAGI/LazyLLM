from lazyllm.tools.rag.store.segment.sqlite_store import SQLiteStore


def test_bulk_upsert_replaces_fts_across_batches_and_preserves_other_documents(tmp_path):
    store = SQLiteStore(str(tmp_path / 'segments.db'))
    store.connect()
    try:
        rows = [dict(uid=f'n{i}', doc_id='doc', content='oldword', group='chunks') for i in range(1100)]
        other = dict(uid='other', doc_id='other-doc', content='oldword', group='chunks')
        assert store.upsert('chunks', rows + [other])
        for row in rows:
            row['content'] = 'newword'
        assert store.upsert('chunks', rows)
        assert store.upsert('chunks', rows)
        assert len(store.get('chunks')) == 1101
        assert len(store.search('chunks', 'newword', topk=2000)) == 1100
        assert [row['uid'] for row in store.search('chunks', 'oldword')] == ['other']
        assert store._open_conn().execute('SELECT count(*) FROM chunks_fts').fetchone()[0] == 1101
    finally:
        store._open_conn().close()
