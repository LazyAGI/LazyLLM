import pytest
import lazyllm
from lazyllm.tools.rag.utils import DocListManager
from lazyllm.tools.rag.doc_manager import DocManager
import shutil
import hashlib
import sqlite3
import unittest
import requests
import io
import json


@pytest.fixture(autouse=True)
def setup_tmpdir(request, tmpdir):
    request.cls.tmpdir = tmpdir


def get_fid(path):
    if isinstance(path, (tuple, list)):
        return type(path)(get_fid(p) for p in path)
    else:
        return hashlib.sha256(f'{path}'.encode()).hexdigest()


@pytest.mark.usefixtures("setup_tmpdir")
class TestDocListManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = test_dir = self.tmpdir.mkdir("test_documents")

        test_file_1, test_file_2 = test_dir.join("test1.txt"), test_dir.join("test2.txt")
        test_file_1.write("This is a test file 1.")
        test_file_2.write("This is a test file 2.")
        self.test_file_1, self.test_file_2 = str(test_file_1), str(test_file_2)

        self.manager = DocListManager(str(test_dir), "TestManager")

    def tearDown(self):
        shutil.rmtree(str(self.test_dir))
        self.manager.release()

    def test_init_tables(self):
        self.manager.init_tables()
        assert self.manager.table_inited() is True

    def test_add_files(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1, self.test_file_2])
        files_list = self.manager.list_files(details=True)
        assert len(files_list) == 2
        assert any(self.test_file_1.endswith(row[1]) for row in files_list)
        assert any(self.test_file_2.endswith(row[1]) for row in files_list)

    def test_list_kb_group_files(self):
        self.manager.init_tables()
        files_list = self.manager.list_kb_group_files(DocListManager.DEFAULT_GROUP_NAME, details=True)
        assert len(files_list) == 2
        files_list = self.manager.list_kb_group_files('group1', details=True)
        assert len(files_list) == 0

        self.manager.add_files_to_kb_group(get_fid([self.test_file_1, self.test_file_2]),
                                           DocListManager.DEFAULT_GROUP_NAME)
        files_list = self.manager.list_kb_group_files(DocListManager.DEFAULT_GROUP_NAME, details=True)
        assert len(files_list) == 2

        self.manager.add_files_to_kb_group(get_fid([self.test_file_1, self.test_file_2]), 'group1')
        files_list = self.manager.list_kb_group_files('group1', details=True)
        assert len(files_list) == 2

    def test_list_kb_groups(self):
        self.manager.init_tables()
        assert len(self.manager.list_all_kb_group()) == 1

        self.manager.add_kb_group('group1')
        self.manager.add_kb_group('group2')
        r = self.manager.list_all_kb_group()
        assert len(r) == 3 and self.manager.DEFAULT_GROUP_NAME in r and 'group2' in r

    def test_delete_files(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1, self.test_file_2])
        self.manager.delete_files([hashlib.sha256(f'{self.test_file_1}'.encode()).hexdigest()])
        files_list = self.manager.list_files(details=True)
        assert len(files_list) == 2
        files_list = self.manager.list_files(details=True, exclude_status=DocListManager.Status.deleted)
        assert len(files_list) == 1
        assert not any(self.test_file_1.endswith(row[1]) for row in files_list)

    def test_update_file_message(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1])
        file_id = hashlib.sha256(f'{self.test_file_1}'.encode()).hexdigest()
        self.manager.update_file_message(file_id, metadata="New metadata", status="processed")

        conn = sqlite3.connect(self.manager._db_path)
        cursor = conn.execute("SELECT metadata, status FROM documents WHERE doc_id = ?", (file_id,))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "New metadata"
        assert row[1] == "processed"

    def test_get_and_update_file_status(self):
        self.manager.init_tables()

        file_id = hashlib.sha256(f'{self.test_file_1}'.encode()).hexdigest()
        status = self.manager.get_file_status(file_id)
        assert status[0] == DocListManager.Status.success

        self.manager.add_files([self.test_file_1], status=DocListManager.Status.waiting)
        status = self.manager.get_file_status(file_id)
        assert status[0] == DocListManager.Status.success

        self.manager.update_file_status([file_id], DocListManager.Status.waiting)
        status = self.manager.get_file_status(file_id)
        assert status[0] == DocListManager.Status.waiting

    def test_add_files_to_kb_group(self):
        self.manager.init_tables()
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 0

        self.manager.add_files([self.test_file_1, self.test_file_2])
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 0

        self.manager.add_files_to_kb_group(get_fid([self.test_file_1, self.test_file_2]), group="group1")
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 2

    def test_delete_files_from_kb_group(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1, self.test_file_2])
        self.manager.add_files_to_kb_group(get_fid([self.test_file_1, self.test_file_2]), group="group1")

        self.manager.delete_files_from_kb_group([hashlib.sha256(f'{self.test_file_1}'.encode()).hexdigest()], "group1")
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 1


@pytest.fixture(scope="class", autouse=True)
def setup_tmpdir_class(request, tmpdir_factory):
    request.cls.tmpdir = tmpdir_factory.mktemp("class_tmpdir")


@pytest.mark.usefixtures("setup_tmpdir_class")
class TestDocListServer(object):

    @classmethod
    def setup_class(cls):
        cls.test_dir = test_dir = cls.tmpdir.mkdir("test_server")

        test_file_1, test_file_2 = test_dir.join("test1.txt"), test_dir.join("test2.txt")
        test_file_1.write("This is a test file 1.")
        test_file_2.write("This is a test file 2.")
        cls.test_file_1, cls.test_file_2 = str(test_file_1), str(test_file_2)

        cls.manager = DocListManager(str(test_dir), "TestManager")
        cls.manager.init_tables()
        cls.manager.add_kb_group('group1')
        cls.manager.add_kb_group('extra_group')
        cls.server = lazyllm.ServerModule(DocManager(cls.manager))
        cls.server.start()
        cls._test_inited = True

        test_file_extra = test_dir.join("test_extra.txt")
        test_file_extra.write("This is a test file extra.")
        cls.test_file_extra = str(test_file_extra)

    def get_url(self, url, **kw):
        url = (self.server._url.rsplit("/", 1)[0] + '/' + url).rstrip('/')
        if kw: url += ('?' + '&'.join([f'{k}={v}' for k, v in kw.items()]))
        return url

    def teardown_class(cls):
        cls.server.stop()
        shutil.rmtree(str(cls.test_dir))
        cls.manager.release()

    @pytest.mark.order(0)
    def test_redirect_to_docs(self):
        assert requests.get(self.get_url('')).status_code == 200
        assert requests.get(self.get_url('docs')).status_code == 200

    @pytest.mark.order(1)
    def test_list_kb_groups(self):
        response = requests.get(self.get_url('list_kb_groups'))
        assert response.status_code == 200
        assert response.json().get('data') == [DocListManager.DEFAULT_GROUP_NAME, 'group1', 'extra_group']

    @pytest.mark.order(2)
    def test_list_files(self):
        response = requests.get(self.get_url('list_files'))
        assert len(response.json().get('data')) == 2
        response = requests.get(self.get_url('list_files', limit=1))
        assert len(response.json().get('data')) == 1
        response = requests.get(self.get_url('list_files_in_group', group_name=DocListManager.DEFAULT_GROUP_NAME))
        assert len(response.json().get('data')) == 2
        response = requests.get(self.get_url('list_files_in_group', group_name='group1'))
        assert len(response.json().get('data')) == 0

    @pytest.mark.order(3)
    def test_upload_files_and_upload_files_to_kb(self):
        files = [('files', ('test1.txt', io.BytesIO(b"file1 content"), 'text/plain')),
                 ('files', ('test2.txt', io.BytesIO(b"file2 content"), 'text/plain'))]

        data = dict(override='false', metadatas=json.dumps([{"key": "value"}, {"key": "value2"}]), user_path='path')
        response = requests.post(self.get_url('upload_files', **data), files=files)
        assert response.status_code == 200 and response.json().get('code') == 200, response.json()
        assert len(response.json().get('data')[0]) == 2

        response = requests.get(self.get_url('list_files', details=False))
        ids = response.json().get('data')
        assert response.status_code == 200 and len(ids) == 4

        # add_files_to_group
        files = [('files', ('test3.txt', io.BytesIO(b"file3 content"), 'text/plain'))]
        data = dict(override='false', metadatas=json.dumps([{"key": "value"}]), group_name='group1')
        response = requests.post(self.get_url('add_files_to_group', **data), files=files)
        assert response.status_code == 200

        response = requests.get(self.get_url('list_files', details=True))
        assert response.status_code == 200 and len(response.json().get('data')) == 5
        response = requests.get(self.get_url('list_files_in_group', group_name='group1'))
        assert response.status_code == 200 and len(response.json().get('data')) == 1

    @pytest.mark.order(4)
    def test_add_files_to_group_and_delete_files_from_group(self):
        response = requests.get(self.get_url('list_files', details=False))
        ids = response.json().get('data')
        assert response.status_code == 200 and len(ids) == 5
        requests.post(self.get_url('add_files_to_group_by_id'), json=dict(file_ids=ids[:2], group_name='group1'))
        response = requests.get(self.get_url('list_files_in_group', group_name='group1'))
        assert response.status_code == 200 and len(response.json().get('data')) == 3

        requests.post(self.get_url('delete_files_from_group'), json=dict(file_ids=ids[:1], group_name='group1'))
        response = requests.get(self.get_url('list_files_in_group', group_name='group1'))
        assert response.status_code == 200 and len(response.json().get('data')) == 3
        response = requests.get(self.get_url('list_files_in_group', group_name='group1', alive=True))
        assert response.status_code == 200 and len(response.json().get('data')) == 2

    @pytest.mark.order(5)
    def test_delete_files(self):
        response = requests.get(self.get_url('list_files', details=False))
        ids = response.json().get('data')
        assert response.status_code == 200 and len(ids) == 5

        response = requests.post(self.get_url('delete_files'), json=dict(file_ids=ids[-1:]))
        lazyllm.LOG.warning(response.json())
        assert response.status_code == 200 and response.json().get('code') == 200

        response = requests.get(self.get_url('list_files'))
        assert response.status_code == 200 and len(response.json().get('data')) == 5
        response = requests.get(self.get_url('list_files', alive=True))
        assert response.status_code == 200 and len(response.json().get('data')) == 4

        response = requests.get(self.get_url('list_files_in_group', group_name='group1'))
        assert response.status_code == 200 and len(response.json().get('data')) == 3
        response = requests.get(self.get_url('list_files_in_group', group_name='group1', alive=True))
        assert response.status_code == 200 and len(response.json().get('data')) == 1

    @pytest.mark.order(6)
    def test_add_files(self):
        json_data = {
            'files': [self.test_file_extra, "fake path"],
            'group_name': "extra_group",
            'metadatas': json.dumps([{"key": "value"}, {"key": "value"}])
        }
        response = requests.post(self.get_url('add_files'), json=json_data)
        assert response.status_code == 200
        assert len(response.json().get('data')) == 2 and response.json().get('data')[1] is None
