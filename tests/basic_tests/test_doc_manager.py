import pytest
from lazyllm.tools.rag.utils import DocListManager
import shutil
import hashlib
import sqlite3
import unittest


@pytest.fixture(autouse=True)
def setup_tmpdir(request, tmpdir):
    request.cls.tmpdir = tmpdir


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
        self.manager.close()

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
        files_list = self.manager.list_kb_group_files(DocListManager.DEDAULT_GROUP_NAME, details=True)
        assert len(files_list) == 2

        self.manager.add_files_to_kb_group([self.test_file_1, self.test_file_2], DocListManager.DEDAULT_GROUP_NAME)
        files_list = self.manager.list_kb_group_files(DocListManager.DEDAULT_GROUP_NAME, details=True)
        assert len(files_list) == 4

    def test_list_kb_groups(self):
        self.manager.init_tables()
        assert len(self.manager.list_all_kb_group()) == 1

        self.manager.add_kb_group('group1')
        self.manager.add_kb_group('group2')
        r = self.manager.list_all_kb_group()
        assert len(r) == 3 and self.manager.DEDAULT_GROUP_NAME in r and 'group2' in r

    def test_delete_files(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1, self.test_file_2])
        self.manager.delete_files([self.test_file_1])
        files_list = self.manager.list_files(details=True)
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

        self.manager.update_file_status(file_id, DocListManager.Status.waiting)
        status = self.manager.get_file_status(file_id)
        assert status[0] == DocListManager.Status.waiting

    def test_add_files_to_kb_group(self):
        self.manager.init_tables()
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 0

        self.manager.add_files([self.test_file_1, self.test_file_2])
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 0

        self.manager.add_files_to_kb_group([self.test_file_1, self.test_file_2], group="group1")
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 2

    def test_delete_files_from_kb_group(self):
        self.manager.init_tables()

        self.manager.add_files([self.test_file_1, self.test_file_2])
        self.manager.add_files_to_kb_group([self.test_file_1, self.test_file_2], group="group1")

        self.manager.delete_files_from_kb_group([self.test_file_1], "group1")
        files_list = self.manager.list_kb_group_files("group1", details=True)
        assert len(files_list) == 1
