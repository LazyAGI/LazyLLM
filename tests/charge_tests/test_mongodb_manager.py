import unittest
from lazyllm.tools import MongoDBManager, DBStatus, SqlCall
import lazyllm
from .utils import MongoDBEgsData, get_db_init_keywords
import datetime
import re


class TestSqlManager(unittest.TestCase):
    @classmethod
    def clean_obsolete_tables(cls, mongodb_manager: MongoDBManager):
        today = datetime.datetime.now()
        pattern = r"^(?:america)_(\d{8})_(\w+)"
        OBSOLETE_DAYS = 2
        db_result = mongodb_manager.get_all_collections()
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        existing_collections = db_result.result
        for collection_name in existing_collections:
            match = re.match(pattern, collection_name)
            if not match:
                continue
            table_create_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
            delta = (today - table_create_date).days
            if delta >= OBSOLETE_DAYS:
                mongodb_manager.drop_collection(collection_name)

    @classmethod
    def setUpClass(cls):
        username, password, host, port, database = get_db_init_keywords("MongoDB")
        cls.mongodb_manager = MongoDBManager(username, password, host, port, database, MongoDBEgsData.COLLECTION_NAME)
        cls.clean_obsolete_tables(cls.mongodb_manager)
        cls.mongodb_manager.delete({})
        cls.mongodb_manager.insert(MongoDBEgsData.COLLECTION_DATA)
        cls.mongodb_manager.set_desc(
            {
                "summary": "美国各个城市的人口情况",
                "schema_type": MongoDBEgsData.COLLECTION_SCHEMA_TYPE,
                "schema_desc": MongoDBEgsData.COLLECTION_SCHEMA_DESC,
            }
        )

        # Recommend to use sensenova, gpt-4o, qwen online model
        sql_llm = lazyllm.OnlineChatModule(source="sensenova")
        cls.sql_call: SqlCall = SqlCall(sql_llm, cls.mongodb_manager, use_llm_for_sql_result=True)

    @classmethod
    def tearDownClass(cls):
        # restore to clean database
        cls.mongodb_manager.drop_collection(MongoDBEgsData.COLLECTION_NAME)

    def test_manager_status(self):
        db_result = self.mongodb_manager.check_connection()
        assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_table_delete_insert_query(self):
        # delete all documents
        self.mongodb_manager.delete({})
        db_result = self.mongodb_manager.select({})
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        assert len(db_result.result) == 0

        # insert one document
        self.mongodb_manager.insert(MongoDBEgsData.COLLECTION_DATA[0])
        # insert many documents
        self.mongodb_manager.insert(MongoDBEgsData.COLLECTION_DATA[1:])

        db_result = self.mongodb_manager.select({})
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        assert len(db_result.result) == len(MongoDBEgsData.COLLECTION_DATA)

    def test_select(self):
        db_result = self.mongodb_manager.select({"state": "TX"})
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        db_result = self.mongodb_manager.select({"state": "TX"}, projection={"city": True})
        assert db_result.status == DBStatus.SUCCESS, db_result.detail
        assert len(db_result.result) == sum([ele["state"] == "TX" for ele in MongoDBEgsData.COLLECTION_DATA])

    def test_llm_query_online(self):
        str_results = self.sql_call("人口超过了300万的州有哪些?")
        self.assertIn("TX", str_results)
        self.assertIn("CA", str_results)
        self.assertIn("NY", str_results)
        print(f"str_results:\n{str_results}")

    @unittest.skip("Charge test has no scc support")
    def test_llm_query_local(self):
        local_llm = lazyllm.TrainableModule("qwen2-7b-instruct").deploy_method(lazyllm.deploy.vllm).start()
        sql_call = SqlCall(local_llm, self.mongodb_manager[0], use_llm_for_sql_result=True, return_trace=True)
        str_results = sql_call("人口超过了300万的州有哪些?")
        self.assertIn("TX", str_results)


if __name__ == "__main__":
    unittest.main()
