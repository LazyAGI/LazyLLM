import unittest
from lazyllm.tools import MongoDBManager, DBStatus, SqlCall
import lazyllm
import datetime
import re
import os
import uuid

UUID_HEX = str(uuid.uuid4().hex)
CURRENT_DAY = datetime.datetime.now().strftime("%Y%m%d")


class MongoDBEgsData:
    COLLECTION_NAME = f"america_{CURRENT_DAY}_{UUID_HEX}"
    COLLECTION_SCHEMA_TYPE = {
        "_id": "string",
        "city": "string",
        "state": "string",
        "pop": "int",
        "loc": {"type": "string", "coordinates": "array of float"},
    }
    COLLECTION_SCHEMA_DESC = {
        "city": "城市名",
        "state": "两个字母的州名缩写",
        "pop": "人口数量",
        "loc": "城市的经纬度",
    }

    COLLECTION_DATA = [
        {
            "city": "New York",
            "state": "NY",
            "pop": 8419600,
            "loc": {"type": "Point", "coordinates": [-74.0060, 40.7128]},
        },
        {
            "city": "Los Angeles",
            "state": "CA",
            "pop": 3980400,
            "loc": {"type": "Point", "coordinates": [-118.2437, 34.0522]},
        },
        {
            "city": "Chicago",
            "state": "IL",
            "pop": 2716000,
            "loc": {"type": "Point", "coordinates": [-87.6298, 41.8781]},
        },
        {
            "city": "Houston",
            "state": "TX",
            "pop": 2328000,
            "loc": {"type": "Point", "coordinates": [-95.3698, 29.7604]},
        },
        {
            "city": "Phoenix",
            "state": "AZ",
            "pop": 1690000,
            "loc": {"type": "Point", "coordinates": [-112.0740, 33.4484]},
        },
        {
            "city": "Philadelphia",
            "state": "PA",
            "pop": 1584200,
            "loc": {"type": "Point", "coordinates": [-75.1652, 39.9526]},
        },
        {
            "city": "San Antonio",
            "state": "TX",
            "pop": 1547000,
            "loc": {"type": "Point", "coordinates": [-98.4936, 29.4241]},
        },
        {
            "city": "San Diego",
            "state": "CA",
            "pop": 1423800,
            "loc": {"type": "Point", "coordinates": [-117.1611, 32.7157]},
        },
        {"city": "Dallas", "state": "TX", "pop": 1343000, "loc": {"type": "Point", "coordinates": [-96.7970, 32.7767]}},
        {
            "city": "San Jose",
            "state": "CA",
            "pop": 1028000,
            "loc": {"type": "Point", "coordinates": [-121.8863, 37.3382]},
        },
    ]


class TestMongoDBManager(unittest.TestCase):
    @classmethod
    def clean_obsolete_tables(cls, mongodb_manager: MongoDBManager):
        today = datetime.datetime.now()
        pattern = r"^(?:america)_(\d{8})_(\w+)"
        OBSOLETE_DAYS = 2
        with mongodb_manager.get_client() as client:
            db = client[mongodb_manager.db_name]
            existing_collections = db.list_collection_names()
            for collection_name in existing_collections:
                match = re.match(pattern, collection_name)
                if not match:
                    continue
                table_create_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
                delta = (today - table_create_date).days
                if delta >= OBSOLETE_DAYS:
                    db.drop_collection(collection_name)

    @classmethod
    def setUpClass(cls):
        conn_url = os.environ.get("LAZYLLM_MongoDB_URL", None)
        assert conn_url is not None
        pattern = r"mongodb://(?P<username>[^:]+):(?P<password>[^@]+)@(?P<host>[^:]+):(?P<port>\d+)/(?P<database>.+)"
        match = re.search(pattern, conn_url)
        assert match is not None
        username = match.group("username")
        password = match.group("password")
        host = match.group("host")
        port = match.group("port")
        database = match.group("database")

        cls.mongodb_manager = MongoDBManager(username, password, host, port, database, MongoDBEgsData.COLLECTION_NAME)
        cls.clean_obsolete_tables(cls.mongodb_manager)
        with cls.mongodb_manager.get_client() as client:
            collection = client[cls.mongodb_manager.db_name][cls.mongodb_manager.collection_name]
            collection.delete_many({})
            collection.insert_many(MongoDBEgsData.COLLECTION_DATA)
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
        with cls.mongodb_manager.get_client() as client:
            collection = client[cls.mongodb_manager.db_name][cls.mongodb_manager.collection_name]
            collection.drop()

    def test_manager_status(self):
        db_result = self.mongodb_manager.check_connection()
        assert db_result.status == DBStatus.SUCCESS, db_result.detail

    def test_manager_table_delete_insert_query(self):
        # delete all documents
        with self.mongodb_manager.get_client() as client:
            collection = client[self.mongodb_manager.db_name][self.mongodb_manager.collection_name]
            collection.delete_many({})
            results = list(collection.find({}))
            assert len(results) == 0

            # insert one document
            collection.insert_one(MongoDBEgsData.COLLECTION_DATA[0])
            # insert many documents
            collection.insert_many(MongoDBEgsData.COLLECTION_DATA[1:])

            results = list(collection.find({}))
            assert len(results) == len(MongoDBEgsData.COLLECTION_DATA)

    def test_select(self):
        with self.mongodb_manager.get_client() as client:
            collection = client[self.mongodb_manager.db_name][self.mongodb_manager.collection_name]
            results = list(collection.find({"state": "TX"}, projection={"city": True}))
            match_count = sum([ele["state"] == "TX" for ele in MongoDBEgsData.COLLECTION_DATA])
            assert len(results) == match_count

    def test_aggregate(self):
        with self.mongodb_manager.get_client() as client:
            collection = client[self.mongodb_manager.db_name][self.mongodb_manager.collection_name]
            results = list(collection.aggregate([{'$group': {'_id': '$state', 'totalPop': {'$sum': '$pop'}}},
                                                 {'$match': {'totalPop': {'$gt': 3000000}}}]))
            print(f"results: {results}")

    @unittest.skip("Just run local model in non-charge test")
    def test_llm_query_online(self):
        str_results = self.sql_call("人口超过了300万的州有哪些?")
        self.assertIn("TX", str_results)
        self.assertIn("CA", str_results)
        self.assertIn("NY", str_results)
        print(f"str_results:\n{str_results}")

    def test_llm_query_local(self):
        local_llm = lazyllm.TrainableModule("qwen2-72b-instruct-awq").deploy_method(lazyllm.deploy.vllm).start()
        sql_call = SqlCall(local_llm, self.mongodb_manager, use_llm_for_sql_result=True, return_trace=True)
        str_results = sql_call("总人口超过了300万的州有哪些?")
        self.assertIn("TX", str_results)
        self.assertIn("CA", str_results)
        self.assertIn("NY", str_results)
        print(f"str_results:\n{str_results}")


if __name__ == "__main__":
    unittest.main()
