import os
import shutil
import unittest
import tempfile
from lazyllm import TrainableModule, Document, deploy
from lazyllm.launcher import cleanup
from lazyllm.tools.rag import SchemaExtractor
from lazyllm.tools.rag.doc_to_db.model import SchemaSetInfo, ExtractResult
from lazyllm.tools.sql_call import SqlCall

from pydantic import BaseModel, Field


class TestSchemaExtractor(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        fd, cls.db_dir = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        cls.text = 'In 2023, Tesla got 14.997 billion USD for the pure profit.'
        cls._temp_dir = tempfile.mkdtemp()
        with open(os.path.join(cls._temp_dir, 'test.txt'), 'w') as f:
            f.write(cls.text)
        cls._file_path = os.path.join(cls._temp_dir, 'test.txt')
        cls.db_config = {
            "db_type": "sqlite",
            "user": None,
            "password": None,
            "host": None,
            "port": None,
            "db_name": cls.db_dir,
        }
        cls.llm = TrainableModule('internlm2-chat-7b').deploy_method(deploy.vllm).start()
        cls.extractor = SchemaExtractor(db_config=cls.db_config, llm=cls.llm)

    @classmethod
    def teardown_class(cls):
        cleanup()
        if os.path.exists(cls.db_dir):
            os.remove(cls.db_dir)
        if os.path.exists(cls._temp_dir):
            shutil.rmtree(cls._temp_dir)

    def test_llm_schema_gen(self):
        res = self.extractor.analyze_schema_and_register(data=self.text, schema_set_id='temp_schema_set')
        assert isinstance(res, SchemaSetInfo)
        assert res.schema_set_id == 'temp_schema_set'

    def test_custom_schema_extract(self):
        class SchemaSet1(BaseModel):
            name: str = Field(description="Name of the people", default='unknown')
            age: int = Field(description="Age of the people", default=0)
        self.extractor.register_schema_set_to_kb(schema_set=SchemaSet1)
        res = self.extractor('Tom is a boy, he is eleven.')
        assert isinstance(res, ExtractResult)
        assert res.data.get('name', '') == 'Tom'
        assert res.data.get('age', 0) == 11

    def test_document_for_sqlcall(self):
        class TestSchema(BaseModel):
            company: str = Field(description="Name of the company", default='unknown')
            profit: float = Field(description="Profit of the company, unit is billion", default=0.0)
        doc = Document(
            dataset_path=self._temp_dir,
            name="test_algo",
            display_name="test_algo",
            description="algo for testing",
            store_conf={'metadata_store': self.db_config},
            schema_extractor=self.llm,
        )
        doc.register_schema_set(schema_set=TestSchema, force_refresh=True)
        doc.start()
        sqlcall = SqlCall.create_from_document(document=doc, llm=self.llm.share())
        result = sqlcall('what is the profit of Tesla?')
        assert '14.997' in result
