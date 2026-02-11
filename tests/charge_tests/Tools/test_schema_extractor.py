import json
import os
import tempfile

import lazyllm
import pytest
from pydantic import BaseModel, Field

from lazyllm.tools.rag import SchemaExtractor
from lazyllm.tools.rag.doc_to_db.model import Table_ALGO_KB_SCHEMA


class ReadingReportSchema(BaseModel):
    reading_time: str = Field(description="The date or time period when the book was read.", default="")
    document_title: str = Field(description="The title of the book being reviewed.", default="")
    author_name: str = Field(description="The name of the author of the book.", default="")
    publication_type: str = Field(description="The type of publication (e.g., book, journal, etc.).", default="")
    publisher_name: str = Field(description="The name of the publisher of the book.", default="")
    publication_date: str = Field(description="The date when the book was published.", default="")
    keywords: str = Field(description="Key terms or themes discussed in the book.", default="")
    content_summary: str = Field(description="A brief summary of the book's main content or arguments.", default="")
    insights: str = Field(description="The reader's insights on the book's content.", default="")
    reflections: str = Field(description="The reader's reflections on the book's content.", default="")


EXPECTED_FIELDS = {
    "reading_time",
    "document_title",
    "author_name",
    "publication_type",
    "publisher_name",
    "publication_date",
    "keywords",
    "content_summary",
    "insights",
    "reflections",
}


def _fetch_bind_row(sql_manager, algo_id):
    bind_table = Table_ALGO_KB_SCHEMA["name"]
    bind_rows = json.loads(
        sql_manager.execute_query(
            f"select * from {bind_table} where algo_id='{algo_id}' limit 1"
        )
    )
    assert isinstance(bind_rows, list)
    return bind_rows[0] if bind_rows else None


def _get_table_name(schema_extractor, schema_set_id):
    return schema_extractor._table_name(schema_set_id)


def _get_count(sql_manager, table_name):
    count_result = json.loads(
        sql_manager.execute_query(f"select count(*) as cnt from {table_name}")
    )
    return count_result[0]["cnt"] if count_result else 0


def _connect_and_get_table(documents, schema_extractor, algo_id, *, force_refresh):
    documents.connect_sql_manager(
        sql_manager=schema_extractor.sql_manager,
        schma=ReadingReportSchema,
        force_refresh=force_refresh,
    )
    sql_manager = schema_extractor.sql_manager
    bind_row = _fetch_bind_row(sql_manager, algo_id)
    assert bind_row is not None
    table_name = _get_table_name(schema_extractor, bind_row["schema_set_id"])
    return sql_manager, table_name


class TestSchemaExtractor:
    @classmethod
    def setup_class(cls):
        cls.llm = lazyllm.OnlineChatModule(source="qwen")
        data_root_dir = os.getenv("LAZYLLM_DATA_PATH")
        assert data_root_dir
        cls.pdf_root = os.path.join(data_root_dir, "rag_master/default/__data/pdfs")
        fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        cls.db_config = {
            "db_type": "sqlite",
            "user": None,
            "password": None,
            "host": None,
            "port": None,
            "db_name": cls.db_path,
        }
        cls.schema_extractor = SchemaExtractor(
            db_config=cls.db_config,
            llm=cls.llm,
            force_refresh=True,
        )

    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)

    def setup_method(self, method):
        self.algo_id = f"doc_to_db_test_{method.__name__}"
        self.documents = lazyllm.Document(
            dataset_path=self.pdf_root,
            name=self.algo_id,
            schema_extractor=self.schema_extractor,
        )

    def test_connect_sql_manager_requires_schema(self):
        # 未提供 schema 时应抛出错误
        with pytest.raises(AssertionError) as excinfo:
            self.documents.connect_sql_manager(
                sql_manager=self.schema_extractor.sql_manager,
                schma=None,
            )
        assert "doc_table_schma must be given" in str(excinfo.value)

    def test_connect_sql_manager_creates_bind_and_table(self):
        # 提供 schema 时应写入绑定映射并创建表
        sql_manager, table_name = _connect_and_get_table(
            self.documents,
            self.schema_extractor,
            self.algo_id,
            force_refresh=True,
        )

        table_cls = sql_manager.get_table_orm_class(table_name)
        assert table_cls is not None
        column_names = {col.name for col in table_cls.__table__.columns}
        assert EXPECTED_FIELDS.issubset(column_names)

    def test_start_triggers_extraction_and_writes_rows(self):
        # 启动流程后应抽取并写入结构化数据
        sql_manager, table_name = _connect_and_get_table(
            self.documents,
            self.schema_extractor,
            self.algo_id,
            force_refresh=True,
        )

        self.documents.start()
        self.documents.extract_db_schema(llm=self.llm, print_schema=True)

        count_before = _get_count(sql_manager, table_name)
        assert count_before > 0

        sample_row_str = sql_manager.execute_query(f"select * from {table_name} limit 1")
        print(f"sample_row: {sample_row_str}")

    def test_extract_db_schema_returns_schema_info(self):
        # extract_db_schema 应返回带 schema_set_id 的结果
        _connect_and_get_table(
            self.documents,
            self.schema_extractor,
            self.algo_id,
            force_refresh=True,
        )

        self.documents.start()
        schema_info = self.documents.extract_db_schema(llm=self.llm, print_schema=True)
        assert schema_info is not None
        assert getattr(schema_info, "schema_set_id", None)

    def test_connect_same_schema_no_refresh(self):
        # 相同 schema 再次绑定且 force_refresh=False 时不应刷新数据
        sql_manager, table_name = _connect_and_get_table(
            self.documents,
            self.schema_extractor,
            self.algo_id,
            force_refresh=True,
        )

        self.documents.start()
        self.documents.extract_db_schema(llm=self.llm, print_schema=True)
        count_before = _get_count(sql_manager, table_name)

        self.documents.connect_sql_manager(
            sql_manager=sql_manager,
            schma=ReadingReportSchema,
            force_refresh=False,
        )
        count_after = _get_count(sql_manager, table_name)
        assert count_after == count_before
