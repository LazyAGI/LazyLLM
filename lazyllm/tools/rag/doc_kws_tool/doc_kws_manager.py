from typing import List
import lazyllm
from lazyllm import ModuleBase
from .doc_kws_prcoessor import DocTypeDetector, DocKWSExtractor, DocKWSGenerator, DocKwDesc, validate_kw_desc
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus, DBResult
from sqlalchemy.orm import DeclarativeBase
import sqlalchemy
import uuid
from datetime import datetime
import re


class TableBase(DeclarativeBase):
    pass


class DocKWSManager:

    DB_TYPE_MAP = {
        "int": sqlalchemy.Integer,
        "text": sqlalchemy.Text,
        "float": sqlalchemy.Float,
    }
    UUID_COL_NAME = "lazyllm_uuid"
    CREATED_AT_COL_NAME = "lazyllm_created_at"
    DOC_PATH_COL_NAME = "lazyllm_doc_path"

    def __init__(self, llm: ModuleBase, sql_manager: SqlManager, table_name="lazyllm_doc_kws"):
        self._doc_type_detector = DocTypeDetector(llm)
        self._kws_generator = DocKWSGenerator(llm, maximum_doc_num=2)
        self._kws_extractor = DocKWSExtractor(llm)
        self._sql_manager = sql_manager
        self._kws_desc: List[DocKwDesc] = None
        self._table_name = table_name
        self._table_class = None
        self._doc_type = None

    @property
    def kws_desc(self):
        return self._kws_desc

    @property
    def table_name(self):
        return self._table_name

    @table_name.setter
    def table_name(self, table_name: str):
        raise NotImplementedError("Invalid to change table name")

    @property
    def doc_type(self):
        return self._doc_type

    @property
    def sql_manager(self):
        return self._sql_manager

    @sql_manager.setter
    def sql_manager(self, sql_manager: SqlManager):
        self._sql_manager = sql_manager

    @doc_type.setter
    def doc_type(self, doc_type: str):
        DOC_TYPE_LEN_LIMIT = 10
        english_words = re.findall(r"[a-zA-Z']+", doc_type)
        english_count = len(english_words)

        # chinese Unicode range
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", doc_type)
        chinese_count = len(chinese_chars)

        total_ct = english_count + chinese_count
        if total_ct > DOC_TYPE_LEN_LIMIT:
            raise ValueError(f"doc_type too long, make it less than {DOC_TYPE_LEN_LIMIT} words")
        else:
            self._doc_type = doc_type

    @kws_desc.setter
    def kws_desc(self, kws_desc: List[DocKwDesc]):
        raise NotImplementedError("As it'a dangerous operation, please use set_kws_desc instead")

    def _clear_table_orm(self):
        if self._table_class is not None:
            self._sql_manager.drop_table(self._table_class)
            TableBase.metadata.remove(self._table_class.__table__)
            TableBase.registry._dispose_cls(self._table_class)
            del self._table_class
            self._table_class = None

    def clear(self):
        self._clear_table_orm()
        self._doc_type = None
        self._table_class = None
        self._kws_desc = None

    # Alert, set kws_desc will drop old result in db
    def set_kws_desc(self, kws_desc: List[DocKwDesc]):
        assert isinstance(kws_desc, list)
        self._clear_table_orm()
        for kw_desc in kws_desc:
            is_success, err_msg = validate_kw_desc(kw_desc, DocKwDesc)
            assert is_success, err_msg
        self._kws_desc = kws_desc
        attrs = {"__tablename__": self._table_name, "__table_args__": {"extend_existing": True}}
        # use uuid as primary key
        attrs[self.UUID_COL_NAME] = sqlalchemy.Column(sqlalchemy.String(36), primary_key=True)
        attrs[self.CREATED_AT_COL_NAME] = sqlalchemy.Column(
            sqlalchemy.DateTime, default=sqlalchemy.func.now(), nullable=False
        )
        attrs[self.DOC_PATH_COL_NAME] = sqlalchemy.Column(
            sqlalchemy.Text, nullable=False, primary_key=False, index=True
        )
        for kw_desc in kws_desc:
            real_type = self.DB_TYPE_MAP.get(kw_desc["type"].lower(), sqlalchemy.Text)
            attrs[kw_desc["key"]] = sqlalchemy.Column(real_type, nullable=True, primary_key=False)
        self._table_class = type(self._table_name.capitalize(), (TableBase,), attrs)
        db_result = self._sql_manager.create_table(self._table_class)
        if db_result.status != DBStatus.SUCCESS:
            lazyllm.LOG.warning(f"Create table failed: {db_result.detail}")
            self.clear()

    def analyze_kws_desc(self, doc_type: str, doc_paths: List[str]) -> List[DocKwDesc]:
        return self._kws_generator.gen_kws_template(doc_type, doc_paths)

    def extract_kws_value(self, doc_path: str, extra_desc: str = ""):
        kws_value = self._kws_extractor.extract_kws_value(doc_path, self._kws_desc, extra_desc)
        if kws_value:
            kws_value[self.DOC_PATH_COL_NAME] = str(doc_path)
        else:
            lazyllm.LOG.warning(f"Extract kws value failed for {doc_path}")
        return kws_value

    def batch_extract_kws_value(self, doc_paths: List[str], extra_desc: str = ""):
        return [self.extract_kws_value(doc_path, extra_desc) for doc_path in doc_paths]

    def export_kws_values_to_db(self, kws_values: List[dict]):
        # Generate uuid explicitly because SQLite doesn't support auto gen uuid
        new_values = []
        for kws_value in kws_values:
            if kws_value:
                kws_value[self.UUID_COL_NAME] = str(uuid.uuid4())
                kws_value[self.CREATED_AT_COL_NAME] = datetime.now()
                new_values.append(kws_value)
        db_result = self._sql_manager.insert_values(self._table_name, new_values)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(f"Insert values failed: {db_result.detail}")

    def extract_and_record_kws(self, doc_paths: List[str], extra_desc: str = "") -> DBResult:
        existent_doc_paths = self.list_existent_doc_paths_in_db(doc_paths)
        # skip docs already in db
        doc_paths = set(doc_paths) - set(existent_doc_paths)
        kws_values = [self.extract_kws_value(doc_path, extra_desc) for doc_path in doc_paths]
        new_values = []
        for kws_value in kws_values:
            if kws_value:
                kws_value[self.UUID_COL_NAME] = str(uuid.uuid4())
                kws_value[self.CREATED_AT_COL_NAME] = datetime.now()
                new_values.append(kws_value)
        db_result = self._sql_manager.insert_values(self._table_name, new_values)
        return db_result

    def analyse_and_init_kws_desc(self, doc_paths: List[str]):
        if len(doc_paths) == 0:
            return
        if self._doc_type is None:
            doc_type = self._doc_type_detector.get_doc_type(doc_paths[0])
            if doc_type == "":
                raise ValueError("Failed to detect doc type")
            self._doc_type = doc_type
        if self._kws_desc is None:
            kws_desc = self.analyze_kws_desc(self._doc_type, doc_paths)
            if not kws_desc:
                raise ValueError("Failed to detect kws_desc")
            self.set_kws_desc(kws_desc)

    def auto_export_docs_to_sql(self, doc_paths: List[str]):
        self.analyse_and_init_kws_desc(doc_paths)
        self.extract_and_record_kws(doc_paths)

    def delete_doc_paths_records(self, doc_paths: list[str]):
        doc_paths = [str(ele) for ele in doc_paths]
        with self._sql_manager.get_session() as session:
            stmt = sqlalchemy.delete(self._table_class).where(
                getattr(self._table_class, self.DOC_PATH_COL_NAME).in_(doc_paths)
            )
            session.execute(stmt)
            session.commit()

    def list_existent_doc_paths_in_db(self, doc_paths: list[str]) -> List[str]:
        doc_paths = [str(ele) for ele in doc_paths]
        with self._sql_manager.get_session() as session:
            stmt = sqlalchemy.select(getattr(self._table_class, self.DOC_PATH_COL_NAME)).where(
                getattr(self._table_class, self.DOC_PATH_COL_NAME).in_(doc_paths)
            )
            result = session.execute(stmt).fetchall()
            return [ele[0] for ele in result]
