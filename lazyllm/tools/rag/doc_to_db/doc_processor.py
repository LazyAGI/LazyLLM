from typing import List, Union
import lazyllm
from lazyllm import OnlineChatModule, TrainableModule
from .doc_analysis import (
    DocGenreAnalyser,
    DocInfoExtractor,
    DocInfoSchemaAnalyser,
    DocInfoSchema,
    DocInfoSchemaItem,
    validate_schema_item,
)
from lazyllm.tools.sql.sql_manager import SqlManager, DBStatus
from sqlalchemy.orm import DeclarativeBase
import sqlalchemy
import uuid
from datetime import datetime
import json


class TableBase(DeclarativeBase):
    pass


class LazyllmDocTableDesc(TableBase):
    __tablename__ = "lazyllm_doc_table_desc"
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True, default=1)
    desc = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)


class DocToDbProcessor:

    DB_TYPE_MAP = {
        "int": sqlalchemy.Integer,
        "text": sqlalchemy.Text,
        "float": sqlalchemy.Float,
    }
    UUID_COL_NAME = "lazyllm_uuid"
    CREATED_AT_COL_NAME = "lazyllm_created_at"
    DOC_PATH_COL_NAME = "lazyllm_doc_path"

    def __init__(self, sql_manager: SqlManager, doc_table_name="lazyllm_doc_elements"):
        self._doc_genre_analyser = DocGenreAnalyser()
        self._doc_info_schema_analyser = DocInfoSchemaAnalyser(maximum_doc_num=2)
        self._doc_info_extractor = DocInfoExtractor()
        self._sql_manager = sql_manager
        self._doc_info_schema: DocInfoSchema = None
        self._doc_table_name = doc_table_name
        self._table_class = None
        all_table_names = set(self._sql_manager.get_all_tables()) if sql_manager else {}
        # If doc_table exists, then desc_table must exist as well
        if self._doc_table_name in all_table_names:
            assert (
                LazyllmDocTableDesc.__tablename__ in all_table_names
            ), "LazyllmDocTableDesc table not found in database"
        # Create desc table for totally new database
        if sql_manager and LazyllmDocTableDesc.__tablename__ not in all_table_names:
            self._sql_manager.create_table(LazyllmDocTableDesc)

    @property
    def doc_info_schema(self):
        return self._doc_info_schema

    @property
    def doc_table_name(self):
        return self._doc_table_name

    @doc_table_name.setter
    def doc_table_name(self, doc_table_name: str):
        raise NotImplementedError("Invalid to change table name")

    @property
    def sql_manager(self):
        return self._sql_manager

    @sql_manager.setter
    def sql_manager(self, sql_manager: SqlManager):
        self._sql_manager = sql_manager

    @doc_info_schema.setter
    def doc_info_schema(self, doc_info_schema: DocInfoSchema):
        raise NotImplementedError("As it'a dangerous operation, please use reset_doc_info_schema instead")

    def _save_description_to_db(self, doc_info_schema: DocInfoSchema):
        assert self._sql_manager is not None, "sqlManager is not initialized"
        json_data = json.dumps(doc_info_schema)  # 直接存储为 JSON（如果数据库支持）
        with self._sql_manager.get_session() as session:
            existing = session.query(LazyllmDocTableDesc).filter_by(id=1).first()
            if existing:
                # 更新现有记录
                existing.desc = json_data
            else:
                # 插入新记录
                new_desc = LazyllmDocTableDesc(id=1, desc=json_data)
                session.add(new_desc)
            session.commit()

    def _clear_table_orm(self, drop_doc_table=True):
        if self._table_class is not None:
            if drop_doc_table:
                self._sql_manager.drop_table(self._table_class)
            TableBase.metadata.remove(self._table_class.__table__)
            TableBase.registry._dispose_cls(self._table_class)
            del self._table_class
            self._table_class = None

    def clear(self):
        self._clear_table_orm()
        self._table_class = None
        self._doc_info_schema = None

    # Alert, reset_doc_info_schema will drop old result in db
    def _reset_doc_info_schema(self, doc_info_schema: DocInfoSchema, recreate_doc_table=True):
        assert isinstance(doc_info_schema, list)
        self._save_description_to_db(doc_info_schema)
        self._clear_table_orm(drop_doc_table=recreate_doc_table)
        for schema_item in doc_info_schema:
            is_success, err_msg = validate_schema_item(schema_item, DocInfoSchemaItem)
            assert is_success, err_msg
        self._doc_info_schema = doc_info_schema
        attrs = {"__tablename__": self._doc_table_name, "__table_args__": {"extend_existing": True}}
        # use uuid as primary key
        attrs[self.UUID_COL_NAME] = sqlalchemy.Column(sqlalchemy.String(36), primary_key=True)
        attrs[self.CREATED_AT_COL_NAME] = sqlalchemy.Column(sqlalchemy.DateTime, nullable=False)
        attrs[self.DOC_PATH_COL_NAME] = sqlalchemy.Column(
            sqlalchemy.Text, nullable=False, primary_key=False, index=True
        )
        for schema_item in doc_info_schema:
            real_type = self.DB_TYPE_MAP.get(schema_item["type"].lower(), sqlalchemy.Text)
            attrs[schema_item["key"]] = sqlalchemy.Column(real_type, nullable=True, primary_key=False)
        self._table_class = type(self._doc_table_name.capitalize(), (TableBase,), attrs)
        if recreate_doc_table:
            # After drop_table, create table in db
            db_result = self._sql_manager.create_table(self._table_class)
            if db_result.status != DBStatus.SUCCESS:
                lazyllm.LOG.warning(f"Create table failed: {db_result.detail}")
                self.clear()

    def analyze_info_schema_by_llm(
        self, llm: Union[OnlineChatModule, TrainableModule], doc_paths: List[str], doc_topic: str = ""
    ) -> DocInfoSchema:
        assert len(doc_paths) > 0, "doc_paths should not be empty"
        if not doc_topic:
            doc_topic = self._doc_genre_analyser.analyse_doc_genre(llm, doc_paths[0])
            if doc_topic == "":
                raise ValueError("Failed to detect doc type")
        return self._doc_info_schema_analyser.analyse_info_schema(llm, doc_topic, doc_paths)

    def extract_info_from_docs(
        self, llm: Union[OnlineChatModule, TrainableModule], doc_paths: List[str], extra_desc: str = ""
    ) -> List[dict]:
        existent_doc_paths = self._list_existent_doc_paths_in_db(doc_paths)
        # skip docs already in db
        doc_paths = list(set(doc_paths) - set(existent_doc_paths))
        info_dicts = []
        for doc_path in doc_paths:
            kws_value = self._doc_info_extractor.extract_doc_info(llm, doc_path, self._doc_info_schema, extra_desc)
            if kws_value:
                kws_value[self.DOC_PATH_COL_NAME] = str(doc_path)
                info_dicts.append(kws_value)
            else:
                lazyllm.LOG.warning(f"Extract kws value failed for {doc_path}")
        return info_dicts

    def export_info_to_db(self, info_dicts: List[dict]):
        # Generate uuid explicitly because SQLite doesn't support auto gen uuid
        new_values = []
        for kws_value in info_dicts:
            if kws_value:
                kws_value[self.UUID_COL_NAME] = str(uuid.uuid4())
                kws_value[self.CREATED_AT_COL_NAME] = datetime.now()
                new_values.append(kws_value)
        db_result = self._sql_manager.insert_values(self._doc_table_name, new_values)
        if db_result.status != DBStatus.SUCCESS:
            raise ValueError(f"Insert values failed: {db_result.detail}")

    def _list_existent_doc_paths_in_db(self, doc_paths: list[str]) -> List[str]:
        doc_paths = [str(ele) for ele in doc_paths]
        with self._sql_manager.get_session() as session:
            stmt = sqlalchemy.select(getattr(self._table_class, self.DOC_PATH_COL_NAME)).where(
                getattr(self._table_class, self.DOC_PATH_COL_NAME).in_(doc_paths)
            )
            result = session.execute(stmt).fetchall()
            return [ele[0] for ele in result]

def extract_db_schema_from_files(file_paths: List[str], llm: Union[OnlineChatModule, TrainableModule]) -> DocInfoSchema:
    return DocToDbProcessor(sql_manager=None).analyze_info_schema_by_llm(llm, file_paths)
