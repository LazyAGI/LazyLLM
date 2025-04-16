import unittest
import lazyllm
from lazyllm.tools import SqlManager
import pytest


class DocToDbTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llm = lazyllm.OnlineChatModule(source="qwen")
        cls.pdf_root = "/mnt/lustre/share_data/lazyllm/data/rag_master/default/__data/pdfs"
        cls.kws_desc_egs01 = [
            {"key": "reader_name", "desc": "The name of the person who wrote the reading report.", "type": "text"},
            {"key": "reading_date", "desc": "The date or time period when the reading was conducted.", "type": "text"},
            {"key": "document_title", "desc": "The title or name of the literature/document read.", "type": "text"},
            {"key": "author_name", "desc": "The name of the author of the literature/document.", "type": "text"},
            {
                "key": "document_form",
                "desc": "The format or type of the literature/document (e.g., book, journal).",
                "type": "text",
            },
            {
                "key": "publication_info",
                "desc": "Details about the publication such as publisher name and publication date.",
                "type": "text",
            },
            {
                "key": "keywords",
                "desc": "Key terms or phrases related to the content of the literature/document.",
                "type": "text",
            },
            {
                "key": "content_summary",
                "desc": "A brief summary or abstract of the main ideas presented in the literature/document.",
                "type": "text",
            },
            {
                "key": "insights_reflections",
                "desc": "The reader's thoughts, insights, or reflections after reading the literature/document.",
                "type": "text",
            },
            {
                "key": "issues_suggestions",
                "desc": "Any questions, problems, or suggestions raised by the reader regarding the literature/document.",  # noqa E501
                "type": "text",
            },
        ]

    def test_doc_to_db_sop(self):
        sql_manager = SqlManager(
            "SQLite",
            None,
            None,
            None,
            None,
            db_name=":memory:",
        )
        documents = lazyllm.Document(
            dataset_path=self.pdf_root,
            create_ui=False,
        )
        # Test-1: Use llm to extract schema
        schema_by_llm = documents.extract_db_schema(llm=self.llm, print_schema=True)
        assert schema_by_llm

        # Test-2: set without schema, assert failed
        with pytest.raises(AssertionError) as excinfo:
            documents.set_doc_to_db_connection(sql_manager=sql_manager, llm=self.llm, doc_table_schma=None)
        assert "doc_table_schma must be given" in str(excinfo.value)

        refined_schema = [
            {"key": "reading_time", "desc": "The date or time period when the book was read.", "type": "text"},
            {"key": "document_title", "desc": "The title of the book being reviewed.", "type": "text"},
            {"key": "author_name", "desc": "The name of the author of the book.", "type": "text"},
            {"key": "publication_type", "desc": "The type of publication (e.g., book, journal, etc.).", "type": "text"},
            {"key": "publisher_name", "desc": "The name of the publisher of the book.", "type": "text"},
            {"key": "publication_date", "desc": "The date when the book was published.", "type": "text"},
            {"key": "keywords", "desc": "Key terms or themes discussed in the book.", "type": "text"},
            {
                "key": "content_summary",
                "desc": "A brief summary of the book's main content or arguments.",
                "type": "text",
            },
            {"key": "insights", "desc": "The reader's insights on the book's content.", "type": "text"},
            {"key": "reflections", "desc": "The reader's reflections on the book's content.", "type": "text"},
        ]
        # Test-3: set sqlmanager, llm, with schema
        documents.set_doc_to_db_connection(
            sql_manager=sql_manager,
            llm=self.llm,
            doc_table_schma=refined_schema,
            force_refresh=True,
            export_doc_instantly=False,
        )
        # Test-4: check update run success (The extracted row exists in db means it definitely fits schema)
        documents.update_doc_to_db(llm=self.llm)
        str_result = sql_manager.execute_query(f"select * from {documents._doc_to_db_handler._processor.doc_table_name}")
        print(f"str_result: {str_result}")
        assert "reading_report_p1" in str_result

if __name__ == "__main__":
    unittest.main()
