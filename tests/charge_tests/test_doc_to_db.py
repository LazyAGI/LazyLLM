import unittest
import lazyllm
from lazyllm.tools.rag.doc_to_db import DocInfoSchemaAnalyser, DocInfoExtractor, DocToDbProcessor
from lazyllm.tools import SqlManager
from pathlib import Path


class TestDocKwsManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llm = lazyllm.OnlineChatModule(source="qwen")
        cls.pdf_root = "/mnt/lustre/share_data/lazyllm/data/rag_master/default/__data/pdfs"
        cls.doc_kws_generator = DocInfoSchemaAnalyser(cls.llm)
        cls.doc_kws_extractor = DocInfoExtractor(cls.llm)
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

        cls.kws_value_egs01 = {
            "reader_name": "韦家朝",
            "reading_date": "2009年11\xad12月",
            "document_title": "论语",
            "author_name": "孔子",
            "document_form": "书籍",
            "content_summary": "《论语》是一部记录孔子及其弟子言语事迹的儒家经典，其内容极其丰富，涉及政治、思想、文化、教育、伦理道德等许多方面，对后世的影响非常大。全篇共分为二十章，篇名一般取自本篇的第一句话的前面二个字，内容经过精心编排，井然又序。概括起来就主要讲了三方面：如何做官、做人（仁、义、礼）和做学问。",  # noqa E501
            "insights_reflections": "反思：一、对儒学既要有同情的了解，又要有深沉的反省。罗素有段话说得好：“研究一个哲学家的时候，正确的态度既不是尊崇也不是蔑视，而是应该首先要有一种假设的同情，直到可能知道在他的理论里有什么东西大概是可以相信的为止；唯有到了这个时候才可以重新采取批判的态度，这种批判的态度应该尽可能地类似于一个人放弃了他所一直坚持的意见之后的那种精神状态。蔑视便妨碍了这一过程的前一部分，而尊崇便妨碍了这一过程的后一部分”。我认为，这也应该是对待儒学传统的正确态度。“五四”揭示儒学对人们的精神奴役是中国现代思想史的光辉一页，其冷峻的批判和沉痛的呐喊，至今依然令人深思，依然振聋发聩。二、对儒学既要借鉴其人生智慧，又不能将其圣贤树为现代的理想人格。中国近代以来对于“新人”、“新民”的呼唤，就是要树立新的人格理想，其趋向是用平民化的自由人格来取代儒学的圣贤人格。我们可以借鉴包括儒学在内的传统的人生智慧，但重要的是构建新时代的理想人格；如果以儒学圣贤为现代人格理想，那么就不可能“道不远人”，而是“人远离道”了。对孔子、儒学只有尊崇而没有批判，其实如果把儒学的智慧作为传统资源而融入与当代中国相适应的价值体系，就不应再是原有价值的翻版。就是说，它是作为水滴流淌在当代价值体系的活水中，而它本身不是活水之源。",  # noqa E501
            "issues_suggestions": "问题：经学在过去的社会里，有装点门面之用，并没有修齐治平的功效。鲁迅说过，20世纪开始以来，孔夫子的运气很坏。但他大概做梦也不会想到从21世纪以来，孔夫子却走运了。于丹《论语》心得的火爆，是儒学走运的重要象征。近年来孔子诞辰日的祭奠已成为政府的大事，然而穿西装的主祭官员、披古服的参拜者，和献花圈的武警战士混杂在一起出场，不免有点滑稽。眼下读经可谓方兴未艾，不仅有儿童的读经学堂，还有企业老板的读经高级研讨班。鲁迅的《老调子已经唱完》写道：有些读书人不觉得孔孟古书怎样有害，其实‘正因为并不觉得怎样有害，我们这才总是觉不出这致死的毛病来。因为这是“软刀子”’。目前学术刊物上阐发儒学现代价值的文章比比皆是，最为时髦的是开列出儒学为今天建设和谐社会而设计的种种良方。傅斯年的《论学校读经》说明了中国历史上的伟大朝代创业都不靠经学，而后来提倡经学之后，国力往往衰弱，汉唐宋明都是实例，经学在过去的 社会里，有装点门面之用，并没有修齐治平的功效。这就提示人们：我们应当用什么态度来对待传统儒学?",  # noqa E501
        }

    def test_kws_gen(self):
        kws_desc = self.doc_kws_generator.analyse_info_schema(
            "reading report",
            [
                str(Path(self.pdf_root, "reading_report_p1.pdf")),
                str(Path(self.pdf_root, "reading_report_p2.pdf")),
            ],
        )
        lazyllm.LOG.info(f"kws_desc: \n{kws_desc}")
        assert kws_desc, "kws_desc is empty"

    def test_kws_extract(self):
        res = self.doc_kws_extractor.extract_doc_info(
            str(Path(self.pdf_root, "reading_report_p2.pdf")), self.kws_desc_egs01
        )
        lazyllm.LOG.info(f"res: \n{res}")
        assert res, "res is empty"

    def test_extract_with_sqlmanager(self):
        sql_manager = SqlManager(
            "SQLite",
            None,
            None,
            None,
            None,
            db_name=":memory:",
        )
        doc_kws_manager: DocToDbProcessor = DocToDbProcessor(self.llm, sql_manager)
        kws_values_temp = self.kws_value_egs01.copy()
        kws_values_temp["lazyllm_doc_path"] = str(Path(self.pdf_root, "reading_report_p2.pdf"))
        doc_kws_manager.reset_doc_info_schema(self.kws_desc_egs01)
        assert doc_kws_manager._table_class is not None
        doc_kws_manager.export_info_to_db([kws_values_temp])

    def test_kws_manager_auto_extract(self):
        sql_manager = SqlManager(
            "SQLite",
            None,
            None,
            None,
            None,
            db_name=":memory:",
        )
        doc_kws_manager: DocToDbProcessor = DocToDbProcessor(self.llm, sql_manager)
        file_paths = list(Path(self.pdf_root).glob("*.pdf"))
        doc_kws_manager.auto_docs_to_sql(file_paths)
        str_result = sql_manager.execute_query(f"select * from {doc_kws_manager.table_name}")
        print(f"str_result: {str_result}")
        assert "reading_report_p1" in str_result

        doc_kws_manager.delete_doc_paths_records(file_paths[0:1])
        str_result = sql_manager.execute_query(f"select * from {doc_kws_manager.table_name}")
        print(f"str_result: {str_result}")
        assert "reading_report_p1" not in str_result

    def test_doc_to_db(self):
        sql_manager = SqlManager(
            "SQLite",
            None,
            None,
            None,
            None,
            db_name="/home/mnt/zhangyongchao/workspace/gitlab/lazyllm-jinan/LazyLLM/test.db",
        )
        documents = lazyllm.Document(
            dataset_path=self.pdf_root,
            create_ui=False,
        )
        # It will run several minutes until update_doc_to_db done
        documents.set_doc_to_db_connection(sql_manager=sql_manager, llm=self.llm)
        schema_by_llm = documents.extract_db_schema(llm=self.llm, print_schema=True)
        assert schema_by_llm
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
        documents.set_doc_to_db_connection(
            sql_manager=sql_manager,
            llm=self.llm,
            doc_table_schma=refined_schema,
            force_refresh=True,
            export_doc_instantly=False,
        )
        documents.update_doc_to_db(llm=self.llm)

if __name__ == "__main__":
    unittest.main()
