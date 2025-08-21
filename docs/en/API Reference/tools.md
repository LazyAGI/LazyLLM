::: lazyllm.tools.IntentClassifier
    members:
    - intent_promt_hook
    - post_process_result
    exclude-members:

::: lazyllm.tools.Document
    members:
    exclude-members:

::: lazyllm.tools.rag.store.ChromadbStore
    members:
    exclude-members:

::: lazyllm.tools.rag.store.MilvusStore
    members:
    exclude-members:
    
::: lazyllm.tools.rag.readers.ReaderBase
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.PandasCSVReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.PandasExcelReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.PDFReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.PPTXReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.VideoAudioReader
    members:
    exclude-members:

::: lazyllm.tools.SqlManager
    members: 
        - get_session
        - check_connection
        - set_desc
        - get_all_tables
        - get_table_orm_class
        - execute_commit
        - execute_query
        - create_table
        - drop_table
        - insert_values
    exclude-members:

::: lazyllm.tools.Reranker
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.readerBase.LazyLLMReaderBase
    members:
	exclude-members:

::: lazyllm.tools.rag.component.bm25
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaItem
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocGenreAnalyser
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaAnalyser
    members: analyse_info_schema
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoExtractor
    members: extract_doc_info
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoExtractor
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocToDbProcessor
    members: 
        - extract_info_from_docs
        - analyze_info_schema_by_llm
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.extract_db_schema_from_files

::: lazyllm.tools.rag.readers.DocxReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.EpubReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.HWPReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.ImageReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.IPYNBReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.MineruPDFReader
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.MarkdownReader
    members:
        - remove_images
        - remove_hyperlinks
    exclude-members:

::: lazyllm.tools.rag.readers.MboxReader
    members:
	exclude-members:

::: lazyllm.tools.SqlCall
    members: 
        - sql_query_promt_hook
        - sql_explain_prompt_hook
        - extract_sql_from_response
	exclude-members:

::: lazyllm.tools.rag.default_index.DefaultIndex
    members:
        - update
        - remove
        - query
    exclude-members: 

::: lazyllm.tools.Reranker
    members: [register_reranker]
    exclude-members: forward

::: lazyllm.tools.Retriever
    members:
    exclude-members: forward

::: lazyllm.tools.rag.retriever.TempDocRetriever
    members: [create_node_group, add_subretriever]
    exclude-members: 

::: lazyllm.tools.rag.retriever.UrlDocument
    members: [find]
    exclude-members: 

::: lazyllm.tools.rag.DocManager
    members:
	exclude-members:

::: lazyllm.tools.rag.utils.SqliteDocListManager
    members:
      - table_inited
      - get_status_cond_and_params
      - validate_paths
      - update_need_reparsing
      - list_files
      - get_docs
      - set_docs_new_meta
      - fetch_docs_changed_meta
      - list_all_kb_group
      - add_kb_group
      - list_kb_group_files
      - delete_unreferenced_doc
      - get_docs_need_reparse
      - get_existing_paths_by_pattern
      - update_file_message
      - update_file_status
      - add_files_to_kb_group
      - delete_files_from_kb_group
      - get_file_status
      - update_kb_group
      - release
	exclude-members:

::: lazyllm.tools.rag.data_loaders.DirectoryReader
    members: load_data
	exclude-members:

::: lazyllm.tools.SentenceSplitter
    members:
    exclude-members:

::: lazyllm.tools.LLMParser
    members:
    exclude-members:

lazyllm.tools.rag.transform.NodeTransform
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.TransformArgs
    members:
    exclude-members:

::: lazyllm.tools.rag.similarity.register_similarity
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_node.DocNode
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_processor.DocumentProcessor
    members: register_algorithm, drop_algorithm
::: lazyllm.tools.rag.doc_node.QADocNode
    members: get_text
    exclude-members:

::: lazyllm.tools.rag.dataReader.SimpleDirectoryReader
    members:
    exclude-members:

::: lazyllm.tools.rag.dataReader.FileReader
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.FuncNodeTransform
    members: transform
    exclude-members:

::: lazyllm.tools.rag.web.DocWebModule
    members:
    exclude-members:    
    
::: lazyllm.tools.WebModule
    members:
    exclude-members: forward

::: lazyllm.tools.CodeGenerator
    members: [choose_prompt]
    exclude-members: forward

::: lazyllm.tools.ParameterExtractor
    members: 
    exclude-members: forward

::: lazyllm.tools.QustionRewrite
    members: choose_prompt
    exclude-members: forward

::: lazyllm.tools.agent.toolsManager.ToolManager
    members: 
    exclude-members: forward

::: lazyllm.tools.ModuleTool
    members: 
    exclude-members: forward

::: lazyllm.tools.FunctionCall
    members: 
    exclude-members: forward

::: lazyllm.tools.FunctionCallFormatter
    members: 
    exclude-members: forward

::: lazyllm.tools.FunctionCallAgent
    members: 
    exclude-members: forward

::: lazyllm.tools.ReactAgent
    members: 
    exclude-members: forward

::: lazyllm.tools.PlanAndSolveAgent
    members: 
    exclude-members: forward

::: lazyllm.tools.ReWOOAgent
    members: 
    exclude-members: forward

::: lazyllm.tools.rag.smart_embedding_index.SmartEmbeddingIndex
    members: update, remove, query
    exclude-members:

::: lazyllm.tools.rag.doc_node.ImageDocNode
    members: do_embedding, get_content, get_text
    exclude-members:

::: lazyllm.tools.rag.transform.AdaptiveTransform
    members: transform
    exclude-members:

::: lazyllm.tools.rag.rerank.ModuleReranker
    members: forward
    exclude-members:
::: lazyllm.tools.rag.utils.DocListManager
    members: 
    exclude-members: 
::: lazyllm.tools.rag.global_metadata.GlobalMetadataDesc
    members: 
    exclude-members: 

::: lazyllm.tools.rag.IndexBase.update
    members:
	exclude-members: 

::: lazyllm.tools.rag.IndexBase.remove
    members:
	exclude-members: 

::: lazyllm.tools.rag.IndexBase.query
    members:
	exclude-members:

::: lazyllm.tools.rag.index_base.IndexBase
    members: 

::: lazyllm.tools.BaseEvaluator
    members: 
    exclude-members:

::: lazyllm.tools.ResponseRelevancy
    members: 
    exclude-members:    

::: lazyllm.tools.Faithfulness
    members: 
    exclude-members: 

::: lazyllm.tools.LLMContextRecall
    members: 
    exclude-members: 

::: lazyllm.tools.NonLLMContextRecall
    members: 
    exclude-members:

::: lazyllm.tools.ContextRelevance
    members: 
    exclude-members:

::: lazyllm.tools.HttpRequest
    members: 
    exclude-members:

::: lazyllm.tools.JobDescription
    members: 
    exclude-members:

::: lazyllm.tools.DBManager
    members: execute_query
    exclude-members:

::: lazyllm.tools.MongoDBManager
    members: 
    exclude-members:

::: lazyllm.tools.HttpTool
    members: 
    exclude-members:

::: lazyllm.tools.agent.functionCall.StreamResponse
    members: 
    exclude-members:

::: lazyllm.tools.MCPClient
    members: [call_tool, list_tools, get_tools, aget_tools, deploy]
    exclude-members:

::: lazyllm.tools.tools.GoogleSearch
    members: forward

::: lazyllm.tools.tools.tencent_search.TencentSearch
    members: 
    exclude-members:

::: lazyllm.tools.rag.web.WebUi
    members: 
    exclude-members:

::: lazyllm.tools.http_request.http_executor_response.HttpExecutorResponse
    members: extract_file, get_content_type
    exclude-members: