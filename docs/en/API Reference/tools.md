::: lazyllm.tools.IntentClassifier
    members:
    - intent_promt_hook
    - post_process_result
    exclude-members:

::: lazyllm.tools.Document
    members: [connect_sql_manager, get_sql_manager, extract_db_schema, update_database, create_kb_group, activate_group, activate_groups, get_store, get_embed, register_index, find, clear_cache, create_node_group, find_parent, find_children, register_global_reader, add_reader]
    exclude-members:

::: lazyllm.tools.rag.store.ChromaStore
    members: [dir, connect, upsert, delete, get, search]
    exclude-members:

::: lazyllm.tools.rag.store.MilvusStore
    members:
    exclude-members:

::: lazyllm.tools.rag.store.hybrid.hybrid_store.HybridStore
    members: connect, upsert, delete, get, search
    exclude-members:

::: lazyllm.tools.rag.store.hybrid.oceanbase_store.OceanBaseStore
    members: connect, upsert, delete, get, search
    exclude-members:

::: lazyllm.tools.rag.store.ElasticSearchStore
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

::: lazyllm.tools.rag.readers.readerBase.TxtReader
    members:
	exclude-members:

::: lazyllm.tools.rag.component.bm25.BM25
    members: retrieve
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaItem
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocGenreAnalyser
    members: gen_detection_query, analyse_doc_genre
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
        - clear
        - export_info_to_db
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
    members: document, list_kb_groups, add_files, reparse_files
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
      - get_status_cond_and_params
	exclude-members:

::: lazyllm.tools.rag.data_loaders.DirectoryReader
    members: load_data
	exclude-members:

::: lazyllm.tools.rag.transform.sentence.SentenceSplitter
    members: [split_text, from_tiktoken_encoder, from_huggingface_tokenizer]
    exclude-members:

::: lazyllm.tools.rag.transform.character.CharacterSplitter
    memebers:
    exclude-members:

::: lazyllm.tools.rag.transform.recursive.RecursiveSplitter
    memebers:
    exclude-members:

::: lazyllm.tools.rag.transform.markdown.MarkdownSplitter
    memebers:
    exclude-members:

::: lazyllm.tools.rag.transform.code.CodeSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.HTMLSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.JSONSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.JSONLSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.YAMLSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.ProgrammingSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.code.XMLSplitter
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.base.NodeTransform
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.factory.LLMParser
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.factory.TransformArgs
    members:
    exclude-members:

::: lazyllm.tools.rag.transform.factory.FuncNodeTransform
    members: transform
    exclude-members:

::: lazyllm.tools.rag.transform.factory.AdaptiveTransform
    members: transform
    exclude-members:

::: lazyllm.tools.rag.similarity.register_similarity
    members:
    exclude-members:

::: lazyllm.tools.rag.doc_node.DocNode
    members: get_children_str, get_parent_id, get_content, to_dict, set_embedding
    exclude-members:

::: lazyllm.tools.rag.doc_node.QADocNode
    members: get_text
    exclude-members:

::: lazyllm.tools.rag.dataReader.SimpleDirectoryReader
    members: [load_file, find_extractor_by_file, get_default_reader, add_post_action_for_default_reader]
    exclude-members:

::: lazyllm.tools.rag.dataReader.FileReader
    members:
    exclude-members:

::: lazyllm.tools.rag.web.DocWebModule
    members:
    exclude-members:    

::: lazyllm.tools.rag.parsing_service.server.DocumentProcessor
    members: [start, register_algorithm, drop_algorithm]
    exclude-members:

::: lazyllm.tools.rag.parsing_service.worker.DocumentProcessorWorker
    members: [start, stop]
    exclude-members:

::: lazyllm.tools.WebModule
    members:
    exclude-members: forward

::: lazyllm.tools.CodeGenerator
    members: [choose_prompt]
    exclude-members: forward

::: lazyllm.tools.ParameterExtractor
    members: [choose_prompt]
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

::: lazyllm.tools.rag.store.hybrid.MapStore
    members: connect, upsert, delete, get, search
    exclude-members:

::: lazyllm.tools.rag.store.segment.opensearch_store.OpenSearchStore
    members: connect, upsert, delete, get, search
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
    members: process_one_data, validate_inputs_key, batch_process, save_res
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

::: lazyllm.tools.tools.TencentSearch
    members: 
    exclude-members:

::: lazyllm.tools.rag.web.WebUi
    members: 
    exclude-members:

::: lazyllm.tools.http_request.http_executor_response.HttpExecutorResponse
    members: extract_file, get_content_type
    exclude-members:


::: lazyllm.tools.StreamCallHelper
    members: [split_text]
    exclude-members:

::: lazyllm.tools.rag.LazyLLMStoreBase
    members: [connect, upsert, delete, get, search]
    exclude-members:


::: lazyllm.tools.rag.doc_impl.DocImpl
    members: create_global_node_group, create_node_group, register_global_reader, register_index, add_reader, worker, activate_group, active_node_groups, retrieve, find, find_parent, find_children, clear_cache
    exclude-members:

::: lazyllm.tools.services.client.ClientBase
    members: uniform_status
    exclude-members:


::: lazyllm.tools.services.services.ServerBase
    members: authorize_current_user
    exclude-members:

::: lazyllm.tools.infer_service.serve.InferServer
    members: create_job, cancel_job, list_jobs, get_job_info, get_job_log
    exclude-members:

::: lazyllm.tools.rag.store.hybrid.sensecore_store.SenseCoreStore
    members:
    exclude-members: