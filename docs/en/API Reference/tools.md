::: lazyllm.tools.Document
    members:
    exclude-members:

::: lazyllm.tools.rag.readers.ReaderBase
    members:
	exclude-members:

::: lazyllm.tools.Reranker
    members:
    exclude-members: forward

::: lazyllm.tools.Retriever
    members:
    exclude-members: forward

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

::: lazyllm.tools.SentenceSplitter
    members:
    exclude-members:

::: lazyllm.tools.LLMParser
    members:
    exclude-members:

::: lazyllm.tools.WebModule
    members:
    exclude-members: forward

::: lazyllm.tools.ToolManager
    members: 
    exclude-members: forward

::: lazyllm.tools.FunctionCall
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

::: lazyllm.tools.IntentClassifier
    members: 
    exclude-members:
