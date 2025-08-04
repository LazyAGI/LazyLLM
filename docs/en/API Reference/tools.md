::: lazyllm.tools.Document
    members:
    exclude-members:

::: lazyllm.tools.rag.store.ChromadbStore
    members:
    exclude-members:

::: lazyllm.tools.rag.store.MilvusStore
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

::: lazyllm.tools.rag.component.bm25.BM25
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaItem
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocGenreAnalyser
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaAnalyser
    members:
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

::: lazyllm.tools.rag.readers.MagicPDFReader
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

::: lazyllm.tools.rag.component.bm25.BM25
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaItem
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocGenreAnalyser
    members:
	exclude-members:

::: lazyllm.tools.rag.doc_to_db.DocInfoSchemaAnalyser
    members:
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

::: lazyllm.tools.rag.readers.MagicPDFReader
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

::: lazyllm.tools.Reranker
    members: register_reranker
    members: register_reranker
    exclude-members: forward

::: lazyllm.tools.Retriever
    members:
    exclude-members: forward

::: lazyllm.tools.rag.retriever.TempDocRetriever
    members:
    exclude-members: 

::: lazyllm.tools.rag.retriever.TempDocRetriever
    members:
    exclude-members: 

::: lazyllm.tools.rag.DocManager
    members:
	exclude-members:

::: lazyllm.tools.IndexBase.update
    members:
	exclude-members: 

::: lazyllm.tools.IndexBase.remove
    members:
	exclude-members: 

::: lazyllm.tools.IndexBase.query
    members:
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

::: lazyllm.tools.rag.dataReader.SimpleDirectoryReader
    members:
    exclude-members:

::: lazyllm.tools.rag.dataReader.FileReader
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

::: lazyllm.tools.rag.dataReader.SimpleDirectoryReader
    members:
    exclude-members:

::: lazyllm.tools.rag.dataReader.FileReader
    members:
    exclude-members:
    
    
::: lazyllm.tools.WebModule
    members:
    exclude-members: forward

::: lazyllm.tools.CodeGenerator
    members: 
    exclude-members: forward

::: lazyllm.tools.ParameterExtractor
    members: 
    exclude-members: forward

::: lazyllm.tools.QustionRewrite
    members: 
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

::: lazyllm.tools.IntentClassifier
    members: 
    exclude-members:
::: lazyllm.tools.rag.utils.DocListManager
    members: 
    exclude-members: 
::: lazyllm.tools.rag.global_metadata.GlobalMetadataDesc
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
    members: 
    exclude-members:

::: lazyllm.tools.MongoDBManager
    members: 
    exclude-members:
::: lazyllm.tools.rag.utils.DocListManager
    members: 
    exclude-members: 
::: lazyllm.tools.rag.global_metadata.GlobalMetadataDesc
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
    members: 
    exclude-members:

::: lazyllm.tools.MongoDBManager
    members: 
    exclude-members:

::: lazyllm.tools.HttpTool
    members: 
    exclude-members: