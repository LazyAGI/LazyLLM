import sys
import argparse

# lazyllm run xx.json / xx.dsl / xx.lazyml
# lazyllm run chatbot --model=xx --framework=xx --source=xx
# lazyllm run rag --model=xx --framework=xx --source=xx --documents=''

def chatbot(llm):
    import lazyllm
    lazyllm.WebModule(llm, port=range(20000, 25000)).start().wait()


def rag(llm, docpath):
    import lazyllm
    from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker
    prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. In this '
              'task, you need to provide your answer based on the given context and question.')

    documents = Document(dataset_path=docpath, embed=lazyllm.OnlineEmbeddingModule(), create_ui=False)
    documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

    with pipeline() as ppl:
        with parallel().sum as ppl.prl:
            ppl.prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
            ppl.prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

        ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
        ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]),
                                                   query=query)) | bind(query=ppl.input)
        ppl.llm = llm.prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))

    lazyllm.WebModule(ppl, port=range(20000, 25000)).start().wait()


def run(commands):
    if not commands:
        print('Usage:\n  lazyllm run graph.json\n  lazyllm run chatbot\n  lazyllm run rag\n')

    parser = argparse.ArgumentParser(description='lazyllm deploy command')
    parser.add_argument('command', type=str, help='command')

    args, _ = parser.parse_known_args(commands)

    if args.command in ('chatbot', 'rag'):
        parser.add_argument('--model', type=str, default=None, help='model name')
        parser.add_argument('--source', type=str, default=None, help='Online model source, conflict with framework',
                            choices=['openai', 'sensenova', 'glm', 'kimi', 'qwen', 'doubao'])
        parser.add_argument('--framework', type=str, default=None, help='Online model source, conflict with source',
                            choices=['lightllm', 'vllm', 'lmdeploy'])
        if args.command == 'rag':
            parser.add_argument('--documents', required=True, type=str, help='document absolute path')

        args = parser.parse_args(commands)
        import lazyllm
        llm = lazyllm.AutoModel(args.model, args.source, args.framework)

        if args.command == 'chatbot':
            chatbot(llm)
        elif args.command == 'rag':
            rag(llm, args.documents)
    else:
        print('lazyllm run is not ready yet.')
        sys.exit(0)
