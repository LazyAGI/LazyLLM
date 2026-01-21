RAG Parsing Service Example

This folder shows how to use a standalone parsing service and connect it to
Document/Retriever flows. The scripts are:
- server_with_worker.py: single service that runs server + worker together
- server_and_separate_workers.py: server and worker started separately
- document.py: Document configured to use the parsing service
- retriever_using_url.py: remote Retriever usage via Document URL

1) Standalone parsing service setup

Start the parsing service first. You can run it with an embedded worker
or run the worker separately.

- Embedded worker:
  - Run `server_with_worker.py` to start the server with a local worker.
- Separate worker:
  - Run `server_and_separate_workers.py` to start the server.
  - Start one or more workers using `DocumentProcessorWorker`.

2) Document registration and remote access

When creating a Document, set the parsing service URL to register and update
algorithm info through the service. The Document can run in server mode so
others can access it remotely by URL, and Retrievers can use that URL directly.

See `document.py` for the setup:
- `manager=DocumentProcessor(url="http://0.0.0.0:9966")` points to the parsing service.
- `server=9977` exposes the Document as a service.

Then use `retriever_using_url.py` to create:
`Document(url="http://127.0.0.1:9977", name="doc_example")` and run a Retriever.

Note: This mode requires independently deployed storage services such as
OpenSearch or Milvus standalone.

3) Server-worker model and scaling

The parsing service uses a server-worker architecture. Workers can be deployed
independently and scaled out, including a Ray-based extension for elasticity.
Make sure the server and all workers share the same database configuration
(`db_config`), otherwise tasks will not be coordinated correctly.
