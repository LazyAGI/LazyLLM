from lazyllm.tools.rag.parsing_service import DocumentProcessor
from lazyllm.tools.rag.parsing_service.base import AddDocRequest, FileInfo, TaskType, TransferParams


def test_resolve_add_task_type_returns_transfer_for_copy_request():
    request = AddDocRequest(
        algo_id='general_algo',
        kb_id='kb_source',
        file_infos=[FileInfo(
            file_path='/tmp/source.pdf',
            doc_id='doc_source',
            transfer_params=TransferParams(
                mode='cp',
                target_algo_id='general_algo',
                target_doc_id='doc_target',
                target_kb_id='kb_target',
            ),
        )],
    )

    assert DocumentProcessor._Impl._resolve_add_task_type(request) == TaskType.DOC_TRANSFER.value


def test_resolve_add_task_type_keeps_add_for_regular_upload():
    request = AddDocRequest(
        algo_id='general_algo',
        kb_id='kb_source',
        file_infos=[FileInfo(file_path='/tmp/source.pdf', doc_id='doc_source')],
    )

    assert DocumentProcessor._Impl._resolve_add_task_type(request) == TaskType.DOC_ADD.value
