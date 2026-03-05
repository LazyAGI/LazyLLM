from ..base_data import data_register
from lazyllm.tools.rag import MineruPDFReader
from lazyllm import LOG

Pdf2Qa = data_register.new_group('pdf2Qa')


class Pdf2Md(Pdf2Qa):
    def __init__(self,
                 input_key='pdf_path',
                 output_key='docs',
                 reader_url=None,
                 backend='vlm-vllm-async-engine',
                 upload_mode=True,
                 use_cache=False,
                 **kwargs):

        super().__init__(_concurrency_mode='thread', **kwargs)
        if not reader_url:
            raise ValueError('You must pass in a reader_url.')

        self.input_key = input_key
        self.output_key = output_key
        self.use_cache = use_cache

        self.reader = MineruPDFReader(
            url=reader_url,
            backend=backend,
            upload_mode=upload_mode
        )

    def forward(self, data):
        pdf_path = data.get(self.input_key)
        if not pdf_path:
            return None

        try:
            docs = self.reader(
                file=pdf_path,
                use_cache=self.use_cache
            )
            data[self.output_key] = docs

        except Exception as e:
            LOG.warning(f'PDF read failed: {e}')
            data[self.output_key] = None
        return data
