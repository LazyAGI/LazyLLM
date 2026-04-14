import json
import os
import shutil
import tempfile

from lazyllm import config
from lazyllm.tools.data import domain_finetune


class SimpleMockLLM:

    def __init__(self, response):
        self._response = response
        self._serve = self

    def share(self, prompt=None, format=None, stream=None, history=None):
        return self._serve

    def prompt(self, system_prompt):
        return self._serve

    def start(self):
        return self._serve

    def __call__(self, prompt):
        return self._response


class TestDomainFinetuneOperators:
    def setup_method(self):
        self.root_dir = tempfile.mkdtemp()
        self.keep_dir = config['data_process_path']
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.root_dir
        config.refresh()

    def teardown_method(self):
        os.environ['LAZYLLM_DATA_PROCESS_PATH'] = self.keep_dir
        config.refresh()
        if os.path.exists(self.root_dir):
            shutil.rmtree(self.root_dir)

    def test_rename_key_remove_input(self):
        op = domain_finetune.rename_key(
            input_key='raw', output_key='cleaned_content', remove_input=True,
        )
        r = op([{'raw': 'text', 'meta': 1}])[0]
        assert 'raw' not in r and r['cleaned_content'] == 'text' and r['meta'] == 1

    def test_rename_key_keep_input(self):
        op = domain_finetune.rename_key(
            input_key='raw', output_key='cleaned_content', remove_input=False,
        )
        r = op([{'raw': 'x'}])[0]
        assert r['raw'] == 'x' and r['cleaned_content'] == 'x'

    def test_prepare_load_path(self):
        op = domain_finetune.prepare_load_path(_save_data=False, _concurrency_mode='single')
        assert op([{'_type': 'text', '_raw_path': '/a'}])[0]['_path_to_load'] == '/a'
        assert op([{'_type': 'html', '_markdown_path': '/m'}])[0]['_path_to_load'] == '/m'
        assert op([{'_type': 'unknown'}])[0]['_path_to_load'] == ''

    def test_normalize_text_strip_and_punct(self):
        op_strip = domain_finetune.normalize_text(
            input_key='content', strip_whitespace=True, fix_chinese_punct=False,
        )
        assert op_strip([{'content': '  ab  '}])[0]['content'] == 'ab'
        op_punct = domain_finetune.normalize_text(
            input_key='content', fix_chinese_punct=True, strip_whitespace=False,
        )
        r = op_punct([{'content': '\uff0c'}])[0]['content']
        assert r == ','

    def test_extract_content_text(self):
        op = domain_finetune.extract_content_text(
            input_key='content', output_key='_filter_text',
            _save_data=False, _concurrency_mode='single',
        )
        d1 = {'content': {'messages': [{'content': 'a'}, {'content': 'b'}]}}
        assert op([d1])[0]['_filter_text'] == 'a b'
        d2 = {'content': {'instruction': 'i', 'input': 'in', 'output': 'out'}}
        assert 'i' in op([d2])[0]['_filter_text'] and 'out' in op([d2])[0]['_filter_text']
        d3 = {'content': 'plain'}
        assert op([d3])[0]['_filter_text'] == 'plain'

    def test_merge_context_and_question(self):
        op = domain_finetune.merge_context_and_question(
            question_key='question', context_key='context', target_key='question',
            drop_context=True,
            _save_data=False, _concurrency_mode='single',
        )
        sep = '\\n\\n'
        ctx, q = '上下文片段', '具体问题内容'
        r = op([{'question': q, 'context': ctx}])[0]
        assert r['question'] == f'Context: {ctx}{sep}Question: {q}'
        assert 'context' not in r
        r2 = op([{'question': '', 'context': ctx}])[0]
        assert r2['question'] == f'Context: {ctx}'

    def test_dataset_format_normalizer_alpaca_and_qa(self):
        op = domain_finetune.DatasetFormatNormalizer(
            output_key='content', text_key='_filter_text',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'instruction': 'i', 'input': 'in', 'output': 'out'}])[0]
        assert r['content']['output'] == 'out'
        assert '_filter_text' in r
        r2 = op([{'question': 'Q?', 'answer': 'A.'}])[0]
        assert r2['content']['input'] == 'Q?' and r2['content']['output'] == 'A.'

    def test_dataset_format_normalizer_messages(self):
        op = domain_finetune.DatasetFormatNormalizer(
            output_key='content', _save_data=False, _concurrency_mode='single',
        )
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'u'},
            {'role': 'assistant', 'content': 'a'},
        ]
        r = op([{'messages': msgs}])[0]
        # system + 一轮 user/assistant 会折叠为 Alpaca，而非保留 messages 列表
        assert r['content']['input'] == 'u'
        assert r['content']['output'] == 'a'
        assert r['content']['instruction'] == 'sys'

    def test_dataset_format_normalizer_field_mapping(self):
        op = domain_finetune.DatasetFormatNormalizer(
            output_key='content',
            field_mapping={'q': 'question', 'a': 'answer'},
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'q': '问题足够长吗', 'a': '回答'}])[0]
        assert r['content']['input'] == '问题足够长吗'
        assert r['content']['output'] == '回答'

    def test_dataset_format_normalizer_skip_existing_dict(self):
        op = domain_finetune.DatasetFormatNormalizer(
            output_key='content', text_key='_filter_text',
            _save_data=False, _concurrency_mode='single',
        )
        data = {'content': {'instruction': 'i', 'input': '', 'output': 'o'}}
        r = op([data])[0]
        assert r['_filter_text']

    def test_hash_deduplicator(self):
        op = domain_finetune.HashDeduplicator(
            input_key='content', _save_data=False,
        )
        rows = [{'content': 'same'}, {'content': 'same'}, {'content': 'other'}]
        out = op(rows)
        assert len(out) == 2

    def test_conversation_list_expander(self):
        op = domain_finetune.ConversationListExpander(
            list_key='data',
            min_question_chars=4,
            min_answer_chars=10,
            _save_data=False,
        )
        long_a = 'a' * 12
        rows = [{'data': ['问：hiqq', f'答：{long_a}', '问：skip', '答：short']}]
        out = op(rows)
        assert len(out) == 1
        assert out[0]['input'].startswith('hiqq') and long_a in out[0]['output']

    def test_domain_format_alpaca_string_and_dict(self):
        op = domain_finetune.DomainFormatAlpaca(
            input_key='content', output_key='formatted_text',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'body'}])[0]
        assert r['formatted_text']['output'] == 'body'
        r2 = op([{'content': {'instruction': 'i', 'input': 'x', 'output': 'y'}}])[0]
        assert r2['formatted_text']['input'] == 'x'

    def test_domain_format_sharegpt(self):
        op = domain_finetune.DomainFormatShareGPT(
            input_key='content', output_key='ft',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'hi'}])[0]
        roles = [m['role'] for m in r['ft']['messages']]
        assert roles == ['system', 'user', 'assistant']

    def test_domain_format_raw(self):
        op = domain_finetune.DomainFormatRaw(
            input_key='content', output_key='ft',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': {'k': 1}}])[0]
        assert r['ft']['system'] and r['ft']['content'] == {'k': 1}

    def test_domain_format_chatml(self):
        op = domain_finetune.DomainFormatChatML(
            input_key='content', output_key='ft',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': 'u'}])[0]
        text = r['ft']['text']
        assert '<|im_start|>system' in text and '<|im_end|>' in text

    def test_train_val_test_splitter(self):
        op = domain_finetune.TrainValTestSplitter(
            train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=0,
            _save_data=False,
        )
        items = [{'i': n} for n in range(10)]
        out = op(items)
        assert len(out['train']) + len(out['validation']) + len(out['test']) == 10
        assert len(out['validation']) >= 1 and len(out['test']) >= 1

    def test_train_val_test_splitter_empty(self):
        op = domain_finetune.TrainValTestSplitter(_save_data=False)
        out = op([])
        assert out == {'train': [], 'validation': [], 'test': []}

    def test_llm_data_extractor_qa_parse(self):
        ans = '答案内容' * 5
        payload = {'qa_pairs': [{'question': '测试问题内容？', 'answer': ans}]}
        mock = SimpleMockLLM(json.dumps(payload, ensure_ascii=False))
        op = domain_finetune.LLMDataExtractor(
            llm=mock, extract_format='qa', input_key='content',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': '原文'}])[0]
        assert len(r['_extracted_samples']) == 1
        assert r['_extracted_samples'][0]['input'] == '测试问题内容？'

    def test_llm_data_extractor_instruction_format(self):
        payload = {
            'samples': [
                {'instruction': '做某事', 'input': '参数', 'output': '完成输出结果'},
            ],
        }
        mock = SimpleMockLLM(json.dumps(payload, ensure_ascii=False))
        op = domain_finetune.LLMDataExtractor(
            llm=mock, extract_format='instruction', input_key='content',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'content': '任务描述文本'}])[0]
        assert r['_extracted_samples'][0]['output'] == '完成输出结果'

    def test_llm_field_mapper(self):
        mock = SimpleMockLLM('{"instruction":"i","input":"in","output":"outvaluelongenough"}')
        op = domain_finetune.LLMFieldMapper(
            llm=mock, output_key='content', text_key='_filter_text',
            _save_data=False, _concurrency_mode='single',
        )
        r = op([{'title': 't', 'body': 'b'}])[0]
        assert r['content']['output'] == 'outvaluelongenough'
        assert '_filter_text' in r

    def test_output_content_filter_forward(self):
        op = domain_finetune.OutputContentFilter(
            input_key='content', min_output_chars=10,
            _save_data=False, _concurrency_mode='single',
        )
        assert op.forward({'content': {'output': 'short'}}) is None
        long_out = 'x' * 20
        kept = op.forward({'content': {'output': long_out}})
        assert kept['content']['output'] == long_out

    def test_input_output_ratio_filter_forward(self):
        op = domain_finetune.InputOutputRatioFilter(
            input_key='content', min_ratio=0.5,
            _save_data=False, _concurrency_mode='single',
        )
        dropped = op.forward({
            'content': {'input': '输入很长很长很长', 'output': '短'},
        })
        assert dropped is None
        kept = op.forward({
            'content': {'input': '短', 'output': '输出足够长'},
        })
        assert kept is not None

    def test_sample_expander(self):
        op = domain_finetune.SampleExpander(
            samples_key='_extracted_samples',
            output_key='content',
            keep_original_keys=['id'],
            drop_empty_extraction=False,
            _save_data=False,
        )
        base = {'id': 1, '_extracted_samples': [{'instruction': '', 'input': 'a', 'output': 'b'}]}
        out = op([base])
        assert len(out) == 1 and out[0]['id'] == 1 and out[0]['content']['output'] == 'b'

    def test_sample_expander_drop_empty(self):
        op = domain_finetune.SampleExpander(
            samples_key='_extracted_samples',
            output_key='content',
            drop_empty_extraction=True,
            _save_data=False,
        )
        out = op([{'_extracted_samples': []}])
        assert out == []
