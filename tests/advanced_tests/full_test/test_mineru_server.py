#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pytest
import requests
import unittest
import lazyllm
from lazyllm import LOG
from lazyllm.components.deploy.mineru.mineru_server_module import MineruServer
from lazyllm.tools.rag.readers.mineru_pdf_reader import MineruPDFReader

os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'

@pytest.fixture(autouse=True)
def setup_tmpdir(request, tmpdir):
    request.cls.tmpdir = tmpdir


@pytest.fixture(scope='class', autouse=True)
def setup_tmpdir_class(request, tmpdir_factory):
    request.cls.tmpdir_class = tmpdir_factory.mktemp('mineru_test')


@pytest.mark.skip(reason='Skip for env issues')
@pytest.mark.skip_on_win
@pytest.mark.skip_on_mac
@pytest.mark.usefixtures('setup_tmpdir_class')
class TestMineruServer(unittest.TestCase):
    TEST_FILES_LOCAL = {
        'pdf1': os.path.join(lazyllm.config['data_path'], 'ci_data/test_mineru/test_mineru1.pdf'),
        'pdf2': os.path.join(lazyllm.config['data_path'], 'ci_data/test_mineru/test_mineru2.pdf'),
        'pdf3': os.path.join(lazyllm.config['data_path'], 'ci_data/test_mineru/test_mineru3.pdf'),
        'docx': os.path.join(lazyllm.config['data_path'], 'ci_data/test_mineru/test_mineru.docx'),
        'pptx': os.path.join(lazyllm.config['data_path'], 'ci_data/test_mineru/test_mineru.pptx'),
    }

    @classmethod
    def setUpClass(cls):
        cls.cache_dir = str(cls.tmpdir_class.mkdir('cache'))
        cls.image_save_dir = str(cls.tmpdir_class.mkdir('images'))
        cls.server = MineruServer(
            cache_dir=cls.cache_dir,
            image_save_dir=cls.image_save_dir,
            pythonpath=None, port=31769
        )
        cls.server.start()
        cls.url = cls.server._url[:-9] + '/api/v1/pdf_parse'
        cls.test_results = {}

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'server'):
            cls.server.stop()

    def setUp(self):
        self.test_files = self.__class__.TEST_FILES
        self.validate_test_files()

    def validate_test_files(self):
        '''Validate that test files exist'''
        missing_files = []
        for file_type, file_path in self.test_files.items():
            if not os.path.exists(file_path):
                missing_files.append(f'{file_type}: {file_path}')

        if missing_files:
            error_msg = '❌ The following test files do not exist:\n' + '\n'.join(missing_files)
            error_msg += '\n\nPlease modify the test_files dictionary in the setUp method to provide correct file paths.'
            raise FileNotFoundError(error_msg)

    def post_pdf_parse(
        self,
        files,
        backend='pipeline',
        return_md=True,
        return_content_list=True,
        use_cache=False,
    ):
        '''Fix: Use correct Form data format to send request'''
        data = {
            'files': files,
            'backend': backend,
            'return_md': return_md,
            'return_content_list': return_content_list,
            'use_cache': use_cache,
        }
        try:
            resp = requests.post(self.__class__.url, data=data)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text
        except Exception as e:
            return 500, str(e)

    def check_result(self, result):
        assert isinstance(result, dict)
        assert 'result' in result, f'result: {result}'
        for res in result['result']:
            assert 'md_content' in res and 'content_list' in res

    @pytest.mark.order(1)
    def test_pdf_parsing(self):
        '''Test 1: Initial PDF parsing (create cache foundation)'''
        initial_files = [str(self.test_files['pdf1']), str(self.test_files['pdf2'])]

        status, result = self.post_pdf_parse(
            files=initial_files,
            backend='pipeline',
            return_md=True,
            return_content_list=True,
            use_cache=False,
        )
        assert status == 200, f'status: {status}, error: {result}'
        self.check_result(result)

        content_list = result['result'][0]['content_list']
        self.__class__.test_results[str(self.test_files['pdf1'])] = result['result'][0]
        types = [node.get('type', '') for node in content_list]
        assert 'text' in types
        assert 'image' in types
        assert 'table' in types
        assert 'equation' in types
        image_paths = [os.path.join(self.image_save_dir, node.get('img_path', ''))
                       for node in content_list if node['type'] == 'image']
        for image_path in image_paths:
            assert os.path.exists(image_path)

        for i, file_result in enumerate(result['result']):
            file_path = initial_files[i]
            self.__class__.test_results[file_path] = file_result

    @pytest.mark.order(2)
    def test_pdf_parsing_with_upload_files(self):
        '''Test 2: Initial upload file object parsing'''
        file_path = str(self.test_files['pdf2'])
        with open(file_path, 'rb') as f:
            upload_files = [
                (
                    'upload_files',
                    (os.path.basename(file_path), f.read(), 'application/pdf'),
                )
            ]
        data = {
            'backend': 'pipeline',
            'return_md': True,
            'return_content_list': True,
            'use_cache': False,
        }
        resp = requests.post(self.__class__.url, data=data, files=upload_files)
        status = resp.status_code
        assert status == 200, f'status: {status}, error: {resp.text}'
        result = resp.json()
        self.check_result(result)
        assert len(result['result'][0]['content_list']) == 2

    @pytest.mark.order(3)
    def test_pdf_parsing_with_cache(self):
        '''Test 3: Mixed PDF cache and new files'''
        mixed_files = [str(self.test_files['pdf1']), str(self.test_files['pdf3'])]
        status, result = self.post_pdf_parse(
            files=mixed_files,
            backend='pipeline',
            return_md=True,
            return_content_list=True,
            use_cache=True,  # Use cache
        )
        assert status == 200, f'status: {status}, error: {result}'
        self.check_result(result)
        assert len(result['result']) == 2
        content_list = result['result'][0]['content_list']
        assert content_list == self.__class__.test_results[mixed_files[0]]['content_list']

    @pytest.mark.order(4)
    def test_office_document_parsing(self):
        '''Test 4: Office document parsing functionality'''
        office_files = [str(self.test_files['docx']), str(self.test_files['pptx'])]
        for file_path in office_files:
            status, result = self.post_pdf_parse(
                files=[file_path],
                backend='pipeline',
                return_md=True,
                return_content_list=True,
                use_cache=False,
            )
            assert status in [200, 400], f'status: {status}, error: {result}'
            if status == 200:
                self.check_result(result)
            else:
                LOG.warning('Skipping office document parsing test')

    @pytest.mark.order(5)
    def test_different_backends(self):
        '''Test 6: Different backend testing'''
        backends = ['vlm-sglang-engine', 'vlm-transformers']
        test_file = str(self.test_files['pdf1'])
        for backend in backends:
            status, result = self.post_pdf_parse(
                files=[test_file],
                backend=backend,
                return_md=True,
                return_content_list=True,
                use_cache=False,
            )

            if status != 200:
                LOG.warning(f'Skipping backend: {backend}, status: {status}, error: {result}')
                continue
            self.check_result(result)

    @pytest.mark.order(6)
    def test_pdf_reader(self):
        '''Test 6: Test pdf reader (file path)'''
        pdf_reader = MineruPDFReader(self.__class__.server._url[:-9])
        pdf_path = str(self.test_files['pdf1'])
        nodes = pdf_reader(pdf_path)
        assert isinstance(nodes, list)
        assert len(nodes) == len(self.__class__.test_results[pdf_path]['content_list'])
        image_paths = [os.path.join(self.image_save_dir, node.metadata.get('image_path', ''))
                       for node in nodes if node.metadata.get('type', '') == 'image']
        for image_path in image_paths:
            print(image_path)
            assert os.path.exists(image_path)

    @pytest.mark.order(7)
    def test_pdf_reader_with_upload_files(self):
        '''Test 7: Test pdf reader (upload files)'''
        pdf_reader = MineruPDFReader(self.__class__.server._url[:-9], upload_mode=True)
        pdf_path = str(self.test_files['pdf1'])
        nodes = pdf_reader(pdf_path)
        assert isinstance(nodes, list)
        assert len(nodes) == len(self.__class__.test_results[pdf_path]['content_list'])

    @pytest.mark.order(8)
    def test_pdf_reader_with_post_func(self):
        '''Test 8: Test pdf reader's post-processing function post_func functionality'''
        def test_post_func(nodes):
            for node in nodes:
                node._content += '[after_process]'
            return nodes

        pdf_reader = MineruPDFReader(
            self.__class__.server._url[:-9],
            post_func=test_post_func
        )

        pdf_path = str(self.test_files['pdf1'])
        nodes = pdf_reader(pdf_path)

        nodes = pdf_reader(pdf_path)
        assert isinstance(nodes, list)
        for node in nodes:
            assert node._content.endswith('[after_process]')
