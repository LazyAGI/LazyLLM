import lazyllm
from lazyllm.components import register as comp_register
from lazyllm.common.registry import LazyLLMRegisterMetaClass
from lazyllm.components.core import ComponentBase
from lazyllm.tools import fc_register
import pytest

def orig_func(self):
    pass

class TestRegistry:

    def test_compoments_register(self):
        assert not hasattr(lazyllm, 'test1')
        comp_register.new_group('test1')
        assert hasattr(lazyllm, 'test1')

        @comp_register('test1')
        def test_add(a, b):
            return a + b

        @comp_register('test1')
        def TestSub(a, b):
            return a - b

        assert hasattr(lazyllm.test1, 'test_add')
        assert lazyllm.test1.test_add()(1, 2) == 3
        assert hasattr(lazyllm.test1, 'testsub')
        assert hasattr(lazyllm.test1, 'TestSub')
        assert lazyllm.test1.testsub()(3, 2) == 1

    def test_compoments_register_subgroup(self):
        assert not hasattr(lazyllm, 'test2')

        class LazyLLMTest2Base(ComponentBase):
            pass

        assert hasattr(lazyllm, 'test2')
        assert not hasattr(lazyllm.test2, 'test3')

        class LazyLLMTest3Base(LazyLLMTest2Base):
            pass

        assert hasattr(lazyllm.test2, 'test3')

        @comp_register('test2.test3')
        def test_add(a, b):
            return a + b + 1

        assert lazyllm.test2.test3.test_add()(1, 2) == 4

    def test_compoments_register_with_default_group(self):
        assert not hasattr(lazyllm, 'test3')
        comp_register.new_group('test3')
        comp_register._default_group = 'test3'
        assert not hasattr(lazyllm.test3, 'test_square')

        @comp_register
        def test_square(x):
            return x * x

        assert lazyllm.test3.test_square()(4) == 16

    def test_capital_letter(self):
        class LazyLLMTest5Base(ComponentBase):
            pass

        assert hasattr(lazyllm, 'test5')
        assert hasattr(lazyllm, 'Test5')

        class a(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'a')
        assert hasattr(lazyllm.test5, 'A')

        class B(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'b')
        assert hasattr(lazyllm.test5, 'B')

        class cTest5(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'c')
        assert hasattr(lazyllm.test5, 'C')
        assert hasattr(lazyllm.test5, 'ctest5')
        assert hasattr(lazyllm.test5, 'CTest5')

        class DTest5(LazyLLMTest5Base):
            pass

        assert hasattr(lazyllm.test5, 'd')
        assert hasattr(lazyllm.test5, 'D')
        assert hasattr(lazyllm.test5, 'dtest5')
        assert hasattr(lazyllm.test5, 'DTest5')

    def test_register(self):
        registered_func = fc_register('tool')(orig_func)
        assert registered_func == orig_func

    def test_register_with_new_func_name(self):
        new_func_name = 'another_func_name'
        registered_func = fc_register('tool')(orig_func, new_func_name)
        assert registered_func != orig_func
        assert registered_func.__name__ == new_func_name

    def test_fc_register_execute_in_sandbox_flag(self):
        @fc_register('tool', execute_in_sandbox=False)
        def sandbox_disabled_tool(x: int):
            '''
            Simple tool for testing.

            Args:
                x (int): input value.
            '''
            return x

        tool = lazyllm.tool.sandbox_disabled_tool()
        assert tool.execute_in_sandbox is False

    def test_fc_register_input_files_parm(self):
        @fc_register('tool', input_files_parm='file_path')
        def input_parm_tool(file_path: str):
            '''
            Tool with input file param.

            Args:
                file_path (str): path to input file.
            '''
            return file_path

        tool = lazyllm.tool.input_parm_tool()
        assert tool.input_files_parm == 'file_path'
        # defaults should remain
        assert tool.output_files_parm is None
        assert tool.output_files == []
        assert tool.execute_in_sandbox is True

    def test_fc_register_output_files_parm(self):
        @fc_register('tool', output_files_parm='out_path')
        def output_parm_tool(out_path: str):
            '''
            Tool with output file param.

            Args:
                out_path (str): path to output file.
            '''
            return out_path

        tool = lazyllm.tool.output_parm_tool()
        assert tool.output_files_parm == 'out_path'
        assert tool.input_files_parm is None

    def test_fc_register_output_files_static(self):
        @fc_register('tool', output_files=['report.txt', 'log.txt'])
        def static_output_tool(text: str):
            '''
            Tool with static output files.

            Args:
                text (str): input text.
            '''
            return text

        tool = lazyllm.tool.static_output_tool()
        assert tool.output_files == ['report.txt', 'log.txt']

    def test_fc_register_all_file_params(self):
        @fc_register('tool', input_files_parm='inputs', output_files_parm='outputs',
                     output_files=['extra.txt'], execute_in_sandbox=True)
        def full_file_tool(inputs: str, outputs: str):
            '''
            Tool with all file params.

            Args:
                inputs (str): input file path.
                outputs (str): output file path.
            '''
            return 'done'

        tool = lazyllm.tool.full_file_tool()
        assert tool.input_files_parm == 'inputs'
        assert tool.output_files_parm == 'outputs'
        assert tool.output_files == ['extra.txt']
        assert tool.execute_in_sandbox is True

    def test_fc_register_disallowed_parameter_raises(self):
        with pytest.raises(AssertionError):
            @fc_register('tool', nonexistent_param='value')
            def bad_tool(x: int):
                '''
                Tool.

                Args:
                    x (int): value.
                '''
                return x


class TestModuleToolProperties:
    def test_execute_in_sandbox_default(self):
        @fc_register('tool')
        def default_sandbox_tool(x: int):
            '''
            Tool.

            Args:
                x (int): value.
            '''
            return x

        tool = lazyllm.tool.default_sandbox_tool()
        assert tool.execute_in_sandbox is True

    def test_input_files_parm_setter_validates_type(self):
        @fc_register('tool')
        def setter_test_tool(x: int):
            '''
            Tool.

            Args:
                x (int): value.
            '''
            return x

        tool = lazyllm.tool.setter_test_tool()
        tool.input_files_parm = 'my_param'
        assert tool.input_files_parm == 'my_param'

        with pytest.raises(AssertionError):
            tool.input_files_parm = 123

    def test_output_files_parm_setter_validates_type(self):
        @fc_register('tool')
        def setter_test_tool2(x: int):
            '''
            Tool.

            Args:
                x (int): value.
            '''
            return x

        tool = lazyllm.tool.setter_test_tool2()
        tool.output_files_parm = 'out_param'
        assert tool.output_files_parm == 'out_param'

        with pytest.raises(AssertionError):
            tool.output_files_parm = ['not', 'a', 'string']

    def test_output_files_setter_validates_type(self):
        @fc_register('tool')
        def setter_test_tool3(x: int):
            '''
            Tool.

            Args:
                x (int): value.
            '''
            return x

        tool = lazyllm.tool.setter_test_tool3()
        tool.output_files = ['file1.txt', 'file2.txt']
        assert tool.output_files == ['file1.txt', 'file2.txt']

        with pytest.raises(AssertionError):
            tool.output_files = 'not_a_list'

        with pytest.raises(AssertionError):
            tool.output_files = [1, 2, 3]


class TestRegistryWithKey(object):
    def test_registry_with_key(self):
        all_configs = []

        class LazyLLMOnlineModuleBase(object, metaclass=LazyLLMRegisterMetaClass):
            __lazyllm_registry_key__ = 'testonline'

            @staticmethod
            def __lazyllm_after_registry_hook__(cls, group_name: str, name: str, isleaf: bool):
                if group_name == '':
                    assert name == 'testonline'
                elif not isleaf:
                    assert group_name == 'testonline'
                    assert name.lower() in ('stt', 'tts', 'texttoimage', 'imageediting'), 'group name error'
                else:
                    subgroup = group_name.split('.')[-1]
                    assert name.lower().endswith(subgroup), 'subgroup error'
                    supplier = name[:-len(subgroup)].lower()
                    all_configs.append(f'config.add({supplier}_api_key)')
                    all_configs.append(f'config.add({supplier}_{subgroup}_model_name)')

            def __init__(self, *args, **kw):
                pass

        class OnlineMultiModalBase(LazyLLMOnlineModuleBase):
            __lazyllm_registry_disable__ = True

        class LazyLLMOnlineSTTModuleBase(OnlineMultiModalBase):
            __lazyllm_registry_key__ = 'STT'

        class LazyLLMOnlineTTSModuleBase(OnlineMultiModalBase):
            __lazyllm_registry_key__ = 'TTS'

        class LazyLLMOnlineTextoImageModuleBase(OnlineMultiModalBase):
            __lazyllm_registry_key__ = 'TextToImage'

        class LazyLLMOnlineImageEditingBase(OnlineMultiModalBase):
            __lazyllm_registry_key__ = 'ImageEditing'

        class AbcBase():
            def __init__(self, api_key: str = None, base_url: str = 'base_url'):
                self._client = 'abc(api_key=api_key, base_url=base_url'

        class AbcSTT(LazyLLMOnlineSTTModuleBase, AbcBase):
            def __init__(self, api_key=None, base_url='base_url'):
                super().__init__(api_key, base_url)
                AbcBase.__init__(self, api_key=api_key, base_url=base_url)

        class AbcTTS(LazyLLMOnlineTTSModuleBase, AbcBase):
            def __init__(self, api_key=None, base_url='base_url'):
                super().__init__(api_key, base_url)
                AbcBase.__init__(self, api_key=api_key, base_url=base_url)

        class AbcTextToImage(LazyLLMOnlineTextoImageModuleBase, AbcBase):
            def __init__(self, api_key=None, base_url='base_url'):
                super().__init__(api_key, base_url)
                AbcBase.__init__(self, api_key=api_key, base_url=base_url)

        class AbcImageEditing(LazyLLMOnlineImageEditingBase, AbcBase):
            def __init__(self, api_key=None, base_url='base_url'):
                super().__init__(api_key, base_url)
                AbcBase.__init__(self, api_key=api_key, base_url=base_url)

        assert hasattr(lazyllm, 'testonline')
        assert hasattr(lazyllm.testonline, 'base')
        assert hasattr(lazyllm.testonline, 'stt')
        assert hasattr(lazyllm.testonline, 'tts')
        assert hasattr(lazyllm.testonline, 'TextToImage')
        assert hasattr(lazyllm.testonline, 'ImageEditing')
        assert len(lazyllm.testonline) == 4

        assert hasattr(lazyllm.testonline.stt, 'abc')
        assert hasattr(lazyllm.testonline.tts, 'abc')
        assert hasattr(lazyllm.testonline.TextToImage, 'abc')
        assert hasattr(lazyllm.testonline.ImageEditing, 'abc')

        assert len(all_configs) == 8
        assert 'config.add(abc_api_key)' in all_configs
        assert 'config.add(abc_stt_model_name)' in all_configs

        with pytest.raises(AssertionError, match='group name error'):
            class LazyLLMErrorNameModuleBase(OnlineMultiModalBase):
                __lazyllm_registry_key__ = 'ErrorName'

        with pytest.raises(AssertionError, match='subgroup error'):
            class AbcImageEditingError(LazyLLMOnlineImageEditingBase, AbcBase):
                def __init__(self, api_key=None, base_url='base_url'):
                    super().__init__(api_key, base_url)
                    AbcBase.__init__(self, api_key=api_key, base_url=base_url)
