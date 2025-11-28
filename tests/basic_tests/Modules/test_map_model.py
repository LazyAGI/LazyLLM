import pytest
from lazyllm.module.llms.onlinemodule.map_model_type import get_model_type

test_models = {
    'llm': [
        'GPT5'
        'Qwen3-32B',
        'qwen3-coder-plus',
        'sensechat-128k',
        'glm-4-5-airx',
        'qwen3-coder-plus-2025-09-23',
        'deepseek-v3-1-terminus'
    ],
    'vlm': [
        'moonshot-v1-128k-vision-preview',
        'qwen3-vl-flash-2025-10-15',
        'doubao-seed-1-6-flash-250828',
        'doubao-1-5-vision-pro-32k-250115'
    ],
    'stt': [
        'qwen3-asr-flash-realtime-2025-10-27'
    ],
    'tts': [
        'qwen3-tts-flash-realtime-2025-09-18'
    ],
    'embed': [
        'text-embedding-v3'
    ],
    'sd': [
        'cogview-4',
        'wanx2.1-t2i-plus',
        'animate-anyone-template-gen2',
        'doubao-seedance-1-0-pro-fast-251015'
    ],
    'cross_modal_embed': [
        'doubao-embedding-vision-250615'
    ]
}

class TestGetModelCategory:

    def infer_model_type(self, rename_func=None):
        errors = []
        total_tests = 0

        for expected_type, model_list in test_models.items():
            for model_name in model_list:
                total_tests += 1
                try:
                    if rename_func:
                        model_name = rename_func(model_name)
                    inferred_type = get_model_type(model_name)
                    if inferred_type != expected_type:
                        errors.append(
                            f'❌ {model_name} misclassified as {inferred_type}, expected {expected_type}'
                        )
                except Exception as e:
                    errors.append(f'❌❌ {model_name} -> {e}')

        if errors:
            error_summary = f'\nTest models: {len(errors)}/{total_tests} failed.\n'
            error_summary += '\n'.join(errors)
            pytest.fail(error_summary)

    def test_all_models_original_names(self):
        self.infer_model_type()
