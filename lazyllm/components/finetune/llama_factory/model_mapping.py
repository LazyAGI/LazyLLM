def match_longest_prefix(model_name):
    """
    Find the longest key in the dictionary that is a prefix of the model_name string.
    The keys in the dictionary are case-insensitive.

    Args:
        model_name (str): The input string of model name to be matched.

    Returns:
        str: The value corresponding to the longest matched key, or '' if no match found.
    """
    keys_sorted = sorted(llamafactory_mapping_dict.keys(), key=lambda k: (-len(k), k))
    model_name_lower = model_name.lower()
    for key in keys_sorted:
        if model_name_lower.startswith(key):
            return llamafactory_mapping_dict[key]
    return ''

llamafactory_mapping_dict = {
    'aya': 'cohere',
    'commandr': 'cohere',
    'bluelm': 'bluelm',
    'baichuan': 'baichuan',
    'baichuan2': 'baichuan2',
    'breeze': 'breeze',
    'chatglm2': 'chatglm2',
    'chatglm3': 'chatglm3',
    'chinese-alpaca-2': 'llama2_zh',
    'codegeex4': 'codegeex4',
    'gemma': 'gemma',
    'codegemma': 'gemma',
    'mistral': 'mistral',
    'codemistral': 'mistral',
    'dbrx': 'dbrx',
    'deepseek': 'deepseek',
    'deepseek-coder': 'deepseekcoder',
    'exaone': 'exaone',
    'falcon': 'falcon',
    'glm-4': 'glm4',
    'index': 'index',
    'internlm': 'intern',
    'internlm2': 'intern2',
    'llama-2': 'llama2',
    'llama-3': 'llama3',
    'llama-3.2': 'mllama',
    'llava': 'llava',
    'llava-next': 'llava_next',
    'llava-next-mistral': 'llava_next_mistral',
    'llava-next-llama3': 'llava_next_llama3',
    'llava-next-34b': 'llava_next_yi',
    'llava-next-110b-chat': 'llava_next_qwen',
    'llava-next-video': 'llava_next_video',
    'llava-next-video-7b': 'llava_next_video_mistral',
    'llava-next-video-34b': 'llava_next_video_yi',
    'minicpm': 'cpm',
    'minicpm3': 'cpm3',
    'openchat': 'openchat',
    'openchat3.6': 'openchat-3.6',
    'opencoder': 'opencoder',
    'orion': 'orion',
    'paligemma': 'paligemma',
    'phi-3': 'phi',
    'phi-3-7b': 'phi_small',
    'pixtral': 'pixtral',
    'qwen': 'qwen',
    'codeqwen': 'qwen',
    'qwen2-vl': 'qwen2_vl',
    'solar': 'solar',
    'telechat': 'telechat',
    'vicuna': 'vicuna',
    'video-llava': 'video_llava',
    'xuanyuan': 'xuanyuan',
    'xverse': 'xverse',
    'yi': 'yi',
    'yi-vl': 'yi_vl',
    'yuan': 'yuan',
    'zephyr': 'zephyr',
}
