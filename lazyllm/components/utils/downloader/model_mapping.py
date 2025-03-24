# flake8: noqa: E501,

model_groups = {
    "meta-llama-3": {
        "prompt_keys": {'sos': '<|start_header_id|>system<|end_header_id|>\n\n', 'soh': '<|start_header_id|>user<|end_header_id|>\n\n', 'soa': '<|start_header_id|>assistant<|end_header_id|>\n\n', 'eos': '<|eot_id|>', 'eoh': '<|eot_id|>', 'eoa': '<|eot_id|>', 'stop_words': ['<|eot_id|>', '<|end_of_text|>']},
    },
    "llama-2": {
        "prompt_keys": {
            'sos': '[INST] <<SYS>>\n', 'soa': ' [/INST] ', 'eos': '\n<</SYS>>\n\n', 'eoa': '</s>', 'separator': '<s>[INST] ',
            'system': ("\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")
            },
    },
    "qwen": {
        "prompt_keys": {
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'soe': '<|im_start|>environment\n', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>'], 'tool_start_token': '<|action_start|><|function|>', 'tool_end_token': '<|action_end|>', 'tool_args_token': "<|args|>",
            'system': "You are a helpful assistant."
            },
    },
    "deepseek-r1-distill-qwen": {
        "prompt_keys": {
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n\n<think>', 'soe': '<|im_start|>environment\n', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>'], 'tool_start_token': '<|action_start|><|function|>', 'tool_end_token': '<|action_end|>', 'tool_args_token': "<|args|>",
            'system': "You are a helpful assistant."
            },
    },
    "deepseek-r1-distill-llama": {
        "prompt_keys": {
            'sos': '<|start_header_id|>system<|end_header_id|>\n\n', 'soh': '<|start_header_id|>user<|end_header_id|>\n\n', 'soa': '<|start_header_id|>assistant<|end_header_id|>\n\n<think>', 'eos': '<|eot_id|>', 'eoh': '<|eot_id|>', 'eoa': '<|eot_id|>', 'stop_words': ['<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'],
            'system': "You are a helpful assistant.",
            },
    },
    "deepseek": {
        "prompt_keys": {
            'sos': '<｜begin▁of▁sentence｜>', 'soh': '<｜User｜>', 'soa': '<｜Assistant｜>', 'eos': '', 'eoh': '', 'eoa': '<｜end▁of▁sentence｜>', 'stop_words': ['<｜end▁of▁sentence｜>'],
            'system': "",
            },
    },    
    "internlm": {
        "prompt_keys": {
            'sos': '<|System|>:', 'soh': '<|User|>:', 'soa': '<|Bot|>:', 'eos': '\n', 'eoh': '\n', 'eoa': '<eoa>', 'separator': '\n', 'stop_words': ['<eoa>'],
            'system': "You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
            },
    },
    "internlm2": {
        "prompt_keys": {
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'soe': '<|im_start|>environment\n', 'plugin': '<|plugin|>', 'interpreter': '<|interpreter|>', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>'], 'tool_start_token': '<|function|>', 'tool_end_token': '<|action_end|>', 'tool_args_token': '<|args|>',
            'system': "You are an AI assistant whose name is InternLM (书生·浦语).\n- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
            },
    },
    "sensechat": {
        "prompt_keys": {
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'soe': '<|im_start|>environment\n', 'plugin': '<|plugin|>', 'interpreter': '<|interpreter|>', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>'], 'tool_start_token': '<|function|>', 'tool_end_token': '<|action_end|>', 'tool_args_token': '<|args|>',
            'system': "You are an AI assistant whose name is SenseChat.\n- SenseChat is a conversational language model that is developed by SenseTime. It is designed to be helpful, honest, and harmless.\n- SenseChat can understand and communicate fluently in the language chosen by the user such as English and 中文."
            },
    },
    "internlm-xcomposer2": {
        "prompt_keys": {
            'sos': '[UNUSED_TOKEN_146]system\n', 'soh': '[UNUSED_TOKEN_146]user\n', 'soa': '[UNUSED_TOKEN_146]assistant\n', 'eos': '[UNUSED_TOKEN_145]\n', 'eoh': '[UNUSED_TOKEN_145]\n', 'eoa': '[UNUSED_TOKEN_145]\n', 'separator': '\n', 'stop_words': ['[UNUSED_TOKEN_145]'],
            'system': ("You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
                    "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.")
            },
    },
    "internlm-xcomposer": {
        "prompt_keys": {},
    },
    "codellama": {
        "prompt_keys": {},
    },
    "baichuan": {
        "prompt_keys": {},
    },
    "baichuan2": {
        "prompt_keys": {'soh': '<reserved_106>', 'soa': '<reserved_107>'},
    },
    "chatglm3": {
        "prompt_keys": {'sos': '<|system|>\n', 'soh': '<|user|>\n', 'soa': '<|assistant|>\n', 'plugin': '<|observation|>\n', 'stop_words': ['<|user|>', '<|observation|>']},
    },
    "glm-4": {
        "prompt_keys": {'sos': '<|system|>\n', 'soh': '<|user|>\n', 'soa': '<|assistant|>\n', 'soe': '<|observation|>\n', 'eoe': '\n', 'plugin': '<|observation|>\n', 'stop_words': ['<|user|>', '<|observation|>'], 'tool_start_token': '✿FUNCTION✿: ', 'tool_args_token': '✿ARGS✿: ', 'tool_end_token': '\n'},
    },
}

model_provider = {
    "baichuan": {           
        "huggingface": "baichuan-inc",
        "modelscope": "baichuan-inc"
    },
    "chatglm": {
        "huggingface": "THUDM",
        "modelscope": "ZhipuAI"
    },
     "glm": {
        "huggingface": "THUDM",
        "modelscope": "ZhipuAI"
    },
    "codellama": {
        "huggingface": "meta-llama",
        "modelscope": "AI-ModelScope"
    },
    "llama": {
        "huggingface": "meta-llama",
        "modelscope": "modelscope"
    },
    "meta-llama": {
        "huggingface": "meta-llama",
        "modelscope": "LLM-Research"
    },
    "internlm": {
        "huggingface": "internlm",
        "modelscope": "Shanghai_AI_Laboratory"
    },
    "sensechat": {
        "huggingface": "sensetime",
        "modelscope": "sensetime"
    },
    "internvl": {
        "huggingface": "OpenGVLab",
        "modelscope": "OpenGVLab"
    },
    "qwen": {
        "huggingface": "Qwen",
        "modelscope": "qwen"
    },
    "deepseek":{
        "huggingface": "deepseek-ai",
        "modelscope": "deepseek-ai"
    }
}

model_name_mapping = {
    "bark": {
        "source": {
            "huggingface": "suno/bark",
            "modelscope": "mapjack/bark"
        },
        "type": "tts"
    },
    "bge-large-zh-v1.5": {
        "source": {
            "huggingface": "BAAI/bge-large-zh-v1.5",
            "modelscope": "AI-ModelScope/bge-large-zh-v1.5"
        },
        "type": "embed"
    },
    "bge-m3": {
        "source": {
            "huggingface": "BAAI/bge-m3",
            "modelscope": "Xorbits/bge-m3"
        },
        "type": "embed"
    },
    "jina-clip-v1": {
        "source": {
            "huggingface": "jinaai/jina-clip-v1",
            "modelscope": "jinaai/jina-clip-v1"
        },
        "type": "cross_modal_embed" # image - text
    },
    "siglip": {
        "source": {
            "huggingface": "google/siglip-so400m-patch14-384",
        },
        "type": "cross_modal_embed" # image - text
    },
    "colqwen2-v0.1": {
        "source": {
            "huggingface": "michaelfeil/colqwen2-v0.1",
        },
        "type": "cross_modal_embed" # image - text
    },
    "bge-reranker-large": {
        "source": {
            "huggingface": "BAAI/bge-reranker-large",
            "modelscope": "Xorbits/bge-reranker-large"
        },
        "type": "reranker"
    },
    "chattts": {
        "source": {
            "huggingface": "2Noise/ChatTTS",
            "modelscope": "AI-ModelScope/ChatTTS"
        },
        "type": "tts"
    },
    "internvl-chat-v1-5": {
        "source": {
            "huggingface": "OpenGVLab/InternVL-Chat-V1-5",
            "modelscope": "OpenGVLab/InternVL-Chat-V1-5"
        },
        "type": "vlm"
    },
    "yi-vl-6b": {
        "source": {
            "huggingface": "01-ai/Yi-VL-6B",
            "modelscope": "01ai/Yi-VL-6B"
        },
        "type": "vlm"
    },
    "llava-1.5-13b-hf": {
        "source": {
            "huggingface": "llava-hf/llava-1.5-13b-hf",
            "modelscope": "huangjintao/llava-1.5-13b-hf"
        },
        "type": "vlm"
    },
    "llava-1.5-7b-hf": {
        "source": {
            "huggingface": "llava-hf/llava-1.5-7b-hf",
            "modelscope": "huangjintao/llava-1.5-7b-hf"
        },
        "type": "vlm"
    },
    "mini-internvl-chat-2b-v1-5": {
        "source": {
            "huggingface": "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
            "modelscope": "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
        },
        "type": "vlm"
    },
    "mini-internvl-chat-4b-v1-5": {
        "source": {
            "huggingface": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
            "modelscope": "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"
        },
        "type": "vlm"
    },
    "musicgen-medium": {
        "source": {
            "huggingface": "facebook/musicgen-medium",
            "modelscope": "AI-ModelScope/musicgen-medium"
        },
        "type": "tts"
    },
    "musicgen-stereo-small": {
        "source": {
            "huggingface": "facebook/musicgen-stereo-small",
            "modelscope": "AI-ModelScope/musicgen-small"
        },
        "type": "tts"
    },
    "sensevoicesmall": {
        "source": {
            "huggingface": "FunAudioLLM/SenseVoiceSmall",
            "modelscope": "iic/SenseVoiceSmall"
        },
        "type": "stt"
    },
    "stable-diffusion-3-medium": {
        "source": {
            "huggingface": "stabilityai/stable-diffusion-3-medium",
            "modelscope": "AI-ModelScope/stable-diffusion-3-medium-diffusers"
        },
        "type": "sd"
    }
}
