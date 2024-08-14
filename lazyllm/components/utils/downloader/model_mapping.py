# flake8: noqa: E501,

model_groups = {
	"Llama3": {
		"prompt_keys": {'sos': '<|start_header_id|>system<|end_header_id|>\n\n', 'soh': '<|start_header_id|>user<|end_header_id|>\n\n', 'soa': '<|start_header_id|>assistant<|end_header_id|>\n\n', 'eos': '<|eot_id|>', 'eoh': '<|eot_id|>', 'eoa': '<|eot_id|>', 'stop_words': ['<|eot_id|>', '<|end_of_text|>']},
	},
	"Llama2": {
		"prompt_keys": {
            'sos': '[INST] <<SYS>>\n', 'soa': ' [/INST] ', 'eos': '\n<</SYS>>\n\n', 'eoa': '</s>', 'separator': '<s>[INST] ',
            'system': ("\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
                    "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.")
            },
	},
	"QWen": {
		"prompt_keys": {
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'soe': '<|im_start|>environment\n', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>'], 'tool_start_token': '<|action_start|><|function|>', 'tool_end_token': '<|action_end|>', 'tool_args_token': "<|args|>",
            'system': "You are a helpful assistant."
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
	"internlm-xcomposer2": {
		"prompt_keys": {
            'sos': '[UNUSED_TOKEN_146]system\n', 'soh': '[UNUSED_TOKEN_146]user\n', 'soa': '[UNUSED_TOKEN_146]assistant\n', 'eos': '[UNUSED_TOKEN_145]\n', 'eoh': '[UNUSED_TOKEN_145]\n', 'eoa': '[UNUSED_TOKEN_145]\n', 'separator': '\n', 'stop_words': ['[UNUSED_TOKEN_145]'],
            'system': ("You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
                    "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.")
            },
	},
	"Baichuan2": {
		"prompt_keys": {'soh': '<reserved_106>', 'soa': '<reserved_107>'},
	},
	"GLM3": {
		"prompt_keys": {'sos': '<|system|>\n', 'soh': '<|user|>\n', 'soa': '<|assistant|>\n', 'plugin': '<|observation|>\n', 'stop_words': ['<|user|>', '<|observation|>']},
	},
	"GLM4": {
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
		"huggingface": "meta-llama"
	},
	"llama": {
		"huggingface": "meta-llama"
	},
	"meta-llama": {
		"huggingface": "meta-llama"
	},
	"internlm": {
		"huggingface": "internlm",
		"modelscope": "Shanghai_AI_Laboratory"
	},
	"qwen": {
		"huggingface": "Qwen",
		"modelscope": "qwen"		
	}
}

model_name_mapping = {
    "baichuan-13b-chat": {
        "prompt_keys": {},
        "source": {
            "huggingface": "baichuan-inc/Baichuan-13B-Chat",
            "modelscope": "baichuan-inc/Baichuan-13B-Chat"
        }
    },
    "baichuan-7b": {
        "prompt_keys": {},
        "source": {
            "huggingface": "baichuan-inc/Baichuan-7B",
            "modelscope": "baichuan-inc/baichuan-7B"
        }
    },
    "baichuan2-13b-chat": {
        "prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
        "source": {
            "huggingface": "baichuan-inc/Baichuan2-13B-Chat",
            "modelscope": "baichuan-inc/Baichuan2-13B-Chat"
        }
    },
    "baichuan2-13b-chat-4bits": {
        "prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
        "source": {
            "huggingface": "baichuan-inc/Baichuan2-13B-Chat-4bits",
            "modelscope": "baichuan-inc/Baichuan2-13B-Chat-4bits"
        }
    },
    "baichuan2-7b-chat": {
        "prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
        "source": {
            "huggingface": "baichuan-inc/Baichuan2-7B-Chat",
            "modelscope": "baichuan-inc/Baichuan2-7B-Chat"
        }
    },
    "baichuan2-7b-chat-4bits": {
        "prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
        "source": {
            "huggingface": "baichuan-inc/Baichuan2-7B-Chat-4bits",
            "modelscope": "baichuan-inc/Baichuan2-7B-Chat-4bits"
        }
    },
    "baichuan2-7b-intermediate-checkpoints": {
        "prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
        "source": {
            "huggingface": "baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints",
            "modelscope": "baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints"
        }
    },
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
    "bge-reranker-large": {
        "source": {
            "huggingface": "BAAI/bge-reranker-large",
            "modelscope": "Xorbits/bge-reranker-large"
        },
        "type": "embed"
    },
    "chatglm3-6b": {
        "prompt_keys": model_groups["GLM3"]["prompt_keys"],
        "source": {
            "huggingface": "THUDM/chatglm3-6b",
            "modelscope": "ZhipuAI/chatglm3-6b"
        }
    },
    "chatglm3-6b-128k": {
        "prompt_keys": model_groups["GLM3"]["prompt_keys"],
        "source": {
            "huggingface": "THUDM/chatglm3-6b-128k",
            "modelscope": "ZhipuAI/chatglm3-6b-128k"
        }
    },
    "chatglm3-6b-32k": {
        "prompt_keys": model_groups["GLM3"]["prompt_keys"],
        "source": {
            "huggingface": "THUDM/chatglm3-6b-32k",
            "modelscope": "ZhipuAI/chatglm3-6b-32k"
        }
    },
    "chattts": {
        "source": {
            "huggingface": "2Noise/ChatTTS",
            "modelscope": "AI-ModelScope/ChatTTS"
        },
        "type": "tts"
    },
    "codellama-13b-hf": {
        "prompt_keys": {},
        "source": {
            "huggingface": "meta-llama/CodeLlama-13b-hf"
        }
    },
    "codellama-34b-hf": {
        "prompt_keys": {},
        "source": {
            "huggingface": "meta-llama/CodeLlama-34b-hf"
        }
    },
    "codellama-70b-hf": {
        "prompt_keys": {},
        "source": {
            "huggingface": "meta-llama/CodeLlama-70b-hf"
        }
    },
    "codellama-7b-hf": {
        "prompt_keys": {},
        "source": {
            "huggingface": "meta-llama/CodeLlama-7b-hf"
        }
    },
    "glm-4-9b-chat": {
        "prompt_keys": model_groups["GLM4"]["prompt_keys"],
        "source": {
            "huggingface": "THUDM/glm-4-9b-chat",
            "modelscope": "ZhipuAI/glm-4-9b-chat"
        }
    },
    "internlm-20b": {
        "prompt_keys": model_groups["internlm"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-20b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-20b"
        }
    },
    "internlm-7b": {
        "prompt_keys": model_groups["internlm"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-7b"
        }
    },
    "internlm-chat-20b": {
        "prompt_keys": model_groups["internlm"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-chat-20b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-chat-20b"
        }
    },
    "internlm-chat-20b-4bit": {
        "prompt_keys": model_groups["internlm"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-chat-20b-4bit",
            "modelscope": "Shanghai_AI_Laboratory/internlm-chat-20b-4bit"
        }
    },
    "internlm-chat-7b": {
        "prompt_keys": model_groups["internlm"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-chat-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-chat-7b"
        }
    },
    "internlm-xcomposer-7b": {
        "prompt_keys": {},
        "source": {
            "huggingface": "internlm/internlm-xcomposer-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-7b"
        }
    },
    "internlm-xcomposer-7b-4bit": {
        "prompt_keys": {},
        "source": {
            "huggingface": "internlm/internlm-xcomposer-7b-4bit",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-7b-4bit"
        }
    },
    "internlm-xcomposer-vl-7b": {
        "prompt_keys": {},
        "source": {
            "huggingface": "internlm/internlm-xcomposer-vl-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b"
        }
    },
    "internlm-xcomposer2-4khd-7b": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-4khd-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b"
        }
    },
    "internlm-xcomposer2-7b": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-7b"
        }
    },
    "internlm-xcomposer2-7b-4bit": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-7b-4bit",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-7b-4bit"
        }
    },
    "internlm-xcomposer2-vl-1_8b": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-vl-1_8b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-1_8b"
        }
    },
    "internlm-xcomposer2-vl-7b": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-vl-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b"
        }
    },
    "internlm-xcomposer2-vl-7b-4bit": {
        "prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm-xcomposer2-vl-7b-4bit",
            "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b-4bit"
        }
    },
    "internlm2-1_8b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-1_8b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-1_8b"
        }
    },
    "internlm2-20b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-20b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-20b"
        }
    },
    "internlm2-7b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-7b"
        }
    },
    "internlm2-chat-1_8b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-1_8b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-1_8b"
        }
    },
    "internlm2-chat-1_8b-sft": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-1_8b-sft",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft"
        }
    },
    "internlm2-chat-20b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-20b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b"
        }
    },
    "internlm2-chat-20b-4bits": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-20b-4bits",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b-4bits"
        }
    },
    "internlm2-chat-20b-sft": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-20b-sft",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b-sft"
        }
    },
    "internlm2-chat-7b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b"
        }
    },
    "internlm2-chat-7b-4bits": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-7b-4bits",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b-4bits"
        }
    },
    "internlm2-chat-7b-sft": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-chat-7b-sft",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b-sft"
        }
    },
    "internlm2-math-20b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-math-20b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-math-20b"
        }
    },
    "internlm2-math-7b": {
        "prompt_keys": model_groups["internlm2"]["prompt_keys"],
        "source": {
            "huggingface": "internlm/internlm2-math-7b",
            "modelscope": "Shanghai_AI_Laboratory/internlm2-math-7b"
        }
    },
    "internvl-chat-v1-5": {
        "source": {
            "huggingface": "OpenGVLab/InternVL-Chat-V1-5",
            "modelscope": "OpenGVLab/InternVL-Chat-V1-5"
        },
        "type": "vlm"
    },
    "llama-2-13b": {
        "prompt_keys": model_groups["Llama2"]["prompt_keys"],
        "source": {
            "huggingface": "meta-llama/Llama-2-13b"
        }
    },
    "llama-2-70b": {
        "prompt_keys": model_groups["Llama2"]["prompt_keys"],
        "source": {
            "huggingface": "meta-llama/Llama-2-70b"
        }
    },
    "llama-2-7b": {
        "prompt_keys": model_groups["Llama2"]["prompt_keys"],
        "source": {
            "huggingface": "meta-llama/Llama-2-7b"
        }
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
    "meta-llama-3-70b": {
        "prompt_keys": model_groups["Llama3"]["prompt_keys"],
        "source": {
            "huggingface": "meta-llama/Meta-Llama-3-70B"
        }
    },
    "meta-llama-3-8b": {
        "prompt_keys": model_groups["Llama3"]["prompt_keys"],
        "source": {
            "huggingface": "meta-llama/Meta-Llama-3-8B"
        }
    },
    "mini-internvl-chat-2b-v1-5": {
        "source": {
            "huggingface": "OpenGVLab/Mini-InternVL-Chat-2B-V1-5",
            "modelscope": "OpenGVLab/Mini-InternVL-Chat-2B-V1-5"
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
    "qwen-14b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen-14B",
            "modelscope": "qwen/Qwen-14B"
        }
    },
    "qwen-1_8b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen-1_8B",
            "modelscope": "qwen/Qwen-1_8B"
        }
    },
    "qwen-72b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen-72B",
            "modelscope": "qwen/Qwen-72B"
        }
    },
    "qwen-7b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen-7B",
            "modelscope": "qwen/Qwen-7B"
        }
    },
    "qwen1.5-0.5b-chat": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-0.5B-Chat",
            "modelscope": "qwen/Qwen1.5-0.5B-Chat"
        }
    },
    "qwen1.5-1.8b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-1.8B",
            "modelscope": "qwen/Qwen1.5-1.8B"
        }
    },
    "qwen1.5-14b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-14B",
            "modelscope": "qwen/Qwen1.5-14B"
        }
    },
    "qwen1.5-14b-chat": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-14B-Chat",
            "modelscope": "qwen/Qwen1.5-14B-Chat"
        }
    },
    "qwen1.5-4b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-4B",
            "modelscope": "qwen/Qwen1.5-4B"
        }
    },
    "qwen1.5-72b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-72B",
            "modelscope": "qwen/Qwen1.5-72B"
        }
    },
    "qwen1.5-7b": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen1.5-7B",
            "modelscope": "qwen/Qwen1.5-7B"
        }
    },
    "qwen2-72b-instruct": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen2-72B-Instruct",
            "modelscope": "qwen/Qwen2-72B-Instruct"
        }
    },
    "qwen2-72b-instruct-awq": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen2-72B-Instruct-AWQ",
            "modelscope": "qwen/Qwen2-72B-Instruct-AWQ"
        }
    },
    "qwen2-7b-instruct": {
        "prompt_keys": model_groups["QWen"]["prompt_keys"],
        "source": {
            "huggingface": "Qwen/Qwen2-7B-Instruct",
            "modelscope": "qwen/Qwen2-7B-Instruct"
        }
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
