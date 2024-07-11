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
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'separator': '\n', 'stop_words': ['<|im_end|>'],
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
            'sos': '<|im_start|>system\n', 'soh': '<|im_start|>user\n', 'soa': '<|im_start|>assistant\n', 'soe': '<|im_start|>environment\n', 'plugin': '<|plugin|>', 'interpreter': '<|interpreter|>', 'eos': '<|im_end|>\n', 'eoh': '<|im_end|>\n', 'eoa': '<|im_end|>', 'eoe': '<|im_end|>\n', 'separator': '\n', 'stop_words': ['<|im_end|>', '<|action_end|>'],
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
}


model_name_mapping = {
    "Llama-3-8B": {
		"source": {"huggingface": "meta-llama/Meta-Llama-3-8B"},
		"prompt_keys": model_groups["Llama3"]["prompt_keys"],
		},
    "Llama-3-70B": {
		"source": {"huggingface": "meta-llama/Meta-Llama-3-70B"},
		"prompt_keys": model_groups["Llama3"]["prompt_keys"],
		},
    "Llama-2-7b": {
		"source": {"huggingface": "meta-llama/Llama-2-7b"},
		"prompt_keys": model_groups["Llama2"]["prompt_keys"],
		},
    "Llama-2-13b": {
		"source": {"huggingface": "meta-llama/Llama-2-13b"},
		"prompt_keys": model_groups["Llama2"]["prompt_keys"],
		},
    "Llama-2-70b": {
		"source": {"huggingface": "meta-llama/Llama-2-70b"},
		"prompt_keys": model_groups["Llama2"]["prompt_keys"],
		},
    "Llama-7b": {
		"source": {"huggingface": "meta-llama/CodeLlama-7b-hf"},
		"prompt_keys": {},
		},
    "Llama-13b": {
		"source": {"huggingface": "meta-llama/CodeLlama-13b-hf"},
		"prompt_keys": {},
		},
    "Llama-34b": {
		"source": {"huggingface": "meta-llama/CodeLlama-34b-hf"},
		"prompt_keys": {},
		},
    "Llama-70b": {
		"source": {"huggingface": "meta-llama/CodeLlama-70b-hf"},
		"prompt_keys": {},
		},

    "GLM3-6B": {
		"source": {"huggingface": "THUDM/chatglm3-6b", "modelscope": "ZhipuAI/chatglm3-6b"},
		"prompt_keys": model_groups["GLM3"]["prompt_keys"],
		},
    "GLM3-6B-32K": {
		"source": {"huggingface": "THUDM/chatglm3-6b-32k", "modelscope": "ZhipuAI/chatglm3-6b-32k"},
		"prompt_keys": model_groups["GLM3"]["prompt_keys"],
		},
    "GLM3-6B-128K": {
		"source": {"huggingface": "THUDM/chatglm3-6b-128k", "modelscope": "ZhipuAI/chatglm3-6b-128k"},
		"prompt_keys": model_groups["GLM3"]["prompt_keys"],
		},

    "Qwen-1.8B": {
		"source": {"huggingface": "Qwen/Qwen-1_8B", "modelscope": "qwen/Qwen-1_8B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen-7B": {
		"source": {"huggingface": "Qwen/Qwen-7B", "modelscope": "qwen/Qwen-7B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen-14B": {
		"source": {"huggingface": "Qwen/Qwen-14B", "modelscope": "qwen/Qwen-14B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen-72B": {
		"source": {"huggingface": "Qwen/Qwen-72B", "modelscope": "qwen/Qwen-72B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-0.5B-Chat": {
		"source": {"huggingface": "Qwen/Qwen1.5-0.5B-Chat", "modelscope": "qwen/Qwen1.5-0.5B-Chat"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-1.8B": {
		"source": {"huggingface": "Qwen/Qwen1.5-1.8B", "modelscope": "qwen/Qwen1.5-1.8B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-4B": {
		"source": {"huggingface": "Qwen/Qwen1.5-4B", "modelscope": "qwen/Qwen1.5-4B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-7B": {
		"source": {"huggingface": "Qwen/Qwen1.5-7B", "modelscope": "qwen/Qwen1.5-7B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-14B": {
		"source": {"huggingface": "Qwen/Qwen1.5-14B", "modelscope": "qwen/Qwen1.5-14B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "Qwen1.5-72B": {
		"source": {"huggingface": "Qwen/Qwen1.5-72B", "modelscope": "qwen/Qwen1.5-72B"},
		"prompt_keys": model_groups["QWen"]["prompt_keys"],
		},
    "internlm-20b": {
		"source": {"huggingface": "internlm/internlm-20b", "modelscope": "Shanghai_AI_Laboratory/internlm-20b"},
		"prompt_keys": model_groups["internlm"]["prompt_keys"],
		},
    "internlm2-1_8b": {
		"source": {"huggingface": "internlm/internlm2-1_8b", "modelscope": "Shanghai_AI_Laboratory/internlm2-1_8b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-20b": {
		"source": {"huggingface": "internlm/internlm2-20b", "modelscope": "Shanghai_AI_Laboratory/internlm2-20b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-7b": {
		"source": {"huggingface": "internlm/internlm2-7b", "modelscope": "Shanghai_AI_Laboratory/internlm2-7b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-1_8b": {
		"source": {"huggingface": "internlm/internlm2-chat-1_8b", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-1_8b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-1_8b-sft": {
		"source": {"huggingface": "internlm/internlm2-chat-1_8b-sft", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-1_8b-sft"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-20b": {
		"source": {"huggingface": "internlm/internlm2-chat-20b", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-20b-4bits": {
		"source": {"huggingface": "internlm/internlm2-chat-20b-4bits", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b-4bits"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-20b-sft": {
		"source": {"huggingface": "internlm/internlm2-chat-20b-sft", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-20b-sft"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-7b": {
		"source": {"huggingface": "internlm/internlm2-chat-7b", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-7b-4bits": {
		"source": {"huggingface": "internlm/internlm2-chat-7b-4bits", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b-4bits"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-chat-7b-sft": {
		"source": {"huggingface": "internlm/internlm2-chat-7b-sft", "modelscope": "Shanghai_AI_Laboratory/internlm2-chat-7b-sft"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-math-20b": {
		"source": {"huggingface": "internlm/internlm2-math-20b", "modelscope": "Shanghai_AI_Laboratory/internlm2-math-20b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm2-math-7b": {
		"source": {"huggingface": "internlm/internlm2-math-7b", "modelscope": "Shanghai_AI_Laboratory/internlm2-math-7b"},
		"prompt_keys": model_groups["internlm2"]["prompt_keys"],
		},
    "internlm-7b": {
		"source": {"huggingface": "internlm/internlm-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-7b"},
		"prompt_keys": model_groups["internlm"]["prompt_keys"],
		},
    "internlm-chat-20b": {
		"source": {"huggingface": "internlm/internlm-chat-20b", "modelscope": "Shanghai_AI_Laboratory/internlm-chat-20b"},
		"prompt_keys": model_groups["internlm"]["prompt_keys"],
		},
    "internlm-chat-20b-4bit": {
		"source": {"huggingface": "internlm/internlm-chat-20b-4bit", "modelscope": "Shanghai_AI_Laboratory/internlm-chat-20b-4bit"},
		"prompt_keys": model_groups["internlm"]["prompt_keys"],
        },
    "internlm-chat-7b": {
		"source": {"huggingface": "internlm/internlm-chat-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-chat-7b"},
		"prompt_keys": model_groups["internlm"]["prompt_keys"],
		},
    "internlm-xcomposer2-4khd-7b": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-4khd-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer2-7b": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-7b"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer2-7b-4bit": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-7b-4bit", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-7b-4bit"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer2-vl-1_8b": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-vl-1_8b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-1_8b"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer2-vl-7b": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-vl-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer2-vl-7b-4bit": {
		"source": {"huggingface": "internlm/internlm-xcomposer2-vl-7b-4bit", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b-4bit"},
		"prompt_keys": model_groups["internlm-xcomposer2"]["prompt_keys"],
		},
    "internlm-xcomposer-7b": {
		"source": {"huggingface": "internlm/internlm-xcomposer-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-7b"},
		"prompt_keys": {},
		},
    "internlm-xcomposer-7b-4bit": {
		"source": {"huggingface": "internlm/internlm-xcomposer-7b-4bit", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-7b-4bit"},
		"prompt_keys": {},
		},
    "internlm-xcomposer-vl-7b": {
		"source": {"huggingface": "internlm/internlm-xcomposer-vl-7b", "modelscope": "Shanghai_AI_Laboratory/internlm-xcomposer-vl-7b"},
		"prompt_keys": {},
		},

    "Baichuan-13B-Chat": {
		"source": {"huggingface": "baichuan-inc/Baichuan-13B-Chat", "modelscope": "baichuan-inc/Baichuan-13B-Chat"},
		"prompt_keys": {},
		},
    "Baichuan2-13B-Chat": {
		"source": {"huggingface": "baichuan-inc/Baichuan2-13B-Chat", "modelscope": "baichuan-inc/Baichuan2-13B-Chat"},
		"prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
		},
    "Baichuan2-13B-Chat-4bits": {
		"source": {"huggingface": "baichuan-inc/Baichuan2-13B-Chat-4bits", "modelscope": "baichuan-inc/Baichuan2-13B-Chat-4bits"},
		"prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
		},
    "Baichuan2-7B-Chat": {
		"source": {"huggingface": "baichuan-inc/Baichuan2-7B-Chat", "modelscope": "baichuan-inc/Baichuan2-7B-Chat"},
		"prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
		},
    "Baichuan2-7B-Chat-4bits": {
		"source": {"huggingface": "baichuan-inc/Baichuan2-7B-Chat-4bits", "modelscope": "baichuan-inc/Baichuan2-7B-Chat-4bits"},
		"prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
		},
    "Baichuan2-7B-Intermediate-Checkpoints": {
		"source": {"huggingface": "baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints", "modelscope": "baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints"},
		"prompt_keys": model_groups["Baichuan2"]["prompt_keys"],
		},
    "baichuan-7B": {
		"source": {"huggingface": "baichuan-inc/Baichuan-7B", "modelscope": "baichuan-inc/baichuan-7B"},
		"prompt_keys": {},
		},

    "bge-large-zh-v1.5": {
		"source": {"huggingface": "BAAI/bge-large-zh-v1.5", "modelscope": "AI-ModelScope/bge-large-zh-v1.5"},
        "type": "embed",
		},
    "bge-reranker-large": {
		"source": {"huggingface": "BAAI/bge-reranker-large", "modelscope": "Xorbits/bge-reranker-large"},
        "type": "embed",
		},
    "stable-diffusion-3-medium": {
        "source": {"huggingface": "stabilityai/stable-diffusion-3-medium", "modelscope": "AI-ModelScope/stable-diffusion-3-medium-diffusers"},
        "type": "sd",
	}
}

