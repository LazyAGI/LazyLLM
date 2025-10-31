import lazyllm
import logging
logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    # ===== OpenAI (LLM) =====
    "gpt-3.5-turbo-0125": "llm",
    "gpt-3.5-turbo-1106": "llm",
    "gpt-3.5-turbo-0613": "llm",
    "babbage-002": "llm",
    "davinci-002": "llm",
    "gpt-4-0613": "llm",

    # ===== SenseNova =====
    "nova-ptc-s-v2": "llm",  
    "SenseNova-V6-Turbo": "multimodal",
    "SenseChat-Vision": "multimodal",
    "SenseNova-V6-Pro": "multimodal",
    "SenseNova-V6-Reasoner": "multimodal",
    "SenseNova-V6-5-Pro": "multimodal",
    "SenseNova-V6-5-Turbo": "multimodal",

    # ===== GLM =====
    "chatglm3-6b": "llm",
    "chatglm_12b": "llm",
    "chatglm_32b": "llm",
    "chatglm_66b": "llm",
    "chatglm_130b": "llm",
    "glm-4.5v": "multimodal",
    "glm-4.1v": "multimodal",
    "glm-4v": "multimodal",

    # ===== Kimi (Moonshot) =====
    "moonshot-v1-8k": "llm",
    "moonshot-v1-32k": "llm",

    # ===== Qwen =====
    "qwen-turbo": "llm",
    "qwen-7b-chat": "llm",
    "qwen-72b-chat": "llm",
    "qwen-plus": "llm",
    "qwen-vl-plus": "multimodal",
    "qwen-vl-max": "multimodal",
    "qvq-max": "multimodal",
    "qvq-plus": "multimodal",

    # ===== Doubao (豆包) =====
    "doubao-1-5-pro-32k-250115": "llm",
    "doubao-seed-1-6-vision": "multimodal",
    "doubao-1-5-ui-tars": "multimodal",

    # ===== DeepSeek =====
    "deepseek-chat": "llm",

    # ===== Defauly =====
    "default": "llm"
}
def get_model_category(params: str) -> str:
        model_name = params.model
        source = params.source
        api_key = params.api_key
        try:
            category = _try_get_category_from_api(api_key)
            if category:
                return category
        except Exception as e:
            logger.debug(f"API category lookup failed for {model_name} ({source}): {e}")

        if model_name in MODEL_MAPPING:
            return MODEL_MAPPING[model_name]
        
        if "embedding" in model_name.lower():
            return "embedding"
        if "whisper" in model_name.lower() or "paraformer" in model_name.lower():
            return "sst"
        if "tts" in model_name.lower():
            return "tts"
        if "vl" in model_name.lower() or "vision" in model_name.lower():
            return "multimodal"
        if "rerank" in model_name.lower():
            return "rerank"
        if "cross_model_embed" in model_name.lower():
            return "cross_model_embed"
        if "ocr" in model_name.lower():
            return "ocr"
       
        return MODEL_MAPPING["default"]

def _try_get_category_from_api(self, model_name: str, source: str) -> str:
    return None




