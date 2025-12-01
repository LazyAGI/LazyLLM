import pytest
from lazyllm.components.utils.downloader.model_directory import infer_model_type


test_models = {
    'vlm': [
        'rhymes-ai/Aria', 'CohereForAI/aya-vision-8b', 'CohereForAI/aya-vision-32b',
        'Open-Bee/Bee-8B-RL', 'Open-Bee/Bee-8B-SFT', 'Salesforce/blip2-opt-2.7b',
        'Salesforce/blip2-opt-6.7b', 'facebook/chameleon-7b', 'CohereLabs/command-a-vision-07-2025',
        'deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2',
        'deepseek-ai/DeepSeek-OCR', 'baidu/ERNIE-4.5-VL-28B-A3B-PT', 'baidu/ERNIE-4.5-VL-424B-A47B-PT',
        'adept/fuyu-8b', 'google/gemma-3-4b-it', 'google/gemma-3-27b-it', 'google/gemma-3n-E2B-it',
        'google/gemma-3n-E4B-it', 'zai-org/glm-4v-9b', 'zai-org/cogagent-9b-20241220', 'Kwai-Keye/Keye-VL-1_5-8B',
        'zai-org/GLM-4.1V-9B-Thinking', 'zai-org/GLM-4.5V', 'ibm-granite/granite-speech-3.3-8b',
        'h2oai/h2ovl-mississippi-800m', 'h2oai/h2ovl-mississippi-2b', 'HuggingFaceM4/Idefics3-8B-Llama3',
        'internlm/Intern-S1', 'internlm/Intern-S1-mini', 'OpenGVLab/InternVL3_5-14B', 'OpenGVLab/InternVL3-9B',
        'OpenGVLab/InternVideo2_5_Chat_8B', 'OpenGVLab/InternVL2_5-4B', 'OpenGVLab/Mono-InternVL-2B',
        'OpenGVLab/InternVL2-4B', 'OpenGVLab/InternVL3-1B-hf', 'Kwai-Keye/Keye-VL-8B-Preview',
        'moonshotai/Kimi-VL-A3B-Instruct', 'moonshotai/Kimi-VL-A3B-Thinking', 'lightonai/LightOnOCR-1B',
        'meta-llama/Llama-4-Scout-17B-16E-Instruct', 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct', 'nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1',
        'llava-hf/llava-1.5-7b-hf', 'TIGER-Lab/Mantis-8B-siglip-llama3', 'mistral-community/pixtral-12b',
        'llava-hf/llava-v1.6-mistral-7b-hf', 'llava-hf/llava-v1.6-vicuna-7b-hf', 'llava-hf/LLaVA-NeXT-Video-7B-hf',
        'llava-hf/llava-onevision-qwen2-7b-ov-hf', 'llava-hf/llava-onevision-qwen2-0.5b-ov-hf', 'YannQi/R-4B',
        'mispeech/midashenglm-7b', 'openbmb/MiniCPM-o-2_6', 'openbmb/MiniCPM-V-2', 'BAAI/Emu3-Chat-hf',
        'openbmb/MiniCPM-Llama3-V-2_5', 'openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-4', 'openbmb/MiniCPM-V-4_5',
        'MiniMaxAI/MiniMax-VL-01', 'mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'allenai/Molmo-7B-D-0924',
        'allenai/Molmo-7B-O-0924', 'nvidia/NVLM-D-72B', 'AIDC-AI/Ovis2-1B', 'AIDC-AI/Ovis1.6-Llama3.2-3B',
        'AIDC-AI/Ovis2.5-9B', 'google/paligemma-3b-pt-224', 'google/paligemma-3b-mix-224', 'stepfun-ai/step3',
        'google/paligemma2-3b-ft-docci-448', 'microsoft/Phi-3-vision-128k-instruct', 'Skywork/Skywork-R1V-38B',
        'microsoft/Phi-3.5-vision-instruct', 'microsoft/Phi-4-multimodal-instruct', 'SmolVLM2-2.2B-Instruct',
        'mistralai/Mistral-Small-3.1-24B-Instruct-2503', 'mistralai/Pixtral-12B-2409', 'Qwen/Qwen-VL',
        'Qwen/Qwen-VL-Chat', 'Qwen/Qwen2-Audio-7B-Instruct', 'Qwen/QVQ-72B-Preview', 'Qwen/Qwen2-VL-7B-Instruct',
        'Qwen/Qwen2-VL-72B-Instruct', 'Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct',
        'Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'Qwen/Qwen3-Omni-30B-A3B-Instruct', 'Qwen/Qwen3-Omni-30B-A3B-Thinking', 'omni-search/Tarsier-7b',
        'omni-search/Tarsier-34b', 'omni-research/Tarsier2-Recap-7b', 'omni-research/Tarsier2-7b-0115',
        'naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B'],
    'stt': [
        'openai/whisper-small', 'openai/whisper-large-v3-turbo', 'mistralai/Voxtral-Mini-3B-2507',
        'mistralai/Voxtral-Small-24B-2507', 'FunAudioLLM/SenseVoiceSmall'],
    'cross_modal_embed': [
        'openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14', 'royokong/e5-v', 'TIGER-Lab/VLM2Vec-Full',
        'google/siglip-base-patch16-224'],
    'rerank': [
        'cross-encoder/ms-marco-MiniLM-L-6-v2', 'BAAI/bge-reranker-v2-gemma',
        'Alibaba-NLP/gte-multilingual-reranker-base', 'mixedbread-ai/mxbai-rerank-base-v2',
        'tomaarsen/Qwen3-Reranker-0.6B-seq-cls', 'Qwen/Qwen3-Reranker-0.6B',
        'cross-encoder/quora-roberta-base', 'BAAI/bge-reranker-v2-m3'],
    'embed': [
        'BAAI/bge-base-en-v1.5', 'Snowflake/snowflake-arctic-embed-xs', 'BAAI/bge-multilingual-gemma2',
        'google/embeddinggemma-300m', 'Snowflake/snowflake-arctic-embed-m-v2.0', 'Alibaba-NLP/gte-multilingual-base',
        'Alibaba-NLP/gte-modernbert-base', 'nomic-ai/nomic-embed-text-v1', 'nomic-ai/nomic-embed-text-v2-moe',
        'Snowflake/snowflake-arctic-embed-m-long', 'intfloat/e5-mistral-7b-instruct', 'Qwen/Qwen3-Embedding-0.6B',
        'ssmits/Qwen2-7B-Instruct-embed-base', 'Alibaba-NLP/gte-Qwen2-7B-instruct',
        'sentence-transformers/all-roberta-large-v1'],
    'tts': [
        'suno/bark', '2Noise/ChatTTS', 'facebook/musicgen-medium', 'facebook/musicgen-stereo-small',
        'hexgrad/Kokoro-82M', 'coqui/XTTS-v2', 'nari-labs/Dia-1.6B', 'sesame/csm-1b', 'microsoft/VibeVoice-1.5B',
        'ResembleAI/chatterbox', 'SWivid/F5-TTS', 'microsoft/speecht5_tts', 'SparkAudio/Spark-TTS-0.5B',
        'fishaudio/fish-speech-1.5'],
    'ocr': [
        'pp-ocrv4_server', 'pp-ocrv5_mobile', 'allenai/olmOCR-7B-0225-preview', 'reducto/RolmOCR',
        'rednote-hilab/dots.ocr'],
    'sd': [
        'Qwen/Qwen-Image-Edit-2509', 'Qwen/Qwen-Image-Edit', 'black-forest-labs/FLUX.1-Kontext-dev',
        'peteromallet/Qwen-Image-Edit-InSubject', 'QuantStack/Qwen-Image-Edit-2509-GGUF',
        'InstantX/Qwen-Image-ControlNet-Inpainting', 'timbrooks/instruct-pix2pix', 'lllyasviel/sd-controlnet-canny',
        'nunchaku-tech/nunchaku-flux.1-kontext-dev', 'stabilityai/stable-diffusion-2-inpainting',
        'jasperai/Flux.1-dev-Controlnet-Upscaler', 'QuantStack/FLUX.1-Kontext-dev-GGUF',
        'stepfun-ai/Step1X-Edit-v1p2-preview', 'yisol/IDM-VTON', 'CompVis/stable-diffusion-v1-4',
        'stabilityai/stable-diffusion-3-medium', 'black-forest-labs/FLUX.1-schnell', 'prompthero/openjourney',
        'hakurei/waifu-diffusion', 'ByteDance/SDXL-Lightning', 'dreamlike-art/dreamlike-photoreal-2.0',
        'ByteDance/Hyper-SD', 'tencent/HunyuanVideo', 'Wan-AI/Wan2.1-T2V-14B', 'genmo/mochi-1-preview',
        'ByteDance/AnimateDiff-Lightning', 'zai-org/CogVideoX-5b', 'ali-vilab/text-to-video-ms-1.7b',
        'lightx2v/Wan2.2-Lightning', 'stepfun-ai/stepvideo-t2v', 'Wan-AI/Wan2.2-TI2V-5B', 'zai-org/CogVideoX-2b',
        'Lightricks/LTX-Video', 'stabilityai/stable-video-diffusion-img2vid', 'Wan-AI/Wan2.1-I2V-14B-720P',
        'Wan-AI/Wan2.1-VACE-14B'],
    'llm': [
        'swiss-ai/Apertus-8B-2509', 'swiss-ai/Apertus-70B-Instruct-2509', 'BAAI/Aquila-7B', 'BAAI/AquilaChat-7B',
        'arcee-ai/AFM-4.5B-Base', 'Snowflake/snowflake-arctic-base', 'Snowflake/snowflake-arctic-instruct',
        'baichuan-inc/Baichuan2-13B-Chat', 'baichuan-inc/Baichuan-7B', 'inclusionAI/Ling-lite-1.5',
        'inclusionAI/Ling-plus', 'inclusionAI/Ling-mini-2.0', 'ibm-ai-platform/Bamba-9B-fp8',
        'ibm-ai-platform/Bamba-9B', 'bigscience/bloom', 'bigscience/bloomz', 'zai-org/chatglm2-6b',
        'zai-org/chatglm3-6b', 'ShieldLM-6B-chatglm3', 'CohereLabs/c4ai-command-r-v01', 'google/gemma-2b',
        'CohereLabs/c4ai-command-r7b-12-2024', 'CohereLabs/c4ai-command-a-03-2025', 'tiiuae/falcon-40b',
        'CohereLabs/command-a-reasoning-08-2025', 'databricks/dbrx-base', 'databricks/dbrx-instruct',
        'nvidia/Llama-3_3-Nemotron-Super-49B-v1', 'deepseek-ai/deepseek-llm-67b-base', 'tiiuae/falcon-rw-7b',
        'deepseek-ai/deepseek-llm-7b-chat', 'deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2-Chat',
        'deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-V3.1', 'ibm/PowerMoE-3b',
        'rednote-hilab/dots.llm1.base', 'rednote-hilab/dots.llm1.inst', 'baidu/ERNIE-4.5-0.3B-PT',
        'baidu/ERNIE-4.5-21B-A3B-PT', 'baidu/ERNIE-4.5-300B-A47B-PT', 'LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct',
        'LGAI-EXAONE/EXAONE-4.0-32B', 'mgleize/fairseq2-dummy-Llama-3.2-1B', 'tiiuae/falcon-7b',
        'tiiuae/falcon-mamba-7b', 'tiiuae/falcon-mamba-7b-instruct', 'tiiuae/Falcon-H1-34B-Base',
        'tiiuae/Falcon-H1-34B-Instruct', 'allenai/FlexOlmo-7x7B-1T', 'allenai/FlexOlmo-7x7B-1T-RT',
        'google/gemma-1.1-2b-it', 'google/gemma-2-9b', 'google/gemma-2-27b', 'zai-org/glm-4-9b-chat-hf',
        'zai-org/GLM-4-32B-0414', 'zai-org/GLM-4.5', 'gpt2', 'gpt2-xl', 'bigcode/starcoder', 'hpcai-tech/grok-1',
        'bigcode/gpt_bigcode-santacoder', 'WizardLM/WizardCoder-15B-V1.0', 'EleutherAI/gpt-j-6b',
        'nomic-ai/gpt4all-j', 'EleutherAI/gpt-neox-20b', 'EleutherAI/pythia-12b', 'Qwen/Qwen1.5-MoE-A2.7B',
        'OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5', 'databricks/dolly-v2-12b', 'Qwen/Qwen3-8B',
        'stabilityai/stablelm-tuned-alpha-7b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'Qwen/Qwen2-7B',
        'ibm-granite/granite-3.0-2b-base', 'ibm-granite/granite-3.1-8b-instruct', 'ibm/PowerLM-3b', 'CofeAI/Tele-FLM',
        'ibm-granite/granite-3.0-1b-a400m-base', 'ibm-granite/granite-3.0-3b-a800m-instruct', 'Tele-AI/TeleChat2-7B',
        'ibm-granite/granite-4.0-tiny-preview', 'ibm-research/moe-7b-1b-active-shared-experts', 'Qwen/Qwen3-30B-A3B',
        'tencent/Hunyuan-7B-Instruct', 'tencent/Hunyuan-A13B-Instruct', 'tencent/Hunyuan-A13B-Pretrain',
        'tencent/Hunyuan-A13B-Instruct-FP8', 'internlm/internlm-7b', 'internlm/internlm-chat-7b',
        'internlm/internlm2-7b', 'internlm/internlm2-chat-7b', 'internlm/internlm3-8b-instruct',
        'inceptionai/jais-13b', 'inceptionai/jais-13b-chat', 'inceptionai/jais-30b-v3', 'microsoft/phi-1_5',
        'inceptionai/jais-30b-chat-v3', 'ai21labs/AI21-Jamba-1.5-Large', 'ai21labs/AI21-Jamba-1.5-Mini',
        'ai21labs/Jamba-v0.1', 'LiquidAI/LFM2-1.2B', 'LiquidAI/LFM2-700M', 'LiquidAI/LFM2-350M', 'microsoft/phi-2',
        'LiquidAI/LFM2-8B-A1B-preview', 'meta-llama/Meta-Llama-3.1-405B-Instruct', 'meta-llama/Meta-Llama-3.1-70B',
        'meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Llama-2-70b-hf', '01-ai/Yi-34B', 'Qwen/Qwen-7B-Chat',
        'state-spaces/mamba-130m-hf', 'state-spaces/mamba-790m-hf', 'state-spaces/mamba-2.8b-hf', 'Qwen/Qwen-7B',
        'mistralai/Mamba-Codestral-7B-v0.1', 'XiaomiMiMo/MiMo-7B-RL', 'openbmb/MiniCPM-2B-sft-bf16',
        'openbmb/MiniCPM-2B-dpo-bf16', 'openbmb/MiniCPM-S-1B-sft', 'openbmb/MiniCPM3-4B', 'Qwen/QwQ-32B-Preview',
        'mistralai/Mistral-7B-v0.1', 'mistralai/Mistral-7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-v0.1',
        'mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistral-community/Mixtral-8x22B-v0.1', 'mosaicml/mpt-7b',
        'mosaicml/mpt-7b-storywriter', 'mosaicml/mpt-30b', 'nvidia/Minitron-8B-Base', 'adept/persimmon-8b-base',
        'mgoin/Nemotron-4-340B-Base-hf-FP8', 'nvidia/Nemotron-H-8B-Base-8K', 'nvidia/Nemotron-H-47B-Base-8K',
        'nvidia/Nemotron-H-56B-Base-8K', 'allenai/OLMo-1B-hf', 'allenai/OLMo-7B-hf', 'allenai/OLMo-2-0425-1B',
        'allenai/OLMoE-1B-7B-0924', 'allenai/OLMoE-1B-7B-0924-Instruct', 'facebook/opt-66b', 'Qwen/Qwen2-7B-Instruct',
        'facebook/opt-iml-max-30b', 'OrionStarAI/Orion-14B-Base', 'OrionStarAI/Orion-14B-Chat', 'pfnet/plamo-2-8b',
        'microsoft/Phi-4-mini-instruct', 'microsoft/Phi-4', 'microsoft/Phi-3-mini-4k-instruct', 'pfnet/plamo-2-1b',
        'microsoft/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct', 'microsoft/Phi-3.5-MoE-instruct',
        'adept/persimmon-8b-chat', 'Qwen/Qwen1.5-MoE-A2.7B-Chat', 'Qwen/Qwen3-Next-80B-A3B-Instruct',
        'ByteDance-Seed/Seed-OSS-36B-Instruct', 'stabilityai/stablelm-3b-4e1t', 'bigcode/starcoder2-7b',
        'stabilityai/stablelm-base-alpha-7b-v2', 'bigcode/starcoder2-3b', 'bigcode/starcoder2-15b',
        'upstage/solar-pro-preview-instruct', 'Tele-AI/TeleChat2-3B', 'Tele-AI/TeleChat2-35B',
        'CofeAI/FLM-2-52B-Instruct-2407', 'xverse/XVERSE-7B-Chat', 'xverse/XVERSE-13B-Chat', 'xverse/XVERSE-65B-Chat',
        'MiniMaxAI/MiniMax-M1-40k', 'MiniMaxAI/MiniMax-M1-80k', 'MiniMaxAI/MiniMax-Text-01',
        'Zyphra/Zamba2-7B-instruct', 'Zyphra/Zamba2-2.7B-instruct', 'Zyphra/Zamba2-1.2B-instruct',
        'meituan-longcat/LongCat-Flash-Chat', 'meituan-longcat/LongCat-Flash-Chat-FP8', 'HuggingFaceTB/SmolLM3-3B'],
}

class TestModelTypeInference:

    def infer_model_type(self, rename_func=None):
        errors = []
        total_tests = 0

        for expected_type, model_list in test_models.items():
            for model_name in model_list:
                total_tests += 1
                try:
                    if rename_func:
                        model_name = rename_func(model_name)
                    inferred_type = infer_model_type(model_name)
                    if inferred_type != expected_type:
                        errors.append(f'❌ {model_name} misclassified as {inferred_type}, expected {expected_type}')
                except Exception as e:
                    errors.append(f'❌❌ {model_name} -> {e}')

        if errors:
            error_summary = f'\nTest models: {len(errors)}/{total_tests} failed.\n'
            error_summary += '\n'.join(errors)
            pytest.fail(error_summary)

    def test_all_models_original_names(self):
        self.infer_model_type()

    def test_all_models_short_names(self):
        def shorten_name(name):
            return name.split('/')[-1]

        self.infer_model_type(rename_func=shorten_name)
