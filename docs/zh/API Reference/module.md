::: lazyllm.module.ModuleBase
    options:
      members:
      - eval
      - evalset
      - forward
      - start
      - restart
      - update
      - stream_output
      - used_by
      - register_hook
      - unregister_hook
      - clear_hooks
      - update_server
      - wait
      - stop
      - for_each

::: lazyllm.module.servermodule.LLMBase
    options:
      members:
      - prompt
      - formatter
      - share

::: lazyllm.module.ActionModule
    options:
      members:
      - evalset
      - forward
      - submodules

::: lazyllm.module.TrainableModule
    options:
      members:
      - start
      - restart
      - update
      - evalset
      - eval
      - wait
      - stop
      - prompt
      - log_path
      - forward_openai
      - forward_standard
      - forward

::: lazyllm.module.UrlModule
    options:
      members:
      - forward

::: lazyllm.module.ServerModule
    options:
      members:
      - start
      - restart
      - evalset
      - wait
      - stop

::: lazyllm.module.AutoModel
    options:
      members:

::: lazyllm.module.TrialModule
    members: [start]
    exclude-members:

::: lazyllm.module.OnlineChatModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.ppio.PPIOModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoMultiModal
    members:
    exclude-members:

::: lazyllm.module.OnlineEmbeddingModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIEmbedding
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenSTTModule
    members:
    exclude-members:

::: lazyllm.module.OnlineChatModuleBase
    options:
      members:
      - set_train_tasks
      - set_specific_finetuned_model

::: lazyllm.module.OnlineEmbeddingModuleBase
    members:
        - forward
        - run_embed_batch

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoEmbedding
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoMultimodalEmbedding
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMTextToImageModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenTextToImageModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.kimi.KimiModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.fileHandler.FileHandlerBase
    members: get_finetune_data
    exclude-members: 

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMReranking
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMMultiModal
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenReranking
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenTTSModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.sensenova.SenseNovaModule
    members: set_deploy_parameters
    exclude-members:

::: lazyllm.module.llms.onlinemodule.base.onlineMultiModalBase.OnlineMultiModalBase
    members:
    exclude-members:
    options:
      members:
        - get_finetune_data 

::: lazyllm.module.llms.onlinemodule.base.utils.OnlineModuleBase
    members:
    exclude-members:

::: lazyllm.module.module.ModuleCache
    members: [get, set, close]
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenModule
    options:
      members:
        - set_deploy_parameters

::: lazyllm.module.llms.onlinemodule.supplier.qwen.QwenEmbedding
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMEmbedding
    options:
      members: 

::: lazyllm.module.llms.onlinemodule.supplier.glm.GLMSTTModule
    options:
      members: 

::: lazyllm.module.llms.onlinemodule.supplier.deepseek.DeepSeekModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoTextToImageModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIReranking
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.sensenova.SenseNovaEmbedding
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowTTSModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowReranking
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.siliconflow.SiliconFlowTextToImageModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingModule
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingEmbedding
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingReranking
    options:
      members:

::: lazyllm.module.llms.onlinemodule.supplier.aiping.AipingTextToImageModule
    options:
      members: