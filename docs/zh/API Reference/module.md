
::: lazyllm.module.ModuleBase
    options:
      members:
      - _get_deploy_tasks
      - _get_train_tasks
      - eval
      - evalset
      - forward
      - start
      - restart
      - update

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

::: lazyllm.module.OnlineEmbeddingModule
    members:
    exclude-members:

::: lazyllm.module.llms.onlinemodule.supplier.openai.OpenAIEmbedding
    members:
    exclude-members:

::: lazyllm.module.OnlineChatModuleBase
    options:
      members:
      filters:
      - '!forward'

::: lazyllm.module.OnlineEmbeddingModuleBase
    members:
    exclude-members: forward

::: lazyllm.module.llms.onlinemodule.supplier.doubao.DoubaoEmbedding
    options:
      members:
    
::: lazyllm.module.llms.onlinemodule.fileHandler.FileHandlerBase
    members: get_finetune_data
    exclude-members: 
