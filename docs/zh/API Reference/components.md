::: lazyllm.components.finetune
    members: 
    exclude-members:

::: lazyllm.components.auto
    members: AutoFinetune
    exclude-members:

::: lazyllm.components.deploy
    members: 
    exclude-members:

::: lazyllm.launcher
    options:
      members:
      - EmptyLauncher
      - RemoteLauncher
      - ScoLauncher
      - SlurmLauncher
      filters:
      - '!get_idle_nodes'

::: lazyllm.components.prompter.LazyLLMPrompterBase
    members: generate_prompt, get_response
    exclude-members:

::: lazyllm.components.AlpacaPrompter
    members: generate_prompt, get_response
    exclude-members:

::: lazyllm.components.ChatPrompter
    members: generate_prompt, get_response
    exclude-members:

::: lazyllm.common.Register
    members: 
    exclude-members:

::: lazyllm.components.ModelManager
    members: 
    exclude-members:

::: lazyllm.components.formatter.LazyLLMFormatterBase
    members:
    exclude-members:

::: lazyllm.components.JsonFormatter
    members:
    exclude-members:

::: lazyllm.components.EmptyFormatter
    members:
    exclude-members:
