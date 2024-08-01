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
      - '!get_idle'

## Prompter

::: lazyllm.components.prompter.LazyLLMPrompterBase
    options:
      heading_level: 3

::: lazyllm.components.AlpacaPrompter
    options:
      heading_level: 3

::: lazyllm.components.ChatPrompter
    options:
      heading_level: 3

::: lazyllm.common.Register
    members: 
    exclude-members:

::: lazyllm.components.ModelManager
    members: 
    exclude-members:

## Formatter

::: lazyllm.components.formatter.LazyLLMFormatterBase
    options:
      heading_level: 3

::: lazyllm.components.JsonFormatter
    options:
      heading_level: 3

::: lazyllm.components.EmptyFormatter
    options:
      heading_level: 3
