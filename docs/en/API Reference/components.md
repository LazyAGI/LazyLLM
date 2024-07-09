lazyllm.Components

Finetune

::: lazyllm.components.finetune
    members: 
    exclude-members:

::: lazyllm.components.auto
    members: AutoFinetune
    exclude-members:

Deploy

::: lazyllm.components.deploy
    members: 
    exclude-members:

::: lazyllm.components.auto
    members: AutoDeploy
    exclude-members:

Launcher

::: lazyllm.launcher
    members: 
    exclude-members: Status, get_idle_nodes


Prompter

::: lazyllm.components.prompter.LazyLLMPrompterBase
    members: generate_prompt, get_response
    exclude-members:

::: lazyllm.components.AlpacaPrompter
    members: generate_prompt, get_response
    exclude-members:

::: lazyllm.components.ChatPrompter
    members: generate_prompt, get_response
    exclude-members:

.. _api.components.register:

Register

.. autofunction:: lazyllm.components.register

<!-- ModelManager

::: lazyllm.components.ModelManager
    members: 
    exclude-members: -->

Formatter

::: lazyllm.components.formatter.LazyLLMFormatterBase
    members:
    exclude-members:

::: lazyllm.components.JsonFormatter
    members:
    exclude-members:

::: lazyllm.components.EmptyFormatter
    members:
    exclude-members:
