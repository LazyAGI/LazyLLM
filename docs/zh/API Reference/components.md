## Finetune

::: lazyllm.components.finetune.AlpacaloraFinetune
    options:
      heading_level: 3

::: lazyllm.components.finetune.CollieFinetune
    options:
      heading_level: 3

::: lazyllm.components.finetune.LlamafactoryFinetune
    options:
      heading_level: 3

::: lazyllm.components.finetune.FlagembeddingFinetune
    options:
      heading_level: 3

::: lazyllm.components.auto.AutoFinetune
    options:
      heading_level: 3

---

## Deploy

::: lazyllm.components.deploy.Lightllm
    options:
      heading_level: 3

::: lazyllm.components.deploy.Vllm
    options:
      heading_level: 3

::: lazyllm.components.deploy.LMDeploy
    options:
      heading_level: 3

::: lazyllm.components.auto.AutoDeploy
    options:
      heading_level: 3

---

## Launcher

::: lazyllm.launcher.EmptyLauncher
    options:
      heading_level: 3

::: lazyllm.launcher.RemoteLauncher
    options:
      heading_level: 3

::: lazyllm.launcher.SlurmLauncher
    options:
      heading_level: 3
      filters:
      - '!get_idle'

::: lazyllm.launcher.ScoLauncher
    options:
      heading_level: 3

---

## Prompter

::: lazyllm.components.prompter.LazyLLMPrompterBase
    options:
      heading_level: 3

::: lazyllm.components.AlpacaPrompter
    options:
      heading_level: 3
	  inherited_members:
	    - generate_prompt
	    - get_response
      members: false

::: lazyllm.components.ChatPrompter
    options:
      heading_level: 3
	  inherited_members:
	    - generate_prompt
	    - get_response
      members: false

---

## Register

::: lazyllm.common.Register
    options:
      heading_level: 3

---

## ModelManager

::: lazyllm.components.ModelManager
    options:
      heading_level: 3

---

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

---

## MultiModal

### Text to Image

::: lazyllm.components.StableDiffusionDeploy
    options:
      heading_level: 4

### Visual Question Answering

Reference [LMDeploy][lazyllm.components.deploy.LMDeploy], which supports the Visual Question Answering model.

### Text to Sound

::: lazyllm.components.TTSDeploy
    options:
      heading_level: 4

::: lazyllm.components.ChatTTSDeploy
    options:
      heading_level: 4

::: lazyllm.components.BarkDeploy
    options:
      heading_level: 4

::: lazyllm.components.MusicGenDeploy
    options:
      heading_level: 4

### Speech to Text

::: lazyllm.components.SenseVoiceDeploy
    options:
      heading_level: 4
