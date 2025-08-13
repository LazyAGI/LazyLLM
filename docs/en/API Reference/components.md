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

::: lazyllm.components.deploy.LazyLLMDeployBase
    options:
      heading_level: 3

::: lazyllm.components.deploy.LazyLLMDeployBase.extract_result
    options:
      heading_level: 3
      
::: lazyllm.components.finetune.FlagembeddingFinetune
    options:
      heading_level: 3

::: lazyllm.components.auto.AutoFinetune
    options:
      heading_level: 3
::: lazyllm.components.finetune.base.DummyFinetune
    options:
      heading_level: 3
---

## Deploy

::: lazyllm.components.deploy.Lightllm
    options:
      heading_level: 3
      members: [cmd, geturl, extract_result]

::: lazyllm.components.deploy.Vllm
    options:
      heading_level: 3

::: lazyllm.components.deploy.LMDeploy
    options:
      heading_level: 3
      members: [cmd, geturl, extract_result]

::: lazyllm.components.deploy.base.DummyDeploy
    options:
      heading_level: 3

::: lazyllm.components.auto.AutoDeploy
    options:
      heading_level: 3

::: lazyllm.components.deploy.embed.AbstractEmbedding
    options:
      heading_level: 3

::: lazyllm.components.deploy.EmbeddingDeploy
    options:
      heading_level: 3

::: lazyllm.components.deploy.embed.RerankDeploy
    options:
      heading_level: 3

::: lazyllm.components.deploy.embed.LazyHuggingFaceRerank
    options:
      heading_level: 3
      members: [load_reranker, rebuild]

::: lazyllm.components.deploy.Mindie
    options:
      heading_level: 3

      
::: lazyllm.components.deploy.OCRDeploy
    options:
      heading_level: 3
---

::: lazyllm.components.deploy.relay.base.RelayServer
    options:
      heading_level: 3
      members: [cmd, geturl]

::: lazyllm.components.deploy.OCRDeploy
    options:
      heading_level: 3

---

## Prompter

::: lazyllm.components.prompter.LazyLLMPrompterBase
    options:
      heading_level: 3
    inherited_members:
      - generate_prompt
      - get_response
    members: [pre_hook]

::: lazyllm.components.prompter.EmptyPrompter
    options:
      heading_level: 3
      members: true

::: lazyllm.components.Prompter
    options:
      heading_level: 3
      members: [from_dict, from_template, from_file, empty, generate_prompt, get_response]
  options:
    heading_level: 3
    inherited_members:
      - generate_prompt
      - get_response
    members: false

::: lazyllm.components.prompter.EmptyPrompter
    options:
      heading_level: 3
      members: true

::: lazyllm.components.Prompter
    options:
      heading_level: 3
      members: [from_dict, from_template, from_file, empty, generate_prompt, get_response]

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

::: lazyllm.components.deploy.speech_to_text.sense_voice.SenseVoice
    options:
      heading_level: 4

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

::: lazyllm.components.formatter.formatterbase.JsonLikeFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.formatterbase.PythonFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.FileFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.YamlFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.encode_query_with_filepaths
    options:
      heading_level: 3

::: lazyllm.components.formatter.decode_query_with_filepaths
    options:
      heading_level: 3

::: lazyllm.components.formatter.lazyllm_merge_query
    options:
      heading_level: 3

::: lazyllm.components.formatter.formatterbase.JsonLikeFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.formatterbase.PythonFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.FileFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.YamlFormatter
    options:
      heading_level: 3

::: lazyllm.components.formatter.encode_query_with_filepaths
    options:
      heading_level: 3

::: lazyllm.components.formatter.decode_query_with_filepaths
    options:
      heading_level: 3

::: lazyllm.components.formatter.lazyllm_merge_query
    options:
      heading_level: 3

::: lazyllm.components.JsonFormatter
    options:
      heading_level: 3

::: lazyllm.components.EmptyFormatter
    options:
      heading_level: 3

---

## ComponentBase

::: lazyllm.components.core.ComponentBase
    options:
      heading_level: 3
      members: [apply, cmd]
